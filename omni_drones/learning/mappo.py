# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.stats import entropy
from tensordict import TensorDict
from tensordict.utils import expand_right
from tensordict.nn import make_functional, TensorDictModule, TensorDictParams
from torch.optim import lr_scheduler

from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    MultiDiscreteTensorSpec,
    DiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec as UnboundedTensorSpec,
)

from omni_drones.utils.torchrl.env import AgentSpec

from .utils import valuenorm
from .utils.gae import compute_gae
from ..controllers.Lee_ctrl import LeeController
from ..controllers.Trajectory import DroneTrajectory
from ..utils.torch import quaternion_to_rotation_matrix, quaternion_to_euler

LR_SCHEDULER = lr_scheduler._LRScheduler


class MAPPOPolicy(object):
    def __init__(
        self, cfg, agent_spec: AgentSpec, device="cuda"
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        print(self.agent_spec.observation_spec)
        print(self.agent_spec.action_spec)

        self.clip_param = cfg.clip_param
        self.ppo_epoch = int(cfg.ppo_epochs)
        self.num_minibatches = int(cfg.num_minibatches)
        self.normalize_advantages = cfg.normalize_advantages

        self.entropy_coef = cfg.entropy_coef
        self.gae_gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda

        self.act_dim = agent_spec.action_spec.shape[-1]

        if cfg.reward_weights is not None:
            self.reward_weights = torch.as_tensor(cfg.reward_weights, device=device).float()
        else:
            self.reward_weights = torch.ones(
                self.agent_spec.reward_spec.shape, device=device
            )

        self.obs_name = ("agents", "observation")
        self.act_name = ("agents", "action")
        self.reward_name = ("agents", "reward")

        self.make_actor()
        self.make_critic()

        self.train_in_keys = list(
            set(
                self.actor_in_keys
                + self.actor_out_keys
                + self.critic_in_keys
                + self.critic_out_keys
                + [
                    "next",
                    self.act_logps_name,
                    ("reward", self.reward_name),
                    "state_value",
                ]
                + ["progress", ("collector", "traj_ids")]
            )
        )

        self.n_updates = 0
        self.max_t = 1000
        self.trajs = [None] * 4
        self.ctrl = LeeController(0.035, 0.6685, 0, [4, 4, 2])

    @property
    def act_logps_name(self):
        return f"{self.agent_spec.name}.action_logp"

    def make_actor(self):
        cfg = self.cfg.actor

        self.actor_in_keys = [self.obs_name, self.act_name]
        self.actor_out_keys = [
            self.act_name,
            self.act_logps_name,
            f"{self.agent_spec.name}.action_entropy",
        ]

        create_actor_fn = lambda: TensorDictModule(
            make_ppo_actor(
                cfg, self.agent_spec.observation_spec, self.agent_spec.action_spec
            ),
            in_keys=self.actor_in_keys,
            out_keys=self.actor_out_keys
        ).to(self.device)

        if self.cfg.share_actor:
            self.actor = create_actor_fn()
            self.actor_params = TensorDictParams(make_functional(self.actor))
        else:
            actors = nn.ModuleList([create_actor_fn() for _ in range(self.agent_spec.n)])
            self.actor = actors[0]
            stacked_params = torch.stack([make_functional(actor) for actor in actors])
            self.actor_params = TensorDictParams(stacked_params.to_tensordict())
        
        self.actor_opt = torch.optim.Adam(self.actor_params.parameters(), lr=cfg.lr)

    def make_critic(self):
        cfg = self.cfg.critic

        if cfg.use_huber_loss:
            self.critic_loss_fn = nn.HuberLoss(delta=cfg.huber_delta)
        else:
            self.critic_loss_fn = nn.MSELoss()

        assert self.cfg.critic_input in ("state", "obs")
        if self.cfg.critic_input == "state" and self.agent_spec.state_spec is not None:
            self.critic_in_keys = ["state"]
            self.critic_out_keys = ["state_value"]
            reward_spec = self.agent_spec.reward_spec
            reward_spec = reward_spec.expand(self.agent_spec.n, *reward_spec.shape)
            critic = make_critic(cfg, self.agent_spec.state_spec, reward_spec, centralized=True)
            self.critic = TensorDictModule(
                critic,
                in_keys=self.critic_in_keys,
                out_keys=self.critic_out_keys,
            ).to(self.device)
            self.value_func = self.critic
        else:
            self.critic_in_keys = [self.obs_name]
            self.critic_out_keys = ["state_value"]
            critic = make_critic(cfg, self.agent_spec.observation_spec, self.agent_spec.reward_spec, centralized=False)
            self.critic = TensorDictModule(
                critic,
                in_keys=self.critic_in_keys,
                out_keys=self.critic_out_keys,
            ).to(self.device)
            self.value_func = torch.vmap(self.critic, in_dims=1, out_dims=1)

        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = cfg.lr_scheduler
        if scheduler is not None:
            scheduler = eval(scheduler)
            self.critic_opt_scheduler: LR_SCHEDULER = scheduler(
                self.critic_opt, **cfg.lr_scheduler_kwargs
            )

        if hasattr(cfg, "value_norm") and cfg.value_norm is not None:
            # The original MAPPO implementation uses ValueNorm1 with a very large beta,
            # and normalizes advantages at batch level.
            # Tianshou (https://github.com/thu-ml/tianshou) uses ValueNorm2 with subtract_mean=False,
            # and normalizes advantages at mini-batch level.
            # Empirically the performance is similar on most of the tasks.
            cls = getattr(valuenorm, cfg.value_norm["class"])
            self.value_normalizer: valuenorm.Normalizer = cls(
                input_shape=self.agent_spec.reward_spec.shape[-2:],
                **cfg.value_norm["kwargs"],
            ).to(self.device)

    def value_op(self, tensordict: TensorDict) -> TensorDict:
        critic_input = tensordict.select(*self.critic_in_keys, strict=False)
        if self.cfg.critic_input == "obs":
            critic_input.batch_size = [*critic_input.batch_size, self.agent_spec.n]
        tensordict = self.value_func(critic_input)
        return tensordict

    def __call__(self, tensordict: TensorDict, deterministic: bool = False):
        state = tensordict['info']['drone_state']
        t = tensordict['agents','state','payload'][0,0,-1].item() * self.max_t

        if t < 5:
            self.trajs = [None] * 4
            tensordict['agents']['action'] = torch.zeros(1, 4, 4, device=self.device)
            return tensordict

        t = t * 0.01
        p = state[0,:,:3].cpu().numpy()
        quat = state[0,:,3:7]
        v = state[0,:,7:10].cpu().numpy()

        if self.trajs[0] is None:
            for i in range(4):
                goal = p[i] + 0.5
                self.trajs[i] = DroneTrajectory([p[i], np.zeros(3), np.zeros(3), 0], [goal, np.zeros(3), np.zeros(3), 0], t, 5)

        tensordict['agents']['action'] = torch.zeros(1,4,4, device=self.device)

        for i in range(4):
            rot = quaternion_to_rotation_matrix(quat[i]).cpu().numpy()
            yaw = quaternion_to_euler(quat[i])[..., -1]
            vec_rot = np.hstack([rot[:, 0], rot[:, 1], rot[:, 2]])

            p_d, v_d, a_d, yaw_d = self.trajs[i].get_trajectory(t, rotation=False)
            cur_state = [p[i], v[i], vec_rot]
            des_state = [p_d, v_d, a_d, yaw_d]

            cmd, _ = self.ctrl.compute_control(cur_state, des_state, type="norm_input")
            thrust = cmd[..., 0]
            omega = cmd[..., 1:4]

            tensordict['agents']['action'][..., 0:3] = torch.tensor(omega/ np.pi, device='cuda')
            tensordict['agents']['action'][..., 3] = torch.tensor(thrust / 0.6685, device='cuda')

        actor_input = tensordict.select(*self.actor_in_keys, strict=False)
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]
        # actor_output = torch.vmap(self.actor, in_dims=(1, 0), out_dims=1, randomness="different")(
        #     actor_input, self.actor_params, deterministic=deterministic
        # )
        #
        # tensordict.update(actor_output)
        tensordict.update(self.value_op(tensordict))
        return tensordict

    def update_actor(self, batch: TensorDict) -> Dict[str, Any]:
        advantages = batch["advantages"]
        actor_input = batch.select(*self.actor_in_keys)
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]

        log_probs_old = batch[self.act_logps_name]
        actor_output = torch.vmap(self.actor, in_dims=(1, 0), out_dims=1)(
            actor_input, self.actor_params, eval_action=True
        )

        log_probs_new = actor_output[self.act_logps_name]
        dist_entropy = actor_output[f"{self.agent_spec.name}.action_entropy"]

        assert advantages.shape == log_probs_new.shape == dist_entropy.shape

        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )
        policy_loss = - torch.mean(torch.min(surr1, surr2) * self.act_dim)
        entropy_loss = torch.mean(dist_entropy)

        self.actor_opt.zero_grad()
        (policy_loss + entropy_loss * self.cfg.entropy_coef).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor_opt.param_groups[0]["params"], self.cfg.max_grad_norm
        )
        self.actor_opt.step()

        ess = (2 * ratio.logsumexp(0) - (2 * ratio).logsumexp(0)).exp().mean() / ratio.shape[0]
        return {
            "policy_loss": policy_loss.item(),
            "actor_grad_norm": grad_norm.item(),
            "entropy": - entropy_loss.item(),
            "ESS": ess.item()
        }

    def update_critic(self, batch: TensorDict) -> Dict[str, Any]:
        critic_input = batch.select(*self.critic_in_keys)
        values = self.value_op(critic_input)["state_value"]
        b_values = batch["state_value"]
        b_returns = batch["returns"]
        assert values.shape == b_values.shape == b_returns.shape
        value_pred_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )

        value_loss_clipped = self.critic_loss_fn(b_returns, value_pred_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)

        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_loss.backward()  # do not multiply weights here
        grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.cfg.max_grad_norm
        )
        self.critic_opt.step()
        self.critic_opt.zero_grad(set_to_none=True)
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return {
            "value_loss": value_loss.mean(),
            "critic_grad_norm": grad_norm.item(),
            "explained_var": explained_var.item()
        }

    def _get_dones(self, tensordict: TensorDict):
        env_done = tensordict[("next", "done")].unsqueeze(-1)
        agent_done = tensordict.get(
            ("next", f"{self.agent_spec.name}.done"),
            env_done.expand(*env_done.shape[:-2], self.agent_spec.n, 1),
        )
        done = agent_done | env_done
        return done

    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.select(*self.train_in_keys, strict=False)
        next_tensordict = tensordict["next"][:, -1]
        with torch.no_grad():
            value_output = self.value_op(next_tensordict)

        rewards = tensordict.get(("next", *self.reward_name))
        if rewards.shape[-1] != 1:
            rewards = rewards.sum(-1, keepdim=True)

        values = tensordict["state_value"]
        next_value = value_output["state_value"].squeeze(0)

        if hasattr(self, "value_normalizer"):
            values = self.value_normalizer.denormalize(values)
            next_value = self.value_normalizer.denormalize(next_value)

        dones = self._get_dones(tensordict)

        tensordict["advantages"], tensordict["returns"] = compute_gae(
            rewards,
            dones,
            values,
            next_value,
            gamma=self.gae_gamma,
            lmbda=self.gae_lambda,
        )

        advantages_mean = tensordict["advantages"].mean()
        advantages_std = tensordict["advantages"].std()
        if self.normalize_advantages:
            tensordict["advantages"] = (tensordict["advantages"] - advantages_mean) / (
                advantages_std + 1e-8
            )

        if hasattr(self, "value_normalizer"):
            self.value_normalizer.update(tensordict["returns"])
            tensordict["returns"] = self.value_normalizer.normalize(
                tensordict["returns"]
            )

        train_info = []
        for ppo_epoch in range(self.ppo_epoch):
            dataset = make_dataset_naive(
                tensordict,
                int(self.cfg.num_minibatches),
                self.minibatch_seq_len if hasattr(self, "minibatch_seq_len") else 1,
            )
            for minibatch in dataset:
                train_info.append(
                    TensorDict(
                        {
                            **self.update_actor(minibatch),
                            **self.update_critic(minibatch),
                        },
                        batch_size=[],
                    )
                )

        train_info = {k: v.mean().item() for k, v in torch.stack(train_info).items()}
        train_info["advantages_mean"] = advantages_mean.item()
        train_info["advantages_std"] = advantages_std.item()
        if isinstance(self.agent_spec.action_spec, (BoundedTensorSpec, UnboundedTensorSpec)):
            train_info["action_norm"] = tensordict[self.act_name].norm(dim=-1).mean().item()
        if hasattr(self, "value_normalizer"):
            train_info["value_running_mean"] = self.value_normalizer.running_mean.mean().item()
        
        self.n_updates += 1
        return {f"{self.agent_spec.name}/{k}": v for k, v in train_info.items()}

    def state_dict(self):
        state_dict = {
            "critic": self.critic.state_dict(),
            "actor_params": self.actor_params,
            "value_normalizer": self.value_normalizer.state_dict()
        }
        return state_dict
    
    def load_state_dict(self, state_dict):
        # self.actor_params = TensorDictParams(state_dict["actor_params"])
        with self.actor_params.unlock():
            test = state_dict["actor_params"].to(self.device)
            self.actor_params = test
        self.actor_opt = torch.optim.Adam(self.actor_params.parameters(), lr=self.cfg.actor.lr)
        self.critic.load_state_dict(state_dict["critic"])
        self.value_normalizer.load_state_dict(state_dict["value_normalizer"])


def make_dataset_naive(
    tensordict: TensorDict, num_minibatches: int = 4, seq_len: int = 1
):
    if seq_len > 1:
        N, T = tensordict.shape
        T = (T // seq_len) * seq_len
        tensordict = tensordict[:, :T].reshape(-1, seq_len)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]
    else:
        tensordict = tensordict.reshape(-1)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]


from .modules.distributions import (
    DiagGaussian,
    MultiCategoricalModule,
)

from .common import make_encoder

def make_ppo_actor(cfg, observation_spec: TensorSpec, action_spec: TensorSpec):
    encoder = make_encoder(cfg, observation_spec)

    if isinstance(action_spec, MultiDiscreteTensorSpec):
        act_dist = MultiCategoricalModule(
            encoder.output_shape.numel(), 
            torch.as_tensor(action_spec.nvec.storage().float()).long()
        )
    elif isinstance(action_spec, DiscreteTensorSpec):
        act_dist = MultiCategoricalModule(encoder.output_shape.numel(), [action_spec.space.n])
    elif isinstance(action_spec, (UnboundedTensorSpec, BoundedTensorSpec)):
        action_dim = action_spec.shape[-1]
        act_dist = DiagGaussian(encoder.output_shape.numel(), action_dim, False, 0.01)
        # act_dist = IndependentNormalModule(encoder.output_shape.numel(), action_dim, False)
    else:
        raise NotImplementedError(action_spec)

    return Actor(cfg, encoder, act_dist)


def make_critic(cfg, state_spec: TensorSpec, reward_spec: TensorSpec, centralized=False):
    assert isinstance(reward_spec, (UnboundedTensorSpec, BoundedTensorSpec))
    encoder = make_encoder(cfg, state_spec)

    if centralized:
        v_out = nn.Linear(encoder.output_shape.numel(), reward_spec.shape[-2:].numel())
        nn.init.orthogonal_(v_out.weight, cfg.gain)
        return Critic(encoder, v_out, reward_spec.shape[-2:])
    else:
        v_out = nn.Linear(encoder.output_shape.numel(), reward_spec.shape[-1])
        nn.init.orthogonal_(v_out.weight, cfg.gain)
        return Critic(encoder, v_out, reward_spec.shape[-1:])


class Actor(nn.Module):
    def __init__(
        self,
        cfg,
        encoder: nn.Module,
        act_dist: nn.Module,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.act_dist = act_dist
        self.t = 0
        self.thr1 = 0.2
        self.thr2 = 0.2
        self.roll_m = 0.5
        self.roll_p = -0.5
        self.mid1 = 60
        self.mid2 = 100
        self.mid3 = 200

    def _squash_correction(self, raw_action):
        """
        log |det J_tanh|  for tanh squash (applied per dimension).
        Ref: Appendix C of SAC paper.
        """
        # 2·(log(2) − x − softplus(−2x))
        return 2. * (math.log(2.) - raw_action - F.softplus(-2. * raw_action))

    def forward(
        self,
        obs: Union[torch.Tensor, TensorDict],
        action: torch.Tensor = None,
        deterministic=False,
        eval_action=False
    ):
        actor_features = self.encoder(obs)
        action_dist = self.act_dist(actor_features)

        if eval_action:
            action_log_probs = action_dist.log_prob(action).unsqueeze(-1)
            dist_entropy = action_dist.entropy().unsqueeze(-1)
            return action, action_log_probs, dist_entropy
        else:
            action = action_dist.mode if deterministic else action_dist.sample()
            action_log_probs = action_dist.log_prob(action).unsqueeze(-1)
            dist_entropy = action_dist.entropy().unsqueeze(-1)
            return action, action_log_probs, dist_entropy


class Critic(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        v_out: nn.Module,
        output_shape: torch.Size=torch.Size((-1,)),
    ):
        super().__init__()
        self.base = base
        self.v_out = v_out
        self.output_shape = output_shape

    def forward(self, critic_input: torch.Tensor):
        critic_features = self.base(critic_input)
        values = self.v_out(critic_features)

        if len(self.output_shape) > 1:
            values = values.unflatten(-1, self.output_shape)
        return values


