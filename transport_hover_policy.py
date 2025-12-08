from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import numpy as np
from tensordict.nn import TensorDictModule
import torch.distributions as D
from tensordict.nn import make_functional, TensorDictModule, TensorDictParams
from tensordict import TensorDict

from omni_drones.utils.torch import manual_batch


def init(module: nn.Module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def off_diag(a: torch.Tensor) -> torch.Tensor:
    assert a.shape[0] == a.shape[1]
    n = a.shape[0]
    return (
        a.flatten(0, 1)[1:]
        .unflatten(0, (n - 1, n + 1))[:, :-1]
        .reshape(n, n - 1, *a.shape[2:])
    )

def cpos(p1: torch.Tensor, p2: torch.Tensor):
    assert p1.shape[1] == p2.shape[1]
    return p1.unsqueeze(1) - p2.unsqueeze(0)


def others(x: torch.Tensor) -> torch.Tensor:
    return off_diag(x.expand(x.shape[0], *x.shape))

def quat_rotate(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

@manual_batch
def quat_axis(q: torch.Tensor, axis: int = 0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=False, gain=0.01):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.log_std = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        # #아래는 sqush 방법
        # log_std = torch.tanh(self.log_std) # output in (-1, 1)
        # # log_std = -2.0 + 2.0 * (log_std + 1) / 2 # scale to (-2, 0) #해당 범위가 옳바른지 모름

        # 아래는 clamp방법
        log_std = torch.clamp(self.log_std, min=-20, max=2) ##해당 clamp 범위가 올바른지 모름

        action_std = torch.exp(log_std)
        action_std = action_std.expand_as(action_mean)
        dist = D.Independent(D.Normal(action_mean, action_std), 1)
        return dist

class MLP(nn.Module):
    def __init__(
        self,
        num_units: Sequence[int],
        normalization: Union[str, nn.Module] = None,
        activation_class: nn.Module = nn.ELU,
        activation_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        layers = []
        if activation_kwargs is not None:
            activation_class = partial(activation_class, **activation_kwargs)
        if isinstance(normalization, str):
            normalization = getattr(nn, normalization, None)
        for i, (in_dim, out_dim) in enumerate(zip(num_units[:-1], num_units[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(num_units) - 1:
                layers.append(activation_class())
            if normalization is not None:
                layers.append(normalization(out_dim))
        self.layers = nn.Sequential(*layers)
        self.input_dim = num_units[0]
        self.output_shape = torch.Size((num_units[-1],))

    def forward(self, x: torch.Tensor):
        return self.layers(x)

class Actor(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        act_dist: nn.Module,
    ) -> None:
        super().__init__()
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

class TransportHoverPolicy:
    def __init__(self, policy_path):
        encoder = nn.Sequential(
            nn.LayerNorm(87),
            MLP(
                num_units=[87] + [256, 256, 256],
                normalization=nn.LayerNorm
            ),
        )
        encoder.output_shape = torch.Size((256,))
        act_dist = DiagGaussian(encoder.output_shape.numel(), 3, False, 0.01)

        create_actor_fn = lambda: TensorDictModule(
            Actor(encoder, act_dist),
            in_keys=[('agents', 'observation'), ('agents', 'action')],
            out_keys=[('agents', 'action'), 'drone.action_logp', 'drone.action_entropy']
        ).to(device)

        actors = nn.ModuleList([create_actor_fn() for _ in range(4)])
        self.actor = actors[0]
        stacked_params = torch.stack([make_functional(actor) for actor in actors])
        self.actor_params = TensorDictParams(stacked_params.to_tensordict())

        with self.actor_params.unlock():
            state_dict = torch.load(policy_path, weights_only=False)
            self.actor_params = state_dict["actor_params"].to(device)

        self.payload_target_pos = torch.tensor([0, 0, 1.5], device=device).unsqueeze(0).unsqueeze(0)
        self.payload_target_heading = torch.zeros(3, device=device).unsqueeze(0).unsqueeze(0)

    # drone_pos: (dim: (4,3))
    # drone_quat: (dim: (4,4)) quarternion orientation
    # drone_vel: (dim: (4,3)) 드론 속도 (m/s)
    # drone_ang_vel: (dim: (4,3)) 드론 각속도 (rad/s)
    # payload_pos: (dim: 3)
    # payload_quat: (dim: 4)
    # payload_vel: (dim: 3)
    # payload_ang_vel: (dim: 3)
    # prev_action: (dim: (4,3)) 이전 action (reset 직후 이전 action이 없는 경우 0으로 채워서 넣어주면 됨)

    def forward(self, drone_pos, drone_quat, drone_vel, drone_ang_vel, payload_pos, payload_quat, payload_vel, payload_ang_vel, prev_action):
        drone_pos = torch.tensor(drone_pos, dtype=torch.float32, device=device).unsqueeze(0)
        drone_quat = torch.tensor(drone_quat, dtype=torch.float32, device=device).unsqueeze(0)
        drone_vel = torch.tensor(drone_vel, dtype=torch.float32, device=device).unsqueeze(0)
        drone_ang_vel = torch.tensor(drone_ang_vel, dtype=torch.float32, device=device).unsqueeze(0)
        payload_pos = torch.tensor(payload_pos, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        payload_quat = torch.tensor(payload_quat, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        payload_vel = torch.tensor(payload_vel, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        payload_ang_vel = torch.tensor(payload_ang_vel, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        payload_drone_rpos = payload_pos - drone_pos

        drone_heading = quat_axis(drone_quat, axis=0)
        drone_up = quat_axis(drone_quat, axis=2)
        payload_heading = quat_axis(payload_quat, axis=0)
        payload_up = quat_axis(payload_quat, axis=2)

        identity = torch.eye(4, device=device).expand(1, -1, -1)
        drone_rpos = torch.vmap(cpos)(drone_pos, drone_pos)
        drone_rpos = torch.vmap(off_diag)(drone_rpos)
        drone_pdist = torch.norm(drone_rpos, dim=-1, keepdim=True)

        payload_target_rpos = self.payload_target_pos - payload_pos
        payload_target_rheading = self.payload_target_heading - payload_heading


        obs_self = torch.cat([payload_drone_rpos, drone_quat, drone_vel, drone_ang_vel, drone_heading, drone_up, identity], dim=-1) # (1,4,_)
        obs_others = torch.cat([drone_rpos, drone_pdist, torch.vmap(others)(torch.cat([drone_quat, drone_vel, drone_ang_vel], dim=-1))], dim=-1).reshape(1, 4, -1) # (1,4,_)
        obs_payload = torch.cat([payload_target_rpos, payload_target_rheading, payload_quat, payload_vel, payload_ang_vel, payload_heading, payload_up], dim=-1).expand(-1, 4, -1) # (1,4,_)

        observation = torch.cat([obs_self, obs_others, obs_payload], dim=-1) # (1,4,_)
        prev_action = torch.tensor(prev_action, dtype=torch.float32).unsqueeze(0) # (1,4,3)

        agents_td = TensorDict(
            {
                "action": prev_action,
                "observation": observation
            },
            batch_size=[1, 4],
            device=device
        )

        # outer tensordict
        td = TensorDict(
            {"agents": agents_td},
            batch_size=[1, 4],
            device=device
        )

        actor_output = torch.vmap(self.actor, in_dims=(1, 0), out_dims=1, randomness="different")(
            td, self.actor_params, deterministic=True
        )

        return actor_output['agents']['action'].squeeze()

if __name__ == "__main__":
    policy = TransportHoverPolicy('checkpoint_a13e4f87.pt')

    action = policy.forward(
        np.random.random((4,3)),
        np.random.random((4,4)),
        np.random.random((4,3)),
        np.random.random((4,3)),
        np.random.random(3),
        np.random.random(4),
        np.random.random(3),
        np.random.random(3),
        np.random.random((4,3)),
    )

    print(action)
