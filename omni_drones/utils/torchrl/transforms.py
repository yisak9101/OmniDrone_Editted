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


from typing import Any, Dict, Optional, Sequence, Union, Tuple

import torch
from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms import (
    TransformedEnv,
    Transform,
    Compose,
    FlattenObservation,
    CatTensors
)
from torchrl.data import (
    TensorSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    CompositeSpec,
)
from .env import AgentSpec
from dataclasses import replace

from ...controllers.Lee_ctrl import LeeController
from ...controllers.traj_generation import DroneTrajectory


def _transform_agent_spec(self: Transform, agent_spec: AgentSpec) -> AgentSpec:
    return agent_spec
Transform.transform_agent_spec = _transform_agent_spec


def _transform_agent_spec(self: Compose, agent_spec: AgentSpec) -> AgentSpec:
    for transform in self.transforms:
        agent_spec = transform.transform_agent_spec(agent_spec)
    return agent_spec
Compose.transform_agent_spec = _transform_agent_spec


def _agent_spec(self: TransformedEnv) -> AgentSpec:
    agent_spec = self.transform.transform_agent_spec(self.base_env.agent_spec)
    return {name: replace(spec, _env=self) for name, spec in agent_spec.items()}
TransformedEnv.agent_spec = property(_agent_spec)


class FromDiscreteAction(Transform):
    def __init__(
        self,
        action_key: Tuple[str] = ("agents", "action"),
        nbins: Union[int, Sequence[int]] = None,
    ):
        if nbins is None:
            nbins = 2
        super().__init__([], in_keys_inv=[action_key])
        if not isinstance(action_key, tuple):
            action_key = (action_key,)
        self.nbins = nbins
        self.action_key = action_key

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        if isinstance(action_spec, BoundedTensorSpec):
            if isinstance(self.nbins, int):
                nbins = [self.nbins] * action_spec.shape[-1]
            elif len(self.nbins) == action_spec.shape[-1]:
                nbins = self.nbins
            else:
                raise ValueError(
                    "nbins must be int or list of length equal to the last dimension of action space."
                )
            self.minimum = action_spec.space.minimum.unsqueeze(-2)
            self.maximum = action_spec.space.maximum.unsqueeze(-2)
            self.mapping = torch.cartesian_prod(
                *[torch.linspace(0, 1, dim_nbins) for dim_nbins in nbins]
            ).to(action_spec.device)  # [prod(nbins), len(nbins)]
            n = self.mapping.shape[0]
            spec = DiscreteTensorSpec(
                n, shape=[*action_spec.shape[:-1], 1], device=action_spec.device
            )
        else:
            NotImplementedError("Only BoundedTensorSpec is supported.")
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        mapping = self.mapping * (self.maximum - self.minimum) + self.minimum
        action = action.unsqueeze(-1)
        action = torch.take_along_dim(mapping, action, dim=-2).squeeze(-2)
        return action


class FromMultiDiscreteAction(Transform):
    def __init__(
        self,
        action_key: Tuple[str] = ("agents", "action"),
        nbins: Union[int, Sequence[int]] = 2,
    ):
        if action_key is None:
            action_key = "action"
        super().__init__([], in_keys_inv=[action_key])
        if not isinstance(action_key, tuple):
            action_key = (action_key,)
        self.nbins = nbins
        self.action_key = action_key

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        if isinstance(action_spec, BoundedTensorSpec):
            if isinstance(self.nbins, int):
                nbins = [self.nbins] * action_spec.shape[-1]
            elif len(self.nbins) == action_spec.shape[-1]:
                nbins = self.nbins
            else:
                raise ValueError(
                    "nbins must be int or list of length equal to the last dimension of action space."
                )
            spec = MultiDiscreteTensorSpec(
                nbins, shape=action_spec.shape, device=action_spec.device
            )
            self.nvec = spec.nvec.to(action_spec.device)
            self.minimum = action_spec.space.minimum
            self.maximum = action_spec.space.maximum
        else:
            NotImplementedError("Only BoundedTensorSpec is supported.")
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        action = action / (self.nvec - 1) * (self.maximum - self.minimum) + self.minimum
        return action

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return super()._inv_call(tensordict)


class DepthImageNorm(Transform):
    def __init__(
        self,
        in_keys: Sequence[str],
        min_range: float,
        max_range: float,
        inverse: bool=False
    ):
        super().__init__(in_keys=in_keys)
        self.max_range = max_range
        self.min_range = min_range
        self.inverse = inverse

    def _apply_transform(self, obs: torch.Tensor) -> None:
        obs = torch.nan_to_num(obs, posinf=self.max_range, neginf=self.min_range)
        obs = obs.clip(self.min_range, self.max_range)
        if self.inverse:
            obs = (obs - self.min_range) / (self.max_range - self.min_range)
        else:
            obs = (self.max_range - obs) / (self.max_range - self.min_range)
        return obs


def ravel_composite(
    spec: CompositeSpec, key: str, start_dim: int=-2, end_dim: int=-1
):
    r"""
    
    Examples:
    >>> obs_spec = CompositeSpec({
    ...     "obs_self": UnboundedContinuousTensorSpec((1, 19)),
    ...     "obs_others": UnboundedContinuousTensorSpec((3, 13)),
    ... })
    >>> spec = CompositeSpec({
            "agents": {
                "observation": obs_spec
            }
    ... })
    >>> t = ravel_composite(spec, ("agents", "observation"))

    """
    composite_spec = spec[key]
    if not isinstance(key, tuple):
        key = (key,)
    if isinstance(composite_spec, CompositeSpec):
        in_keys = [k for k in spec.keys(True, True) if k[:len(key)] == key]
        return Compose(
            FlattenObservation(start_dim, end_dim, in_keys),
            CatTensors(in_keys, out_key=key, del_keys=False)
        )
    else:
        raise TypeError


class VelController(Transform):
    def __init__(
        self,
        controller,
        action_key: str = ("agents", "action"),
    ):
        super().__init__([], in_keys_inv=[("info", "drone_state")])
        self.controller = controller
        self.action_key = action_key
    
    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(action_spec.shape[:-1]+(4,), device=action_spec.device)
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec
    
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]
        target_vel, target_yaw = action.split([3, 1], -1)
        cmds = self.controller(
            drone_state, 
            target_vel=target_vel, 
            target_yaw=target_yaw*torch.pi
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict


class RateController(Transform):
    def __init__(
        self,
        controller,
        action_key: str = ("agents", "action"),
    ):
        super().__init__([], in_keys_inv=[("info", "drone_state")])
        self.controller = controller
        self.action_key = action_key
        self.max_thrust = self.controller.max_thrusts.sum(-1)

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(action_spec.shape[:-1]+(4,), device=action_spec.device)
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]
        target_rate, target_thrust = action.split([3, 1], -1)
        target_thrust = ((target_thrust + 1) / 2).clip(0.) * self.max_thrust
        cmds = self.controller(
            drone_state,
            target_rate=target_rate * torch.pi,
            target_thrust=target_thrust
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict

import numpy as np
from scipy.spatial.transform import Rotation as R

class PositionController(RateController):
    def __init__(
        self,
        cfg,
        rotor_config,
        controller,
        action_key: str = ("agents", "action"),
    ):
        super().__init__(controller, action_key)
        self.alpha = 0.9
        self.cfg = cfg
        self.dt = self.cfg.sim.dt
        self.n_env = self.cfg.env.num_envs
        self.n_drones = 4
        self.device = cfg.sim.device
        self.yaw = np.zeros((self.n_env, self.n_drones, 1))
        self.max_delta_pos = 0.8333 * self.dt # 30 km/h -> 0.8333 m/s. 0.83

        mass = rotor_config['mass']
        thrust_map = [rotor_config['rotor_configuration']['force_constants'][0], 0, 0]
        motor_omega_max = rotor_config['rotor_configuration']['max_rotation_velocities'][0]
        motor_omega_min = 0
        f_max = (
            thrust_map[0] * motor_omega_max**2
            + thrust_map[1] * motor_omega_max
            + thrust_map[2]
        ) * 4
        f_min = (
            thrust_map[0] * motor_omega_min**2
            + thrust_map[1] * motor_omega_min
            + thrust_map[2]
        ) * 4
        self.omega_max = [3.141592, 3.141592, 3.141592]
        self.std = (f_max - f_min) / 2
        self.mean = (f_max + f_min) / 2

        self.ctrl = LeeController(mass, f_max, f_min, self.omega_max)
        self.init = np.array([[[0.5, 0.24999934434890747, 2.599824905395508], [0.5, -0.25000065565109253, 2.599823474884033],[-0.5000000596046448, -0.25000065565109253, 2.599818229675293],[-0.5000000596046448, 0.24999934434890747, 2.5998196601867676]]])


    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(action_spec.shape[:-1]+(3,), device=action_spec.device)
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = torch.zeros(self.n_env, self.n_drones, 4, device=self.device)
        state = tensordict['agents']['state']['drones']
        pos = state[...,:3].cpu().numpy()
        quat = state[..., 3:7].cpu().numpy()
        vel = state[...,7:10].cpu().numpy()
        omega = state[...,10:13].cpu().numpy() * 180 / np.pi
        goal_pos_delta = tensordict['agents']['action'].cpu().numpy().clip(-1, 1) * self.max_delta_pos
        goal_pos = pos + goal_pos_delta
        goal_vel = np.zeros_like(pos)
        goal_acc = np.zeros_like(pos)

        for i in range(self.n_env):
            for j in range(self.n_drones):
                rot = R.from_quat(quat[i,j], scalar_first=True).as_matrix()
                vec_rot = np.hstack([rot[:, 0], rot[:, 1], rot[:, 2]])
                cur_state = [pos[i,j], vel[i,j], vec_rot, omega[i,j]]
                goal_state = [goal_pos[i,j], goal_vel[i,j], goal_acc[i,j], 0]
                cmd, _ = self.ctrl.compute_control(cur_state, goal_state, type="norm_input")
                cmd = cmd[0]
                # action -> [f, omega]
                # actual_f = cmd[0] # N - real_input
                # omega = cmd[1:] # rad/s - real_input
                actual_f = cmd[0]  # N - norm_input
                target_omega = cmd[1:]  # rad/s - norm_input
                action[i, j, 0] = target_omega[0].item()
                action[i, j, 1] = target_omega[1].item()
                action[i, j, 2] = target_omega[2].item()
                action[i, j, 3] = actual_f.item()

        tensordict.set(self.action_key, action)
        return super()._inv_call(tensordict)


class AttitudeController(Transform):
    def __init__(
        self,
        controller,
        action_key: str = ("agents", "action"),
    ):
        super().__init__([], in_keys_inv=[("info", "drone_state")])
        self.controller = controller
        self.action_key = action_key
        self.max_thrust = self.controller.max_thrusts.sum(-1)
    
    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(action_spec.shape[:-1]+(4,), device=action_spec.device)
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec
    
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]
        target_thrust, target_yaw_rate, target_roll, target_pitch = action.split(1, dim=-1)
        cmds = self.controller(
            drone_state,
            target_thrust=((target_thrust+1)/2).clip(0.) * self.max_thrust,
            target_yaw_rate=target_yaw_rate * torch.pi,
            target_roll=target_roll * torch.pi,
            target_pitch=target_pitch * torch.pi
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict


class History(Transform):
    def __init__(
        self,
        in_keys: Sequence[str],
        out_keys: Sequence[str]=None,
        steps: int = 32,
    ):
        if out_keys is None:
            out_keys = [
                f"{key}_h" if isinstance(key, str) else key[:-1] + (f"{key[-1]}_h",)
                for key in in_keys
            ]
        if any(key in in_keys for key in out_keys):
            raise ValueError
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.steps = steps
    
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            is_tuple = isinstance(in_key, tuple)
            if in_key in observation_spec.keys(include_nested=is_tuple):
                spec = observation_spec[in_key]
                spec = spec.unsqueeze(-1).expand(*spec.shape, self.steps)
                observation_spec[out_key] = spec
        return observation_spec

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            item = tensordict.get(in_key)
            item_history = tensordict.get(out_key)
            item_history[..., :-1] = item_history[..., 1:]
            item_history[..., -1] = item
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            item = tensordict.get(in_key)
            item_history = tensordict.get(out_key).clone()
            item_history[..., :-1] = item_history[..., 1:]
            item_history[..., -1] = item
            tensordict.set(("next", out_key), item_history)
        return tensordict

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        _reset = tensordict.get("_reset", None)
        if _reset is None:
            _reset = torch.ones(tensordict.batch_size, dtype=bool, device=tensordict.device)
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if out_key not in tensordict.keys(True, True):
                item = tensordict.get(in_key)
                item_history = (
                    item.unsqueeze(-1)
                    .expand(*item.shape, self.steps)
                    .clone()
                    .zero_()
                )
                tensordict.set(out_key, item_history)
            else:
                item_history = tensordict.get(out_key)
                item_history[_reset] = 0.
        return tensordict

