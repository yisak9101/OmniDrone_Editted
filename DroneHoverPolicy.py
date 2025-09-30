import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale

def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

def quat_rotate(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def quat_axis(q: torch.Tensor, axis: int=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

class DroneHoverPolicy:

    def __init__(self, policy_path):
        self.policy = nn.Sequential(make_mlp([256, 256, 256]), Actor(4))
        self.policy.load_state_dict(torch.load(policy_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = self.policy.to(self.device)

    # rel_pos: (dim: 3) 목표 위치와의 상대거리 (m)
    # quat: (dim: 4) quarternion orientation
    # vel: (dim: 3) 드론 속도 (m/s)
    # ang_vel: (dim: 3) 드론 각속도 (rad/s)
    # prev_action: (dim: 4) 이전 action (reset 직후 이전 action이 없는 경우 0으로 채워서 넣어주면 됨)
    # output: (dim: 4) first 3는 rate ([-1, 1]), 마지막은 thrust ([-1, 1])
    def forward(self, rel_pos, quat, vel, ang_vel, prev_action):
        rel_pos = torch.tensor(rel_pos, dtype=torch.float32).unsqueeze(0)
        quat = torch.tensor(quat, dtype=torch.float32).unsqueeze(0)
        vel = torch.tensor(vel, dtype=torch.float32).unsqueeze(0)
        ang_vel = torch.tensor(ang_vel, dtype=torch.float32).unsqueeze(0)
        prev_action = torch.tensor(prev_action, dtype=torch.float32).unsqueeze(0)

        heading = quat_axis(quat, axis=0)
        up = quat_axis(quat, axis=2)
        rel_heading = -heading

        obs = torch.cat([rel_pos, quat, vel, ang_vel, heading, up, prev_action, rel_heading], dim=-1)
        action, _ = self.policy(obs.to(self.device))

        return action