# agents/networks.py
import torch
import torch.nn as nn
import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class Actor(nn.Module):
    """
    Actor: π(a|s)
    Output Gaussian policy, dùng tanh để squash action ∈ [-1,1]
    """
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)

        logp = dist.log_prob(z).sum(-1, keepdim=True)
        logp -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)

        return action, logp
class Critic(nn.Module):
    """
    Critic: Q(s,a)
    """
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)
