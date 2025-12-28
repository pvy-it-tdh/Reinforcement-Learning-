# agents/replay_buffer.py
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=1_000_000):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, o, a, r, no, d):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.next_obs[self.ptr] = no
        self.done[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx], device=device),
            torch.tensor(self.act[idx], device=device),
            torch.tensor(self.rew[idx], device=device),
            torch.tensor(self.next_obs[idx], device=device),
            torch.tensor(self.done[idx], device=device),
        )
