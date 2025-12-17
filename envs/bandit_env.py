import numpy as np

class KArmedBandit:
    def __init__(self, k=10):
        self.k = k
        self.true_means = np.random.normal(0, 1, k)

    def step(self, action):
        # reward từ Gaussian
        reward = np.random.normal(self.true_means[action], 1.0)
        return reward
