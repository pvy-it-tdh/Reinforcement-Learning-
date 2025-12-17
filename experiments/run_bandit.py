import numpy as np
from envs.bandit_env import KArmedBandit
from agents.bandit_agent import EpsilonGreedyAgent

env = KArmedBandit(k=10)
agent = EpsilonGreedyAgent(k=10, epsilon=0.1, alpha=0.1)

rewards = []

for step in range(10000):
    action = agent.select_action()
    reward = env.step(action)
    agent.update(action, reward)
    rewards.append(reward)

print("Estimated Q:", agent.Q)
print("True means:", env.true_means)
print("Best arm:", np.argmax(agent.Q))
