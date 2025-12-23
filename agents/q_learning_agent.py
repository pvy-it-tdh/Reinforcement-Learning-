# agents/q_learning_agent.py
import numpy as np


class QLearningAgent:
    """
    Tabular Q-learning cho GridWorld:
    Q(s,a) cập nhật theo:
      Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s',a') - Q(s,a) ]
    """

    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_min=0.02, epsilon_decay=0.995):
        self.env = env
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.Q = np.zeros((env.size, env.size, len(env.actions)), dtype=np.float32)

    def select_action(self, state):
        # ε-greedy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        best_next = 0.0 if done else float(np.max(self.Q[next_state]))
        td_target = float(reward) + self.gamma * best_next
        td_error = td_target - float(self.Q[state][action])
        self.Q[state][action] = float(self.Q[state][action]) + self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def derive_policy(self):
        # π(s)=argmax_a Q(s,a)
        policy = np.zeros((self.env.size, self.env.size), dtype=np.int32)
        for x in range(self.env.size):
            for y in range(self.env.size):
                policy[x, y] = int(np.argmax(self.Q[(x, y)]))
        return policy
