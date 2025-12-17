import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-4):
        self.env = env
        self.gamma = gamma
        self.theta = theta

        # Value function V(s)
        self.V = np.zeros((env.size, env.size))

        # Policy π(s)
        self.policy = np.zeros((env.size, env.size), dtype=int)

    def value_iteration(self):
        while True:
            delta = 0

            for x in range(self.env.size):
                for y in range(self.env.size):
                    state = (x, y)

                    if state == self.env.goal:
                        continue

                    v_old = self.V[state]
                    q_values = []

                    for action in self.env.actions:
                        next_state, reward, _ = self.env.step(state, action)
                        q = reward + self.gamma * self.V[next_state]
                        q_values.append(q)

                    self.V[state] = max(q_values)
                    delta = max(delta, abs(v_old - self.V[state]))

            if delta < self.theta:
                break

    def extract_policy(self):
        for x in range(self.env.size):
            for y in range(self.env.size):
                state = (x, y)

                if state == self.env.goal:
                    continue

                q_values = []
                for action in self.env.actions:
                    next_state, reward, _ = self.env.step(state, action)
                    q = reward + self.gamma * self.V[next_state]
                    q_values.append(q)

                self.policy[state] = np.argmax(q_values)

        return self.policy
