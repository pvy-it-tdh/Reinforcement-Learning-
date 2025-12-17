# envs/gridworld_env.py
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.actions = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT

    def step(self, state, action):
        x, y = state

        if action == 0:      # UP
            x = max(x - 1, 0)
        elif action == 1:    # DOWN
            x = min(x + 1, self.size - 1)
        elif action == 2:    # LEFT
            y = max(y - 1, 0)
        elif action == 3:    # RIGHT
            y = min(y + 1, self.size - 1)

        next_state = (x, y)
        reward = 10 if next_state == self.goal else -1
        done = next_state == self.goal
        return next_state, reward, done
