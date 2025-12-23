# envs/gridworld_env.py
import numpy as np


class GridWorld:
    """
    GridWorld MDP đơn giản (deterministic):
    - State: (x, y)
    - Action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    - Reward: -1 mỗi bước, +10 khi tới goal
    - Terminal: goal
    """

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

    def __init__(self, size: int = 4, start=(0, 0), goal=None):
        self.size = size
        self.start = tuple(start)
        self.goal = tuple(goal) if goal is not None else (size - 1, size - 1)
        self.actions = [self.UP, self.DOWN, self.LEFT, self.RIGHT]

    def reset(self):
        return self.start

    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def step(self, state, action):
        x, y = state

        if action == self.UP:
            x2, y2 = max(x - 1, 0), y
        elif action == self.DOWN:
            x2, y2 = min(x + 1, self.size - 1), y
        elif action == self.LEFT:
            x2, y2 = x, max(y - 1, 0)
        elif action == self.RIGHT:
            x2, y2 = x, min(y + 1, self.size - 1)
        else:
            raise ValueError(f"Unknown action: {action}")

        next_state = (x2, y2)
        done = (next_state == self.goal)

        reward = 10.0 if done else -1.0
        return next_state, reward, done

    def render_policy(self, policy):
        """
        policy: np.ndarray shape (size, size) action int
        """
        arrow = {self.UP: "↑", self.DOWN: "↓", self.LEFT: "←", self.RIGHT: "→"}
        grid = []
        for x in range(self.size):
            row = []
            for y in range(self.size):
                s = (x, y)
                if s == self.goal:
                    row.append("G")
                else:
                    row.append(arrow[int(policy[x, y])])
            grid.append(row)
        return grid
