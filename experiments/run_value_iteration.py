# experiments/run_value_iteration.py
from envs.gridworld_env import GridWorld
from agents.value_iteration_agent import ValueIterationAgent

env = GridWorld(size=4)
agent = ValueIterationAgent(env, gamma=0.9)

print("Running Value Iteration...")
agent.value_iteration()

policy = agent.extract_policy()

print("\nOptimal Value Function V(s):")
for row in agent.V:
    print(["{:+.2f}".format(v) for v in row])

print("\nOptimal Policy (0=UP,1=DOWN,2=LEFT,3=RIGHT):")
for row in policy:
    print(row)
