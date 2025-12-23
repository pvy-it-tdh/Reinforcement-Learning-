# experiments/run_q_learning.py
import numpy as np

from envs.gridworld_env import GridWorld
from agents.q_learning_agent import QLearningAgent


def rollout_greedy(env, agent, max_steps=100):
    """Chạy 1 episode theo policy greedy (deterministic) để xem agent đi thế nào."""
    state = env.reset()
    total_reward = 0.0
    path = [state]
    for _ in range(max_steps):
        action = int(np.argmax(agent.Q[state]))  # greedy
        next_state, reward, done = env.step(state, action)
        total_reward += reward
        state = next_state
        path.append(state)
        if done:
            break
    return total_reward, path


def main():
    env = GridWorld(size=4, start=(0, 0), goal=(3, 3))

    agent = QLearningAgent(
        env,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.2,
        epsilon_min=0.02,
        epsilon_decay=0.995,
    )

    episodes = 1500
    max_steps_per_ep = 200

    success_count = 0
    ep_returns = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        ep_return = 0.0

        for _ in range(max_steps_per_ep):
            action = agent.select_action(state)
            next_state, reward, done = env.step(state, action)

            agent.update(state, action, reward, next_state, done)

            ep_return += reward
            state = next_state

            if done:
                success_count += 1
                break

        agent.decay_epsilon()
        ep_returns.append(ep_return)

        if ep % 100 == 0:
            mean100 = float(np.mean(ep_returns[-100:]))
            succ_rate = success_count / 100.0
            print(f"Episode {ep:4d} | mean_return(100)={mean100:7.2f} | success/100={succ_rate:5.2f} | eps={agent.epsilon:.3f}")
            success_count = 0

    # In policy
    policy = agent.derive_policy()
    print("\nPolicy (0=UP,1=DOWN,2=LEFT,3=RIGHT):")
    print(policy)

    print("\nPolicy arrows:")
    grid = env.render_policy(policy)
    for row in grid:
        print(" ".join(row))

    # Rollout greedy để xem đường đi
    total_reward, path = rollout_greedy(env, agent, max_steps=50)
    print("\nGreedy rollout total_reward:", total_reward)
    print("Path:", path)


if __name__ == "__main__":
    main()
