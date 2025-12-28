# experiments/eval_sac.py
import gymnasium as gym
import torch
import time

from agents.sac_agent import SACAgent

# =============================
# SETUP ENV
# =============================
env = gym.make("Pendulum-v1", render_mode="human")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# LOAD AGENT
# =============================
agent = SACAgent(obs_dim, act_dim, device)

MODEL_PATH = "models/sac_step_300000.pt"  # ← đổi nếu cần
agent.load(MODEL_PATH)

print("✅ Model loaded:", MODEL_PATH)

# =============================
# TEST LOOP
# =============================
obs, _ = env.reset()
total_reward = 0.0
episode = 1

while True:
    # -------- ACT (DETERMINISTIC) --------
    action = agent.act(obs, deterministic=True)

    obs, reward, done, truncated, _ = env.step(action)
    total_reward += reward

    time.sleep(0.02)  # cho dễ nhìn (GUI)

    if done or truncated:
        print(f"🎯 Episode {episode} | Total Reward: {total_reward:.2f}")
        episode += 1
        total_reward = 0.0
        obs, _ = env.reset()
