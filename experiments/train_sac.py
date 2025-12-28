# experiments/train_sac.py
import os
import gymnasium as gym
import torch

from agents.sac_agent import SACAgent
from agents.replay_bufer import ReplayBuffer

# =============================
# SETUP
# =============================
env = gym.make("Pendulum-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

device = "cuda" if torch.cuda.is_available() else "cpu"

agent = SACAgent(obs_dim, act_dim, device)
buffer = ReplayBuffer(obs_dim, act_dim)

os.makedirs("models", exist_ok=True)

# =============================
# TRAINING PARAMS
# =============================
TOTAL_STEPS = 300_000
START_STEPS = 5_000
BATCH_SIZE = 256
UPDATE_AFTER = 1_000
UPDATE_EVERY = 1
SAVE_EVERY = 50_000

obs, _ = env.reset()

# =============================
# TRAIN LOOP
# =============================
for step in range(1, TOTAL_STEPS + 1):

    # -------- ACT --------
    if step < START_STEPS:
        action = env.action_space.sample()
    else:
        action = agent.act(obs, deterministic=False)

    # -------- STEP ENV --------
    next_obs, reward, done, truncated, _ = env.step(action)
    buffer.add(obs, action, reward, next_obs, float(done or truncated))
    obs = next_obs

    if done or truncated:
        obs, _ = env.reset()

    # -------- UPDATE --------
    if step >= UPDATE_AFTER and buffer.size >= BATCH_SIZE:
        for _ in range(UPDATE_EVERY):
            batch = buffer.sample(BATCH_SIZE, device)
            log = agent.update(batch)

    # -------- SAVE MODEL --------
    if step % SAVE_EVERY == 0:
        agent.save(f"models/sac_step_{step}.pt")
        print(
            f"💾 Saved model at step {step} | "
            f"actor_loss={log['actor_loss']:.3f} | "
            f"alpha={log['alpha']:.3f}"
        )

env.close()
print("✅ Training finished.")
