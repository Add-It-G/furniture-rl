"""Phase 1 smoke test: verifies the toolchain works end to end."""
import gymnasium as gym
import numpy as np
import matplotlib
import stable_baselines3 as sb3
from stable_baselines3 import PPO

from config import SEED, ensure_dirs

ensure_dirs()
print(f"gymnasium {gym.__version__} | sb3 {sb3.__version__} | "
      f"numpy {np.__version__} | matplotlib {matplotlib.__version__}")

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, seed=SEED, verbose=0)
model.learn(total_timesteps=2_000)

obs, _ = env.reset(seed=SEED)
total_reward, done = 0.0, False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated
print(f"Episode reward after 2k steps of training: {total_reward}")
print("Smoke test passed.")
