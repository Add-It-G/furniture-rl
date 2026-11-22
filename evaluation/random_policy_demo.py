"""Run the environment with a random policy. From the project root:
    python -m evaluation.random_policy_demo
"""
import time

import numpy as np

from config import SEED, TRAIN_ROOM_SEEDS
from environment.furniture_env import FurnitureEnv

env = FurnitureEnv(room_seeds=TRAIN_ROOM_SEEDS, render_mode="ansi")
returns, finals, valid = [], [], 0
steps_total, t0 = 0, time.time()
for ep in range(30):
    _, info = env.reset(seed=SEED + ep)
    start, ret, done = info["score"], 0.0, False
    while not done:
        _, r, term, trunc, info = env.step(env.action_space.sample())
        ret += r
        steps_total += 1
        done = term or trunc
    returns.append(ret); finals.append(info["score"]); valid += info["is_valid"]
dt = time.time() - t0
print(f"30 random-policy episodes: mean return {np.mean(returns):.2f}, "
      f"mean final score {np.mean(finals):.2f}, valid layouts {valid}/30")
print(f"speed: {steps_total / dt:,.0f} steps/sec")
print("\nLast final layout:")
print(env.render())
