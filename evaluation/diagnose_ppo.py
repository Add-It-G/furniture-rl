"""Find out WHY a PPO run behaves as it does. From the project root:

    python -m evaluation.diagnose_ppo
    python -m evaluation.diagnose_ppo --model models/long3m/ppo_final.zip --logs results/ppo_logs_long3m

Part 1 prints the training log at evenly spaced points.
Part 2 replays the policy on the VALIDATION rooms (never the test rooms), once
greedily (argmax) and once sampling, and reports which actions it really takes.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from config import (LOG_DIR, MAX_EPISODE_STEPS, MODELS_DIR, PPO_MIN_FINISH_STEP, SEED,
                    VAL_ROOM_SEEDS)
from environment.actions import FINISH_ACTION, N_OPS, describe_action
from environment.furniture_env import FurnitureEnv
from evaluation.starts import make_start

LOG_COLUMNS = ["time/total_timesteps", "rollout/ep_rew_mean", "rollout/ep_len_mean",
               "train/entropy_loss", "train/explained_variance", "train/approx_kl",
               "train/clip_fraction", "eval/mean_reward"]


def summarize_log(path: Path, n_points: int = 7) -> list[dict[str, str]]:
    """Return up to n_points evenly spaced rows of the SB3 progress.csv."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    idx = sorted({round(i * (len(rows) - 1) / max(1, n_points - 1)) for i in range(n_points)})
    return [rows[i] for i in idx]


def print_log(path: Path) -> None:
    rows = summarize_log(path)
    short = ["steps", "ep_rew", "ep_len", "ent_loss", "expl_var", "kl", "clip_frac", "val_rew"]
    print("  ".join(f"{h:>10}" for h in short))
    for r in rows:
        cells = []
        for col in LOG_COLUMNS:
            v = r.get(col, "")
            try:
                cells.append(f"{float(v):>10.3f}" if col != LOG_COLUMNS[0] else f"{float(v):>10.0f}")
            except ValueError:
                cells.append(f"{'-':>10}")
        print("  ".join(cells))


def rollout_stats(model, room_seeds, deterministic: bool,
                  max_steps: int = MAX_EPISODE_STEPS,
                  min_finish: int = PPO_MIN_FINISH_STEP) -> dict:
    counts: Counter = Counter()
    starts, finals, bests, valid, invalid_steps, total_steps = [], [], [], 0, 0, 0
    for seed in room_seeds:
        room, start = make_start(seed)
        env = FurnitureEnv(max_steps=max_steps, min_steps_before_finish=min_finish)
        obs, info = env.reset(options={"room": room, "layout": start})
        starts.append(info["score"])
        best, done = info["score"], False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)
            obs, _, term, trunc, info = env.step(action)
            counts[action] += 1
            total_steps += 1
            invalid_steps += info["invalid_action"]
            best = max(best, info["score"])
            done = term or trunc
        finals.append(info["score"]); bests.append(best); valid += info["is_valid"]
    n = max(1, total_steps)
    rotations = sum(c for a, c in counts.items() if a != FINISH_ACTION and a % N_OPS >= 4)
    return {"start": float(np.mean(starts)), "final": float(np.mean(finals)),
            "best_visited": float(np.mean(bests)), "valid": valid, "n_rooms": len(starts),
            "finish_attempts": counts[FINISH_ACTION] / n, "invalid": invalid_steps / n,
            "rotations": rotations / n,
            "top_actions": [(describe_action(a), c / n) for a, c in counts.most_common(5)]}


def print_stats(label: str, st: dict) -> None:
    print(f"\n{label}")
    print(f"  mean score: start {st['start']:.2f} -> final {st['final']:.2f} "
          f"(best visited {st['best_visited']:.2f}); valid {st['valid']}/{st['n_rooms']}")
    print(f"  steps that were: FINISH attempts {st['finish_attempts']:.0%}, "
          f"invalid {st['invalid']:.0%}, rotations {st['rotations']:.0%}")
    print("  most common actions: " + ", ".join(f"{a} {f:.0%}" for a, f in st["top_actions"]))


if __name__ == "__main__":
    from algorithms.ppo import load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS_DIR / "ppo_final.zip"))
    ap.add_argument("--logs", default=str(LOG_DIR))
    ap.add_argument("--rooms", type=int, default=len(VAL_ROOM_SEEDS))
    args = ap.parse_args()

    log = Path(args.logs) / "progress.csv"
    print(f"=== Training log: {log} ===")
    print_log(log) if log.exists() else print("  (not found)")

    model = load_model(args.model)
    model.set_random_seed(SEED)
    seeds = VAL_ROOM_SEEDS[:args.rooms]
    print(f"\n=== Policy behaviour on {len(seeds)} validation rooms ===")
    print_stats("Greedy (argmax) actions  <- what run_ppo.py measures",
                rollout_stats(model, seeds, deterministic=True))
    print_stats("Sampled actions          <- what PPO actually trained",
                rollout_stats(model, seeds, deterministic=False))
