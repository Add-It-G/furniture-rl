"""Evaluate trained PPO on the 50 test rooms. From the project root:
    python -m evaluation.run_ppo
Needs: models/ppo_final.zip (and optionally models/best_model.zip).
"""
import argparse
import json

import numpy as np

from algorithms.ppo import load_model, ppo_solve
from config import MODELS_DIR, RESULTS_DIR, TEST_ROOM_SEEDS, ensure_dirs
from evaluation.criteria import classify
from evaluation.starts import make_start

ensure_dirs()
load = lambda n: {r["room_seed"]: r for r in json.loads((RESULTS_DIR / f"{n}.json").read_text())}


ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(MODELS_DIR / "ppo_final.zip"))
ap.add_argument("--obs-mode", default="vector", choices=["vector", "grid"])
ap.add_argument("--label", default="PPO (final model)")
ap.add_argument("--out", default="ppo.json", help="results file written to results/")
ARGS = ap.parse_args()


def run(model_path):
    model = load_model(model_path)
    rows = []
    for seed in TEST_ROOM_SEEDS:
        room, start = make_start(seed)
        res = ppo_solve(model, room, start, obs_mode=ARGS.obs_mode)
        b = res.breakdown
        rows.append({"room_seed": seed, "score": res.score, "valid": b.is_valid,
                     "n_violations": b.n_violations, "seconds": res.seconds,
                     "n_evals": res.n_evals, **res.info})
    return rows


def summary(name, rows):
    sc = [r["score"] for r in rows]
    n_valid = sum(r["valid"] for r in rows)
    print(f"{name:<24}{np.mean(sc):>9.2f}{np.std(sc):>7.2f}{n_valid:>5}/{len(rows)}"
          f"{np.mean([r['n_evals'] for r in rows]):>8.0f}"
          f"{np.mean([r['seconds'] for r in rows]):>10.3f}")
    return np.mean(sc), n_valid


print(f"{'algorithm':<24}{'mean score':>9}{'std':>7}{'valid':>9}{'evals':>8}{'sec/room':>10}")
for n, label in (("random_search", "random search"), ("greedy", "greedy"),
                 ("simulated_annealing", "simulated annealing")):
    rows = list(load(n).values())
    summary(label, rows)

primary = run(ARGS.model)
mean, n_valid = summary(ARGS.label, primary)
(RESULTS_DIR / ARGS.out).write_text(json.dumps(primary, indent=1))
print(f"\nPPO secondary: mean best-visited score "
      f"{np.mean([r['best_visited_score'] for r in primary]):.2f}; "
      f"FINISH chosen in {sum(r['finished'] for r in primary)}/{len(primary)} rooms; "
      f"mean steps {np.mean([r['steps'] for r in primary]):.0f}")
print(f"Pre-registered verdict: {classify(mean, n_valid).upper()}")

best = MODELS_DIR / "best_model.zip"
if best.exists() and ARGS.out == "ppo.json":
    summary("PPO (best on val rooms)", run(best))

