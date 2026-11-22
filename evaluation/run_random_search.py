"""Run random search on every test room. From the project root:
    python -m evaluation.run_random_search
"""
import json

import numpy as np

from algorithms.random_search import random_search
from config import EVAL_BUDGET, RESULTS_DIR, TEST_ROOM_SEEDS, ensure_dirs
from environment.reward import evaluate_layout
from evaluation.starts import make_start, solver_rng

ensure_dirs()
CHECKPOINTS = [100, 500, 1000, 2500, EVAL_BUDGET]
rows, start_scores = [], []
curves = {k: [] for k in CHECKPOINTS}

for seed in TEST_ROOM_SEEDS:
    room, start_layout = make_start(seed)
    start_scores.append(evaluate_layout(room, start_layout).total)
    res = random_search(room, start_layout, solver_rng(seed, 0), EVAL_BUDGET)
    for k in CHECKPOINTS:
        curves[k].append(res.history[k - 1])
    b = res.breakdown
    rows.append({"room_seed": seed, "score": res.score, "valid": b.is_valid,
                 "n_violations": b.n_violations, "seconds": res.seconds,
                 "n_evals": res.n_evals})

scores = np.array([r["score"] for r in rows])
print(f"Random search, {len(rows)} unseen rooms, budget {EVAL_BUDGET} evals each")
print(f"  mean best score : {scores.mean():6.2f}  (std {scores.std():.2f})")
print(f"  valid layouts   : {sum(r['valid'] for r in rows)}/{len(rows)}")
print(f"  mean time/room  : {np.mean([r['seconds'] for r in rows]):.2f} s")
print(f"  (starting layouts averaged {np.mean(start_scores):.2f})")
print("\nMean best score vs. evaluations used:")
for k in CHECKPOINTS:
    print(f"  {k:>5} evals -> {np.mean(curves[k]):6.2f}")

(RESULTS_DIR / "random_search.json").write_text(json.dumps(rows, indent=1))
print(f"\nSaved {RESULTS_DIR / 'random_search.json'}")

