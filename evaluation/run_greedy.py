"""Run greedy on every test room and compare with random search. From the project root:
    python -m evaluation.run_greedy   (run run_random_search first)
"""
import json
from collections import Counter

import numpy as np

from algorithms.greedy import greedy
from config import EVAL_BUDGET, RESULTS_DIR, TEST_ROOM_SEEDS, ensure_dirs
from environment.reward import evaluate_layout
from evaluation.starts import make_start, solver_rng

ensure_dirs()
rand = {r["room_seed"]: r for r in json.loads((RESULTS_DIR / "random_search.json").read_text())}
rows, start_scores = [], []
for seed in TEST_ROOM_SEEDS:
    room, start_layout = make_start(seed)
    start_scores.append(evaluate_layout(room, start_layout).total)
    res = greedy(room, start_layout, solver_rng(seed, 1), EVAL_BUDGET)
    b = res.breakdown
    rows.append({"room_seed": seed, "score": res.score, "valid": b.is_valid,
                 "n_violations": b.n_violations, "seconds": res.seconds,
                 "n_evals": res.n_evals, **res.info})

scores = np.array([r["score"] for r in rows])
r_scores = np.array([rand[r["room_seed"]]["score"] for r in rows])
print(f"Greedy, {len(rows)} unseen rooms (same starts as every other algorithm)")
print(f"  start layouts   : {np.mean(start_scores):6.2f}")
print(f"  mean final score: {scores.mean():6.2f}  (std {scores.std():.2f})   "
      f"[random search: {r_scores.mean():.2f}]")
print(f"  valid layouts   : {sum(r['valid'] for r in rows)}/{len(rows)}   "
      f"[random search: {sum(r['valid'] for r in rows if False) or sum(v['valid'] for v in rand.values())}/{len(rows)}]")
print(f"  mean evals used : {np.mean([r['n_evals'] for r in rows]):.0f} of {EVAL_BUDGET}")
print(f"  mean turns      : {np.mean([r['turns'] for r in rows]):.1f}")
print(f"  mean time/room  : {np.mean([r['seconds'] for r in rows]):.2f} s")
print(f"  stopped because : {dict(Counter(r['stop_reason'] for r in rows))}")
print(f"  greedy beat random search in {int((scores > r_scores).sum())}/{len(rows)} rooms")

(RESULTS_DIR / "greedy.json").write_text(json.dumps(rows, indent=1))
