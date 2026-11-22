"""Run simulated annealing on every test room. From the project root:
    python -m evaluation.run_annealing   (run the random + greedy scripts first)
"""
import json

import numpy as np

from algorithms.simulated_annealing import simulated_annealing
from config import EVAL_BUDGET, RESULTS_DIR, TEST_ROOM_SEEDS, ensure_dirs
from evaluation.starts import make_start, solver_rng

ensure_dirs()
load = lambda n: {r["room_seed"]: r for r in json.loads((RESULTS_DIR / f"{n}.json").read_text())}
rand, gre = load("random_search"), load("greedy")

rows = []
for seed in TEST_ROOM_SEEDS:
    room, start_layout = make_start(seed)
    res = simulated_annealing(room, start_layout, solver_rng(seed, 2), EVAL_BUDGET)
    b = res.breakdown
    rows.append({"room_seed": seed, "score": res.score, "valid": b.is_valid,
                 "n_violations": b.n_violations, "seconds": res.seconds,
                 "n_evals": res.n_evals, **res.info})

sa = np.array([r["score"] for r in rows])
rs = np.array([rand[r["room_seed"]]["score"] for r in rows])
gr = np.array([gre[r["room_seed"]]["score"] for r in rows])
print(f"{'algorithm':<22}{'mean score':>11}{'std':>7}{'valid':>8}{'evals':>8}{'sec/room':>10}")
for name, d in (("random search", rand), ("greedy", gre),
                ("simulated annealing", {r['room_seed']: r for r in rows})):
    sc = [d[s]["score"] for s in TEST_ROOM_SEEDS]
    print(f"{name:<22}{np.mean(sc):>11.2f}{np.std(sc):>7.2f}"
          f"{sum(d[s]['valid'] for s in TEST_ROOM_SEEDS):>5}/{len(sc)}"
          f"{np.mean([d[s]['n_evals'] for s in TEST_ROOM_SEEDS]):>8.0f}"
          f"{np.mean([d[s]['seconds'] for s in TEST_ROOM_SEEDS]):>10.2f}")
print(f"\nSA beat greedy in {int((sa > gr + 1e-9).sum())}/{len(rows)} rooms "
      f"(tied {int((abs(sa - gr) <= 1e-9).sum())}), beat random search in {int((sa > rs).sum())}/{len(rows)}")
print(f"mean acceptance rate: {np.mean([r['acceptance_rate'] for r in rows]):.1%}   "
      f"worse moves accepted per room: {np.mean([r['accepted_worse'] for r in rows]):.0f}")
print(f"mean score of FINAL state (not best): {np.mean([r['final_state_score'] for r in rows]):.2f}")
best_known = np.maximum.reduce([sa, rs, gr])
print(f"mean best-known score across the three solvers: {best_known.mean():.2f}")

(RESULTS_DIR / "simulated_annealing.json").write_text(json.dumps(rows, indent=1))
