"""Command-line entry point.

    python main.py compare                       # statistics table -> results/comparison.md
    python main.py figures --ppo-model models/long3m/ppo_final.zip

Pass PPO result files with --ppo name=Label (name = results/<name>.json from run_ppo --out).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from algorithms.greedy import greedy
from algorithms.random_search import random_search
from algorithms.simulated_annealing import simulated_annealing
from config import EVAL_BUDGET, RESULTS_DIR, TEST_ROOM_SEEDS, ensure_dirs
from environment.reward import evaluate_layout
from evaluation.compare import (build_report, load_all, most_divergent_room)
from evaluation.evaluate import load_results
from evaluation.starts import make_start, solver_rng

FIG_DIR = RESULTS_DIR / "figures"
DEFAULT_PPO = ["ppo=PPO"]


def parse_specs(specs: list[str]) -> list[tuple[str, str]]:
    return [tuple(s.split("=", 1)) if "=" in s else (s, s) for s in specs]


def cmd_compare(args) -> None:
    ensure_dirs()
    report = build_report(parse_specs(args.ppo))
    (RESULTS_DIR / "comparison.md").write_text(report)
    print(report)
    print(f"Saved {RESULTS_DIR / 'comparison.md'}")


def cmd_figures(args) -> None:
    from visualization.plots import (convergence_figure, layouts_figure,
                                     score_distribution_figure, training_curves_figure)
    ensure_dirs()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    specs = parse_specs(args.ppo)
    results = load_all(specs)

    # 1. Score distributions
    scores = {l: [rows[s]["score"] for s in sorted(rows)] for l, rows in results.items()}
    valid = {l: sum(rows[s]["valid"] for s in rows) for l, rows in results.items()}
    score_distribution_figure(scores, valid, FIG_DIR / "score_distributions.png")

    # 2. Side-by-side layouts: first three test rooms + largest |PPO - SA| (rule fixed in advance)
    rooms = list(TEST_ROOM_SEEDS[:3])
    primary_label = specs[0][1]
    if primary_label in results:
        extra = most_divergent_room(results[primary_label], results["Simulated annealing"])
        if extra not in rooms:
            rooms.append(extra)
        print(f"Most divergent PPO vs SA room: {extra}")
    model = None
    if args.ppo_model:
        from algorithms.ppo import load_model, ppo_solve
        model = load_model(args.ppo_model)
    panels: dict[str, dict[str, tuple]] = {}
    for seed in rooms:
        room, start = make_start(seed)
        row = {"Start": (room, start, evaluate_layout(room, start))}
        for tag, (label, fn) in enumerate((("Random search", random_search),
                                           ("Greedy", greedy),
                                           ("Annealing", simulated_annealing))):
            res = fn(room, start, solver_rng(seed, tag), EVAL_BUDGET)
            row[label] = (room, res.layout, res.breakdown)
        if model is not None:
            res = ppo_solve(model, room, start, obs_mode=args.obs_mode)
            row["PPO"] = (room, res.layout, res.breakdown)
        panels[f"room {seed}"] = row
    layouts_figure(panels, FIG_DIR / "layouts.png",
                   columns=["Start", "Random search", "Greedy", "Annealing", "PPO"])

    # 3. Convergence (deterministic re-runs on the first 20 test rooms)
    n_conv = 20
    curves = {}
    for tag, (label, fn) in enumerate((("Random search", random_search), ("Greedy", greedy),
                                       ("Annealing", simulated_annealing))):
        hist = []
        for seed in TEST_ROOM_SEEDS[:n_conv]:
            room, start = make_start(seed)
            h = fn(room, start, solver_rng(seed, tag), EVAL_BUDGET).history
            hist.append(np.pad(h, (0, EVAL_BUDGET - len(h)), mode="edge"))
        curves[label] = np.mean(hist, axis=0)
    convergence_figure(curves, FIG_DIR / "convergence.png", n_conv)

    # 4. Training curves
    logs = {}
    for item in args.training_logs:
        label, path = item.split("=", 1)
        if Path(path).exists():
            logs[label] = Path(path)
    if logs:
        starts = {s: evaluate_layout(*make_start(s)).total for s in TEST_ROOM_SEEDS}
        refs = {}
        for name, label in (("greedy", "greedy"), ("simulated_annealing", "annealing")):
            rows = load_results(name)
            refs[label] = float(np.mean([rows[s]["score"] - starts[s] for s in rows]))
        training_curves_figure(logs, FIG_DIR / "training_curves.png", refs)
    else:
        print("No training logs found; skipped training_curves.png")
    print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("compare", cmd_compare), ("figures", cmd_figures)):
        sp = sub.add_parser(name)
        sp.add_argument("--ppo", nargs="*", default=DEFAULT_PPO,
                        help="PPO result files as name=Label (results/<name>.json)")
        if name == "figures":
            sp.add_argument("--ppo-model", default=None)
            sp.add_argument("--obs-mode", default="vector", choices=["vector", "grid"])
            sp.add_argument("--training-logs", nargs="*", default=[
                "PPO 300k=results/ppo_logs/progress.csv",
                "PPO 3M=results/ppo_logs_long3m/progress.csv"])
        sp.set_defaults(func=fn)
    a = ap.parse_args()
    a.func(a)
