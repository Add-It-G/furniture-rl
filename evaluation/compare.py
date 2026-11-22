"""Paired statistics and the comparison report."""
from __future__ import annotations

import json

import numpy as np

from config import RESULTS_DIR, SEED
from evaluation.criteria import classify
from evaluation.evaluate import load_results, summarize

BASELINES = [("random_search", "Random search"), ("greedy", "Greedy"),
             ("simulated_annealing", "Simulated annealing")]


def bootstrap_ci(diffs, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = SEED) -> tuple[float, float, float]:
    """Mean of `diffs` with a percentile bootstrap (1 - alpha) confidence interval.

    Resample the ROOMS with replacement n_boot times, take the mean each time,
    and read off the 2.5th and 97.5th percentiles of those means.
    """
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(diffs.mean()), float(np.percentile(means, 100 * alpha / 2)), \
        float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_summary(a: dict[int, dict], b: dict[int, dict], tol: float = 1e-9) -> dict:
    """Compare algorithm A to B on the rooms they share (difference = A - B)."""
    rooms = sorted(set(a) & set(b))
    diffs = np.array([a[r]["score"] - b[r]["score"] for r in rooms])
    mean, lo, hi = bootstrap_ci(diffs)
    return {"n": len(rooms), "a_wins": int((diffs > tol).sum()),
            "ties": int((abs(diffs) <= tol).sum()), "b_wins": int((diffs < -tol).sum()),
            "mean_diff": mean, "ci_lo": lo, "ci_hi": hi}


def most_divergent_room(a: dict[int, dict], b: dict[int, dict]) -> int:
    """Room seed with the largest |score_a - score_b| (the pre-agreed rule)."""
    rooms = sorted(set(a) & set(b))
    return max(rooms, key=lambda r: abs(a[r]["score"] - b[r]["score"]))


def load_all(ppo_specs: list[tuple[str, str]]) -> dict[str, dict[int, dict]]:
    """{label: rows} for the baselines plus every PPO result file that exists."""
    out = {label: load_results(name) for name, label in BASELINES}
    for name, label in ppo_specs:
        if (RESULTS_DIR / f"{name}.json").exists():
            out[label] = load_results(name)
    return out


def build_report(ppo_specs: list[tuple[str, str]]) -> str:
    results = load_all(ppo_specs)
    lines = ["## Results on the 50 unseen test rooms", "",
             "| Algorithm | Mean score | Std | Median | Valid layouts | Avg violations | Evals | Sec/room |",
             "|---|---|---|---|---|---|---|---|"]
    for label, rows in results.items():
        s = summarize(rows)
        lines.append(f"| {label} | {s['mean']:.2f} | {s['std']:.2f} | {s['median']:.2f} | "
                     f"{s['n_valid']}/{s['n']} | {s['mean_violations']:.1f} | "
                     f"{s['mean_evals']:.0f} | {s['mean_seconds']:.3f} |")

    labels = list(results)
    lines += ["", "## Paired comparison (same rooms, same starting layouts)", "",
              "Difference = first minus second; positive favours the first. "
              "95% bootstrap CI on the mean paired difference (10,000 resamples of rooms).", "",
              "| First vs second | First wins | Ties | Second wins | Mean diff [95% CI] |",
              "|---|---|---|---|---|"]
    for i in range(len(labels) - 1, 0, -1):
        for j in range(i):
            p = paired_summary(results[labels[i]], results[labels[j]])
            lines.append(f"| {labels[i]} vs {labels[j]} | {p['a_wins']} | {p['ties']} | "
                         f"{p['b_wins']} | {p['mean_diff']:+.2f} [{p['ci_lo']:+.2f}, {p['ci_hi']:+.2f}] |")

    for name, label in ppo_specs:
        if label in results:
            s = summarize(results[label])
            lines += ["", f"**Pre-registered criteria applied to {label}:** "
                          f"{classify(s['mean'], s['n_valid']).upper()} "
                          f"(mean {s['mean']:.2f}, {s['n_valid']}/{s['n']} valid)"]
        train = RESULTS_DIR / f"{name.replace('ppo', 'ppo_training', 1)}.json"
        if train.exists():
            t = json.loads(train.read_text())
            lines.append(f"- {label} one-time training cost: {t['timesteps']:,} steps, "
                         f"{t['seconds']:.0f} s")
    return "\n".join(lines) + "\n"
