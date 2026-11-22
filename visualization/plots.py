"""Figures: layout comparison, score distributions, convergence, training curves."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visualization.renderer import draw_layout


def layouts_figure(panels: dict[str, dict[str, tuple]], path: Path,
                   columns: list[str] | None = None) -> None:
    """panels[row_label][column_label] = (room, layout, breakdown). One row per room.
    A column missing from a row is drawn as an empty "(not run)" panel."""
    row_labels = list(panels)
    col_labels = columns or list(panels[row_labels[0]])
    fig, axes = plt.subplots(len(row_labels), len(col_labels), squeeze=False,
                             figsize=(3.3 * len(col_labels), 3.3 * len(row_labels)))
    for i, rl in enumerate(row_labels):
        for j, cl in enumerate(col_labels):
            ax = axes[i][j]
            if cl not in panels[rl]:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{cl}\n(not run)", ha="center", va="center", fontsize=8)
                continue
            room, layout, b = panels[rl][cl]
            status = "valid" if b.is_valid else f"{b.n_violations} violations"
            draw_layout(ax, room, layout, f"{cl} | {b.total:.2f} | {status}")
            if j == 0:
                ax.set_ylabel(rl, fontsize=9)
    fig.suptitle("Final layouts from the same room and starting layout  "
                 "(green=door, blue=window, red=overlap, orange=blocked door, dashed=walking path)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def score_distribution_figure(scores: dict[str, list[float]], n_valid: dict[str, int],
                              path: Path) -> None:
    labels = list(scores)
    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 2, 4.5))
    ax.boxplot([scores[l] for l in labels], showfliers=False, widths=0.5)
    rng = np.random.default_rng(0)
    for i, l in enumerate(labels, start=1):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(scores[l])), scores[l], s=12, alpha=0.5)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([f"{l}\n{n_valid[l]}/{len(scores[l])} valid" for l in labels], fontsize=8)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Final layout score (higher is better)")
    ax.set_title("Score per test room")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def convergence_figure(curves: dict[str, np.ndarray], path: Path, n_rooms: int) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label, y in curves.items():
        ax.plot(np.arange(1, len(y) + 1), y, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("evaluate_layout() calls used (log scale)")
    ax.set_ylabel("Mean best score so far")
    ax.set_title(f"Search efficiency (first {n_rooms} test rooms; greedy stops at its local optimum)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def read_training_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(timesteps, ep_rew_mean) from an SB3 progress.csv, skipping rows without both."""
    xs, ys = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                x, y = float(row["time/total_timesteps"]), float(row["rollout/ep_rew_mean"])
            except (KeyError, ValueError):
                continue
            xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)


def training_curves_figure(logs: dict[str, Path], path: Path,
                           references: dict[str, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for label, p in logs.items():
        x, y = read_training_log(p)
        if len(x):
            ax.plot(x, y, label=label)
    for label, v in (references or {}).items():
        ax.axhline(v, ls="--", lw=1, color="gray")
        ax.text(ax.get_xlim()[0], v + 0.4, f" {label} ({v:+.1f})", fontsize=7, color="dimgray")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Mean episode return = score improvement")
    ax.set_title("PPO training curves (training rooms)")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
