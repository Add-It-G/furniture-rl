"""Load saved per-room results and summarise them."""
from __future__ import annotations

import json

import numpy as np

from config import RESULTS_DIR


def load_results(name: str) -> dict[int, dict]:
    """results/<name>.json -> {room_seed: row}."""
    path = RESULTS_DIR / f"{name}.json"
    return {r["room_seed"]: r for r in json.loads(path.read_text())}


def summarize(rows: dict[int, dict]) -> dict:
    scores = np.array([r["score"] for r in rows.values()])
    return {"n": len(scores), "mean": float(scores.mean()), "std": float(scores.std()),
            "median": float(np.median(scores)),
            "n_valid": int(sum(r["valid"] for r in rows.values())),
            "mean_violations": float(np.mean([r["n_violations"] for r in rows.values()])),
            "mean_seconds": float(np.mean([r["seconds"] for r in rows.values()])),
            "mean_evals": float(np.mean([r["n_evals"] for r in rows.values()]))}
