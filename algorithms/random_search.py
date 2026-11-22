"""Baseline A: random search.

Draw independent random layouts, score each, keep the best. No learning, no
memory, no use of the starting layout. Any smarter algorithm should beat this.
"""
from __future__ import annotations

import time

import numpy as np

from algorithms.base import BudgetedEvaluator, SolverResult
from config import EVAL_BUDGET
from environment.layout import Layout, random_layout
from environment.reward import RewardWeights
from environment.room import Room


def random_search(room: Room, initial_layout: Layout | None, rng: np.random.Generator,
                  budget: int = EVAL_BUDGET,
                  weights: RewardWeights = RewardWeights()) -> SolverResult:
    """Standard solver signature (shared by every algorithm).

    initial_layout is accepted but deliberately ignored: random search samples
    fresh layouts, so every one of its `budget` evaluations is a new sample.
    """
    evaluator = BudgetedEvaluator(room, budget, weights)
    start = time.perf_counter()
    while not evaluator.exhausted:
        evaluator(random_layout(room, rng))
    seconds = time.perf_counter() - start
    return SolverResult("random_search", room, evaluator.best_layout,
                        evaluator.best_breakdown, evaluator.n_evals, seconds,
                        evaluator.history)
