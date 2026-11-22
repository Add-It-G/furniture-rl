"""Baseline B: greedy hill climbing.

Each turn: try every legal single move (30 actions), score each resulting
layout, and take the best one -- but only if it strictly improves on the
current score. When no move improves, STOP. That is a local optimum: a layout
that is better than all of its neighbours, though not necessarily the best
layout overall.
"""
from __future__ import annotations

import time

import numpy as np

from algorithms.base import BudgetedEvaluator, SolverResult
from config import EVAL_BUDGET
from environment.actions import FINISH_ACTION, apply_action
from environment.layout import Layout
from environment.reward import RewardWeights
from environment.room import Room

EPS = 1e-9   # improvements smaller than this count as "no improvement" (float noise)


def greedy(room: Room, initial_layout: Layout, rng: np.random.Generator,
           budget: int = EVAL_BUDGET,
           weights: RewardWeights = RewardWeights()) -> SolverResult:
    """Pure greedy (no restarts). Ties between equally good moves are broken at random.

    Stops when (a) no move improves the score -> "local_optimum", or
    (b) the budget cannot cover another full sweep of neighbours -> "budget".
    Invalid moves are skipped without spending budget.
    """
    evaluator = BudgetedEvaluator(room, budget, weights)
    start = time.perf_counter()

    layout = dict(initial_layout)
    current = evaluator(layout).total
    turns, stop_reason = 0, "budget"

    while True:
        neighbours = [n for a in range(FINISH_ACTION)
                      if (n := apply_action(room, layout, a)) is not None]
        if evaluator.budget - evaluator.n_evals < len(neighbours):
            stop_reason = "budget"
            break
        scores = [evaluator(n).total for n in neighbours]
        best = max(scores)
        if best <= current + EPS:
            stop_reason = "local_optimum"
            break
        tied = [i for i, sc in enumerate(scores) if sc >= best - EPS]
        layout = neighbours[int(rng.choice(tied))]
        current = best
        turns += 1

    seconds = time.perf_counter() - start
    return SolverResult("greedy", room, evaluator.best_layout, evaluator.best_breakdown,
                        evaluator.n_evals, seconds, evaluator.history,
                        info={"turns": turns, "stop_reason": stop_reason})
