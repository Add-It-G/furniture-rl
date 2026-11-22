"""Baseline C: simulated annealing.

Like greedy, but a WORSE layout is sometimes accepted, with probability
exp(delta / T), where delta < 0 is the score change and T is a "temperature"
that shrinks over the run. Hot early = explore (cross valleys); cold late =
exploit (polish).
"""
from __future__ import annotations

import math
import time

import numpy as np

from algorithms.base import BudgetedEvaluator, SolverResult
from config import (EVAL_BUDGET, SA_P_TRANSLATE, SA_T_END, SA_T_START)
from environment.actions import FINISH_ACTION, N_OPS, N_PIECES, apply_action
from environment.layout import Layout
from environment.reward import RewardWeights
from environment.room import Room


def temperature(step: int, n_steps: int, t_start: float, t_end: float) -> float:
    """Geometric cooling: T falls by the same FACTOR every step.

    T_k = t_start * (t_end / t_start) ** (k / (n_steps - 1)),   k = 0 .. n_steps-1
    """
    frac = step / max(1, n_steps - 1)
    return t_start * (t_end / t_start) ** frac


def propose_action(rng: np.random.Generator, p_translate: float) -> int:
    """Pick a piece uniformly, then a translation (prob p_translate) or a rotation."""
    piece = int(rng.integers(0, N_PIECES))
    if rng.random() < p_translate:
        op = int(rng.integers(0, 4))            # up / down / left / right
    else:
        op = int(rng.integers(4, N_OPS))        # rotate cw / ccw
    return piece * N_OPS + op


def simulated_annealing(room: Room, initial_layout: Layout, rng: np.random.Generator,
                        budget: int = EVAL_BUDGET,
                        weights: RewardWeights = RewardWeights(),
                        t_start: float = SA_T_START, t_end: float = SA_T_END,
                        p_translate: float = SA_P_TRANSLATE) -> SolverResult:
    """One evaluation per step (plus one for the start). Returns the BEST layout
    ever visited, not the final state: SA can wander downhill at the end."""
    evaluator = BudgetedEvaluator(room, budget, weights)
    start = time.perf_counter()

    layout = dict(initial_layout)
    current = evaluator(layout).total
    n_steps = budget - 1
    accepted = accepted_worse = 0

    for step in range(n_steps):
        temp = temperature(step, n_steps, t_start, t_end)
        # Invalid proposals (piece would leave the room entirely) are redrawn
        # for free: they never reach the evaluator, so they cost no budget.
        for _ in range(100):
            candidate = apply_action(room, layout, propose_action(rng, p_translate))
            if candidate is not None:
                break
        else:
            continue
        new_score = evaluator(candidate).total
        delta = new_score - current
        if delta >= 0 or rng.random() < math.exp(delta / temp):
            layout, current = candidate, new_score
            accepted += 1
            accepted_worse += delta < 0

    seconds = time.perf_counter() - start
    return SolverResult("simulated_annealing", room, evaluator.best_layout,
                        evaluator.best_breakdown, evaluator.n_evals, seconds,
                        evaluator.history,
                        info={"accepted": accepted, "accepted_worse": accepted_worse,
                              "acceptance_rate": accepted / max(1, n_steps),
                              "final_state_score": current})
