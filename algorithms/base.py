"""Shared plumbing so every solver is measured the same way.

BudgetedEvaluator wraps evaluate_layout(): it COUNTS every call, remembers the
best layout seen, and refuses to go over budget. Solvers can't cheat the
budget because they have no other way to score a layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import EVAL_BUDGET
from environment.layout import Layout
from environment.reward import RewardBreakdown, RewardWeights, evaluate_layout
from environment.room import Room


class BudgetExhausted(RuntimeError):
    """Raised if a solver tries to evaluate more layouts than it was given."""


class BudgetedEvaluator:
    def __init__(self, room: Room, budget: int = EVAL_BUDGET,
                 weights: RewardWeights = RewardWeights()) -> None:
        self.room, self.budget, self.weights = room, budget, weights
        self.n_evals = 0
        self.best_score = float("-inf")
        self.best_layout: Layout | None = None
        self.best_breakdown: RewardBreakdown | None = None
        self.history: list[float] = []      # best-so-far score after each evaluation

    @property
    def exhausted(self) -> bool:
        return self.n_evals >= self.budget

    def __call__(self, layout: Layout) -> RewardBreakdown:
        if self.exhausted:
            raise BudgetExhausted(f"budget of {self.budget} evaluations used up")
        breakdown = evaluate_layout(self.room, layout, self.weights)
        self.n_evals += 1
        if breakdown.total > self.best_score:
            self.best_score = breakdown.total
            self.best_layout = dict(layout)
            self.best_breakdown = breakdown
        self.history.append(self.best_score)
        return breakdown


@dataclass
class SolverResult:
    name: str
    room: Room
    layout: Layout                  # best layout found
    breakdown: RewardBreakdown
    n_evals: int
    seconds: float
    history: list[float] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)   # solver-specific extras

    @property
    def score(self) -> float:
        return self.breakdown.total
