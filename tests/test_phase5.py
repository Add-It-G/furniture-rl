"""Phase 5 tests: budget accounting and random search."""
import numpy as np
import pytest

from algorithms.base import BudgetedEvaluator, BudgetExhausted
from algorithms.random_search import random_search
from environment.examples import bad_layout, example_room, good_layout
from environment.reward import evaluate_layout
from evaluation.starts import make_start


def test_evaluator_counts_tracks_best_and_refuses_overspend():
    ev = BudgetedEvaluator(example_room(), budget=2)
    ev(bad_layout())
    ev(good_layout())
    assert ev.n_evals == 2 and ev.exhausted
    assert ev.best_score == pytest.approx(evaluate_layout(example_room(), good_layout()).total)
    assert ev.history[0] < ev.history[1]
    with pytest.raises(BudgetExhausted):
        ev(good_layout())


def test_random_search_uses_exactly_its_budget_and_history_never_drops():
    room = example_room()
    res = random_search(room, None, np.random.default_rng(0), budget=300)
    assert res.n_evals == 300 and len(res.history) == 300
    assert all(b >= a for a, b in zip(res.history, res.history[1:]))
    assert res.score == pytest.approx(res.history[-1])
    assert res.score == pytest.approx(evaluate_layout(room, res.layout).total)


def test_random_search_is_reproducible_and_ignores_the_start():
    room = example_room()
    a = random_search(room, good_layout(), np.random.default_rng(3), budget=200)
    b = random_search(room, bad_layout(), np.random.default_rng(3), budget=200)
    assert a.score == pytest.approx(b.score) and a.layout == b.layout


def test_make_start_is_deterministic():
    r1, l1 = make_start(10_000)
    r2, l2 = make_start(10_000)
    assert r1 == r2 and l1 == l2
