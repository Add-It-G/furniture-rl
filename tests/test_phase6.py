"""Phase 6 tests: greedy hill climbing."""
import numpy as np
import pytest

from algorithms.greedy import greedy
from environment.examples import example_room, good_layout
from environment.reward import evaluate_layout
from evaluation.starts import make_start


def test_greedy_stops_immediately_at_a_local_optimum():
    room = example_room()
    res = greedy(room, good_layout(), np.random.default_rng(0), budget=5000)
    assert res.info["stop_reason"] == "local_optimum" and res.info["turns"] == 0
    assert res.score == pytest.approx(evaluate_layout(room, good_layout()).total)
    assert res.n_evals <= 31        # 1 for the start + at most 30 neighbours


def test_greedy_only_improves_and_respects_budget():
    room, start = make_start(10_001)
    start_score = evaluate_layout(room, start).total
    res = greedy(room, start, np.random.default_rng(1), budget=1000)
    assert res.n_evals <= 1000
    assert res.score >= start_score
    assert all(b >= a for a, b in zip(res.history, res.history[1:]))
    assert res.score == pytest.approx(evaluate_layout(room, res.layout).total)


def test_greedy_is_reproducible_with_same_seed():
    room, start = make_start(10_002)
    a = greedy(room, start, np.random.default_rng(5))
    b = greedy(room, start, np.random.default_rng(5))
    assert a.layout == b.layout and a.n_evals == b.n_evals


def test_greedy_makes_real_progress_from_a_bad_start():
    room, start = make_start(10_003)
    res = greedy(room, start, np.random.default_rng(2))
    assert res.info["turns"] > 5
    assert res.score > evaluate_layout(room, start).total + 5
