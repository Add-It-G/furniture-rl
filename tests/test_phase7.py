"""Phase 7 tests: simulated annealing."""
import numpy as np
import pytest

from algorithms.greedy import greedy
from algorithms.simulated_annealing import (propose_action, simulated_annealing,
                                            temperature)
from environment.actions import N_OPS
from environment.reward import evaluate_layout
from evaluation.starts import make_start


def test_geometric_cooling_endpoints_and_constant_ratio():
    n = 1000
    assert temperature(0, n, 1.0, 0.02) == pytest.approx(1.0)
    assert temperature(n - 1, n, 1.0, 0.02) == pytest.approx(0.02)
    r1 = temperature(1, n, 1.0, 0.02) / temperature(0, n, 1.0, 0.02)
    r2 = temperature(501, n, 1.0, 0.02) / temperature(500, n, 1.0, 0.02)
    assert r1 == pytest.approx(r2)


def test_proposals_are_about_80_percent_translations_and_uniform_over_pieces():
    rng = np.random.default_rng(0)
    acts = [propose_action(rng, 0.8) for _ in range(20_000)]
    rot_frac = np.mean([a % N_OPS >= 4 for a in acts])
    assert rot_frac == pytest.approx(0.2, abs=0.015)
    counts = np.bincount([a // N_OPS for a in acts], minlength=5) / len(acts)
    assert np.allclose(counts, 0.2, atol=0.015)


def test_sa_budget_history_and_best_tracking():
    room, start = make_start(10_004)
    res = simulated_annealing(room, start, np.random.default_rng(0), budget=800)
    assert res.n_evals == 800
    assert all(b >= a for a, b in zip(res.history, res.history[1:]))
    assert res.score >= evaluate_layout(room, start).total
    assert res.score == pytest.approx(evaluate_layout(room, res.layout).total)
    assert res.score >= res.info["final_state_score"] - 1e-9   # best >= wherever it ended


def test_sa_is_reproducible():
    room, start = make_start(10_005)
    a = simulated_annealing(room, start, np.random.default_rng(9), budget=500)
    b = simulated_annealing(room, start, np.random.default_rng(9), budget=500)
    assert a.layout == b.layout and a.info == b.info


def test_sa_accepts_some_worse_moves_when_hot_but_none_when_frozen():
    room, start = make_start(10_006)
    hot = simulated_annealing(room, start, np.random.default_rng(1), budget=600,
                              t_start=1.0, t_end=1.0)
    cold = simulated_annealing(room, start, np.random.default_rng(1), budget=600,
                               t_start=1e-6, t_end=1e-6)
    assert hot.info["accepted_worse"] > 0
    assert cold.info["accepted_worse"] == 0


def test_sa_escapes_the_local_optimum_that_traps_greedy():
    """Room 10003 (Phase 6): greedy stalls at ~2.27; annealing crosses the valley."""
    from evaluation.starts import solver_rng
    room, start = make_start(10_003)
    g = greedy(room, start, solver_rng(10_003, 1))
    sa = simulated_annealing(room, start, solver_rng(10_003, 2))
    assert g.info["stop_reason"] == "local_optimum"
    assert sa.score > g.score + 1.0
