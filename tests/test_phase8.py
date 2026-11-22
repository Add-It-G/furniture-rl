"""Phase 8 tests (the training smoke test is skipped if stable-baselines3 is missing)."""
import numpy as np
import pytest

from algorithms.ppo import ScaleReward, make_ppo_solver, ppo_solve
from config import TEST_ROOM_SEEDS, TRAIN_ROOM_SEEDS, VAL_ROOM_SEEDS
from environment.actions import FINISH_ACTION, N_OPS, UP
from environment.examples import example_room, good_layout
from environment.furniture import DEFAULT_ITEMS
from environment.furniture_env import FurnitureEnv
from environment.reward import evaluate_layout
from evaluation.criteria import classify
from evaluation.starts import make_start


class FixedPolicy:
    """Stand-in for a trained model: replays a list of actions, then FINISH."""
    def __init__(self, actions):
        self.actions, self.i = list(actions), 0

    def predict(self, obs, deterministic=True):
        a = self.actions[self.i] if self.i < len(self.actions) else FINISH_ACTION
        self.i += 1
        return np.array(a), None


def test_seed_pools_are_disjoint():
    train, val, test = map(set, (TRAIN_ROOM_SEEDS, VAL_ROOM_SEEDS, TEST_ROOM_SEEDS))
    assert not (train & val) and not (train & test) and not (val & test)


def test_reward_wrapper_scales_rewards():
    env = ScaleReward(FurnitureEnv(), scale=0.1)
    raw = FurnitureEnv()
    for e in (env, raw):
        e.reset(options={"room": example_room(), "layout": good_layout()})
    a = DEFAULT_ITEMS.index("bed") * N_OPS + UP
    _, r_scaled, *_ = env.step(a)
    _, r_raw, *_ = raw.step(a)
    assert r_scaled == pytest.approx(0.1 * r_raw)


def test_finish_is_refused_until_min_steps():
    env = FurnitureEnv(min_steps_before_finish=3, invalid_action_penalty=0.1)
    env.reset(seed=0)
    for _ in range(3):
        _, r, term, _, info = env.step(FINISH_ACTION)
        assert not term and info["invalid_action"] and r == pytest.approx(-0.1)
    _, _, term, _, _ = env.step(FINISH_ACTION)
    assert term


def test_ppo_solve_reports_final_layout_and_counts_evaluations():
    room, start = make_start(10_007)
    move = DEFAULT_ITEMS.index("bed") * N_OPS + UP
    model = FixedPolicy([move] * 10)
    res = ppo_solve(model, room, start, min_finish_step=0)
    assert res.info["finished"] and res.info["steps"] == 11      # 10 moves + FINISH
    assert res.n_evals <= 11                                      # start + valid moves
    assert res.score == pytest.approx(evaluate_layout(room, res.layout).total)
    assert res.info["best_visited_score"] >= res.score - 1e-9


def test_ppo_solve_respects_the_finish_guard_and_truncation():
    room, start = make_start(10_008)
    res = ppo_solve(FixedPolicy([]), room, start, max_steps=60, min_finish_step=50)
    assert res.info["steps"] == 51 and res.info["finished"]       # refused 50 times, then allowed
    res2 = ppo_solve(FixedPolicy([]), room, start, max_steps=20, min_finish_step=50)
    assert res2.info["steps"] == 20 and not res2.info["finished"]  # truncated first


def test_solver_wrapper_has_the_shared_signature():
    room, start = make_start(10_009)
    res = make_ppo_solver(FixedPolicy([]))(room, start, np.random.default_rng(0), 5000, None)
    assert res.name == "ppo"


def test_criteria_boundaries():
    assert classify(3.8, 40) == "full win"
    assert classify(3.79, 45) == "useful trade-off"
    assert classify(2.5, 30) == "useful trade-off"
    assert classify(2.5, 29) == "loss"
    assert classify(1.99, 50) == "loss"
    assert classify(5.0, 10) == "loss"


def test_training_smoke(tmp_path):
    pytest.importorskip("stable_baselines3")
    from algorithms.ppo import train_ppo
    path = train_ppo(timesteps=2048, models_dir=tmp_path / "m", results_dir=tmp_path / "r")
    assert path.exists()
