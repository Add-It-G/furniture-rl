"""Phase 4 tests: actions and the Gymnasium environment."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from config import TEST_ROOM_SEEDS, TRAIN_ROOM_SEEDS
from environment.actions import (DOWN, FINISH_ACTION, N_ACTIONS, N_OPS, ROTATE_CCW,
                                 ROTATE_CW, UP, apply_action, describe_action)
from environment.examples import example_room, good_layout
from environment.furniture import DEFAULT_ITEMS
from environment.furniture_env import OBS_DIM, FurnitureEnv, room_for_seed


def act(piece: str, op: int) -> int:
    return DEFAULT_ITEMS.index(piece) * N_OPS + op


def test_action_space_size_and_names():
    assert N_ACTIONS == 31
    assert describe_action(act("sofa", ROTATE_CW)) == "sofa:rotate_cw"
    assert describe_action(FINISH_ACTION) == "finish"


def test_move_and_rotate_round_trips():
    room, layout = example_room(), good_layout()
    moved = apply_action(room, layout, act("bed", UP))
    assert moved["bed"].row == layout["bed"].row - 1
    assert layout["bed"].row == 8                       # original untouched
    back = apply_action(room, moved, act("bed", DOWN))
    assert back["bed"] == layout["bed"]
    r = apply_action(room, apply_action(room, layout, act("sofa", ROTATE_CW)),
                     act("sofa", ROTATE_CCW))
    assert r["sofa"] == layout["sofa"]


def test_fully_leaving_the_room_is_invalid():
    room, layout = example_room(), good_layout()
    tv = layout["tv_stand"]                             # 5 rows tall
    layout = dict(layout, tv_stand=tv.moved(-(tv.row + 4), 0))   # only its last row inside
    assert apply_action(room, layout, act("tv_stand", UP)) is None


def test_env_passes_gymnasium_checker():
    check_env(FurnitureEnv(room_seeds=TRAIN_ROOM_SEEDS), skip_render_check=True)


def test_reset_is_deterministic_and_obs_in_space():
    env = FurnitureEnv(room_seeds=TRAIN_ROOM_SEEDS)
    o1, _ = env.reset(seed=5)
    o2, _ = env.reset(seed=5)
    assert o1.shape == (OBS_DIM,) and np.array_equal(o1, o2)
    assert env.observation_space.contains(o1)


def test_return_telescopes_to_score_change():
    env = FurnitureEnv(room_seeds=TRAIN_ROOM_SEEDS, invalid_action_penalty=0.0)
    _, info = env.reset(seed=1)
    start = info["score"]
    total = 0.0
    for _ in range(80):
        _, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        if term or trunc:
            break
    assert total == pytest.approx(info["score"] - start, abs=1e-6)


def test_back_and_forth_earns_nothing():
    env = FurnitureEnv()
    env.reset(options={"room": example_room(), "layout": good_layout()})
    _, r1, *_ = env.step(act("bed", UP))
    _, r2, *_ = env.step(act("bed", DOWN))
    assert r1 + r2 == pytest.approx(0.0, abs=1e-9)


def test_finish_terminates_and_truncation_happens():
    env = FurnitureEnv(max_steps=3)
    env.reset(seed=0)
    _, _, term, trunc, _ = env.step(FINISH_ACTION)
    assert term and not trunc
    env.reset(seed=0)
    for i in range(3):
        _, _, term, trunc, _ = env.step(act("bed", UP) if i < 2 else act("bed", DOWN))
    assert trunc and not term


def test_invalid_action_is_penalised_and_changes_nothing():
    env = FurnitureEnv(invalid_action_penalty=0.1)
    room, layout = example_room(), good_layout()
    tv = layout["tv_stand"]
    layout = dict(layout, tv_stand=tv.moved(-(tv.row + 4), 0))
    env.reset(options={"room": room, "layout": layout})
    _, r, _, _, info = env.step(act("tv_stand", UP))
    assert info["invalid_action"] and r == pytest.approx(-0.1)
    assert env.layout["tv_stand"] == layout["tv_stand"]


def test_train_and_test_room_pools_are_disjoint():
    assert not set(TRAIN_ROOM_SEEDS) & set(TEST_ROOM_SEEDS)
    assert room_for_seed(TEST_ROOM_SEEDS[0]) == room_for_seed(TEST_ROOM_SEEDS[0])
