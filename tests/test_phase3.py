"""Phase 3 tests: BFS, derived chair, and the reward function."""
from dataclasses import replace

import numpy as np

from environment.examples import bad_layout, example_room, good_layout
from environment.furniture import CATALOG, EAST, NORTH, SOUTH, WEST, Placement
from environment.layout import build_pieces, chair_for_desk
from environment.pathfinding import bfs_distances
from environment.reward import evaluate_layout


def test_bfs_distances_and_unreachable():
    free = np.ones((3, 4), dtype=bool)
    free[:, 2] = False                     # a wall splitting the grid
    dist = bfs_distances(free, [(0, 0)])
    assert dist[0, 0] == 0 and dist[2, 1] == 3
    assert dist[0, 3] == -1                # other side of the wall


def test_bfs_ignores_blocked_start():
    free = np.ones((2, 2), dtype=bool)
    free[0, 0] = False
    assert (bfs_distances(free, [(0, 0)]) == -1).all()


def test_chair_sits_against_desk_front_for_every_rotation():
    for rot in (NORTH, EAST, SOUTH, WEST):
        desk = Placement(CATALOG["desk"], 10, 10, rot)
        chair = chair_for_desk(desk)
        assert set(chair.cells()) <= set(desk.front_cells())   # inside the clearance zone
        assert not set(chair.cells()) & set(desk.cells())       # not on the desk
        assert chair.rotation == (rot + 2) % 4


def test_good_layout_is_valid_and_scores_high():
    b = evaluate_layout(example_room(), good_layout())
    assert b.is_valid
    assert b.total > 4.5


def test_bad_layout_is_invalid_and_strongly_negative():
    b = evaluate_layout(example_room(), bad_layout())
    assert not b.is_valid and b.overlap_cells > 0 and b.total < 0


def test_out_of_bounds_is_penalised_per_cell():
    room, layout = example_room(), good_layout()
    one = dict(layout, tv_stand=Placement(CATALOG["tv_stand"], 2, -1, EAST))   # 1 col out
    two = dict(layout, tv_stand=Placement(CATALOG["tv_stand"], 2, -2, EAST))   # 2 cols out
    b1, b2 = evaluate_layout(room, one), evaluate_layout(room, two)
    assert b1.out_of_bounds_cells == 5 and b2.out_of_bounds_cells == 10   # 5 rows tall
    assert b2.total < b1.total                                           # dense signal


def test_blocking_the_door_counts_cells_and_cuts_off_everything():
    room = example_room()
    blocked = dict(good_layout(), wardrobe=Placement(CATALOG["wardrobe"], 0, 2, NORTH))
    b = evaluate_layout(room, blocked)
    assert b.door_blocked_cells == 12          # the whole 4x3 entry zone
    assert b.unreachable_pieces == 5           # nobody can walk in


def test_moving_desk_away_from_window_lowers_that_score_only():
    room, layout = example_room(), good_layout()
    far = dict(layout, desk=Placement(CATALOG["desk"], 2, 9, WEST))   # still valid, off the wall
    a, b = evaluate_layout(room, layout), evaluate_layout(room, far)
    assert b.scores["desk_window"] < a.scores["desk_window"]


def test_weights_scale_the_penalty():
    room = example_room()
    from environment.reward import RewardWeights
    light = evaluate_layout(room, bad_layout(), RewardWeights())
    heavy = evaluate_layout(room, bad_layout(), replace(RewardWeights(), overlap_cell=2.0))
    assert heavy.penalty > light.penalty
