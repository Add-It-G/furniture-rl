"""Phase 2 tests: run with `pytest -v` from the furniture-rl folder."""
import numpy as np

from environment.furniture import (CATALOG, EAST, NORTH, Placement,
                                   meters_to_cells)
from environment.room import Opening, Room, random_room


def test_meters_to_cells_rounds_up_without_float_noise():
    assert meters_to_cells(1.0) == 4      # exact, must not become 5
    assert meters_to_cells(1.4) == 6      # 5.6 -> 6
    assert meters_to_cells(0.1) == 1      # never zero


def test_bed_footprint_and_rotation_swap():
    bed = CATALOG["bed"]
    assert (bed.width_cells, bed.depth_cells) == (6, 8)
    assert Placement(bed, 0, 0, NORTH).size == (8, 6)
    assert Placement(bed, 0, 0, EAST).size == (6, 8)
    assert len(Placement(bed, 0, 0).cells()) == 48


def test_front_cells_point_the_right_way():
    desk = Placement(CATALOG["desk"], 5, 5, NORTH)     # size (3 rows, 5 cols)
    assert all(r < 5 for r, _ in desk.front_cells())   # above the top edge
    assert len(desk.front_cells()) == desk.spec.clearance_cells * 5
    assert all(r > 7 for r, _ in desk.rotated(2).front_cells())  # SOUTH: below
    assert all(c > 5 + 3 - 1 for _, c in desk.rotated(1).front_cells())  # EAST
    assert all(c < 5 for _, c in desk.rotated(3).front_cells())  # WEST


def test_occupancy_detects_overlap_and_out_of_bounds():
    room = Room(10, 10, Opening("N", 2, 4), ())
    a = Placement(CATALOG["chair"], 0, 0)      # 2x2 at top-left
    b = Placement(CATALOG["chair"], 1, 1)      # overlaps one cell with a
    grid, oob = room.occupancy([a, b])
    assert (grid > 1).sum() == 1 and oob == 0
    c = Placement(CATALOG["chair"], 9, 9)      # 3 of its 4 cells are outside
    _, oob = room.occupancy([c])
    assert oob == 3


def test_random_room_is_reproducible_and_valid():
    r1 = random_room(np.random.default_rng(7))
    r2 = random_room(np.random.default_rng(7))
    assert r1 == r2
    for seed in range(200):
        room = random_room(np.random.default_rng(seed))
        assert all(w.wall != room.door.wall for w in room.windows)
        for op in (room.door, *room.windows):
            limit = room.n_cols if op.wall in ("N", "S") else room.n_rows
            assert 1 <= op.offset and op.offset + op.length < limit
