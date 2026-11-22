"""A Layout = the movable furniture placed in a room.

Layout is a plain dict {name: Placement} holding the pieces the optimizer
controls. The chair is derived from the desk, so it never appears in a Layout
but always appears in build_pieces().
"""
from __future__ import annotations

import numpy as np

from environment.furniture import (CATALOG, DEFAULT_ITEMS, EAST, NORTH, SOUTH,
                                   WEST, Placement)
from environment.room import Room

Layout = dict[str, Placement]


def chair_for_desk(desk: Placement) -> Placement:
    """Place the chair flush against the desk's front edge, centred on it."""
    spec = CATALOG["chair"]
    desk_rows, desk_cols = desk.size
    ch_rows, ch_cols = spec.depth_cells, spec.width_cells   # chair is square-ish
    facing_desk = (desk.rotation + 2) % 4                   # chair faces the desk

    if desk.rotation == NORTH:
        row, col = desk.row - ch_rows, desk.col + (desk_cols - ch_cols) // 2
    elif desk.rotation == SOUTH:
        row, col = desk.row + desk_rows, desk.col + (desk_cols - ch_cols) // 2
    elif desk.rotation == EAST:
        row, col = desk.row + (desk_rows - ch_rows) // 2, desk.col + desk_cols
    else:  # WEST
        row, col = desk.row + (desk_rows - ch_rows) // 2, desk.col - ch_cols
    return Placement(spec, row, col, facing_desk)


def build_pieces(layout: Layout) -> dict[str, Placement]:
    """Everything physically in the room: the layout plus the derived chair."""
    pieces = dict(layout)
    pieces["chair"] = chair_for_desk(layout["desk"])
    return pieces


def random_layout(room: Room, rng: np.random.Generator) -> Layout:
    """Random layout with every piece inside the room (overlaps still possible)."""
    layout: Layout = {}
    for name in DEFAULT_ITEMS:
        rotation = int(rng.integers(0, 4))
        probe = Placement(CATALOG[name], 0, 0, rotation)
        n_rows, n_cols = probe.size
        row = int(rng.integers(0, max(1, room.n_rows - n_rows + 1)))
        col = int(rng.integers(0, max(1, room.n_cols - n_cols + 1)))
        layout[name] = Placement(CATALOG[name], row, col, rotation)
    return layout
