"""Hand-built example room and layouts, used by tests and demos."""
from __future__ import annotations

from environment.furniture import CATALOG, EAST, NORTH, WEST, Placement
from environment.layout import Layout
from environment.room import Opening, Room


def example_room() -> Room:
    """16 x 18 cells (4.0 x 4.5 m); door on the north wall, window on the east."""
    return Room(16, 18, Opening("N", 2, 4), (Opening("E", 5, 4),))


def good_layout() -> Layout:
    c = CATALOG
    return {
        "bed":      Placement(c["bed"], 8, 12, NORTH),
        "wardrobe": Placement(c["wardrobe"], 13, 6, NORTH),
        "desk":     Placement(c["desk"], 2, 15, WEST),      # by the window
        "sofa":     Placement(c["sofa"], 1, 8, WEST),       # faces the TV
        "tv_stand": Placement(c["tv_stand"], 2, 0, EAST),   # faces the sofa
    }


def bad_layout() -> Layout:
    """Everything piled in the top-left corner."""
    return {n: Placement(CATALOG[n], 0, 0, NORTH)
            for n in ("bed", "wardrobe", "desk", "sofa", "tv_stand")}
