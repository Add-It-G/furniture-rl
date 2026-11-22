"""Room geometry: size, door, windows, occupancy, random room generation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (DOOR_LENGTH_CELLS, MAX_WINDOWS, ROOM_MAX_CELLS,
                    ROOM_MIN_CELLS, WINDOW_LENGTH_CELLS)
from environment.furniture import Cell, Placement

WALLS = ("N", "E", "S", "W")


@dataclass(frozen=True)
class Opening:
    """A door or window cut into one wall.

    wall:   which wall ("N","E","S","W").
    offset: cells from the wall's start (left end for N/S, top end for E/W).
    length: how many cells wide.
    """
    wall: str
    offset: int
    length: int

    def inside_cells(self, n_rows: int, n_cols: int, depth: int = 1) -> list[Cell]:
        """Room cells right next to the opening, `depth` layers into the room."""
        span = range(self.offset, self.offset + self.length)
        if self.wall == "N":
            return [(d, c) for d in range(depth) for c in span]
        if self.wall == "S":
            return [(n_rows - 1 - d, c) for d in range(depth) for c in span]
        if self.wall == "W":
            return [(r, d) for d in range(depth) for r in span]
        return [(r, n_cols - 1 - d) for d in range(depth) for r in span]  # "E"


@dataclass(frozen=True)
class Room:
    n_rows: int
    n_cols: int
    door: Opening
    windows: tuple[Opening, ...]

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.n_rows and 0 <= col < self.n_cols

    def door_zone(self, depth: int = 3) -> list[Cell]:
        """Cells just inside the door that must stay free (swing + entry)."""
        return self.door.inside_cells(self.n_rows, self.n_cols, depth)

    def window_zone(self, depth: int = 1) -> list[Cell]:
        """Cells just inside every window (used for 'desk near window')."""
        cells: list[Cell] = []
        for w in self.windows:
            cells += w.inside_cells(self.n_rows, self.n_cols, depth)
        return cells

    def occupancy(self, placements: list[Placement]) -> tuple[np.ndarray, int]:
        """Count how many pieces cover each cell.

        Returns (grid, out_of_bounds_cells). A cell value > 1 means overlap.
        """
        grid = np.zeros((self.n_rows, self.n_cols), dtype=np.int16)
        out_of_bounds = 0
        for p in placements:
            for r, c in p.cells():
                if self.in_bounds(r, c):
                    grid[r, c] += 1
                else:
                    out_of_bounds += 1
        return grid, out_of_bounds


def random_room(rng: np.random.Generator) -> Room:
    """Generate a random valid room. Same seed -> same room (reproducible)."""
    n_rows = int(rng.integers(ROOM_MIN_CELLS, ROOM_MAX_CELLS + 1))
    n_cols = int(rng.integers(ROOM_MIN_CELLS, ROOM_MAX_CELLS + 1))

    def wall_len(wall: str) -> int:
        return n_cols if wall in ("N", "S") else n_rows

    def make_opening(wall: str, length: int) -> Opening:
        # Keep 1 cell away from the corners so openings sit on a straight wall.
        offset = int(rng.integers(1, wall_len(wall) - length))
        return Opening(wall, offset, length)

    door_wall = str(rng.choice(WALLS))
    door = make_opening(door_wall, DOOR_LENGTH_CELLS)

    other_walls = [w for w in WALLS if w != door_wall]
    n_windows = int(rng.integers(1, MAX_WINDOWS + 1))
    chosen = rng.choice(other_walls, size=n_windows, replace=False)
    windows = tuple(make_opening(str(w), WINDOW_LENGTH_CELLS) for w in chosen)
    return Room(n_rows, n_cols, door, windows)


def render_ascii(room: Room, placements: list[Placement]) -> str:
    """Quick text picture of a layout. '#' wall, '=' door, '~' window, 'X' overlap."""
    canvas = [["#"] * (room.n_cols + 2) for _ in range(room.n_rows + 2)]
    for r in range(room.n_rows):
        for c in range(room.n_cols):
            canvas[r + 1][c + 1] = "."

    def mark(op: Opening, ch: str) -> None:
        for i in range(op.offset, op.offset + op.length):
            if op.wall == "N":
                canvas[0][i + 1] = ch
            elif op.wall == "S":
                canvas[room.n_rows + 1][i + 1] = ch
            elif op.wall == "W":
                canvas[i + 1][0] = ch
            else:
                canvas[i + 1][room.n_cols + 1] = ch

    mark(room.door, "=")
    for w in room.windows:
        mark(w, "~")

    grid, _ = room.occupancy(placements)
    for p in placements:
        letter = p.spec.name[0].upper()
        for r, c in p.cells():
            if room.in_bounds(r, c):
                canvas[r + 1][c + 1] = "X" if grid[r, c] > 1 else letter
    return "\n".join("".join(row) for row in canvas)
