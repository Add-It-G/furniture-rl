"""Furniture specs and placement geometry on the grid.

A FurnitureSpec says what a piece of furniture IS (name, real size, how much
free space it needs in front). A Placement says where one piece currently IS
(row, column, rotation). Specs never change; placements change constantly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from config import CELL_SIZE_M

# Rotation codes = the direction the furniture's FRONT faces.
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
ROTATIONS = (NORTH, EAST, SOUTH, WEST)

# Unit vector (d_row, d_col) of the direction each rotation faces.
FRONT_VECTORS = {NORTH: (-1, 0), EAST: (0, 1), SOUTH: (1, 0), WEST: (0, -1)}

Cell = tuple[int, int]  # (row, col); row 0 is the top of the room


def meters_to_cells(meters: float) -> int:
    """Convert a real length to grid cells, rounding UP (never smaller than real).

    round(..., 6) first so float noise like 4.0000000001 doesn't become 5.
    """
    return max(1, math.ceil(round(meters / CELL_SIZE_M, 6)))


@dataclass(frozen=True)
class FurnitureSpec:
    """Immutable description of a furniture type.

    width_m: size along the front edge (e.g. a sofa's long side).
    depth_m: size front-to-back.
    clearance_cells: free cells needed directly in front (to sit, open doors...).
    """
    name: str
    width_m: float
    depth_m: float
    clearance_cells: int = 0

    @property
    def width_cells(self) -> int:
        return meters_to_cells(self.width_m)

    @property
    def depth_cells(self) -> int:
        return meters_to_cells(self.depth_m)


@dataclass(frozen=True)
class Placement:
    """One furniture piece at a position. (row, col) is its TOP-LEFT cell.

    Frozen on purpose: to "move" a piece you create a new Placement. That makes
    it safe for search algorithms to try neighbours without corrupting the
    current layout.
    """
    spec: FurnitureSpec
    row: int
    col: int
    rotation: int = NORTH

    @property
    def size(self) -> tuple[int, int]:
        """(n_rows, n_cols) of the footprint. Rotating 90 degrees swaps them."""
        if self.rotation in (NORTH, SOUTH):
            return self.spec.depth_cells, self.spec.width_cells
        return self.spec.width_cells, self.spec.depth_cells

    def cells(self) -> list[Cell]:
        """Every grid cell this piece covers (may include out-of-room cells)."""
        n_rows, n_cols = self.size
        return [(self.row + r, self.col + c)
                for r in range(n_rows) for c in range(n_cols)]

    def front_cells(self) -> list[Cell]:
        """Cells in the clearance zone directly in front of the piece."""
        n_rows, n_cols = self.size
        k = self.spec.clearance_cells
        if k == 0:
            return []
        if self.rotation == NORTH:   # front is the top edge
            return [(self.row - i, self.col + c)
                    for i in range(1, k + 1) for c in range(n_cols)]
        if self.rotation == SOUTH:   # front is the bottom edge
            return [(self.row + n_rows - 1 + i, self.col + c)
                    for i in range(1, k + 1) for c in range(n_cols)]
        if self.rotation == EAST:    # front is the right edge
            return [(self.row + r, self.col + n_cols - 1 + i)
                    for i in range(1, k + 1) for r in range(n_rows)]
        # WEST: front is the left edge
        return [(self.row + r, self.col - i)
                for i in range(1, k + 1) for r in range(n_rows)]

    def bounds(self) -> tuple[int, int, int, int]:
        """Inclusive bounding box (row_min, col_min, row_max, col_max)."""
        n_rows, n_cols = self.size
        return self.row, self.col, self.row + n_rows - 1, self.col + n_cols - 1

    def center(self) -> tuple[float, float]:
        """Centre of the footprint in (row, col) cell coordinates."""
        n_rows, n_cols = self.size
        return self.row + (n_rows - 1) / 2, self.col + (n_cols - 1) / 2

    def moved(self, d_row: int, d_col: int) -> "Placement":
        return replace(self, row=self.row + d_row, col=self.col + d_col)

    def rotated(self, steps: int = 1) -> "Placement":
        """Rotate clockwise by 90 degrees * steps, keeping the top-left fixed."""
        return replace(self, rotation=(self.rotation + steps) % 4)


# Realistic sizes in metres (width x depth).
CATALOG: dict[str, FurnitureSpec] = {
    "bed":        FurnitureSpec("bed", 1.4, 2.0, clearance_cells=1),
    "wardrobe":   FurnitureSpec("wardrobe", 1.0, 0.6, clearance_cells=2),
    "desk":       FurnitureSpec("desk", 1.2, 0.6, clearance_cells=3),
    "chair":      FurnitureSpec("chair", 0.5, 0.5),
    "sofa":       FurnitureSpec("sofa", 2.0, 0.9, clearance_cells=2),
    "tv_stand":   FurnitureSpec("tv_stand", 1.2, 0.4),
    "nightstand": FurnitureSpec("nightstand", 0.45, 0.4),
}

# The pieces the optimizer moves. The chair is NOT here: it is derived from the
# desk (see layout.py), so the agent controls 5 pieces and the room holds 6.
DEFAULT_ITEMS = ["bed", "wardrobe", "desk", "sofa", "tv_stand"]
