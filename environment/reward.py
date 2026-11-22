"""The reward function: scores how good a furniture layout is.

Every optimizer in this project (random, greedy, annealing, PPO) calls
evaluate_layout(). Total = (weighted soft bonuses) - (weighted hard penalties).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from environment.furniture import FRONT_VECTORS, Cell, Placement
from environment.layout import Layout, build_pieces
from environment.pathfinding import bfs_distances
from environment.room import Room

IMPORTANT_PIECES = ("bed", "wardrobe", "desk", "sofa", "tv_stand")


@dataclass(frozen=True)
class RewardWeights:
    """Illustrative starting weights. See the tuning notes in the Phase 3 write-up."""
    # Hard penalties: charged PER UNIT of violation (dense on purpose).
    overlap_cell: float = 0.5
    out_of_bounds_cell: float = 0.5
    door_blocked_cell: float = 1.0
    unreachable_piece: float = 3.0
    # Soft bonuses: each underlying score is in [0, 1].
    desk_window: float = 1.0
    sofa_tv_facing: float = 1.0
    sofa_tv_distance: float = 0.5
    bed_wardrobe_distance: float = 0.5
    clearance: float = 1.5
    path_directness: float = 0.5


@dataclass(frozen=True)
class RewardBreakdown:
    """Everything that went into a score, so we can debug and report it."""
    overlap_cells: int
    out_of_bounds_cells: int
    door_blocked_cells: int
    unreachable_pieces: int
    scores: dict[str, float]
    penalty: float
    bonus: float

    @property
    def total(self) -> float:
        return self.bonus - self.penalty

    @property
    def n_violations(self) -> int:
        return (self.overlap_cells + self.out_of_bounds_cells
                + self.door_blocked_cells + self.unreachable_pieces)

    @property
    def is_valid(self) -> bool:
        return self.n_violations == 0


# ---------- small geometry helpers ----------

def _rect_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Empty cells between two inclusive boxes (Manhattan). 0 = touching/overlap."""
    row_gap = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    col_gap = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    return row_gap + col_gap


def _range_score(value: float, lo: float, hi: float, slack: float) -> float:
    """1.0 inside [lo, hi]; fades linearly to 0 over `slack` outside it."""
    if lo <= value <= hi:
        return 1.0
    miss = lo - value if value < lo else value - hi
    return max(0.0, 1.0 - miss / slack)


def _facing_score(a: Placement, b: Placement) -> float:
    """cos(angle between a's front and the direction a->b), clipped at 0."""
    (ar, ac), (br, bc) = a.center(), b.center()
    dr, dc = br - ar, bc - ac
    norm = math.hypot(dr, dc)
    if norm == 0:
        return 0.0
    fr, fc = FRONT_VECTORS[a.rotation]
    return max(0.0, (fr * dr + fc * dc) / norm)


def _adjacent_cells(p: Placement) -> list[Cell]:
    own = set(p.cells())
    out: set[Cell] = set()
    for r, c in own:
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nb not in own:
                out.add(nb)
    return list(out)


def _access_cells(room: Room, p: Placement, ignore: set[Cell]) -> list[Cell]:
    """Where a person stands to use a piece: its front, or any side if it has no front."""
    cells = p.front_cells() if p.spec.clearance_cells > 0 else _adjacent_cells(p)
    return [(r, c) for r, c in cells if room.in_bounds(r, c) and (r, c) not in ignore]


def _clearance_fraction(room: Room, grid: np.ndarray, p: Placement,
                        ignore: set[Cell]) -> float:
    cells = [c for c in p.front_cells() if c not in ignore]
    if not cells:
        return 1.0
    free = sum(1 for r, c in cells if room.in_bounds(r, c) and grid[r, c] == 0)
    return free / len(cells)


# ---------- the main entry point ----------

def evaluate_layout(room: Room, layout: Layout,
                    weights: RewardWeights = RewardWeights()) -> RewardBreakdown:
    pieces = build_pieces(layout)
    grid, out_of_bounds = room.occupancy(list(pieces.values()))

    # --- hard constraints (counted per cell / per piece) ---
    overlap = int(np.clip(grid - 1, 0, None).sum())
    door_blocked = sum(1 for r, c in room.door_zone(3) if grid[r, c] > 0)

    # --- walking paths ---
    free = grid == 0
    starts = [(r, c) for r, c in room.door.inside_cells(room.n_rows, room.n_cols, 1)
              if free[r, c]]
    dist = bfs_distances(free, starts)
    chair_cells = set(pieces["chair"].cells())      # the desk's own chair isn't an obstacle to it

    unreachable = 0
    ratios: list[float] = []
    for name in IMPORTANT_PIECES:
        ignore = chair_cells if name == "desk" else set()
        reachable = [(int(dist[r, c]), r, c)
                     for r, c in _access_cells(room, pieces[name], ignore)
                     if dist[r, c] >= 0]
        if not reachable:
            unreachable += 1
            continue
        d, r, c = min(reachable)
        straight = min(abs(r - sr) + abs(c - sc) for sr, sc in starts)
        ratios.append((straight + 1) / (d + 1))     # 1.0 = perfectly direct route

    # --- soft scores, each in [0, 1] ---
    desk_rect = pieces["desk"].bounds()
    window_gaps = [_rect_gap(desk_rect, (r, c, r, c)) for r, c in room.window_zone(1)]
    clear = [_clearance_fraction(room, grid, pieces[n],
                                 chair_cells if n == "desk" else set())
             for n in ("bed", "wardrobe", "desk", "sofa")]
    sofa, tv = pieces["sofa"], pieces["tv_stand"]
    scores = {
        "desk_window": _range_score(min(window_gaps), 0, 0, 8) if window_gaps else 0.0,
        "sofa_tv_facing": 0.5 * (_facing_score(sofa, tv) + _facing_score(tv, sofa)),
        "sofa_tv_distance": _range_score(_rect_gap(sofa.bounds(), tv.bounds()), 6, 12, 6),
        "bed_wardrobe_distance": _range_score(
            _rect_gap(pieces["bed"].bounds(), pieces["wardrobe"].bounds()), 1, 6, 8),
        "clearance": sum(clear) / len(clear),
        "path_directness": sum(ratios) / len(ratios) if ratios else 0.0,
    }

    bonus = sum(getattr(weights, k) * v for k, v in scores.items())
    penalty = (weights.overlap_cell * overlap
               + weights.out_of_bounds_cell * out_of_bounds
               + weights.door_blocked_cell * door_blocked
               + weights.unreachable_piece * unreachable)
    return RewardBreakdown(overlap, out_of_bounds, door_blocked, unreachable,
                           scores, penalty, bonus)


def compute_reward(room: Room, layout: Layout,
                   weights: RewardWeights = RewardWeights()) -> float:
    """Just the number. This is what the optimizers maximize."""
    return evaluate_layout(room, layout, weights).total
