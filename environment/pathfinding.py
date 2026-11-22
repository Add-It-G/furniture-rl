"""Breadth-first search on the room grid.

One BFS from the door gives the walking distance to EVERY cell at once, which
is exactly what the reward needs ("can I reach each piece of furniture, and
how direct is the route?"). A* is for one start -> one goal; it would run once
per piece and gain nothing on a grid this small.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from environment.furniture import Cell


def bfs_distances(free: np.ndarray, starts: Iterable[Cell]) -> np.ndarray:
    """Walking distance (in steps) from the nearest start cell to every cell.

    free:   2D bool array, True where a person can stand (no furniture).
    starts: cells to start from (e.g. the cells just inside the door).
    Returns an int array; -1 means "unreachable".
    Movement is 4-directional (up/down/left/right), each step costs 1.
    """
    n_rows, n_cols = free.shape
    free_l = free.tolist()                       # list lookups are faster than numpy's
    dist = [[-1] * n_cols for _ in range(n_rows)]
    queue: deque[Cell] = deque()

    for r, c in starts:
        if free_l[r][c] and dist[r][c] == -1:
            dist[r][c] = 0
            queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        d = dist[r][c] + 1
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < n_rows and 0 <= nc < n_cols \
                    and free_l[nr][nc] and dist[nr][nc] == -1:
                dist[nr][nc] = d
                queue.append((nr, nc))
    return np.array(dist, dtype=np.int32)


def reconstruct_path(dist: np.ndarray, goal: Cell) -> list[Cell]:
    """Walk back from `goal` to a start cell, one step closer each time.

    Works because bfs_distances numbers every cell with its distance from the
    nearest start: from any reachable cell, a neighbour with distance-1 exists.
    Returns the path ordered start -> goal (empty if the goal is unreachable).
    """
    r, c = goal
    if dist[r, c] < 0:
        return []
    path = [goal]
    n_rows, n_cols = dist.shape
    while dist[r, c] > 0:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < n_rows and 0 <= nc < n_cols and dist[nr, nc] == dist[r, c] - 1:
                r, c = nr, nc
                path.append((r, c))
                break
    return path[::-1]


def shortest_path(free: np.ndarray, starts: Iterable[Cell],
                  goals: Iterable[Cell]) -> list[Cell] | None:
    """Cells of a shortest walk from some start to the NEAREST goal (None if impossible).

    Same BFS as above, but remembering each cell's parent so the route can be
    traced back. Used only for drawing the walking paths.
    """
    n_rows, n_cols = free.shape
    goal_set = set(goals)
    parent: dict[Cell, Cell | None] = {}
    queue: deque[Cell] = deque()
    for s in starts:
        if free[s] and s not in parent:
            parent[s] = None
            queue.append(s)
    while queue:
        cell = queue.popleft()
        if cell in goal_set:
            path: list[Cell] = []
            node: Cell | None = cell
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]
        r, c = cell
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nb[0] < n_rows and 0 <= nb[1] < n_cols \
                    and free[nb] and nb not in parent:
                parent[nb] = cell
                queue.append(nb)
    return None
