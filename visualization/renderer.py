"""Top-down Matplotlib drawing of a room and a layout."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")          # draw to files; no window needed
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from environment.furniture import FRONT_VECTORS
from environment.layout import Layout, build_pieces
from environment.pathfinding import shortest_path
from environment.reward import IMPORTANT_PIECES, _access_cells
from environment.room import Opening, Room

COLORS = {"bed": "#8ecae6", "wardrobe": "#b08968", "desk": "#ffb703", "chair": "#fb8500",
          "sofa": "#90be6d", "tv_stand": "#8d99ae"}
LABELS = {"bed": "bed", "wardrobe": "ward.", "desk": "desk", "chair": "chair",
          "sofa": "sofa", "tv_stand": "TV"}


def _opening_line(room: Room, op: Opening) -> tuple[list[float], list[float]]:
    a, b = op.offset, op.offset + op.length
    if op.wall == "N":
        return [a, b], [0, 0]
    if op.wall == "S":
        return [a, b], [room.n_rows, room.n_rows]
    if op.wall == "W":
        return [0, 0], [a, b]
    return [room.n_cols, room.n_cols], [a, b]


def draw_layout(ax: Axes, room: Room, layout: Layout, title: str = "",
                show_paths: bool = True) -> None:
    """Draw walls, door (green) and windows (blue), furniture with orientation
    arrows, walking paths (dashed), overlaps (red) and blocked door cells (orange)."""
    pieces = build_pieces(layout)
    grid, _ = room.occupancy(list(pieces.values()))

    bounds = [p.bounds() for p in pieces.values()]
    min_r = min(0, *(b[0] for b in bounds)) - 1
    min_c = min(0, *(b[1] for b in bounds)) - 1
    max_r = max(room.n_rows, *(b[2] + 1 for b in bounds)) + 1
    max_c = max(room.n_cols, *(b[3] + 1 for b in bounds)) + 1
    ax.set_xlim(min_c, max_c)
    ax.set_ylim(max_r, min_r)                     # row 0 at the top
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)

    ax.add_patch(Rectangle((0, 0), room.n_cols, room.n_rows, facecolor="#f7f3ea",
                           edgecolor="black", lw=2.5, zorder=0))
    for r, c in room.door_zone(3):                # the area that must stay clear
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#2a9d8f", alpha=0.13, lw=0, zorder=1))
    xs, ys = _opening_line(room, room.door)
    ax.plot(xs, ys, color="#2a9d8f", lw=6, solid_capstyle="butt", zorder=5)
    for w in room.windows:
        xs, ys = _opening_line(room, w)
        ax.plot(xs, ys, color="#4cc9f0", lw=5, solid_capstyle="butt", zorder=5)

    for name, p in pieces.items():
        r0, c0, r1, c1 = p.bounds()
        outside = any(not room.in_bounds(r, c) for r, c in p.cells())
        ax.add_patch(Rectangle((c0, r0), c1 - c0 + 1, r1 - r0 + 1, facecolor=COLORS[name],
                               alpha=0.85, edgecolor="red" if outside else "black",
                               lw=2.5 if outside else 1.0, zorder=2))
        cy, cx = p.center()
        cy += 0.5; cx += 0.5
        ax.text(cx, cy, LABELS[name], ha="center", va="center", fontsize=7, zorder=6)
        dr, dc = FRONT_VECTORS[p.rotation]
        reach = 0.42 * ((c1 - c0 + 1) if dc else (r1 - r0 + 1))
        ax.annotate("", xy=(cx + dc * reach, cy + dr * reach), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", lw=1.1, color="#333333"), zorder=6)

    for r in range(room.n_rows):
        for c in range(room.n_cols):
            if grid[r, c] > 1:
                ax.add_patch(Rectangle((c, r), 1, 1, facecolor="red", alpha=0.65,
                                       hatch="///", edgecolor="darkred", lw=0, zorder=3))
    for r, c in room.door_zone(3):
        if grid[r, c] > 0:
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor="orange", alpha=0.6, lw=0, zorder=3))

    if show_paths:
        free = grid == 0
        starts = [(r, c) for r, c in room.door.inside_cells(room.n_rows, room.n_cols, 1)
                  if free[r, c]]
        chair_cells = set(pieces["chair"].cells())
        for name in IMPORTANT_PIECES:
            ignore = chair_cells if name == "desk" else set()
            goals = _access_cells(room, pieces[name], ignore)
            path = shortest_path(free, starts, goals) if starts else None
            if path:
                ax.plot([c + 0.5 for _, c in path], [r + 0.5 for r, _ in path], "--",
                        color="#264653", lw=1.1, alpha=0.6, zorder=4)
                ax.plot(path[-1][1] + 0.5, path[-1][0] + 0.5, "o", ms=2.5,
                        color="#264653", zorder=4)
            else:
                cy, cx = pieces[name].center()
                ax.text(cx + 0.5, cy + 1.6, "no path", color="red", fontsize=6,
                        ha="center", zorder=7)
    ax.set_title(title, fontsize=8)
