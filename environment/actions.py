"""The action set, shared by the Gymnasium env and the greedy / annealing solvers.

30 piece-actions + 1 FINISH = Discrete(31).  action = piece_index * 6 + op.
"""
from __future__ import annotations

from environment.furniture import DEFAULT_ITEMS
from environment.layout import Layout
from environment.room import Room

# op codes
UP, DOWN, LEFT, RIGHT, ROTATE_CW, ROTATE_CCW = range(6)
N_OPS = 6
OP_NAMES = ("up", "down", "left", "right", "rotate_cw", "rotate_ccw")
_MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

N_PIECES = len(DEFAULT_ITEMS)
FINISH_ACTION = N_PIECES * N_OPS           # 30
N_ACTIONS = FINISH_ACTION + 1              # 31


def describe_action(action: int) -> str:
    if action == FINISH_ACTION:
        return "finish"
    return f"{DEFAULT_ITEMS[action // N_OPS]}:{OP_NAMES[action % N_OPS]}"


def apply_action(room: Room, layout: Layout, action: int) -> Layout | None:
    """Return the layout after `action`, or None if the action is invalid.

    Invalid = would leave the piece with ZERO cells inside the room. Partly
    leaving is allowed (and penalised per cell by the reward) so the agent can
    feel the gradient of "less outside is better". FINISH is handled by the caller.
    """
    if not 0 <= action < FINISH_ACTION:
        raise ValueError(f"apply_action expects 0..{FINISH_ACTION - 1}, got {action}")
    name = DEFAULT_ITEMS[action // N_OPS]
    op = action % N_OPS
    piece = layout[name]
    if op in _MOVES:
        new_piece = piece.moved(*_MOVES[op])
    elif op == ROTATE_CW:
        new_piece = piece.rotated(1)
    else:
        new_piece = piece.rotated(-1)
    if not any(room.in_bounds(r, c) for r, c in new_piece.cells()):
        return None
    new_layout = dict(layout)
    new_layout[name] = new_piece
    return new_layout
