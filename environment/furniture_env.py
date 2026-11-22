"""Gymnasium environment: arrange furniture one small move at a time.

MDP:  state  = room + current layout (+ progress through the episode)
      action = move / rotate one piece, or FINISH        (Discrete(31))
      reward = score(new layout) - score(old layout)     (potential-based)
"""
from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (INVALID_ACTION_PENALTY, MAX_EPISODE_STEPS, MAX_WINDOWS,
                    ROOM_MAX_CELLS)
from environment.actions import (FINISH_ACTION, N_ACTIONS, apply_action,
                                 describe_action)
from environment.furniture import DEFAULT_ITEMS
from environment.layout import Layout, build_pieces, random_layout
from environment.reward import RewardBreakdown, RewardWeights, evaluate_layout
from environment.room import WALLS, Opening, Room, random_room, render_ascii

SCALE = float(ROOM_MAX_CELLS)          # normalise cell counts to roughly [0, 1]
_PER_OPENING = 1 + len(WALLS) + 2      # present flag, wall one-hot, offset, length
_PER_PIECE = 2 + 4 + 2                 # row, col, rotation one-hot, footprint rows/cols
OBS_DIM = 2 + _PER_OPENING + MAX_WINDOWS * _PER_OPENING + len(DEFAULT_ITEMS) * _PER_PIECE + 1
GRID_DIM = ROOM_MAX_CELLS * ROOM_MAX_CELLS        # 20 x 20 occupancy picture, flattened
OBS_DIM_GRID = OBS_DIM + GRID_DIM
OBS_MODES = ("vector", "grid")


def room_for_seed(seed: int) -> Room:
    """Deterministic room from a seed (this is what makes train/test pools work)."""
    return random_room(np.random.default_rng(seed))


def _opening_features(op: Opening | None) -> list[float]:
    if op is None:
        return [0.0] * _PER_OPENING
    wall = [1.0 if w == op.wall else 0.0 for w in WALLS]
    return [1.0, *wall, op.offset / SCALE, op.length / SCALE]


def _occupancy_features(room: Room, layout: Layout) -> np.ndarray:
    """A 20x20 'picture' of the room: 0 empty, 0.5 one piece, 1.0 overlap (2+ pieces),
    -1 outside the room (padding). Gives the network direct sight of overlaps."""
    grid, _ = room.occupancy(list(build_pieces(layout).values()))
    canvas = np.full((ROOM_MAX_CELLS, ROOM_MAX_CELLS), -1.0, dtype=np.float32)
    canvas[:room.n_rows, :room.n_cols] = np.minimum(grid, 2) / 2.0
    return canvas.ravel()


def encode_observation(room: Room, layout: Layout, progress: float,
                       with_grid: bool = False) -> np.ndarray:
    """Flatten the whole situation into a fixed-length float vector.

    Layout:  room size (2) | door (7) | window slots (2 x 7) |
             per piece (5 x 8): row, col, rotation one-hot (4), footprint rows, cols |
             progress (1) = steps used / max steps
    """
    feats: list[float] = [room.n_rows / SCALE, room.n_cols / SCALE]
    feats += _opening_features(room.door)
    slots = list(room.windows) + [None] * (MAX_WINDOWS - len(room.windows))
    for w in slots[:MAX_WINDOWS]:
        feats += _opening_features(w)
    for name in DEFAULT_ITEMS:
        p = layout[name]
        n_rows, n_cols = p.size
        feats += [p.row / SCALE, p.col / SCALE]
        feats += [1.0 if p.rotation == k else 0.0 for k in range(4)]
        feats += [n_rows / SCALE, n_cols / SCALE]
    feats.append(progress)
    vec = np.asarray(feats, dtype=np.float32)
    if with_grid:
        vec = np.concatenate([vec, _occupancy_features(room, layout)])
    return vec


class FurnitureEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, room_seeds: Sequence[int] | None = None,
                 max_steps: int = MAX_EPISODE_STEPS,
                 invalid_action_penalty: float = INVALID_ACTION_PENALTY,
                 weights: RewardWeights = RewardWeights(),
                 min_steps_before_finish: int = 0,
                 obs_mode: str = "vector",
                 render_mode: str | None = None) -> None:
        super().__init__()
        if obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode must be one of {OBS_MODES}, got {obs_mode!r}")
        self.obs_mode = obs_mode
        self.room_seeds = list(room_seeds) if room_seeds is not None else None
        self.max_steps = max_steps
        self.invalid_action_penalty = invalid_action_penalty
        self.weights = weights
        # FINISH is refused (treated as an invalid action) until this many steps
        # have been taken. 0 = always allowed. See the Phase 8 notes for why.
        self.min_steps_before_finish = min_steps_before_finish
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(N_ACTIONS)
        obs_dim = OBS_DIM_GRID if obs_mode == "grid" else OBS_DIM
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(obs_dim,), dtype=np.float32)

        self.room: Room | None = None
        self.layout: Layout = {}
        self._breakdown: RewardBreakdown | None = None
        self._steps = 0

    # ---- Gymnasium API ----
    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        """Start a new episode.

        options may contain "room" and/or "layout" to force a specific start
        (used so every algorithm is compared from the SAME starting point).
        """
        super().reset(seed=seed)            # seeds self.np_random
        options = options or {}
        if "room" in options:
            self.room = options["room"]
        elif self.room_seeds is not None:
            self.room = room_for_seed(int(self.np_random.choice(self.room_seeds)))
        else:
            self.room = random_room(self.np_random)
        self.layout = dict(options["layout"]) if "layout" in options \
            else random_layout(self.room, self.np_random)
        self._steps = 0
        self._breakdown = evaluate_layout(self.room, self.layout, self.weights)
        return self._observe(), self._info(invalid=False)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0..{N_ACTIONS - 1}")
        too_early = action == FINISH_ACTION and self._steps < self.min_steps_before_finish
        self._steps += 1
        reward, terminated, invalid = 0.0, False, False

        if too_early:
            invalid = True
            reward = -self.invalid_action_penalty
        elif action == FINISH_ACTION:
            terminated = True               # no reward: the score is already banked in the deltas
        else:
            new_layout = apply_action(self.room, self.layout, action)
            if new_layout is None:
                invalid = True
                reward = -self.invalid_action_penalty
            else:
                new_breakdown = evaluate_layout(self.room, new_layout, self.weights)
                reward = new_breakdown.total - self._breakdown.total
                self.layout, self._breakdown = new_layout, new_breakdown

        truncated = (not terminated) and self._steps >= self.max_steps
        return self._observe(), float(reward), terminated, truncated, self._info(invalid)

    def render(self) -> str | None:
        if self.render_mode == "ansi" and self.room is not None:
            from environment.layout import build_pieces
            return render_ascii(self.room, list(build_pieces(self.layout).values()))
        return None

    # ---- helpers ----
    def _observe(self) -> np.ndarray:
        return encode_observation(self.room, self.layout, self._steps / self.max_steps,
                                  with_grid=self.obs_mode == "grid")

    def _info(self, invalid: bool) -> dict[str, Any]:
        b = self._breakdown
        return {"score": b.total, "is_valid": b.is_valid, "n_violations": b.n_violations,
                "invalid_action": invalid, "breakdown": b, "layout": self.layout}
