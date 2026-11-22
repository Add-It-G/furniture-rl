"""Identical (room, initial layout) pairs for every algorithm.

make_start(seed) always returns the same pair for the same seed, so greedy,
annealing and PPO all begin each test room from exactly the same position.
"""
from __future__ import annotations

import numpy as np

from config import SEED
from environment.furniture_env import room_for_seed
from environment.layout import Layout, random_layout
from environment.room import Room


def make_start(room_seed: int) -> tuple[Room, Layout]:
    room = room_for_seed(room_seed)
    layout = random_layout(room, np.random.default_rng((SEED, room_seed, 1)))
    return room, layout


def solver_rng(room_seed: int, tag: int) -> np.random.Generator:
    """Per-(room, algorithm) random stream: reproducible and independent."""
    return np.random.default_rng((SEED, room_seed, 100 + tag))
