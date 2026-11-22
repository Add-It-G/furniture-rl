"""Look at how the reward behaves. Run from the project root:
    python -m evaluation.inspect_reward
"""
import numpy as np

from config import SEED
from environment.examples import bad_layout, example_room, good_layout
from environment.layout import random_layout
from environment.reward import RewardWeights, evaluate_layout
from environment.room import random_room


def show(label, b):
    print(f"{label:<12} total={b.total:7.2f}  bonus={b.bonus:5.2f}  penalty={b.penalty:6.2f}  "
          f"overlap={b.overlap_cells} oob={b.out_of_bounds_cells} "
          f"door={b.door_blocked_cells} unreachable={b.unreachable_pieces}")
    print("             " + "  ".join(f"{k}={v:.2f}" for k, v in b.scores.items()))


room = example_room()
show("good", evaluate_layout(room, good_layout()))
show("bad", evaluate_layout(room, bad_layout()))

w = RewardWeights()
max_bonus = sum(getattr(w, k) for k in w.__dataclass_fields__ if k in
                ("desk_window", "sofa_tv_facing", "sofa_tv_distance",
                 "bed_wardrobe_distance", "clearance", "path_directness"))
print(f"\nMaximum possible bonus: {max_bonus}")

rng = np.random.default_rng(SEED)
totals, valid = [], 0
N = 1000
for _ in range(N):
    r = random_room(rng)
    b = evaluate_layout(r, random_layout(r, rng))
    totals.append(b.total)
    valid += b.is_valid
totals = np.array(totals)
print(f"\n{N} random layouts in random rooms:")
print(f"  valid: {valid}/{N}   mean total: {totals.mean():.2f}   "
      f"best: {totals.max():.2f}   worst: {totals.min():.2f}")
