"""Central configuration. Every tunable number lives here."""
from pathlib import Path

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SEED = 42            # fixed seed so experiments are reproducible

# Grid representation: the room is a grid of square cells.
CELL_SIZE_M = 0.25   # each cell is 0.25 m x 0.25 m

# Room generation (in cells): 12 cells = 3.0 m, 20 cells = 5.0 m
ROOM_MIN_CELLS = 12
ROOM_MAX_CELLS = 20

DOOR_LENGTH_CELLS = 4     # 4 * 0.25 m = 1.0 m wide door
WINDOW_LENGTH_CELLS = 4   # 1.0 m wide windows
MAX_WINDOWS = 2


def ensure_dirs() -> None:
    """Create output folders if they don't exist."""
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

# --- Environment / experiment settings (Phase 4) ---
MAX_EPISODE_STEPS = 200          # truncate an episode after this many actions
INVALID_ACTION_PENALTY = 0.1     # cost of an action that can't be applied

# Rooms come from seeds. Train and test pools must never overlap.
TRAIN_ROOM_SEEDS = list(range(0, 200))
TEST_ROOM_SEEDS = list(range(10_000, 10_050))

# --- Algorithm comparison (Phase 5+) ---
EVAL_BUDGET = 5_000      # max evaluate_layout() calls per solver per room

# --- Simulated annealing (Phase 7) ---
SA_T_START = 1.0         # accepts a -1 move ~37% of the time
SA_T_END = 0.02          # essentially hill climbing
SA_P_TRANSLATE = 0.8     # 80% of proposals are translations, 20% rotations

# --- PPO (Phase 8) ---
VAL_ROOM_SEEDS = list(range(20_000, 20_020))   # for monitoring/model selection, never for the final test
LOG_DIR = RESULTS_DIR / "ppo_logs"
PPO_TIMESTEPS = 300_000
PPO_N_ENVS = 4                 # 4 environments x 512 steps = 2048 steps per PPO update
PPO_MIN_FINISH_STEP = 50       # FINISH is refused before step 50 (see Phase 8 notes)
REWARD_SCALE = 0.1             # PPO sees rewards x0.1 (returns of ~20 become ~2)

# --- Success criteria, written down BEFORE seeing PPO's results ---
FULL_WIN_MEAN, FULL_WIN_VALID = 3.8, 40        # >= both: matches annealing
USEFUL_MEAN, USEFUL_VALID = 2.0, 30            # >= both: speed-quality trade-off
