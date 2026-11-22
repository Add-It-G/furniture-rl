"""Success criteria, fixed before PPO was trained (see config.py)."""
from config import FULL_WIN_MEAN, FULL_WIN_VALID, USEFUL_MEAN, USEFUL_VALID


def classify(mean_score: float, n_valid: int) -> str:
    """'full win', 'useful trade-off' or 'loss'. Both conditions must hold."""
    if mean_score >= FULL_WIN_MEAN and n_valid >= FULL_WIN_VALID:
        return "full win"
    if mean_score >= USEFUL_MEAN and n_valid >= USEFUL_VALID:
        return "useful trade-off"
    return "loss"
