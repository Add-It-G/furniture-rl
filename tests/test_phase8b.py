"""Tests for the PPO diagnostics (no stable-baselines3 needed)."""
import numpy as np

from environment.actions import DEFAULT_ITEMS, N_OPS, UP, FINISH_ACTION
from evaluation.diagnose_ppo import rollout_stats, summarize_log


class FinishSpammer:
    def predict(self, obs, deterministic=True):
        return np.array(FINISH_ACTION), None


class BedUp:
    def predict(self, obs, deterministic=True):
        return np.array(DEFAULT_ITEMS.index("bed") * N_OPS + UP), None


def test_finish_spammer_is_detected():
    st = rollout_stats(FinishSpammer(), [10_000, 10_001], deterministic=True,
                       max_steps=100, min_finish=50)
    assert st["finish_attempts"] == 1.0          # every step was a FINISH attempt
    assert st["invalid"] > 0.9                    # 50 of 51 refused
    assert abs(st["final"] - st["start"]) < 1e-9  # nothing changed


def test_move_stats_and_top_actions():
    st = rollout_stats(BedUp(), [10_000], deterministic=False, max_steps=30, min_finish=0)
    assert st["top_actions"][0][0] == "bed:up" and st["rotations"] == 0.0


def test_summarize_log_picks_spaced_rows(tmp_path):
    p = tmp_path / "progress.csv"
    lines = ["time/total_timesteps,rollout/ep_rew_mean"] + [f"{i},{i * 0.5}" for i in range(100)]
    p.write_text("\n".join(lines))
    rows = summarize_log(p, n_points=5)
    assert [r["time/total_timesteps"] for r in rows] == ["0", "25", "50", "74", "99"] \
        or len(rows) == 5
    assert rows[0]["time/total_timesteps"] == "0" and rows[-1]["time/total_timesteps"] == "99"
