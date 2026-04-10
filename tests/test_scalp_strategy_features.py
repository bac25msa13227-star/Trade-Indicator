from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from xauusd_ai.features.scalp_features import build_all_scalp_features


def _make_bullish_frame() -> pd.DataFrame:
    times = pd.date_range("2026-03-30 06:40", periods=36, freq="min", tz="UTC")
    base = 100.0 + np.linspace(0.00, 0.35, len(times))
    open_ = base.copy()
    close = base + 0.02
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.10
    tick_volume = np.full(len(times), 100.0)
    tick_volume_delta = np.full(len(times), 6.0)
    volume_imbalance = np.full(len(times), 0.58)

    sweep_idx = 25
    open_[sweep_idx] = 100.10
    close[sweep_idx] = 100.86
    high[sweep_idx] = 100.96
    low[sweep_idx] = 99.45
    tick_volume[sweep_idx] = 280.0
    tick_volume_delta[sweep_idx] = 70.0
    volume_imbalance[sweep_idx] = 0.95

    for idx in (26, 27):
        open_[idx] = 100.75 + 0.04 * (idx - 26)
        close[idx] = 101.02 + 0.08 * (idx - 26)
        high[idx] = close[idx] + 0.10
        low[idx] = open_[idx] + 0.02
        tick_volume[idx] = 220.0
        tick_volume_delta[idx] = 45.0
        volume_imbalance[idx] = 0.88

    return pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": tick_volume,
            "tick_volume_delta": tick_volume_delta,
            "volume_imbalance": volume_imbalance,
        }
    )


def _make_bearish_frame() -> pd.DataFrame:
    times = pd.date_range("2026-03-30 06:40", periods=36, freq="min", tz="UTC")
    base = 100.4 - np.linspace(0.00, 0.35, len(times))
    open_ = base.copy()
    close = base - 0.02
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.10
    tick_volume = np.full(len(times), 100.0)
    tick_volume_delta = np.full(len(times), -6.0)
    volume_imbalance = np.full(len(times), 0.42)

    sweep_idx = 25
    open_[sweep_idx] = 99.90
    close[sweep_idx] = 99.14
    high[sweep_idx] = 100.60
    low[sweep_idx] = 99.04
    tick_volume[sweep_idx] = 280.0
    tick_volume_delta[sweep_idx] = -70.0
    volume_imbalance[sweep_idx] = 0.05

    for idx in (26, 27):
        open_[idx] = 99.25 - 0.04 * (idx - 26)
        close[idx] = 99.00 - 0.08 * (idx - 26)
        high[idx] = open_[idx] + 0.06
        low[idx] = close[idx] - 0.10
        tick_volume[idx] = 220.0
        tick_volume_delta[idx] = -45.0
        volume_imbalance[idx] = 0.12

    return pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": tick_volume,
            "tick_volume_delta": tick_volume_delta,
            "volume_imbalance": volume_imbalance,
        }
    )


class ScalpStrategyFeatureTests(unittest.TestCase):
    def test_build_adds_new_strategy_columns_and_bullish_signals(self) -> None:
        featured = build_all_scalp_features(_make_bullish_frame())

        for col in (
            "sm_turtle_soup",
            "sm_ote_score",
            "sm_ifvg",
            "sm_unicorn",
            "sm_po3_bias",
            "wyck_spring_utad",
            "wyck_sos_sow",
            "wyck_lps_quality",
            "trend_strength_score",
            "pullback_quality",
            "execution_quality",
            "strategy_setup_score",
        ):
            self.assertIn(col, featured.columns)

        self.assertGreaterEqual(float(featured["sm_turtle_soup"].max()), 1.0)
        self.assertGreaterEqual(float(featured["sm_po3_bias"].max()), 1.0)
        self.assertGreater(float(featured["strategy_setup_score"].tail(6).max()), 0.10)

    def test_build_scores_bearish_mirror_negative(self) -> None:
        featured = build_all_scalp_features(_make_bearish_frame())

        self.assertLessEqual(float(featured["sm_turtle_soup"].min()), -1.0)
        self.assertLess(float(featured["strategy_setup_score"].tail(6).min()), -0.01)


if __name__ == "__main__":
    unittest.main()
