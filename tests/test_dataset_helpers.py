from __future__ import annotations

import unittest

import pandas as pd

from xauusd_ai.config import Settings
from xauusd_ai.features.dataset import (
    _build_date_mask,
    _build_sltp_label,
    _compute_sltp_realized_rr,
    _expected_direction_from_context,
    _session_spread_multiplier,
)


class DatasetHelperTests(unittest.TestCase):
    def test_build_date_mask_handles_utc_series(self) -> None:
        series = pd.to_datetime(
            [
                "2026-03-01 00:00:00+00:00",
                "2026-03-02 00:00:00+00:00",
                "2026-03-03 00:00:00+00:00",
            ]
        ).to_series(index=[0, 1, 2])
        mask = _build_date_mask(series, "2026-03-02", "2026-03-03")
        self.assertEqual(mask.tolist(), [False, True, True])

    def test_session_spread_multiplier_matches_ranges(self) -> None:
        hours = pd.Series([1, 8, 16, 22])
        mult = _session_spread_multiplier(hours)
        self.assertEqual(mult.tolist(), [1.5, 1.0, 1.2, 2.0])

    def test_expected_direction_uses_context_votes(self) -> None:
        df = pd.DataFrame(
            {
                "close": [100, 101, 102, 103],
                "daily_bias": [1, 1, 1, 1],
                "hourly_bias": [1, 1, 1, 1],
                "h4_market_structure_bias": [1, 1, 1, 1],
                "h4_ict_confluence": [2, 2, 2, 2],
                "strategy_score": [0.4, 0.4, 0.4, 0.4],
                "macd_hist": [0.1, 0.1, 0.1, 0.1],
                "rsi": [55, 56, 57, 58],
                "rsi_slope": [0.2, 0.2, 0.2, 0.2],
                "execution_quality": [0.2, 0.2, 0.2, 0.2],
            }
        )
        direction = _expected_direction_from_context(df)
        self.assertTrue((direction == 1).all())

    def test_sltp_label_marks_tp_first(self) -> None:
        settings = Settings()
        settings.training.sltp_label_max_horizon = 2
        settings.risk.take_profit_rr = 2.0
        settings.risk.stop_loss_atr_multiple = 1.0

        dataset = pd.DataFrame(
            {
                "open": [100.0, 100.4, 100.1, 99.4],
                "close": [100.0, 100.5, 100.2, 99.5],
                "high": [100.2, 102.2, 101.0, 100.0],
                "low": [99.8, 100.0, 98.8, 98.9],
                "atr": [1.0, 1.0, 1.0, 1.0],
                "expected_direction": [1, -1, 1, 1],
            }
        )
        labels = _build_sltp_label(dataset, settings)
        self.assertEqual(labels.iloc[0], 1)
        self.assertEqual(labels.iloc[1], 0)

    def test_compute_sltp_realized_rr_returns_series(self) -> None:
        settings = Settings()
        settings.training.sltp_label_max_horizon = 2
        settings.risk.take_profit_rr = 2.0
        settings.risk.stop_loss_atr_multiple = 1.0

        dataset = pd.DataFrame(
            {
                "open": [100.0, 100.4, 100.1, 99.4],
                "close": [100.0, 100.5, 100.2, 99.5],
                "high": [100.2, 102.2, 101.0, 100.0],
                "low": [99.8, 100.0, 98.8, 98.9],
                "atr": [1.0, 1.0, 1.0, 1.0],
                "expected_direction": [1, -1, 1, 1],
            }
        )
        rr, bars, peak_rr = _compute_sltp_realized_rr(dataset, settings)
        self.assertEqual(len(rr), len(dataset))
        self.assertEqual(len(bars), len(dataset))
        self.assertGreaterEqual(float(rr.iloc[0]), 1.0)
        self.assertGreaterEqual(int(bars.iloc[0]), 1)


if __name__ == "__main__":
    unittest.main()
