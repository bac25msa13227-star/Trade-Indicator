from __future__ import annotations

import unittest

import pandas as pd

from xauusd_ai.features.indicators import (
    atr_expansion,
    close_position_in_range,
    fair_value_gap,
    kill_zone,
    macd,
    wick_rejection,
)


class IndicatorsTests(unittest.TestCase):
    def test_macd_returns_three_series_same_length(self) -> None:
        close = pd.Series([100 + i * 0.5 for i in range(40)])
        line, signal, hist = macd(close)
        self.assertEqual(len(line), len(close))
        self.assertEqual(len(signal), len(close))
        self.assertEqual(len(hist), len(close))

    def test_kill_zone_flags_expected_utc_hours(self) -> None:
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        "2026-03-30 00:00:00+00:00",
                        "2026-03-30 05:00:00+00:00",
                        "2026-03-30 08:00:00+00:00",
                        "2026-03-30 13:00:00+00:00",
                    ]
                )
            }
        )
        out = kill_zone(frame)
        self.assertEqual(out.tolist(), [1, 0, 1, 1])

    def test_fair_value_gap_output_is_bounded(self) -> None:
        frame = pd.DataFrame(
            {
                "high": [100, 101, 105, 106, 107, 103, 102],
                "low": [99, 100, 104, 105, 106, 100, 99],
            }
        )
        out = fair_value_gap(frame, decay_bars=5)
        self.assertTrue(((out >= -1.0) & (out <= 1.0)).all())

    def test_atr_expansion_is_clipped_to_expected_range(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [100 + i for i in range(60)],
                "high": [101 + i for i in range(60)],
                "low": [99 + i for i in range(60)],
                "close": [100.5 + i for i in range(60)],
            }
        )
        out = atr_expansion(frame, fast=7, slow=28)
        self.assertTrue(((out >= 0.3) & (out <= 3.0)).all())

    def test_wick_rejection_and_close_in_range_are_bounded(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [102, 103, 104],
                "low": [99, 100, 101],
                "close": [101, 100.5, 103.5],
            }
        )
        wick = wick_rejection(frame)
        cir = close_position_in_range(frame)
        self.assertTrue(((wick >= -1.0) & (wick <= 1.0)).all())
        self.assertTrue(((cir >= 0.0) & (cir <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
