from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from xauusd_ai.config import Settings
from xauusd_ai.features.dataset import (
    _build_date_mask,
    _build_sltp_label,
    _compute_sltp_realized_rr,
    _enrich_execution_frame,
    _expected_direction_from_context,
    _frame_bias,
    _merge_context,
    _session_spread_multiplier,
    build_live_feature_frame,
    build_merged_context,
    get_label_lookahead_bars,
    prepare_training_dataset,
)


class DatasetTests(unittest.TestCase):
    """Test feature engineering and dataset preparation."""

    def setUp(self) -> None:
        """Load real config to avoid validation errors."""
        from pathlib import Path
        from xauusd_ai.config import load_settings
        
        config_path = Path("configs/acc1_v14pp_profit.yaml")
        if config_path.exists():
            self.settings = load_settings(config_path)
        else:
            # Fallback to default Settings
            self.settings = Settings()

    # ==================== Helper Function Tests ====================
    def test_build_date_mask_no_bounds(self) -> None:
        series = pd.Series(
            pd.date_range("2024-01-01", periods=10, freq="d", tz="UTC")
        )
        mask = _build_date_mask(series, start=None, end=None)
        self.assertTrue(mask.all())  # All True when no bounds

    def test_build_date_mask_with_start(self) -> None:
        series = pd.Series(
            pd.date_range("2024-01-01", periods=10, freq="d", tz="UTC")
        )
        mask = _build_date_mask(series, start="2024-01-05", end=None)
        self.assertEqual(mask.sum(), 6)  # 5th to 10th = 6 days

    def test_build_date_mask_with_end(self) -> None:
        series = pd.Series(
            pd.date_range("2024-01-01", periods=10, freq="d", tz="UTC")
        )
        mask = _build_date_mask(series, start=None, end="2024-01-05")
        self.assertEqual(mask.sum(), 5)  # 1st to 5th = 5 days

    def test_build_date_mask_with_both_bounds(self) -> None:
        series = pd.Series(
            pd.date_range("2024-01-01", periods=10, freq="d", tz="UTC")
        )
        mask = _build_date_mask(series, start="2024-01-03", end="2024-01-07")
        self.assertEqual(mask.sum(), 5)  # 3rd to 7th = 5 days

    def test_frame_bias_returns_series(self) -> None:
        frame = pd.DataFrame({
            "close": [100 + i for i in range(100)]
        })
        bias = _frame_bias(frame, fast_window=10, slow_window=30)
        self.assertEqual(len(bias), len(frame))
        self.assertTrue(bias.isin([1, 0, -1]).all())

    def test_frame_bias_uptrend_gives_positive(self) -> None:
        frame = pd.DataFrame({
            "close": [100 + i * 2 for i in range(100)]
        })
        bias = _frame_bias(frame, fast_window=10, slow_window=30)
        # Most of uptrend should be bullish
        self.assertGreater((bias == 1).sum(), (bias == -1).sum())

    def test_session_spread_multiplier_london_high(self) -> None:
        hours = pd.Series([8, 9, 10, 15, 16, 17])  # London + NY
        multipliers = _session_spread_multiplier(hours)
        # London hours (8-10) should have lower multiplier (1.0)
        self.assertEqual(multipliers.iloc[0], 1.0)
        # Other hours higher multiplier
        self.assertGreater(multipliers.iloc[-1], 1.0)

    def test_get_label_lookahead_bars_returns_positive(self) -> None:
        bars = get_label_lookahead_bars(self.settings)
        self.assertGreater(bars, 0)
        self.assertIsInstance(bars, int)

    # ==================== Context Merge Tests ====================
    def test_merge_context_requires_m15_frame(self) -> None:
        frames = {"H4": pd.DataFrame(), "H1": pd.DataFrame()}
        with self.assertRaises(KeyError):
            _merge_context(self.settings, frames)

    def test_merge_context_returns_dataframe(self) -> None:
        # Create minimal frames with required columns
        m15_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=100, freq="15min"),
            "open": 2000.0 + np.random.randn(100),
            "high": 2005.0 + np.random.randn(100),
            "low": 1995.0 + np.random.randn(100),
            "close": 2000.0 + np.random.randn(100),
            "volume": 1000 + np.random.randint(-100, 100, 100),
            "tick_volume": 1000 + np.random.randint(-100, 100, 100),
            "spread": 2,
        })
        h1_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=25, freq="h", tz="UTC"),
            "open": 2000.0 + np.random.randn(25),
            "high": 2005.0 + np.random.randn(25),
            "low": 1995.0 + np.random.randn(25),
            "close": 2000.0 + np.random.randn(25),
            "volume": 5000 + np.random.randint(-500, 500, 25),
            "tick_volume": 5000 + np.random.randint(-500, 500, 25),
            "spread": 2,
        })
        h4_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC"),
            "open": 2000.0 + np.random.randn(6),
            "high": 2005.0 + np.random.randn(6),
            "low": 1995.0 + np.random.randn(6),
            "close": 2000.0 + np.random.randn(6),
            "volume": 20000 + np.random.randint(-2000, 2000, 6),
            "tick_volume": 20000 + np.random.randint(-2000, 2000, 6),
            "spread": 2,
        })
        d1_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=2, freq="d", tz="UTC"),
            "open": 2000.0 + np.random.randn(2),
            "high": 2005.0 + np.random.randn(2),
            "low": 1995.0 + np.random.randn(2),
            "close": 2000.0 + np.random.randn(2),
            "volume": 100000 + np.random.randint(-10000, 10000, 2),
            "tick_volume": 100000 + np.random.randint(-10000, 10000, 2),
            "spread": 2,
        })

        frames = {"M15": m15_frame, "H1": h1_frame, "H4": h4_frame, "D1": d1_frame}
        result = _merge_context(self.settings, frames)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_build_merged_context_adds_multi_tf_features(self) -> None:
        # Create minimal valid frames
        m15_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=100, freq="15min"),
            "open": 2000.0 + np.random.randn(100),
            "high": 2005.0 + np.random.randn(100),
            "low": 1995.0 + np.random.randn(100),
            "close": 2000.0 + np.random.randn(100),
            "volume": 1000 + np.random.randint(-100, 100, 100),
            "tick_volume": 1000 + np.random.randint(-100, 100, 100),
            "spread": 2,
        })
        h1_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=25, freq="h", tz="UTC"),
            "open": 2000.0 + np.random.randn(25),
            "high": 2005.0 + np.random.randn(25),
            "low": 1995.0 + np.random.randn(25),
            "close": 2000.0 + np.random.randn(25),
            "volume": 5000 + np.random.randint(-500, 500, 25),
            "tick_volume": 5000 + np.random.randint(-500, 500, 25),
            "spread": 2,
        })
        h4_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC"),
            "open": 2000.0 + np.random.randn(6),
            "high": 2005.0 + np.random.randn(6),
            "low": 1995.0 + np.random.randn(6),
            "close": 2000.0 + np.random.randn(6),
            "volume": 20000 + np.random.randint(-2000, 2000, 6),
            "tick_volume": 20000 + np.random.randint(-2000, 2000, 6),
            "spread": 2,
        })
        d1_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=2, freq="d", tz="UTC"),
            "open": 2000.0 + np.random.randn(2),
            "high": 2005.0 + np.random.randn(2),
            "low": 1995.0 + np.random.randn(2),
            "close": 2000.0 + np.random.randn(2),
            "volume": 100000 + np.random.randint(-10000, 10000, 2),
            "tick_volume": 100000 + np.random.randint(-10000, 10000, 2),
            "spread": 2,
        })

        frames = {"M15": m15_frame, "H1": h1_frame, "H4": h4_frame, "D1": d1_frame}
        result = build_merged_context(self.settings, frames)
        
        # Should have time columns
        self.assertIn("time", result.columns)
        # Should have price columns
        self.assertIn("close", result.columns)

    # ==================== Label Generation Tests ====================
    def test_build_sltp_label_returns_series(self) -> None:
        dataset = pd.DataFrame({
            "open": [2000, 2001, 2002, 2003, 2004],
            "high": [2005, 2006, 2007, 2008, 2009],
            "low": [1995, 1996, 1997, 1998, 1999],
            "close": [2000, 2001, 2002, 2003, 2004],
            "atr": [10.0, 10.5, 11.0, 10.0, 9.5],
            "expected_direction": [1, 1, -1, -1, 0],
        })
        labels = _build_sltp_label(dataset, self.settings)
        self.assertEqual(len(labels), len(dataset))
        self.assertTrue(labels.isin([1, 0, -1]).all())

    def test_compute_sltp_realized_rr_returns_three_series(self) -> None:
        dataset = pd.DataFrame({
            "open": [2000, 2001, 2002, 2003, 2004],
            "high": [2010, 2011, 2012, 2013, 2014],
            "low": [1990, 1991, 1992, 1993, 1994],
            "close": [2000, 2001, 2002, 2003, 2004],
            "atr": [10.0, 10.5, 11.0, 10.0, 9.5],
            "expected_direction": [1, 1, -1, -1, 0],
        })
        rr_long, rr_short, win_flag = _compute_sltp_realized_rr(dataset, self.settings)
        self.assertEqual(len(rr_long), len(dataset))
        self.assertEqual(len(rr_short), len(dataset))
        self.assertEqual(len(win_flag), len(dataset))

    # ==================== Expected Direction Tests ====================
    def test_expected_direction_from_context_returns_series(self) -> None:
        dataset = pd.DataFrame({
            "close": [2000, 2001, 2002, 2003, 2004],
            "daily_bias": [1, 1, -1, -1, 0],
            "hourly_bias": [1, -1, 1, -1, 0],
            "h4_market_structure_bias": [1, 1, -1, -1, 0],
        })
        direction = _expected_direction_from_context(dataset)
        self.assertEqual(len(direction), len(dataset))
        self.assertTrue(direction.isin([1, 0, -1]).all())

    # ==================== Execution Frame Enrichment Tests ====================
    def test_enrich_execution_frame_adds_features(self) -> None:
        frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01 08:00", periods=50, freq="15min", tz="UTC"),
            "open": 2000.0 + np.random.randn(50),
            "high": 2005.0 + np.random.randn(50),
            "low": 1995.0 + np.random.randn(50),
            "close": 2000.0 + np.random.randn(50),
            "volume": 1000 + np.random.randint(-100, 100, 50),
            "tick_volume": 1000 + np.random.randint(-100, 100, 50),
            "spread": 2,
        })
        enriched = _enrich_execution_frame(self.settings, frame)
        # Should add RSI, MACD, etc
        self.assertIn("rsi", enriched.columns)
        self.assertIn("macd_hist", enriched.columns)
        self.assertIn("atr_ratio", enriched.columns)

    # ==================== Prepare Training Dataset Tests ====================
    @pytest.mark.requires_strategy
    def test_prepare_training_dataset_returns_xy(self) -> None:
        # Mock strategy object
        class MockStrategy:
            def compute_score(self, df):
                return pd.Series([0.5] * len(df))

        strategy = MockStrategy()

        # Create minimal frames
        m15_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=200, freq="15min"),
            "open": 2000.0 + np.cumsum(np.random.randn(200) * 0.1),
            "high": 2005.0 + np.cumsum(np.random.randn(200) * 0.1),
            "low": 1995.0 + np.cumsum(np.random.randn(200) * 0.1),
            "close": 2000.0 + np.cumsum(np.random.randn(200) * 0.1),
            "volume": 1000 + np.random.randint(-100, 100, 200),
            "tick_volume": 1000 + np.random.randint(-100, 100, 200),
            "spread": 2,
        })
        h1_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
            "open": 2000.0 + np.cumsum(np.random.randn(50) * 0.5),
            "high": 2005.0 + np.cumsum(np.random.randn(50) * 0.5),
            "low": 1995.0 + np.cumsum(np.random.randn(50) * 0.5),
            "close": 2000.0 + np.cumsum(np.random.randn(50) * 0.5),
            "volume": 5000 + np.random.randint(-500, 500, 50),
            "tick_volume": 5000 + np.random.randint(-500, 500, 50),
            "spread": 2,
        })
        h4_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=12, freq="4h", tz="UTC"),
            "open": 2000.0 + np.cumsum(np.random.randn(12)),
            "high": 2005.0 + np.cumsum(np.random.randn(12)),
            "low": 1995.0 + np.cumsum(np.random.randn(12)),
            "close": 2000.0 + np.cumsum(np.random.randn(12)),
            "volume": 20000 + np.random.randint(-2000, 2000, 12),
            "tick_volume": 20000 + np.random.randint(-2000, 2000, 12),
            "spread": 2,
        })
        d1_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=3, freq="d", tz="UTC"),
            "open": 2000.0 + np.cumsum(np.random.randn(3) * 5),
            "high": 2005.0 + np.cumsum(np.random.randn(3) * 5),
            "low": 1995.0 + np.cumsum(np.random.randn(3) * 5),
            "close": 2000.0 + np.cumsum(np.random.randn(3) * 5),
            "volume": 100000 + np.random.randint(-10000, 10000, 3),
            "tick_volume": 100000 + np.random.randint(-10000, 10000, 3),
            "spread": 2,
        })

        frames = {"M15": m15_frame, "H1": h1_frame, "H4": h4_frame, "D1": d1_frame}
        
        X, y = prepare_training_dataset(
            self.settings, frames, strategy,
            train_start="2024-01-01", train_end="2024-01-02"
        )
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))

    # ==================== Build Live Feature Frame Tests ====================
    @pytest.mark.requires_strategy
    def test_build_live_feature_frame_returns_dataframe(self) -> None:
        class MockStrategy:
            def compute_score(self, df):
                return pd.Series([0.5] * len(df))

        strategy = MockStrategy()

        # Create minimal frames for live
        m15_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=50, freq="15min"),
            "open": 2000.0 + np.random.randn(50),
            "high": 2005.0 + np.random.randn(50),
            "low": 1995.0 + np.random.randn(50),
            "close": 2000.0 + np.random.randn(50),
            "volume": 1000 + np.random.randint(-100, 100, 50),
            "tick_volume": 1000 + np.random.randint(-100, 100, 50),
            "spread": 2,
        })
        h1_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=13, freq="h", tz="UTC"),
            "open": 2000.0 + np.random.randn(13),
            "high": 2005.0 + np.random.randn(13),
            "low": 1995.0 + np.random.randn(13),
            "close": 2000.0 + np.random.randn(13),
            "volume": 5000 + np.random.randint(-500, 500, 13),
            "tick_volume": 5000 + np.random.randint(-500, 500, 13),
            "spread": 2,
        })
        h4_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC"),
            "open": 2000.0 + np.random.randn(3),
            "high": 2005.0 + np.random.randn(3),
            "low": 1995.0 + np.random.randn(3),
            "close": 2000.0 + np.random.randn(3),
            "volume": 20000 + np.random.randint(-2000, 2000, 3),
            "tick_volume": 20000 + np.random.randint(-2000, 2000, 3),
            "spread": 2,
        })
        d1_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=1, freq="d", tz="UTC"),
            "open": [2000.0],
            "high": [2005.0],
            "low": [1995.0],
            "close": [2000.0],
            "volume": [100000],
            "tick_volume": [100000],
            "spread": 2,
        })

        frames = {"M15": m15_frame, "H1": h1_frame, "H4": h4_frame, "D1": d1_frame}
        
        live_features = build_live_feature_frame(self.settings, frames, strategy)
        
        self.assertIsInstance(live_features, pd.DataFrame)
        self.assertGreater(len(live_features), 0)

    # ==================== Edge Cases & Validation ====================
    def test_dataset_with_empty_frames_raises_error(self) -> None:
        frames = {"M15": pd.DataFrame()}
        with self.assertRaises((ValueError, KeyError, IndexError)):
            _merge_context(self.settings, frames)

    def test_dataset_with_missing_required_columns_raises_error(self) -> None:
        # Frame missing 'close' column
        m15_frame = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=10, freq="15min"),
            "open": [2000] * 10,
        })
        frames = {"M15": m15_frame}
        with self.assertRaises((KeyError, ValueError)):
            _merge_context(self.settings, frames)

    def test_label_with_insufficient_lookahead_bars(self) -> None:
        # Dataset with only 3 bars but lookahead needs 100
        dataset = pd.DataFrame({
            "open": [2000, 2001, 2002],
            "high": [2005, 2006, 2007],
            "low": [1995, 1996, 1997],
            "close": [2000, 2001, 2002],
            "atr": [10.0, 10.5, 11.0],
            "expected_direction": [1, 1, -1],
        })
        labels = _build_sltp_label(dataset, self.settings)
        # Should handle gracefully, likely return mostly 0 (no trade)
        self.assertEqual(len(labels), len(dataset))

    def test_frame_bias_with_single_row_returns_neutral(self) -> None:
        frame = pd.DataFrame({"close": [2000]})
        bias = _frame_bias(frame, fast_window=10, slow_window=30)
        self.assertEqual(len(bias), 1)
        # Single row should be neutral (0) due to insufficient data
        self.assertEqual(bias.iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
