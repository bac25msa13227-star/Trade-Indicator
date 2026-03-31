from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from xauusd_ai.config import Settings
from xauusd_ai.learning.self_learner import SelfLearner


class SelfLearnerLossSnapshotTests(unittest.TestCase):
    def _make_learner(self, temp_dir: Path) -> SelfLearner:
        settings = Settings()
        settings.app.live_learning_log_path = str(temp_dir / "live_learning.jsonl")
        settings.app.model_meta_path = str(temp_dir / "model_meta.json")
        settings.market.csv_folder_path = str(temp_dir / "empty_csv")
        (temp_dir / "empty_csv").mkdir(parents=True, exist_ok=True)
        with patch.object(SelfLearner, "_preload_csv_cache", lambda self: None):
            learner = SelfLearner(
                settings=settings,
                trainer=MagicMock(),
                strategy=MagicMock(),
                notifier=None,
            )
        learner._loss_log_path = temp_dir / "loss_analysis.jsonl"
        learner._loss_log_path.parent.mkdir(parents=True, exist_ok=True)
        return learner

    def test_analyze_loss_prefers_entry_snapshot_over_live_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            learner = self._make_learner(Path(td))
            live_row = pd.Series(
                {
                    "strategy_score": 0.02,
                    "volatility_regime": 0,
                    "rsi": 78.0,
                    "macd_hist": -0.5,
                    "atr": 20.0,
                    "trend_alignment": 0,
                    "hourly_bias": -0.8,
                    "daily_bias": -0.7,
                    "liquidity_sweep": 1,
                    "wyckoff_phase": -1,
                    "order_flow_proxy": -0.9,
                }
            )
            entry_snapshot = {
                "time": "2026-03-30T00:00:00+00:00",
                "strategy_score": 0.8,
                "volatility_regime": 1,
                "rsi": 55.0,
                "macd_hist": 0.4,
                "atr": 9.5,
                "trend_alignment": 1,
                "hourly_bias": 0.4,
                "daily_bias": 0.3,
                "liquidity_sweep": 0,
                "wyckoff_phase": 0,
                "order_flow_proxy": 0.1,
            }
            position = {"side": "buy", "profit": -15.0, "swap": 0.0, "commission": 0.0}

            analysis = learner.analyze_loss(position, live_row, entry_snapshot=entry_snapshot)
            reasons = analysis["reasons"]

            self.assertEqual(analysis["features"]["snapshot_source"], "entry_snapshot")
            self.assertEqual(analysis["features"]["entry_time"], "2026-03-30T00:00:00+00:00")
            self.assertAlmostEqual(analysis["features"]["rsi"], 55.0, places=3)
            self.assertAlmostEqual(analysis["features"]["atr"], 9.5, places=3)
            self.assertFalse(any("RSI_OVERBOUGHT" in reason for reason in reasons))
            self.assertFalse(any("TREND_MISALIGNED" in reason for reason in reasons))
            self.assertFalse(any("DAILY_BIAS_BEARISH" in reason for reason in reasons))

    def test_analyze_loss_falls_back_to_live_row_when_no_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            learner = self._make_learner(Path(td))
            live_row = pd.Series(
                {
                    "strategy_score": 0.2,
                    "volatility_regime": 1,
                    "rsi": 75.0,
                    "macd_hist": -0.2,
                    "atr": 8.0,
                    "trend_alignment": 0,
                    "hourly_bias": -0.2,
                    "daily_bias": -0.1,
                    "liquidity_sweep": 0,
                    "wyckoff_phase": 0,
                    "order_flow_proxy": -0.6,
                }
            )
            position = {"side": "buy", "profit": -10.0, "swap": 0.0, "commission": 0.0}

            analysis = learner.analyze_loss(position, live_row, entry_snapshot=None)
            reasons = analysis["reasons"]

            self.assertEqual(analysis["features"]["snapshot_source"], "live_row_fallback")
            self.assertTrue(any("RSI_OVERBOUGHT" in reason for reason in reasons))
            self.assertTrue(any("TREND_MISALIGNED" in reason for reason in reasons))
            self.assertTrue(any("DAILY_BIAS_BEARISH" in reason for reason in reasons))

    def test_log_loss_analysis_marks_entry_snapshot_flag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            learner = self._make_learner(Path(td))
            live_row = pd.Series(
                {
                    "strategy_score": 0.4,
                    "volatility_regime": 1,
                    "rsi": 50.0,
                    "macd_hist": 0.1,
                    "atr": 7.2,
                    "trend_alignment": 1,
                    "hourly_bias": 0.1,
                    "daily_bias": 0.2,
                    "liquidity_sweep": 0,
                    "wyckoff_phase": 0,
                    "order_flow_proxy": 0.0,
                }
            )
            position = {
                "ticket": 123,
                "side": "buy",
                "volume": 0.01,
                "profit": -2.0,
                "swap": 0.0,
                "commission": -0.1,
            }
            snapshot = {"time": "2026-03-30T01:00:00+00:00", "rsi": 49.0, "atr": 6.5}

            learner.log_loss_analysis(position, live_row, entry_snapshot=snapshot)
            rows = learner._loss_log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 1)
            event = json.loads(rows[0])
            self.assertTrue(event["entry_snapshot_available"])
            self.assertEqual(event["features"]["snapshot_source"], "entry_snapshot")


if __name__ == "__main__":
    unittest.main()
