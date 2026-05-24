from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from xauusd_ai.backtesting.engine import (
    simulate_dynamic_concurrent_backtest,
    simulate_prediction_backtest,
    write_backtest_outputs,
)
from xauusd_ai.config import Settings
from xauusd_ai.execution.risk import RiskManager


class BacktestingEngineTests(unittest.TestCase):
    def _settings(self) -> Settings:
        settings = Settings()
        settings.training.backtest_initial_balance = 100.0
        settings.risk.risk_per_trade = 0.01
        settings.risk.min_confidence = 0.6
        settings.risk.spread_cost_rr = 0.0
        settings.risk.slippage_rr = 0.0
        settings.risk.commission_rr = 0.0
        settings.risk.kill_switch_enabled = False
        settings.strategy.require_trend_alignment = True
        return settings

    def _predictions(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "split": ["test", "test"],
                "time": pd.to_datetime(
                    ["2026-03-30 09:00:00+00:00", "2026-03-30 09:15:00+00:00"]
                ),
                "prediction": [1, 1],
                "probability": [0.9, 0.9],
                "trade_side": ["buy", "sell"],
                "close": [2000.0, 2001.0],
                "future_return": [0.001, -0.001],
                "directional_return": [0.001, 0.001],
                "realized_rr": [2.0, -1.0],
                "bars_held": [1, 1],
                "strategy_score": [0.6, -0.6],
                "volatility_regime": [1, 1],
                "trend_alignment": [1, 1],
                "session_spread_mult": [1.0, 1.0],
                "adx": [30.0, 30.0],
            }
        )

    def test_simulate_prediction_backtest_returns_expected_profit(self) -> None:
        settings = self._settings()
        risk_manager = RiskManager(settings)
        result = simulate_prediction_backtest(self._predictions(), settings, risk_manager, compound=False)
        self.assertEqual(result.report["trades"], 2)
        self.assertAlmostEqual(float(result.report["ending_balance"]), 101.0, places=2)
        self.assertAlmostEqual(float(result.report["net_profit"]), 1.0, places=2)

    def test_simulate_dynamic_concurrent_backtest_runs(self) -> None:
        settings = self._settings()
        risk_manager = RiskManager(settings)
        result = simulate_dynamic_concurrent_backtest(self._predictions(), settings, risk_manager, label="test", compound=False)
        self.assertEqual(result.report["simulation_mode"], "dynamic_concurrent")
        self.assertEqual(result.report["trades"], 2)
        self.assertIn("max_concurrent_positions", result.report)

    def test_dynamic_concurrent_backtest_skips_when_min_lot_exceeds_risk_cap(self) -> None:
        settings = self._settings()
        settings.risk.max_risk_fraction = 0.02
        predictions = self._predictions()
        predictions["atr"] = 20.0
        risk_manager = RiskManager(settings)

        result = simulate_dynamic_concurrent_backtest(predictions, settings, risk_manager, label="test", compound=False)

        self.assertEqual(result.report["trades"], 0)
        self.assertEqual(result.report["signals_risk_cap"], 2)

    def test_write_backtest_outputs_creates_artifacts(self) -> None:
        predictions = self._predictions()
        trades = pd.DataFrame(
            {
                "time": predictions["time"].astype(str),
                "balance_after": [101.0, 100.0],
                "side": ["buy", "sell"],
                "entry_price": [2000.0, 2001.0],
            }
        )
        report = {"ok": True, "trades": 2}

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            report_path = td_path / "report.json"
            trades_path = td_path / "trades.csv"
            equity = td_path / "equity.png"
            markers = td_path / "markers.png"
            write_backtest_outputs(report, trades, predictions, report_path, trades_path, equity, markers)
            self.assertTrue(report_path.exists())
            self.assertTrue(trades_path.exists())
            self.assertTrue(equity.exists())
            self.assertTrue(markers.exists())


if __name__ == "__main__":
    unittest.main()
