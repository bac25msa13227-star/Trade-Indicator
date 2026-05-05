from __future__ import annotations

import unittest

import pandas as pd

from xauusd_ai.config import RiskThrottleRuleSettings, Settings
from xauusd_ai.execution.risk import RiskManager


class RiskManagerCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings()
        self.settings.risk.kill_switch_enabled = False
        self.rm = RiskManager(self.settings)

    def test_risk_throttle_rule_matches_and_applies_multiplier(self) -> None:
        self.settings.risk.risk_throttle_rules = [
            RiskThrottleRuleSettings(
                name="guard_2utc",
                risk_multiplier=0.5,
                weekdays_utc=["monday"],
                hours_utc=[2],
                sides=["buy"],
                trend_alignment_values=[1],
                volatility_regimes=[1],
                probability_min=0.6,
                adx_min=20.0,
                trend_strength_min=0.2,
                pullback_quality_min=0.1,
                execution_quality_min=0.1,
                strategy_score_min=0.2,
            )
        ]
        row = {
            "time": pd.Timestamp("2026-03-30 02:10:00+00:00"),
            "trade_side": "buy",
            "trend_alignment": 1,
            "volatility_regime": 1,
            "adx": 25.0,
            "trend_strength_score": 0.4,
            "pullback_quality": 0.3,
            "execution_quality": 0.4,
            "strategy_score": 0.5,
        }
        mult, reason = self.rm.risk_throttle_multiplier(row, side="buy", probability=0.75)
        self.assertAlmostEqual(mult, 0.5, places=6)
        self.assertEqual(reason, "guard_2utc")

    def test_risk_fraction_uses_anti_martingale_factor(self) -> None:
        self.settings.risk.min_confidence = 0.5
        self.settings.risk.risk_per_trade = 0.02
        self.settings.risk.max_risk_fraction = 0.03
        self.settings.risk.anti_martingale_factor = 0.5
        self.settings.risk.anti_martingale_max_reductions = 3
        self.rm._consecutive_losses = 2

        rf = self.rm.risk_fraction(
            confidence=0.9,
            volatility_regime=1,
            strategy_score=1.0,
            current_balance=1000.0,
            market_row=None,
            side="buy",
        )
        self.assertAlmostEqual(rf, 0.009, places=6)

    def test_compute_trailing_sl_trail_and_breakeven(self) -> None:
        self.settings.execution.trailing_sl.enabled = True
        self.settings.execution.trailing_sl.breakeven_at_rr = 0.5
        self.settings.execution.trailing_sl.activation_rr = 1.0
        self.settings.execution.trailing_sl.trail_atr_multiple = 1.0
        self.settings.risk.stop_loss_atr_multiple = 2.0

        trail = self.rm.compute_trailing_sl(
            position={"side": "buy", "open_price": 100.0, "sl": 98.0, "current_price": 102.0},
            atr=1.0,
        )
        self.assertEqual(trail, 101.0)

        breakeven = self.rm.compute_trailing_sl(
            position={"side": "buy", "open_price": 100.0, "sl": 99.0, "current_price": 101.2},
            atr=1.0,
        )
        self.assertEqual(breakeven, 100.0)

    def test_should_dca_and_build_dca_plan(self) -> None:
        self.settings.execution.dca.enabled = True
        self.settings.execution.dca.max_dca_count = 2
        self.settings.execution.dca.trigger_atr_multiple = 1.0
        self.settings.execution.dca.lot_multiplier = 2.0
        self.settings.execution.dca.max_total_risk_pct = 0.05
        self.settings.risk.stop_loss_atr_multiple = 2.0
        self.settings.risk.take_profit_rr = 2.0

        position = {"side": "buy", "open_price": 2000.0, "current_price": 1998.5, "volume": 0.01}
        should = self.rm.should_dca(
            position=position,
            atr=1.0,
            dca_count=0,
            account_balance=1000.0,
            total_open_lots=0.02,
        )
        self.assertTrue(should)

        plan = self.rm.build_dca_plan(
            position={**position, "ticket": 123},
            atr=1.0,
            dca_count=1,
            account_balance=1000.0,
        )
        self.assertEqual(plan.side, "buy")
        self.assertAlmostEqual(plan.volume, 0.02, places=6)
        self.assertLess(plan.stop_loss, plan.entry_price)
        self.assertGreater(plan.take_profit, plan.entry_price)

    def test_can_open_position_blocks_when_slots_full(self) -> None:
        self.settings.risk.max_open_positions = 3
        allowed, reason = self.rm.can_open_position(balance=300.0, current_open_positions=3, volatility_regime=1)
        self.assertFalse(allowed)
        self.assertIn("Đã đủ lệnh", reason)


if __name__ == "__main__":
    unittest.main()
