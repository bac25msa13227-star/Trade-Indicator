from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from xauusd_ai.config import RegimeShutdownRuleSettings, Settings
from xauusd_ai.strategies.hybrid import HybridStrategy


class HybridStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings()
        self.strategy = HybridStrategy(self.settings)

    def test_blocked_by_allowed_weekday_hours(self) -> None:
        self.settings.strategy.allowed_weekday_hours_utc = {"monday": [9]}
        blocked, reason = self.strategy._blocked_by_time(pd.Timestamp("2026-03-30 10:00:00+00:00"))
        self.assertTrue(blocked)
        self.assertEqual(reason, "not_in_allowed_weekday_hour")

    def test_regime_shutdown_rule_blocks_matching_row(self) -> None:
        self.settings.strategy.regime_shutdown_rules = [
            RegimeShutdownRuleSettings(
                name="early_week_guard",
                weekdays_utc=["monday"],
                hours_utc=[2],
                sides=["buy"],
                volatility_regimes=[1],
                trend_alignment_values=[1],
                probability_min=0.6,
                adx_min=20,
            )
        ]
        row = SimpleNamespace(
            time=pd.Timestamp("2026-03-30 02:15:00+00:00"),
            trade_side="buy",
            volatility_regime=1,
            trend_alignment=1,
            adx=25.0,
        )
        blocked, reason = self.strategy._regime_shutdown(row, probability=0.7, side="buy")
        self.assertTrue(blocked)
        self.assertEqual(reason, "early_week_guard")

    def test_should_allow_row_checks_confidence_and_alignment(self) -> None:
        row = SimpleNamespace(
            time=pd.Timestamp("2026-03-30 09:00:00+00:00"),
            strategy_score=0.4,
            volatility_regime=1,
            trend_alignment=0,
            trade_side="buy",
            adx=30.0,
        )
        allowed, reason = self.strategy.should_allow_row(row, probability=0.8)
        self.assertFalse(allowed)
        self.assertEqual(reason, "trend_misaligned")

    def test_build_trade_decision_returns_blocked_on_news_blackout(self) -> None:
        live_row = pd.Series(
            {
                "strategy_score": 0.4,
                "volatility_regime": 1,
                "trend_alignment": 1,
                "atr": 2.0,
                "close": 2000.0,
                "news_is_blackout": 1,
                "news_impact_ahead": 2,
                "news_hours_ahead": 0.1,
            }
        )
        decision = self.strategy.build_trade_decision(
            frames={},
            live_row=live_row,
            model_signal={"prediction": 1, "probability": 0.9},
        )
        self.assertFalse(decision.should_trade)
        self.assertIn("News filter blocked trade", decision.reason)

    def test_build_trade_decision_happy_path(self) -> None:
        live_row = pd.Series(
            {
                "strategy_score": 0.7,
                "volatility_regime": 1,
                "trend_alignment": 1,
                "atr": 1.5,
                "close": 2050.0,
                "news_is_blackout": 0,
                "news_impact_ahead": 0,
                "news_hours_ahead": 10.0,
            }
        )
        decision = self.strategy.build_trade_decision(
            frames={},
            live_row=live_row,
            model_signal={"prediction": 1, "probability": 0.9},
        )
        self.assertTrue(decision.should_trade)
        self.assertEqual(decision.side, "buy")
        self.assertGreater(decision.take_profit, decision.entry_price)
        self.assertLess(decision.stop_loss, decision.entry_price)

    def test_build_trade_decision_applies_min_stop_loss_points(self) -> None:
        self.settings.risk.min_stop_loss_points = 5.0
        self.settings.risk.min_stop_loss_atr_multiple = 0.0
        live_row = pd.Series(
            {
                "strategy_score": 0.7,
                "volatility_regime": 1,
                "trend_alignment": 1,
                "atr": 1.0,  # base stop = 1.8 * ATR = 1.8 (default settings)
                "close": 2050.0,
                "news_is_blackout": 0,
                "news_impact_ahead": 0,
                "news_hours_ahead": 10.0,
            }
        )
        decision = self.strategy.build_trade_decision(
            frames={},
            live_row=live_row,
            model_signal={"prediction": 1, "probability": 0.9},
        )
        self.assertTrue(decision.should_trade)
        self.assertAlmostEqual(decision.entry_price - decision.stop_loss, 5.0, places=6)
        self.assertIn("minSL$5.0", decision.reason)

    def test_build_trade_decision_applies_sl_volatility_boost(self) -> None:
        self.settings.risk.sl_volatility_boost_enabled = True
        self.settings.risk.sl_volatility_boost_trigger_percentile = 0.8
        self.settings.risk.sl_volatility_boost_max_multiplier = 2.0
        self.settings.risk.min_stop_loss_points = 0.0
        self.settings.risk.min_stop_loss_atr_multiple = 0.0
        live_row = pd.Series(
            {
                "strategy_score": 0.7,
                "volatility_regime": 1,
                "trend_alignment": 1,
                "atr": 2.0,
                "atr_percentile": 0.9,
                "close": 2050.0,
                "news_is_blackout": 0,
                "news_impact_ahead": 0,
                "news_hours_ahead": 10.0,
            }
        )
        decision = self.strategy.build_trade_decision(
            frames={},
            live_row=live_row,
            model_signal={"prediction": 1, "probability": 0.9},
        )
        self.assertTrue(decision.should_trade)
        # default base sl_mult=1.8; boost at atr_percentile=0.9 with trigger=0.8,max=2.0 -> x1.5
        # effective_sl_mult=2.7 => stop distance = 2.0 * 2.7 = 5.4
        self.assertAlmostEqual(decision.entry_price - decision.stop_loss, 5.4, places=6)
        self.assertIn("volBoost", decision.reason)


if __name__ == "__main__":
    unittest.main()
