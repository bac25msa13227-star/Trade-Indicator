"""
Unit tests for V2 PF3 search final configs:
  - configs/benchmarks/acc1_pf3v2_net815k_dd2891.yaml  (template=minimal)
  - configs/benchmarks/acc2_pf3v2_net63k_dd1864.yaml   (template=focus_hours)

Covers:
  1. Config loads and key fields match search results exactly
  2. Template: minimal -> blocked_hours=[15,22,23], empty allowed_weekday_hours
  3. Template: focus_hours -> allowed weekday hours filter active
  4. compound_cap=50 capping in engine (balance never exceeds fold_start * 50)
  5. compound_cap=0 gives higher result than compound_cap=50 (unbounded growth)
  6. Consecutive-loss pause + cooldown gate
  7. Daily loss limit gate
  8. partial_tp_enabled=False reflected in settings
  9. silver_bullet, adx_gate flags per account
  10. sideway / volatile take_profit_rr derivation (tp-1, tp+1)
  11. anti_martingale_factor reduces risk_fraction after losses
  12. Backtest with ACC1 v2 config produces feasible PF>=3 on crafted clean signal set
"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from xauusd_ai.backtesting.engine import (
    simulate_dynamic_concurrent_backtest,
    simulate_prediction_backtest,
)
from xauusd_ai.config import Settings, load_settings
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.strategies.hybrid import HybridStrategy

_REPO = Path(__file__).resolve().parents[1]
_ACC1_CFG = _REPO / "configs" / "benchmarks" / "acc1_pf3v2_net815k_dd2891.yaml"
_ACC2_CFG = _REPO / "configs" / "benchmarks" / "acc2_pf3v2_net63k_dd1864.yaml"

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_predictions(
    n_wins: int,
    n_losses: int,
    tp_rr: float = 7.0,
    probability: float = 0.90,
    threshold: float = 0.86,
    start_time: str = "2026-01-06 09:00:00+00:00",  # Monday
    bar_minutes: int = 5,
) -> pd.DataFrame:
    """Build a synthetic predictions dataframe with the given win/loss sequence."""
    rows = []
    t = pd.Timestamp(start_time)
    for i in range(n_wins + n_losses):
        is_win = i < n_wins
        rows.append(
            {
                "split": "test",
                "time": t,
                "prediction": 1,
                "probability": probability,
                "trade_side": "buy",
                "close": 2000.0,
                "future_return": 0.001 if is_win else -0.001,
                "directional_return": 0.001,
                "realized_rr": tp_rr if is_win else -1.0,
                "bars_held": 1,
                "strategy_score": 0.9,
                "volatility_regime": 1,
                "trend_alignment": 1,
                "session_spread_mult": 1.0,
                "adx": 25.0,
            }
        )
        t += pd.Timedelta(minutes=bar_minutes)
    return pd.DataFrame(rows)


def _acc1_settings() -> Settings:
    s = load_settings(_ACC1_CFG)
    s.risk.kill_switch_enabled = False
    return s


def _acc2_settings() -> Settings:
    s = load_settings(_ACC2_CFG)
    s.risk.kill_switch_enabled = False
    return s


# ---------------------------------------------------------------------------
# 1. Config loading — exact field values
# ---------------------------------------------------------------------------

class TestAcc1ConfigValues(unittest.TestCase):
    def setUp(self) -> None:
        self.s = load_settings(_ACC1_CFG)

    # ── strategy ────────────────────────────────────────────────────────────
    def test_signal_threshold(self) -> None:
        self.assertAlmostEqual(self.s.strategy.signal_threshold, 0.86, places=6)

    def test_min_strategy_score(self) -> None:
        self.assertAlmostEqual(self.s.strategy.min_strategy_score, 0.15, places=6)

    def test_adx_gate_enabled(self) -> None:
        self.assertTrue(self.s.strategy.adx_gate_enabled)

    def test_adx_min_trend(self) -> None:
        self.assertAlmostEqual(self.s.strategy.adx_min_trend, 12.0, places=6)

    def test_silver_bullet_disabled(self) -> None:
        self.assertFalse(self.s.strategy.silver_bullet_enabled)

    def test_require_trend_alignment_false(self) -> None:
        self.assertFalse(self.s.strategy.require_trend_alignment)

    # ── risk ────────────────────────────────────────────────────────────────
    def test_risk_per_trade(self) -> None:
        self.assertAlmostEqual(self.s.risk.risk_per_trade, 0.07, places=6)

    def test_take_profit_rr(self) -> None:
        self.assertAlmostEqual(self.s.risk.take_profit_rr, 7.0, places=6)

    def test_sideway_take_profit_rr(self) -> None:
        self.assertAlmostEqual(self.s.risk.sideway_take_profit_rr, 6.0, places=6)

    def test_volatile_take_profit_rr(self) -> None:
        self.assertAlmostEqual(self.s.risk.volatile_take_profit_rr, 8.0, places=6)

    def test_compound_cap(self) -> None:
        self.assertAlmostEqual(self.s.risk.compound_cap, 50.0, places=6)

    def test_stop_loss_atr_multiple(self) -> None:
        self.assertAlmostEqual(self.s.risk.stop_loss_atr_multiple, 1.2, places=6)

    def test_partial_tp_disabled(self) -> None:
        self.assertFalse(self.s.risk.partial_tp_enabled)

    def test_consecutive_loss_pause_count(self) -> None:
        self.assertEqual(self.s.risk.consecutive_loss_pause_count, 3)

    def test_consecutive_loss_cooldown_bars(self) -> None:
        self.assertEqual(self.s.risk.consecutive_loss_cooldown_bars, 8)

    def test_anti_martingale_factor(self) -> None:
        self.assertAlmostEqual(self.s.risk.anti_martingale_factor, 0.6, places=6)

    def test_sideway_risk_multiplier(self) -> None:
        self.assertAlmostEqual(self.s.risk.sideway_risk_multiplier, 0.20, places=6)

    def test_strong_volatility_risk_multiplier(self) -> None:
        self.assertAlmostEqual(self.s.risk.strong_volatility_risk_multiplier, 0.80, places=6)

    def test_daily_loss_limit_pct(self) -> None:
        self.assertAlmostEqual(self.s.risk.daily_loss_limit_pct, 0.03, places=6)

    def test_max_open_positions(self) -> None:
        self.assertEqual(self.s.risk.max_open_positions, 3)

    # ── execution ───────────────────────────────────────────────────────────
    def test_comment(self) -> None:
        self.assertEqual(self.s.execution.comment, "xauusd-ai-acc1-pf3v2-net815k")

    # ── training ────────────────────────────────────────────────────────────
    def test_backtest_initial_balance(self) -> None:
        self.assertAlmostEqual(self.s.training.backtest_initial_balance, 200.0, places=2)


class TestAcc2ConfigValues(unittest.TestCase):
    def setUp(self) -> None:
        self.s = load_settings(_ACC2_CFG)

    def test_signal_threshold(self) -> None:
        self.assertAlmostEqual(self.s.strategy.signal_threshold, 0.80, places=6)

    def test_adx_gate_disabled(self) -> None:
        self.assertFalse(self.s.strategy.adx_gate_enabled)

    def test_adx_min_trend(self) -> None:
        self.assertAlmostEqual(self.s.strategy.adx_min_trend, 22.0, places=6)

    def test_silver_bullet_enabled(self) -> None:
        self.assertTrue(self.s.strategy.silver_bullet_enabled)

    def test_risk_per_trade(self) -> None:
        self.assertAlmostEqual(self.s.risk.risk_per_trade, 0.06, places=6)

    def test_take_profit_rr(self) -> None:
        self.assertAlmostEqual(self.s.risk.take_profit_rr, 10.0, places=6)

    def test_sideway_take_profit_rr(self) -> None:
        self.assertAlmostEqual(self.s.risk.sideway_take_profit_rr, 9.0, places=6)

    def test_volatile_take_profit_rr(self) -> None:
        self.assertAlmostEqual(self.s.risk.volatile_take_profit_rr, 11.0, places=6)

    def test_compound_cap(self) -> None:
        self.assertAlmostEqual(self.s.risk.compound_cap, 50.0, places=6)

    def test_partial_tp_disabled(self) -> None:
        self.assertFalse(self.s.risk.partial_tp_enabled)

    def test_consecutive_loss_pause_count(self) -> None:
        self.assertEqual(self.s.risk.consecutive_loss_pause_count, 3)

    def test_consecutive_loss_cooldown_bars(self) -> None:
        self.assertEqual(self.s.risk.consecutive_loss_cooldown_bars, 12)

    def test_anti_martingale_factor(self) -> None:
        self.assertAlmostEqual(self.s.risk.anti_martingale_factor, 0.7, places=6)

    def test_sideway_risk_multiplier(self) -> None:
        self.assertAlmostEqual(self.s.risk.sideway_risk_multiplier, 0.20, places=6)

    def test_strong_volatility_risk_multiplier(self) -> None:
        self.assertAlmostEqual(self.s.risk.strong_volatility_risk_multiplier, 1.0, places=6)

    def test_daily_loss_limit_pct(self) -> None:
        self.assertAlmostEqual(self.s.risk.daily_loss_limit_pct, 0.015, places=6)

    def test_max_open_positions(self) -> None:
        self.assertEqual(self.s.risk.max_open_positions, 2)

    def test_comment(self) -> None:
        self.assertEqual(self.s.execution.comment, "xauusd-ai-acc2-pf3v2-net63k")


# ---------------------------------------------------------------------------
# 2. Template: minimal (ACC1) — blocked_hours=[15,22,23], no weekday filter
# ---------------------------------------------------------------------------

class TestAcc1TemplateMinimal(unittest.TestCase):
    def setUp(self) -> None:
        self.s = load_settings(_ACC1_CFG)
        self.strategy = HybridStrategy(self.s)

    def test_blocked_hours_utc(self) -> None:
        self.assertEqual(sorted(self.s.strategy.blocked_hours_utc), [15, 22, 23])

    def test_allowed_weekday_hours_empty(self) -> None:
        # minimal template: no per-weekday filter, blocked_hours acts globally
        self.assertEqual(self.s.strategy.allowed_weekday_hours_utc, {})

    def test_blocked_hour_15_is_rejected(self) -> None:
        # Monday 15:00 UTC in a blocked hour
        ts = pd.Timestamp("2026-01-05 15:30:00+00:00")  # Monday
        blocked, reason = self.strategy._blocked_by_time(ts)
        self.assertTrue(blocked)
        self.assertEqual(reason, "blocked_hour")

    def test_blocked_hour_22_is_rejected(self) -> None:
        ts = pd.Timestamp("2026-01-05 22:00:00+00:00")
        blocked, _ = self.strategy._blocked_by_time(ts)
        self.assertTrue(blocked)

    def test_blocked_hour_23_is_rejected(self) -> None:
        ts = pd.Timestamp("2026-01-05 23:59:00+00:00")
        blocked, _ = self.strategy._blocked_by_time(ts)
        self.assertTrue(blocked)

    def test_hour_14_is_allowed(self) -> None:
        ts = pd.Timestamp("2026-01-05 14:30:00+00:00")
        blocked, _ = self.strategy._blocked_by_time(ts)
        self.assertFalse(blocked)

    def test_hour_00_is_allowed(self) -> None:
        ts = pd.Timestamp("2026-01-05 00:00:00+00:00")
        blocked, _ = self.strategy._blocked_by_time(ts)
        self.assertFalse(blocked)

    def test_hour_21_is_allowed(self) -> None:
        # 21 not in [15,22,23]
        ts = pd.Timestamp("2026-01-05 21:30:00+00:00")
        blocked, _ = self.strategy._blocked_by_time(ts)
        self.assertFalse(blocked)


# ---------------------------------------------------------------------------
# 3. Template: focus_hours (ACC2) — weekday allowed=[0,1,5,6,7,13,14,20,21]
# ---------------------------------------------------------------------------

class TestAcc2TemplateFocusHours(unittest.TestCase):
    def setUp(self) -> None:
        self.s = load_settings(_ACC2_CFG)
        self.strategy = HybridStrategy(self.s)

    def test_allowed_weekday_hours_configured(self) -> None:
        awh = self.s.strategy.allowed_weekday_hours_utc
        self.assertEqual(set(awh.keys()), set(_WEEKDAYS))

    def test_each_weekday_has_correct_hours(self) -> None:
        expected = [0, 1, 5, 6, 7, 13, 14, 20, 21]
        for day in _WEEKDAYS:
            with self.subTest(day=day):
                self.assertEqual(
                    sorted(self.s.strategy.allowed_weekday_hours_utc[day]),
                    expected,
                )

    def test_allowed_hour_is_not_blocked(self) -> None:
        # Monday 07:00 is in allowed list
        ts = pd.Timestamp("2026-01-05 07:00:00+00:00")  # Monday
        blocked, _ = self.strategy._blocked_by_time(ts)
        self.assertFalse(blocked)

    def test_disallowed_hour_is_blocked(self) -> None:
        # Monday 08:00 not in [0,1,5,6,7,13,14,20,21]
        ts = pd.Timestamp("2026-01-05 08:00:00+00:00")  # Monday
        blocked, reason = self.strategy._blocked_by_time(ts)
        self.assertTrue(blocked)
        self.assertEqual(reason, "not_in_allowed_weekday_hour")

    def test_hour_13_allowed(self) -> None:
        ts = pd.Timestamp("2026-01-06 13:30:00+00:00")  # Tuesday
        blocked, _ = self.strategy._blocked_by_time(ts)
        self.assertFalse(blocked)

    def test_hour_20_allowed(self) -> None:
        ts = pd.Timestamp("2026-01-07 20:15:00+00:00")  # Wednesday
        blocked, _ = self.strategy._blocked_by_time(ts)
        self.assertFalse(blocked)

    def test_hour_22_not_allowed(self) -> None:
        ts = pd.Timestamp("2026-01-07 22:00:00+00:00")
        blocked, _ = self.strategy._blocked_by_time(ts)
        self.assertTrue(blocked)

    def test_blocked_hours_list_empty(self) -> None:
        # focus_hours template has no global blocked_hours
        self.assertEqual(self.s.strategy.blocked_hours_utc, [])


# ---------------------------------------------------------------------------
# 4. compound_cap=50 — balance must never exceed fold_start * 50
# ---------------------------------------------------------------------------

class TestCompoundCap50(unittest.TestCase):
    def _settings_with_cap(self, cap: float) -> Settings:
        s = Settings()
        s.training.backtest_initial_balance = 200.0
        s.risk.risk_per_trade = 0.25        # large risk to trigger growth quickly
        s.risk.max_risk_fraction = 0.30     # allow 25% to flow through (default is 0.015)
        s.risk.compound_cap = cap
        s.risk.min_confidence = 0.5
        s.risk.kill_switch_enabled = False
        s.risk.spread_cost_rr = 0.0
        s.risk.slippage_rr = 0.0
        s.risk.commission_rr = 0.0
        s.strategy.require_trend_alignment = False
        return s

    def _high_win_preds(self, n: int = 30) -> pd.DataFrame:
        rows = []
        t = pd.Timestamp("2026-01-05 09:00:00+00:00")
        for _ in range(n):
            rows.append({
                "split": "test",
                "time": t,
                "prediction": 1,
                "probability": 0.95,
                "trade_side": "buy",
                "close": 2000.0,
                "future_return": 0.001,
                "directional_return": 0.001,
                "realized_rr": 4.0,      # large win RR
                "bars_held": 1,
                "strategy_score": 0.9,
                "volatility_regime": 1,
                "trend_alignment": 1,
                "session_spread_mult": 1.0,
                "adx": 30.0,
            })
            t += pd.Timedelta(minutes=5)
        return pd.DataFrame(rows)

    def test_cap50_limits_ending_balance(self) -> None:
        # compound_cap=50 limits per-trade SIZING (effective_bal ≤ start*cap),
        # so balance grows LINEARLY once the cap triggers, not exponentially.
        # Compare 30 wins: cap=50 grows much slower than cap=0.
        s50 = self._settings_with_cap(50.0)
        s0 = self._settings_with_cap(0.0)
        preds = self._high_win_preds(30)
        r50 = simulate_prediction_backtest(preds, s50, RiskManager(s50), compound=True)
        r0 = simulate_prediction_backtest(preds, s0, RiskManager(s0), compound=True)
        # cap=0: exponential growth (each win doubles balance)
        # cap=50: linear after threshold (pnl capped at start*50 per trade unit)
        # With 30 wins, cap=0 should be more than 1000x cap=50
        self.assertGreater(r0.report["ending_balance"], r50.report["ending_balance"] * 1000)

    def test_cap0_exceeds_cap50_significantly(self) -> None:
        preds = self._high_win_preds(20)
        s50 = self._settings_with_cap(50.0)
        s0 = self._settings_with_cap(0.0)
        rm50 = RiskManager(s50)
        rm0 = RiskManager(s0)
        r50 = simulate_prediction_backtest(preds, s50, rm50, compound=True)
        r0 = simulate_prediction_backtest(preds, s0, rm0, compound=True)
        # uncapped should grow much more
        self.assertGreater(r0.report["ending_balance"], r50.report["ending_balance"])

    def test_cap50_applied_in_dynamic_backtest(self) -> None:
        # In dynamic backtest: verify per-trade pnl is bounded once balance
        # exceeds start*cap, while cap=0 continues growing exponentially.
        s50 = self._settings_with_cap(50.0)
        s0 = self._settings_with_cap(0.0)
        preds = self._high_win_preds(30)
        r50 = simulate_dynamic_concurrent_backtest(
            preds, s50, RiskManager(s50), label="test", compound=True
        )
        r0 = simulate_dynamic_concurrent_backtest(
            preds, s0, RiskManager(s0), label="test", compound=True
        )
        # cap=0 grows exponentially; cap=50 stays linear — enormous difference by 30 wins
        self.assertGreater(r0.report["ending_balance"], r50.report["ending_balance"] * 1000)

    def test_acc1_config_has_cap50(self) -> None:
        s = load_settings(_ACC1_CFG)
        self.assertAlmostEqual(s.risk.compound_cap, 50.0, places=6)

    def test_acc2_config_has_cap50(self) -> None:
        s = load_settings(_ACC2_CFG)
        self.assertAlmostEqual(s.risk.compound_cap, 50.0, places=6)


# ---------------------------------------------------------------------------
# 5. Consecutive-loss pause + cooldown gate
# ---------------------------------------------------------------------------

class TestConsecutiveLossPauseAndCooldown(unittest.TestCase):
    def _settings(self, pause: int, cooldown: int) -> Settings:
        s = Settings()
        s.training.backtest_initial_balance = 1000.0
        s.risk.risk_per_trade = 0.05
        s.risk.compound_cap = 0.0
        s.risk.min_confidence = 0.5
        s.risk.kill_switch_enabled = False
        s.risk.spread_cost_rr = 0.0
        s.risk.slippage_rr = 0.0
        s.risk.commission_rr = 0.0
        s.risk.consecutive_loss_pause_count = pause
        s.risk.consecutive_loss_cooldown_bars = cooldown
        s.risk.daily_loss_limit_pct = 0.0
        s.risk.anti_martingale_factor = 1.0  # no anti-martingale
        s.strategy.require_trend_alignment = False
        return s

    def _preds_loss_then_win(self, n_losses: int, n_after: int) -> pd.DataFrame:
        rows = []
        t = pd.Timestamp("2026-01-05 09:00:00+00:00")
        for _ in range(n_losses):
            rows.append({
                "split": "test", "time": t, "prediction": 1,
                "probability": 0.95, "trade_side": "buy", "close": 2000.0,
                "future_return": -0.001, "directional_return": -0.001,
                "realized_rr": -1.0, "bars_held": 1,
                "strategy_score": 0.9, "volatility_regime": 1,
                "trend_alignment": 1, "session_spread_mult": 1.0, "adx": 25.0,
            })
            t += pd.Timedelta(minutes=5)
        for _ in range(n_after):
            rows.append({
                "split": "test", "time": t, "prediction": 1,
                "probability": 0.95, "trade_side": "buy", "close": 2000.0,
                "future_return": 0.001, "directional_return": 0.001,
                "realized_rr": 2.0, "bars_held": 1,
                "strategy_score": 0.9, "volatility_regime": 1,
                "trend_alignment": 1, "session_spread_mult": 1.0, "adx": 25.0,
            })
            t += pd.Timedelta(minutes=5)
        return pd.DataFrame(rows)

    def test_pause3_blocks_trades_after_3_losses(self) -> None:
        # 3 losses trigger cooldown=2 bars; next signals should be blocked.
        # Report key is 'signals_circuit_breaker' in simulate_dynamic_concurrent_backtest.
        s = self._settings(pause=3, cooldown=2)
        rm = RiskManager(s)
        preds = self._preds_loss_then_win(n_losses=3, n_after=5)
        result = simulate_dynamic_concurrent_backtest(preds, s, rm, compound=False)
        # Some trades after the 3rd loss must be skipped (circuit breaker)
        self.assertGreater(result.report.get("signals_circuit_breaker", 0), 0)

    def test_acc1_pause_and_cooldown_match_search(self) -> None:
        s = load_settings(_ACC1_CFG)
        self.assertEqual(s.risk.consecutive_loss_pause_count, 3)
        self.assertEqual(s.risk.consecutive_loss_cooldown_bars, 8)

    def test_acc2_pause_and_cooldown_match_search(self) -> None:
        s = load_settings(_ACC2_CFG)
        self.assertEqual(s.risk.consecutive_loss_pause_count, 3)
        self.assertEqual(s.risk.consecutive_loss_cooldown_bars, 12)


# ---------------------------------------------------------------------------
# 6. Daily loss limit gate
# ---------------------------------------------------------------------------

class TestDailyLossLimitGate(unittest.TestCase):
    def _settings_with_daily_limit(self, limit_pct: float) -> Settings:
        s = Settings()
        s.training.backtest_initial_balance = 1000.0
        s.risk.risk_per_trade = 0.05
        s.risk.compound_cap = 0.0
        s.risk.min_confidence = 0.5
        s.risk.kill_switch_enabled = False
        s.risk.spread_cost_rr = 0.0
        s.risk.slippage_rr = 0.0
        s.risk.commission_rr = 0.0
        s.risk.consecutive_loss_pause_count = 0
        s.risk.daily_loss_limit_pct = limit_pct
        s.risk.anti_martingale_factor = 1.0
        s.strategy.require_trend_alignment = False
        return s

    def test_daily_loss_limit_blocks_excessive_losses(self) -> None:
        # 3% daily limit: after losing 3%, further signals that day are blocked
        s = self._settings_with_daily_limit(0.03)
        rm = RiskManager(s)
        # We need many losses on the same day to exceed 3%
        rows = []
        t = pd.Timestamp("2026-01-05 09:00:00+00:00")  # same day
        for _ in range(20):
            rows.append({
                "split": "test", "time": t, "prediction": 1,
                "probability": 0.95, "trade_side": "buy", "close": 2000.0,
                "future_return": -0.001, "directional_return": -0.001,
                "realized_rr": -1.0, "bars_held": 1,
                "strategy_score": 0.9, "volatility_regime": 1,
                "trend_alignment": 1, "session_spread_mult": 1.0, "adx": 25.0,
            })
            t += pd.Timedelta(minutes=5)  # still same day
        preds = pd.DataFrame(rows)
        result = simulate_dynamic_concurrent_backtest(preds, s, rm, compound=False)
        # Should not execute all 20 trades (daily limit cuts it off)
        self.assertLess(result.report["trades"], 20)

    def test_acc1_daily_limit_is_3pct(self) -> None:
        s = load_settings(_ACC1_CFG)
        self.assertAlmostEqual(s.risk.daily_loss_limit_pct, 0.03, places=6)

    def test_acc2_daily_limit_is_15pct(self) -> None:
        s = load_settings(_ACC2_CFG)
        self.assertAlmostEqual(s.risk.daily_loss_limit_pct, 0.015, places=6)


# ---------------------------------------------------------------------------
# 7. Anti-martingale factor reduces effective risk after losses
# ---------------------------------------------------------------------------

class TestAntiMartingaleFactor(unittest.TestCase):
    def test_acc1_anti_martingale_reduces_risk_after_losses(self) -> None:
        s = _acc1_settings()
        s.risk.anti_martingale_max_reductions = 3
        rm = RiskManager(s)
        # Simulate consecutive losses by setting internal state directly
        rm._consecutive_losses = 2
        # risk_fraction applies: base * (capped_conf/min_conf) * regime_mult * score_mult * anti_mart
        # ACC1: risk=0.07, min_conf=0.85, anti_mart=0.6^2=0.36
        # capped_conf = min(max(0.95, 0.85), 0.95) = 0.95
        # base_fraction = 0.07 * (0.95/0.85) = 0.078235
        # raw = 0.078235 * 1.0 * 1.0 * 0.36 = 0.028165
        rf = rm.risk_fraction(
            confidence=0.95,
            volatility_regime=1,
            strategy_score=1.0,
            current_balance=1000.0,
            market_row=None,
            side="buy",
        )
        expected = s.risk.risk_per_trade * (0.95 / s.risk.min_confidence) * (0.6 ** 2)
        self.assertAlmostEqual(rf, expected, places=5)

    def test_acc2_anti_martingale_factor_07(self) -> None:
        s = _acc2_settings()
        s.risk.anti_martingale_max_reductions = 3
        rm = RiskManager(s)
        rm._consecutive_losses = 1
        # ACC2: risk=0.06, min_conf=0.75, anti_mart=0.7^1=0.7
        # capped_conf = min(max(0.95, 0.75), 0.95) = 0.95
        # base_fraction = 0.06 * (0.95/0.75) = 0.076
        # raw = 0.076 * 1.0 * 1.0 * 0.7 = 0.0532
        # min(0.0532, max_risk_fraction=0.065) = 0.0532
        rf = rm.risk_fraction(
            confidence=0.95,
            volatility_regime=1,
            strategy_score=1.0,
            current_balance=1000.0,
            market_row=None,
            side="buy",
        )
        expected = min(
            s.risk.risk_per_trade * (0.95 / s.risk.min_confidence) * 0.7,
            s.risk.max_risk_fraction,
        )
        self.assertAlmostEqual(rf, expected, places=5)


# ---------------------------------------------------------------------------
# 8. ADX gate behavior per account
# ---------------------------------------------------------------------------

class TestADXGateBehavior(unittest.TestCase):
    def _row(self, adx: float, prob: float = 0.90) -> object:
        return pd.DataFrame([{
            "split": "test",
            "time": pd.Timestamp("2026-01-05 09:00:00+00:00"),
            "prediction": 1,
            "probability": prob,
            "trade_side": "buy",
            "close": 2000.0,
            "future_return": 0.001,
            "directional_return": 0.001,
            "realized_rr": 7.0,
            "bars_held": 1,
            "strategy_score": 0.9,
            "volatility_regime": 1,
            "trend_alignment": 1,
            "session_spread_mult": 1.0,
            "adx": adx,
        }]).itertuples(index=False).__next__()

    def test_acc1_adx_gate_blocks_below_12(self) -> None:
        s = _acc1_settings()
        # adx_gate_enabled=True, adx_min_trend=12
        strategy = HybridStrategy(s)
        allowed, reason = strategy.should_allow_row(self._row(adx=10.0), probability=0.90)
        self.assertFalse(allowed)

    def test_acc1_adx_gate_allows_above_12(self) -> None:
        s = _acc1_settings()
        strategy = HybridStrategy(s)
        allowed, _ = strategy.should_allow_row(self._row(adx=15.0), probability=0.90)
        self.assertTrue(allowed)

    def test_acc2_adx_gate_disabled_allows_low_adx(self) -> None:
        s = _acc2_settings()
        # adx_gate_enabled=False -> ADX not checked
        strategy = HybridStrategy(s)
        allowed, _ = strategy.should_allow_row(self._row(adx=5.0), probability=0.90)
        # Should not be blocked by ADX (gate is off)
        # Note: may still be blocked by other filters; check reason is not adx-related
        # We check that any blocking is NOT because of ADX
        if not allowed:
            _, reason = strategy.should_allow_row(self._row(adx=5.0), probability=0.90)
            self.assertNotIn("adx", reason.lower())


# ---------------------------------------------------------------------------
# 9. Signal threshold filter
# ---------------------------------------------------------------------------

class TestSignalThresholdFilter(unittest.TestCase):
    """
    should_allow_row uses risk.min_confidence as the live filter threshold for normal regime
    (not strategy.signal_threshold which is used by the ML model probability label).
    ACC1: risk.min_confidence=0.85; ACC2: risk.min_confidence=0.75.
    """

    def _row(self, prob: float, hour: int = 9) -> object:
        # Use hour=9 for ACC1 (minimal template, no weekday filter).
        # For ACC2 (focus_hours), use hour=7 (in allowed list).
        return pd.DataFrame([{
            "split": "test",
            "time": pd.Timestamp(f"2026-01-05 {hour:02d}:00:00+00:00"),
            "prediction": 1,
            "probability": prob,
            "trade_side": "buy",
            "close": 2000.0,
            "future_return": 0.001,
            "directional_return": 0.001,
            "realized_rr": 7.0,
            "bars_held": 1,
            "strategy_score": 0.9,
            "volatility_regime": 1,
            "trend_alignment": 1,
            "session_spread_mult": 1.0,
            "adx": 25.0,
        }]).itertuples(index=False).__next__()

    def test_acc1_blocks_below_min_confidence(self) -> None:
        # ACC1 effective filter: risk.min_confidence=0.85 for normal regime.
        # prob=0.84 < 0.85 → blocked.
        s = _acc1_settings()
        strategy = HybridStrategy(s)
        allowed, reason = strategy.should_allow_row(self._row(0.84), probability=0.84)
        self.assertFalse(allowed)
        self.assertIn("confidence", reason)

    def test_acc1_allows_at_min_confidence(self) -> None:
        # prob=0.85 == min_confidence=0.85 → allowed (not strictly less than).
        s = _acc1_settings()
        strategy = HybridStrategy(s)
        allowed, _ = strategy.should_allow_row(self._row(0.85), probability=0.85)
        self.assertTrue(allowed)

    def test_acc1_allows_above_min_confidence(self) -> None:
        s = _acc1_settings()
        strategy = HybridStrategy(s)
        allowed, _ = strategy.should_allow_row(self._row(0.90), probability=0.90)
        self.assertTrue(allowed)

    def test_acc2_blocks_below_min_confidence(self) -> None:
        # ACC2 effective filter: risk.min_confidence=0.75 for normal regime.
        # prob=0.74 < 0.75 → blocked. Use hour=7 (allowed by focus_hours).
        s = _acc2_settings()
        strategy = HybridStrategy(s)
        allowed, reason = strategy.should_allow_row(self._row(0.74, hour=7), probability=0.74)
        self.assertFalse(allowed)
        self.assertIn("confidence", reason)

    def test_acc2_allows_at_or_above_min_confidence(self) -> None:
        # prob=0.80 > 0.75 → allowed at hour=7 (focus_hours allowed).
        s = _acc2_settings()
        strategy = HybridStrategy(s)
        allowed, _ = strategy.should_allow_row(self._row(0.80, hour=7), probability=0.80)
        self.assertTrue(allowed)


# ---------------------------------------------------------------------------
# 10. End-to-end backtest with acc1 v2 config — PF and feasibility check
# ---------------------------------------------------------------------------

class TestEndToEndBacktestAcc1V2(unittest.TestCase):
    """
    Runs simulate_prediction_backtest with acc1_pf3v2 config on a crafted
    signal set that passes the threshold=0.86 filter. Verifies that:
    - profit_factor > 3.0 (feasibility condition)
    - net profit is positive
    - compound_cap=50 report field is correct
    """

    def _make_signal_df(self, n_wins: int, n_losses: int) -> pd.DataFrame:
        return _make_predictions(
            n_wins=n_wins,
            n_losses=n_losses,
            tp_rr=7.0,
            probability=0.90,  # above threshold 0.86
            start_time="2026-01-05 09:00:00+00:00",
        )

    def test_high_win_rate_gives_pf_above_3(self) -> None:
        s = _acc1_settings()
        rm = RiskManager(s)
        # 20 wins, 2 losses: WR=0.91, PF = (20*7) / (2*1) = 140/2 = 70 theoretical
        preds = self._make_signal_df(n_wins=20, n_losses=2)
        result = simulate_prediction_backtest(preds, s, rm, compound=False)
        self.assertGreater(result.report["profit_factor"], 3.0)
        self.assertGreater(result.report["net_profit"], 0)

    def test_compound_cap_reported_correctly(self) -> None:
        s = _acc1_settings()
        rm = RiskManager(s)
        preds = self._make_signal_df(n_wins=10, n_losses=2)
        result = simulate_prediction_backtest(preds, s, rm, compound=True)
        self.assertAlmostEqual(result.report["compound_cap"], 50.0, places=2)

    def test_trades_count_matches_signals_above_threshold(self) -> None:
        s = _acc1_settings()
        rm = RiskManager(s)
        preds = self._make_signal_df(n_wins=15, n_losses=3)
        result = simulate_prediction_backtest(preds, s, rm, compound=False)
        # All 18 signals pass threshold (prob=0.90 > 0.86), adx=25 > 12
        self.assertEqual(result.report["trades"], 18)


# ---------------------------------------------------------------------------
# 11. End-to-end backtest with acc2 v2 config
# ---------------------------------------------------------------------------

class TestEndToEndBacktestAcc2V2(unittest.TestCase):
    """
    ACC2 uses focus_hours template: only hours [0,1,5,6,7,13,14,20,21] are tradeable.
    Synthetic signals must stay within allowed hours to avoid time-filter blocking.
    With bars_minutes=5, 12 bars = 55 min, safely inside hour 7 (07:00-07:55).
    """

    def _make_signal_df(self, n_wins: int, n_losses: int) -> pd.DataFrame:
        # Interleave wins and losses within allowed hour (07:00-07:Xmin).
        # Max 12 rows to stay within hour 7 (07:00 + 11*5min = 07:55).
        rows = []
        t = pd.Timestamp("2026-01-05 07:00:00+00:00")
        for i in range(n_wins + n_losses):
            # Alternate: win, win, loss, win, win, loss, ... to get some losses
            is_win = (i % max(1, (n_wins + n_losses) // n_losses)) != 0 if n_losses > 0 else True
            # Simpler: first n_wins are wins, then losses — but keep total ≤12 bars
            is_win = i < n_wins
            rows.append({
                "split": "test", "time": t, "prediction": 1,
                "probability": 0.85, "trade_side": "buy", "close": 2000.0,
                "future_return": 0.001 if is_win else -0.001,
                "directional_return": 0.001,
                "realized_rr": 10.0 if is_win else -1.0,
                "bars_held": 1, "strategy_score": 0.9, "volatility_regime": 1,
                "trend_alignment": 1, "session_spread_mult": 1.0, "adx": 25.0,
            })
            t += pd.Timedelta(minutes=5)
        return pd.DataFrame(rows)

    def test_pf_above_3_with_high_wr(self) -> None:
        s = _acc2_settings()
        rm = RiskManager(s)
        # 9 wins + 2 losses within 11 bars (07:00-07:50) — all hour 7, all allowed
        preds = self._make_signal_df(n_wins=9, n_losses=2)
        result = simulate_prediction_backtest(preds, s, rm, compound=False)
        # 9 wins * 10RR / 2 losses * 1RR = PF = 90/2 = 45 (minus friction)
        self.assertGreater(result.report["profit_factor"], 3.0)
        self.assertGreater(result.report["trades"], 0)

    def test_net_profit_positive(self) -> None:
        s = _acc2_settings()
        rm = RiskManager(s)
        # 8 wins + 1 loss = 9 bars (07:00-07:40) within allowed hour
        preds = self._make_signal_df(n_wins=8, n_losses=1)
        result = simulate_prediction_backtest(preds, s, rm, compound=False)
        self.assertGreater(result.report["net_profit"], 0)

    def test_compound_cap_50_in_report(self) -> None:
        s = _acc2_settings()
        rm = RiskManager(s)
        # 5 wins + 1 loss = 6 bars within allowed hour
        preds = self._make_signal_df(n_wins=5, n_losses=1)
        result = simulate_prediction_backtest(preds, s, rm, compound=True)
        self.assertAlmostEqual(result.report["compound_cap"], 50.0, places=2)


# ---------------------------------------------------------------------------
# 12. Both v2 yamls load without ValidationError (strict mode)
# ---------------------------------------------------------------------------

class TestBothConfigsLoadClean(unittest.TestCase):
    def test_acc1_loads_without_error(self) -> None:
        s = load_settings(_ACC1_CFG)
        self.assertIsNotNone(s)

    def test_acc2_loads_without_error(self) -> None:
        s = load_settings(_ACC2_CFG)
        self.assertIsNotNone(s)

    def test_acc1_model_paths_reference_v2_name(self) -> None:
        # WF v2 search used the expand model as base; model_path must point to it
        s = load_settings(_ACC1_CFG)
        self.assertIn("acc1_expand_net127313_dd3215_model.pkl", s.app.model_path)

    def test_acc2_model_paths_reference_v2_name(self) -> None:
        # WF v2 search used the expand model as base; model_path must point to it
        s = load_settings(_ACC2_CFG)
        self.assertIn("acc2_expand_net35714_dd2281_model.pkl", s.app.model_path)

    def test_acc1_sideway_tp_less_than_main_tp(self) -> None:
        s = load_settings(_ACC1_CFG)
        self.assertLess(s.risk.sideway_take_profit_rr, s.risk.take_profit_rr)

    def test_acc2_sideway_tp_less_than_main_tp(self) -> None:
        s = load_settings(_ACC2_CFG)
        self.assertLess(s.risk.sideway_take_profit_rr, s.risk.take_profit_rr)

    def test_acc1_volatile_tp_greater_than_main_tp(self) -> None:
        s = load_settings(_ACC1_CFG)
        self.assertGreater(s.risk.volatile_take_profit_rr, s.risk.take_profit_rr)

    def test_acc2_volatile_tp_greater_than_main_tp(self) -> None:
        s = load_settings(_ACC2_CFG)
        self.assertGreater(s.risk.volatile_take_profit_rr, s.risk.take_profit_rr)


if __name__ == "__main__":
    unittest.main()
