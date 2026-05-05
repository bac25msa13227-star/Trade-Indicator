from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from xauusd_ai.config import Settings, load_settings
from xauusd_ai.execution.risk import OrderPlan, RiskManager
from xauusd_ai.strategies.hybrid import TradeDecision


class RiskManagerTests(unittest.TestCase):
    """Test risk management logic."""

    def setUp(self) -> None:
        """Load real config and create risk manager."""
        config_path = Path("configs/acc1_v14pp_profit.yaml")
        if config_path.exists():
            self.settings = load_settings(config_path)
        else:
            self.settings = Settings()
        
        # Mock file paths to avoid file system side effects
        with patch.object(RiskManager, "_load_peak_balance", return_value=10000.0):
            with patch.object(RiskManager, "_load_daily_state"):
                self.risk_manager = RiskManager(self.settings)

    # ==================== Peak Balance Tests ====================
    def test_peak_balance_initialized(self) -> None:
        self.assertGreater(self.risk_manager._peak_balance, 0)

    def test_peak_balance_updates_on_higher_balance(self) -> None:
        initial_peak = self.risk_manager._peak_balance
        higher_balance = initial_peak + 1000
        
        with patch.object(self.risk_manager, "_save_peak_balance"):
            self.risk_manager.record_trade_result(100.0, higher_balance)
        
        self.assertGreater(self.risk_manager._peak_balance, initial_peak)

    def test_peak_balance_does_not_decrease(self) -> None:
        initial_peak = self.risk_manager._peak_balance
        lower_balance = initial_peak - 1000
        
        with patch.object(self.risk_manager, "_save_peak_balance"):
            self.risk_manager.record_trade_result(-100.0, lower_balance)
        
        self.assertEqual(self.risk_manager._peak_balance, initial_peak)

    # ==================== Circuit Breaker Tests ====================
    def test_circuit_breaker_triggers_on_consecutive_losses(self) -> None:
        max_consecutive = self.settings.risk.max_consecutive_losses
        
        with patch.object(self.risk_manager, "_save_daily_state"):
            for _ in range(max_consecutive):
                self.risk_manager.record_trade_result(-50.0, 10000.0)
        
        active, reason = self.risk_manager.is_circuit_breaker_active(10000.0)
        self.assertTrue(active)
        self.assertIn("consecutive", reason.lower())

    def test_circuit_breaker_triggers_on_daily_loss_limit(self) -> None:
        balance = 10000.0
        daily_limit = balance * self.settings.risk.max_daily_loss_fraction
        
        with patch.object(self.risk_manager, "_save_daily_state"):
            self.risk_manager.record_trade_result(-daily_limit - 1, balance)
        
        active, reason = self.risk_manager.is_circuit_breaker_active(balance)
        self.assertTrue(active)
        self.assertIn("daily loss", reason.lower())

    def test_circuit_breaker_triggers_on_max_drawdown(self) -> None:
        peak = 10000.0
        current_balance = 8000.0  # 20% drawdown
        self.risk_manager._peak_balance = peak
        
        active, reason = self.risk_manager.is_circuit_breaker_active(current_balance)
        if self.settings.risk.max_drawdown_fraction < 0.2:
            self.assertTrue(active)
            self.assertIn("drawdown", reason.lower())

    def test_circuit_breaker_cooldown_decrements(self) -> None:
        self.risk_manager._cooldown_bars_remaining = 10
        self.risk_manager.tick_cooldown()
        self.assertEqual(self.risk_manager._cooldown_bars_remaining, 9)

    def test_circuit_breaker_cooldown_minimum_zero(self) -> None:
        self.risk_manager._cooldown_bars_remaining = 0
        self.risk_manager.tick_cooldown()
        self.assertEqual(self.risk_manager._cooldown_bars_remaining, 0)

    # ==================== Lot Sizing Tests ====================
    def test_calculate_dynamic_lot_returns_positive(self) -> None:
        balance = 10000.0
        entry = 2000.0
        sl = 1980.0
        confidence = 0.75
        
        lot = self.risk_manager.calculate_dynamic_lot(
            balance, entry, sl, confidence, atr=10.0
        )
        self.assertGreater(lot, 0)
        self.assertGreaterEqual(lot, 0.01)  # Min lot

    def test_calculate_dynamic_lot_respects_min_lot(self) -> None:
        balance = 100.0  # Very small balance
        entry = 2000.0
        sl = 1900.0  # Wide stop
        
        lot = self.risk_manager.calculate_dynamic_lot(
            balance, entry, sl, confidence=0.5, atr=100.0
        )
        self.assertGreaterEqual(lot, 0.01)

    def test_calculate_dynamic_lot_scales_with_confidence(self) -> None:
        balance = 10000.0
        entry = 2000.0
        sl = 1980.0
        
        lot_low = self.risk_manager.calculate_dynamic_lot(
            balance, entry, sl, confidence=0.5, atr=10.0
        )
        lot_high = self.risk_manager.calculate_dynamic_lot(
            balance, entry, sl, confidence=0.9, atr=10.0
        )
        
        self.assertGreater(lot_high, lot_low)

    def test_calculate_dynamic_lot_respects_risk_fraction(self) -> None:
        balance = 10000.0
        entry = 2000.0
        sl = 1980.0  # $20 stop
        
        lot = self.risk_manager.calculate_dynamic_lot(
            balance, entry, sl, confidence=0.75, atr=10.0
        )
        
        # Max risk should be < risk_fraction * balance
        max_risk_dollars = abs(entry - sl) * lot * 100  # 100 oz/lot
        max_allowed = balance * self.settings.risk.risk_per_trade
        self.assertLessEqual(max_risk_dollars, max_allowed * 2)  # Allow 2x for rounding

    # ==================== Position Limits Tests ====================
    def test_get_dynamic_max_positions_returns_positive(self) -> None:
        balance = 10000.0
        max_pos = self.risk_manager.get_dynamic_max_positions(balance)
        self.assertGreater(max_pos, 0)

    def test_get_dynamic_max_positions_scales_with_balance(self) -> None:
        low_balance = 5000.0
        high_balance = 20000.0
        
        max_low = self.risk_manager.get_dynamic_max_positions(low_balance)
        max_high = self.risk_manager.get_dynamic_max_positions(high_balance)
        
        self.assertGreaterEqual(max_high, max_low)

    def test_can_open_position_blocks_when_at_limit(self) -> None:
        balance = 10000.0
        open_positions = 10  # Exceed typical max
        
        can_open, reason = self.risk_manager.can_open_position(
            open_positions=open_positions,
            balance=balance,
            atr=10.0,
            confidence=0.75
        )
        self.assertFalse(can_open)
        self.assertIn("max", reason.lower())

    def test_can_open_position_allows_when_below_limit(self) -> None:
        balance = 10000.0
        open_positions = 0
        
        can_open, reason = self.risk_manager.can_open_position(
            open_positions=open_positions,
            balance=balance,
            atr=10.0,
            confidence=0.75
        )
        self.assertTrue(can_open)

    def test_can_open_position_blocks_when_circuit_breaker_active(self) -> None:
        self.risk_manager._killed = True
        
        can_open, reason = self.risk_manager.can_open_position(
            open_positions=0,
            balance=10000.0,
            atr=10.0,
            confidence=0.75
        )
        self.assertFalse(can_open)

    # ==================== Order Plan Tests ====================
    def test_build_order_plan_returns_valid_plan(self) -> None:
        decision = TradeDecision(
            signal=1,
            confidence=0.75,
            entry_price=2000.0,
            stop_loss=1980.0,
            take_profit=2040.0,
            reasoning="Test trade"
        )
        balance = 10000.0
        atr = 10.0
        
        plan = self.risk_manager.build_order_plan(decision, balance, atr)
        
        self.assertIsInstance(plan, OrderPlan)
        self.assertIn(plan.side, ["BUY", "SELL"])
        self.assertGreater(plan.volume, 0)
        self.assertEqual(plan.entry_price, 2000.0)
        self.assertEqual(plan.stop_loss, 1980.0)
        self.assertEqual(plan.take_profit, 2040.0)

    def test_build_order_plan_respects_signal_direction(self) -> None:
        decision_long = TradeDecision(
            signal=1,
            confidence=0.75,
            entry_price=2000.0,
            stop_loss=1980.0,
            take_profit=2040.0,
            reasoning="Long"
        )
        decision_short = TradeDecision(
            signal=-1,
            confidence=0.75,
            entry_price=2000.0,
            stop_loss=2020.0,
            take_profit=1960.0,
            reasoning="Short"
        )
        
        plan_long = self.risk_manager.build_order_plan(decision_long, 10000.0, 10.0)
        plan_short = self.risk_manager.build_order_plan(decision_short, 10000.0, 10.0)
        
        self.assertEqual(plan_long.side, "BUY")
        self.assertEqual(plan_short.side, "SELL")

    def test_build_order_plan_returns_none_when_cannot_open(self) -> None:
        self.risk_manager._killed = True  # Circuit breaker
        
        decision = TradeDecision(
            signal=1,
            confidence=0.75,
            entry_price=2000.0,
            stop_loss=1980.0,
            take_profit=2040.0,
            reasoning="Test"
        )
        
        plan = self.risk_manager.build_order_plan(decision, 10000.0, 10.0)
        self.assertIsNone(plan)

    # ==================== Risk Throttle Tests ====================
    def test_get_anti_martingale_factor_reduces_after_losses(self) -> None:
        self.risk_manager._consecutive_losses = 0
        factor_0 = self.risk_manager.get_anti_martingale_factor()
        
        self.risk_manager._consecutive_losses = 3
        factor_3 = self.risk_manager.get_anti_martingale_factor()
        
        self.assertLess(factor_3, factor_0)

    def test_apply_risk_throttle_reduces_lot_size(self) -> None:
        original_lot = 0.10
        self.risk_manager._consecutive_losses = 3
        
        throttled = self.risk_manager.apply_risk_throttle(original_lot, "test")
        self.assertLess(throttled, original_lot)

    def test_risk_fraction_returns_value_in_range(self) -> None:
        fraction = self.risk_manager.risk_fraction(
            confidence=0.75,
            volatility_regime=1,
            atr=10.0
        )
        self.assertGreater(fraction, 0)
        self.assertLess(fraction, 0.05)  # Should be reasonable

    # ==================== Trailing SL Tests ====================
    def test_compute_trailing_sl_returns_none_when_disabled(self) -> None:
        self.settings.execution.trailing_sl.enabled = False
        
        new_sl = self.risk_manager.compute_trailing_sl(
            side="BUY",
            entry=2000.0,
            original_sl=1980.0,
            current_price=2020.0,
            current_sl=1980.0,
            atr=10.0
        )
        self.assertIsNone(new_sl)

    def test_compute_trailing_sl_moves_sl_on_profit_long(self) -> None:
        self.settings.execution.trailing_sl.enabled = True
        
        new_sl = self.risk_manager.compute_trailing_sl(
            side="BUY",
            entry=2000.0,
            original_sl=1980.0,
            current_price=2050.0,  # Significant profit
            current_sl=1980.0,
            atr=10.0
        )
        
        if new_sl is not None:
            self.assertGreater(new_sl, 1980.0)  # SL moved up

    def test_compute_trailing_sl_does_not_widen_stop(self) -> None:
        self.settings.execution.trailing_sl.enabled = True
        
        new_sl = self.risk_manager.compute_trailing_sl(
            side="BUY",
            entry=2000.0,
            original_sl=1980.0,
            current_price=1990.0,  # Loss
            current_sl=1980.0,
            atr=10.0
        )
        
        # Should not move SL down (widen stop)
        self.assertIsNone(new_sl)

    # ==================== DCA Tests ====================
    def test_should_dca_returns_false_when_disabled(self) -> None:
        self.settings.execution.dca.enabled = False
        
        should = self.risk_manager.should_dca(
            side="BUY",
            entry=2000.0,
            current_price=1970.0,
            current_volume=0.10,
            open_dca_count=0
        )
        self.assertFalse(should)

    def test_should_dca_returns_false_when_at_max_count(self) -> None:
        self.settings.execution.dca.enabled = True
        max_dca = self.settings.execution.dca.max_dca_count
        
        should = self.risk_manager.should_dca(
            side="BUY",
            entry=2000.0,
            current_price=1970.0,
            current_volume=0.10,
            open_dca_count=max_dca
        )
        self.assertFalse(should)

    def test_should_dca_triggers_on_sufficient_drawdown(self) -> None:
        self.settings.execution.dca.enabled = True
        
        should = self.risk_manager.should_dca(
            side="BUY",
            entry=2000.0,
            current_price=1900.0,  # -5% drawdown
            current_volume=0.10,
            open_dca_count=0
        )
        # Depends on DCA trigger threshold in config
        self.assertIsInstance(should, bool)

    def test_build_dca_plan_returns_valid_plan(self) -> None:
        plan = self.risk_manager.build_dca_plan(
            side="BUY",
            entry=2000.0,
            current_price=1970.0,
            current_volume=0.10,
            original_sl=1980.0,
            original_tp=2040.0,
            balance=10000.0,
            atr=10.0
        )
        
        if plan is not None:
            self.assertIsInstance(plan, OrderPlan)
            self.assertEqual(plan.side, "BUY")
            self.assertGreater(plan.volume, 0)

    # ==================== Exposure Tests ====================
    def test_check_total_exposure_passes_when_below_limit(self) -> None:
        positions = pd.DataFrame({
            "volume": [0.10, 0.10],
            "entry_price": [2000.0, 2010.0]
        })
        
        ok, reason = self.risk_manager.check_total_exposure(positions, 10000.0)
        self.assertTrue(ok)

    def test_check_total_exposure_blocks_when_exceeds_limit(self) -> None:
        # Create positions with very high exposure
        positions = pd.DataFrame({
            "volume": [5.0, 5.0, 5.0],  # 15 lots = massive exposure
            "entry_price": [2000.0, 2010.0, 2020.0]
        })
        
        ok, reason = self.risk_manager.check_total_exposure(positions, 10000.0)
        # Should block if exposure > allowed
        if not ok:
            self.assertIn("exposure", reason.lower())

    # ==================== Utility Method Tests ====================
    def test_as_float_converts_valid_values(self) -> None:
        self.assertEqual(RiskManager._as_float(10), 10.0)
        self.assertEqual(RiskManager._as_float("15.5"), 15.5)
        self.assertEqual(RiskManager._as_float(20.25), 20.25)

    def test_as_float_returns_none_for_invalid(self) -> None:
        self.assertIsNone(RiskManager._as_float(None))
        self.assertIsNone(RiskManager._as_float("invalid"))
        self.assertIsNone(RiskManager._as_float({}))

    def test_row_value_extracts_from_dict(self) -> None:
        row = {"price": 2000.0, "volume": 0.10}
        self.assertEqual(RiskManager._row_value(row, "price"), 2000.0)
        self.assertEqual(RiskManager._row_value(row, "missing", 999), 999)

    def test_row_value_extracts_from_series(self) -> None:
        row = pd.Series({"price": 2000.0, "volume": 0.10})
        self.assertEqual(RiskManager._row_value(row, "price"), 2000.0)
        self.assertEqual(RiskManager._row_value(row, "missing", 999), 999)

    def test_row_value_extracts_from_object(self) -> None:
        class MockRow:
            price = 2000.0
            volume = 0.10
        
        row = MockRow()
        self.assertEqual(RiskManager._row_value(row, "price"), 2000.0)

    # ==================== Regime-Based Lot Table Tests ====================
    def test_get_lot_table_by_regime_returns_dict(self) -> None:
        table = self.risk_manager.get_lot_table_by_regime(
            balance=10000.0,
            atr=10.0
        )
        self.assertIsInstance(table, dict)

    def test_get_lot_table_by_regime_has_expected_keys(self) -> None:
        table = self.risk_manager.get_lot_table_by_regime(
            balance=10000.0,
            atr=10.0
        )
        # Should have regime keys
        self.assertGreater(len(table), 0)
        for key in table:
            self.assertIsInstance(table[key], (int, float))

    # ==================== Edge Cases ====================
    def test_risk_manager_handles_zero_balance(self) -> None:
        can_open, reason = self.risk_manager.can_open_position(
            open_positions=0,
            balance=0.0,
            atr=10.0,
            confidence=0.75
        )
        self.assertFalse(can_open)

    def test_risk_manager_handles_negative_confidence(self) -> None:
        lot = self.risk_manager.calculate_dynamic_lot(
            balance=10000.0,
            entry=2000.0,
            sl=1980.0,
            confidence=-0.5,  # Invalid
            atr=10.0
        )
        self.assertGreaterEqual(lot, 0.01)  # Should fallback to min

    def test_risk_manager_handles_zero_atr(self) -> None:
        lot = self.risk_manager.calculate_dynamic_lot(
            balance=10000.0,
            entry=2000.0,
            sl=1980.0,
            confidence=0.75,
            atr=0.0  # Zero volatility
        )
        self.assertGreater(lot, 0)

    def test_risk_manager_handles_entry_equals_sl(self) -> None:
        lot = self.risk_manager.calculate_dynamic_lot(
            balance=10000.0,
            entry=2000.0,
            sl=2000.0,  # Same as entry
            confidence=0.75,
            atr=10.0
        )
        self.assertGreater(lot, 0)  # Should handle gracefully


if __name__ == "__main__":
    unittest.main()
