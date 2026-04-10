from __future__ import annotations

import unittest

from xauusd_ai.config import Settings
from xauusd_ai.execution.sltp import compute_dynamic_sltp, resolve_setup_exit_targets


class DynamicSltpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings()
        self.base_sl = 1.2
        self.base_rr = 1.8

    def test_returns_base_when_disabled(self) -> None:
        self.settings.risk.dynamic_sltp_enabled = False
        sl_mult, tp_rr, tags, metrics = compute_dynamic_sltp(
            self.settings,
            {"execution_quality": 0.8},
            confidence=0.9,
            base_sl_mult=self.base_sl,
            base_tp_rr=self.base_rr,
        )
        self.assertAlmostEqual(sl_mult, self.base_sl, places=8)
        self.assertAlmostEqual(tp_rr, self.base_rr, places=8)
        self.assertEqual(tags, [])
        self.assertIn("combined_score", metrics)

    def test_high_confidence_high_quality_tightens_sl_and_expands_tp(self) -> None:
        self.settings.risk.dynamic_sltp_enabled = True
        self.settings.risk.min_confidence = 0.58
        sl_mult, tp_rr, _, _ = compute_dynamic_sltp(
            self.settings,
            {
                "execution_quality": 0.9,
                "trend_strength_score": 0.8,
                "pullback_quality": 0.7,
            },
            confidence=0.93,
            base_sl_mult=self.base_sl,
            base_tp_rr=self.base_rr,
        )
        self.assertLess(sl_mult, self.base_sl)
        self.assertGreater(tp_rr, self.base_rr)

    def test_low_confidence_weak_quality_widens_sl_and_reduces_tp(self) -> None:
        self.settings.risk.dynamic_sltp_enabled = True
        self.settings.risk.min_confidence = 0.58
        sl_mult, tp_rr, _, _ = compute_dynamic_sltp(
            self.settings,
            {
                "execution_quality": -0.7,
                "trend_strength_score": -0.6,
                "pullback_quality": -0.4,
            },
            confidence=0.59,
            base_sl_mult=self.base_sl,
            base_tp_rr=self.base_rr,
        )
        self.assertGreater(sl_mult, self.base_sl)
        self.assertLess(tp_rr, self.base_rr)

    def test_respects_configured_bounds(self) -> None:
        self.settings.risk.dynamic_sltp_enabled = True
        self.settings.risk.dynamic_sltp_sl_mult_min = 0.95
        self.settings.risk.dynamic_sltp_sl_mult_max = 1.05
        self.settings.risk.dynamic_sltp_tp_rr_min = 1.4
        self.settings.risk.dynamic_sltp_tp_rr_max = 1.6

        sl_mult, tp_rr, _, _ = compute_dynamic_sltp(
            self.settings,
            {"execution_quality": 1.0, "trend_strength_score": 1.0},
            confidence=0.99,
            base_sl_mult=self.base_sl,
            base_tp_rr=self.base_rr,
        )
        self.assertGreaterEqual(sl_mult, self.base_sl * 0.95)
        self.assertLessEqual(sl_mult, self.base_sl * 1.05)
        self.assertGreaterEqual(tp_rr, 1.4)
        self.assertLessEqual(tp_rr, 1.6)

    def test_sell_side_negative_setup_is_treated_as_aligned_strength(self) -> None:
        self.settings.risk.dynamic_sltp_enabled = True
        self.settings.risk.min_confidence = 0.58
        sl_mult, tp_rr, _, metrics = compute_dynamic_sltp(
            self.settings,
            {
                "trade_side": "sell",
                "expected_direction": -1,
                "execution_quality": -0.9,
                "trend_strength_score": -0.8,
                "pullback_quality": -0.7,
                "strategy_setup_score": -0.8,
                "sm_unicorn": -1.0,
                "sm_ifvg": -1.0,
                "sm_po3_bias": -1.0,
                "wyck_sos_sow": -1.0,
            },
            confidence=0.92,
            base_sl_mult=self.base_sl,
            base_tp_rr=self.base_rr,
        )
        self.assertLess(sl_mult, self.base_sl)
        self.assertGreater(tp_rr, self.base_rr)
        self.assertEqual(metrics["setup_type"], "continuation")

    def test_continuation_setup_pushes_farther_tp_and_tighter_sl(self) -> None:
        self.settings.risk.dynamic_sltp_enabled = True
        sl_mult, tp_rr, _, metrics = compute_dynamic_sltp(
            self.settings,
            {
                "trade_side": "buy",
                "expected_direction": 1,
                "execution_quality": 0.7,
                "trend_strength_score": 0.8,
                "pullback_quality": 0.6,
                "sm_unicorn": 1.0,
                "sm_ifvg": 1.0,
                "sm_po3_bias": 0.8,
                "wyck_sos_sow": 0.8,
                "sm_ote_score": 0.9,
            },
            confidence=0.9,
            base_sl_mult=self.base_sl,
            base_tp_rr=self.base_rr,
        )
        self.assertLess(sl_mult, self.base_sl)
        self.assertGreater(tp_rr, self.base_rr)
        self.assertEqual(metrics["setup_type"], "continuation")

    def test_reversal_setup_widens_sl_and_softens_tp(self) -> None:
        self.settings.risk.dynamic_sltp_enabled = True
        sl_mult, tp_rr, _, metrics = compute_dynamic_sltp(
            self.settings,
            {
                "trade_side": "buy",
                "expected_direction": 1,
                "execution_quality": 0.2,
                "trend_strength_score": 0.1,
                "pullback_quality": 0.3,
                "sm_turtle_soup": 1.0,
                "wyck_spring_utad": 1.0,
                "wyck_lps_quality": 1.0,
            },
            confidence=0.72,
            base_sl_mult=self.base_sl,
            base_tp_rr=self.base_rr,
        )
        self.assertGreater(sl_mult, self.base_sl)
        self.assertLess(tp_rr, self.base_rr)
        self.assertEqual(metrics["setup_type"], "reversal")

    def test_resolve_setup_exit_targets_scales_tp_and_preserves_tier_sl(self) -> None:
        tp_rr, sl_mult, tier, tags, metrics = resolve_setup_exit_targets(
            {
                "trade_side": "buy",
                "expected_direction": 1,
                "sm_unicorn": 1.0,
                "m1_bos": 1.0,
            },
            enabled=True,
            tp_scale=0.75,
        )
        self.assertEqual(tier, 3)
        self.assertAlmostEqual(tp_rr, 2.5 * 0.75, places=8)
        self.assertAlmostEqual(sl_mult, 0.65, places=8)
        self.assertIn("setupT3", tags)
        self.assertEqual(metrics["setup_tier"], 3)


if __name__ == "__main__":
    unittest.main()
