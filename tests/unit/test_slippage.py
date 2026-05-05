"""
Unit tests for slippage calculation module.

Tests dynamic slippage model based on:
- ATR (volatility)
- Spread
- Volume ratio
- Time of day (session)
"""
import unittest
from datetime import datetime, time

import numpy as np
import pandas as pd

from xauusd_ai.backtesting.slippage import (
    calculate_slippage_pips,
    calculate_slippage_rr,
    get_session_multiplier,
)


class SlippageCalculationTests(unittest.TestCase):
    """Test suite for slippage calculation functions."""

    def test_calculate_slippage_pips_baseline(self):
        """Test baseline slippage with normal conditions."""
        # Normal conditions: avg ATR, normal spread, normal volume
        slippage = calculate_slippage_pips(
            atr=10.0,
            atr_mean=10.0,
            spread_pips=2.0,
            volume_ratio=1.0,
            session_multiplier=1.0,
        )
        
        # Baseline should be ~1-2 pips
        self.assertGreater(slippage, 0.5)
        self.assertLess(slippage, 3.0)

    def test_calculate_slippage_pips_high_volatility(self):
        """Test slippage increases with high ATR."""
        # High volatility: ATR 2x normal
        slippage_high = calculate_slippage_pips(
            atr=20.0,
            atr_mean=10.0,
            spread_pips=2.0,
            volume_ratio=1.0,
            session_multiplier=1.0,
        )
        
        # Normal volatility
        slippage_normal = calculate_slippage_pips(
            atr=10.0,
            atr_mean=10.0,
            spread_pips=2.0,
            volume_ratio=1.0,
            session_multiplier=1.0,
        )
        
        # High volatility should have MORE slippage
        self.assertGreater(slippage_high, slippage_normal)
        self.assertGreater(slippage_high, 2.0)

    def test_calculate_slippage_pips_low_volatility(self):
        """Test slippage decreases with low ATR."""
        # Low volatility: ATR 0.5x normal
        slippage_low = calculate_slippage_pips(
            atr=5.0,
            atr_mean=10.0,
            spread_pips=2.0,
            volume_ratio=1.0,
            session_multiplier=1.0,
        )
        
        # Normal volatility
        slippage_normal = calculate_slippage_pips(
            atr=10.0,
            atr_mean=10.0,
            spread_pips=2.0,
            volume_ratio=1.0,
            session_multiplier=1.0,
        )
        
        # Low volatility should have LESS slippage
        self.assertLess(slippage_low, slippage_normal)

    def test_calculate_slippage_pips_wide_spread(self):
        """Test slippage increases with wide spread."""
        slippage_wide = calculate_slippage_pips(
            atr=10.0,
            atr_mean=10.0,
            spread_pips=5.0,  # Wide spread
            volume_ratio=1.0,
            session_multiplier=1.0,
        )
        
        slippage_normal = calculate_slippage_pips(
            atr=10.0,
            atr_mean=10.0,
            spread_pips=2.0,  # Normal spread
            volume_ratio=1.0,
            session_multiplier=1.0,
        )
        
        # Wide spread should have MORE slippage
        self.assertGreater(slippage_wide, slippage_normal)

    def test_calculate_slippage_pips_low_volume(self):
        """Test slippage increases with low volume."""
        slippage_low_vol = calculate_slippage_pips(
            atr=10.0,
            atr_mean=10.0,
            spread_pips=2.0,
            volume_ratio=0.5,  # Low volume
            session_multiplier=1.0,
        )
        
        slippage_normal_vol = calculate_slippage_pips(
            atr=10.0,
            atr_mean=10.0,
            spread_pips=2.0,
            volume_ratio=1.0,  # Normal volume
            session_multiplier=1.0,
        )
        
        # Low volume should have MORE slippage
        self.assertGreater(slippage_low_vol, slippage_normal_vol)

    def test_calculate_slippage_pips_asian_session(self):
        """Test slippage increases during Asian session (low liquidity)."""
        slippage_asian = calculate_slippage_pips(
            atr=10.0,
            atr_mean=10.0,
            spread_pips=2.0,
            volume_ratio=1.0,
            session_multiplier=1.5,  # Asian session multiplier
        )
        
        slippage_london = calculate_slippage_pips(
            atr=10.0,
            atr_mean=10.0,
            spread_pips=2.0,
            volume_ratio=1.0,
            session_multiplier=1.0,  # London/NY session
        )
        
        # Asian session should have MORE slippage
        self.assertGreater(slippage_asian, slippage_london)

    def test_calculate_slippage_pips_extreme_conditions(self):
        """Test slippage in extreme market conditions."""
        # Extreme: high volatility + wide spread + low volume + Asian session
        slippage_extreme = calculate_slippage_pips(
            atr=30.0,
            atr_mean=10.0,
            spread_pips=8.0,
            volume_ratio=0.3,
            session_multiplier=1.5,
        )
        
        # Extreme conditions should have significant slippage
        self.assertGreater(slippage_extreme, 4.0)
        self.assertLess(slippage_extreme, 15.0)  # But still realistic

    def test_calculate_slippage_rr_converts_pips_to_rr(self):
        """Test conversion from pips to RR (risk multiples)."""
        # Entry at 2000, SL at 1990 (10 pips risk)
        # Slippage 2 pips = 0.2 R (20% of 1R)
        slippage_rr = calculate_slippage_rr(
            slippage_pips=2.0,
            entry_price=2000.0,
            stop_loss_price=1990.0,
        )
        
        # 2 pips / 10 pips = 0.2 R
        self.assertAlmostEqual(slippage_rr, 0.2, places=2)

    def test_calculate_slippage_rr_handles_tight_stop(self):
        """Test slippage RR with tight stop loss."""
        # Entry at 2000, SL at 1995 (5 pips risk)
        # Slippage 2 pips = 0.4 R (40% of 1R)
        slippage_rr = calculate_slippage_rr(
            slippage_pips=2.0,
            entry_price=2000.0,
            stop_loss_price=1995.0,
        )
        
        # 2 pips / 5 pips = 0.4 R
        self.assertAlmostEqual(slippage_rr, 0.4, places=2)

    def test_calculate_slippage_rr_handles_short_position(self):
        """Test slippage RR for short positions."""
        # Short entry at 2000, SL at 2010 (10 pips risk)
        # Slippage 2 pips = 0.2 R
        slippage_rr = calculate_slippage_rr(
            slippage_pips=2.0,
            entry_price=2000.0,
            stop_loss_price=2010.0,
        )
        
        # 2 pips / 10 pips = 0.2 R
        self.assertAlmostEqual(slippage_rr, 0.2, places=2)

    def test_get_session_multiplier_asian(self):
        """Test session multiplier for Asian session."""
        # Asian session: 22:00-07:00 UTC
        multiplier_asian = get_session_multiplier(time(1, 0))  # 01:00 UTC
        
        # Asian session should have higher multiplier (1.3-1.5)
        self.assertGreater(multiplier_asian, 1.2)
        self.assertLess(multiplier_asian, 1.6)

    def test_get_session_multiplier_london(self):
        """Test session multiplier for London session."""
        # London session: 08:00-16:00 UTC
        multiplier_london = get_session_multiplier(time(10, 0))  # 10:00 UTC
        
        # London session should have normal multiplier (1.0)
        self.assertAlmostEqual(multiplier_london, 1.0, places=1)

    def test_get_session_multiplier_ny(self):
        """Test session multiplier for NY session."""
        # NY session: 13:00-21:00 UTC
        multiplier_ny = get_session_multiplier(time(15, 0))  # 15:00 UTC
        
        # NY session should have normal or slightly better multiplier (0.9-1.0)
        self.assertGreaterEqual(multiplier_ny, 0.8)
        self.assertLessEqual(multiplier_ny, 1.1)

    def test_slippage_is_always_positive(self):
        """Test slippage is never negative."""
        # Even with perfect conditions
        slippage = calculate_slippage_pips(
            atr=5.0,
            atr_mean=10.0,
            spread_pips=1.0,
            volume_ratio=2.0,  # High volume
            session_multiplier=0.8,  # Best session
        )
        
        # Slippage should never be negative
        self.assertGreaterEqual(slippage, 0.0)

    def test_slippage_handles_zero_atr(self):
        """Test slippage handles zero ATR gracefully."""
        slippage = calculate_slippage_pips(
            atr=0.0,
            atr_mean=10.0,
            spread_pips=2.0,
            volume_ratio=1.0,
            session_multiplier=1.0,
        )
        
        # Should still return baseline slippage
        self.assertGreater(slippage, 0.0)
        self.assertLess(slippage, 5.0)

    def test_slippage_handles_zero_spread(self):
        """Test slippage handles zero spread gracefully."""
        slippage = calculate_slippage_pips(
            atr=10.0,
            atr_mean=10.0,
            spread_pips=0.0,
            volume_ratio=1.0,
            session_multiplier=1.0,
        )
        
        # Should still return baseline slippage
        self.assertGreater(slippage, 0.0)


if __name__ == "__main__":
    unittest.main()
