"""
Tests for Minimum Profit Filter

TDD approach: Write tests first, then implement feature.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from xauusd_ai.strategies.profit_filter import MinimumProfitFilter


class TestMinimumProfitFilter:
    """Test minimum profit filter functionality."""
    
    def test_filter_initialization(self):
        """Test filter initializes with correct defaults."""
        filter_obj = MinimumProfitFilter()
        
        assert filter_obj.min_expected_profit == 15.0
        assert filter_obj.spread_pips == 0.5
        assert filter_obj.pip_value == 10.0
        assert filter_obj.enabled is True
        assert filter_obj.spread_cost_per_trade == 10.0  # 2 × 0.5 × 10
    
    def test_skip_low_profit_trade(self):
        """Test that trades below minimum profit are skipped."""
        filter_obj = MinimumProfitFilter(min_expected_profit=15.0)
        
        # Trade with $12 profit < $15 min → SKIP
        should_skip, reason = filter_obj.should_skip_trade(predicted_profit=12.0)
        
        assert should_skip is True
        assert "profit_too_low" in reason
        assert "predicted=$12.00" in reason
        assert "min=$15.00" in reason
    
    def test_allow_high_profit_trade(self):
        """Test that trades above minimum profit are allowed."""
        filter_obj = MinimumProfitFilter(min_expected_profit=15.0)
        
        # Trade with $20 profit > $15 min → ALLOW
        should_skip, reason = filter_obj.should_skip_trade(predicted_profit=20.0)
        
        assert should_skip is False
        assert reason == "ok"
    
    def test_filter_disabled(self):
        """Test that filter allows all trades when disabled."""
        filter_obj = MinimumProfitFilter(min_expected_profit=15.0, enabled=False)
        
        # Even $5 profit should pass when filter disabled
        should_skip, reason = filter_obj.should_skip_trade(predicted_profit=5.0)
        
        assert should_skip is False
        assert reason == "filter_disabled"
    
    def test_lot_size_adjustment(self):
        """Test that minimum profit scales with lot size."""
        filter_obj = MinimumProfitFilter(min_expected_profit=15.0)
        
        # 2.0 lots → need $30 profit (2× × $15)
        should_skip_1, _ = filter_obj.should_skip_trade(
            predicted_profit=20.0, lot_size=2.0
        )
        should_skip_2, _ = filter_obj.should_skip_trade(
            predicted_profit=35.0, lot_size=2.0
        )
        
        assert should_skip_1 is True   # $20 < $30 required
        assert should_skip_2 is False  # $35 > $30 required
    
    def test_filter_signals_dataframe(self):
        """Test filtering a DataFrame of signals."""
        filter_obj = MinimumProfitFilter(min_expected_profit=15.0)
        
        # Create test signals
        signals = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=5, freq="1h"),
            "predicted_profit": [20.0, 12.0, 18.0, 8.0, 25.0],
            "predicted_rr": [2.0, 1.5, 2.2, 1.0, 3.0],
            "lot_size": [1.0, 1.0, 1.0, 1.0, 1.0],
        })
        
        filtered_df, stats = filter_obj.filter_signals(signals)
        
        # Should keep signals with profit >= $15: [20, 18, 25]
        assert len(filtered_df) == 3
        assert stats["total_signals"] == 5
        assert stats["filtered_signals"] == 3
        assert stats["skipped_signals"] == 2
        assert stats["skip_rate"] == 0.4  # 2/5 = 40%
        
        # Verify correct signals kept
        assert list(filtered_df["predicted_profit"]) == [20.0, 18.0, 25.0]
    
    def test_filter_signals_missing_column(self):
        """Test handling of missing predicted_profit column."""
        filter_obj = MinimumProfitFilter(min_expected_profit=15.0)
        
        # Missing predicted_profit column
        signals = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=3, freq="1h"),
            "side": ["BUY", "SELL", "BUY"],
        })
        
        filtered_df, stats = filter_obj.filter_signals(signals)
        
        # Should return all signals unchanged
        assert len(filtered_df) == 3
        assert stats["total_signals"] == 3
        assert stats["filtered_signals"] == 3
        assert stats["skipped_signals"] == 0
    
    def test_filter_signals_when_disabled(self):
        """Test that disabled filter returns all signals."""
        filter_obj = MinimumProfitFilter(min_expected_profit=15.0, enabled=False)
        
        signals = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=5, freq="1h"),
            "predicted_profit": [5.0, 8.0, 10.0, 12.0, 14.0],  # All below $15
            "lot_size": [1.0] * 5,
        })
        
        filtered_df, stats = filter_obj.filter_signals(signals)
        
        # All signals should pass when filter disabled
        assert len(filtered_df) == 5
        assert stats["skipped_signals"] == 0
    
    def test_update_config(self):
        """Test dynamic config updates."""
        filter_obj = MinimumProfitFilter(min_expected_profit=15.0, enabled=True)
        
        # Update minimum profit threshold
        filter_obj.update_config(min_expected_profit=20.0)
        assert filter_obj.min_expected_profit == 20.0
        
        # Verify new threshold is enforced
        should_skip, _ = filter_obj.should_skip_trade(predicted_profit=18.0)
        assert should_skip is True  # $18 < $20 new threshold
        
        # Disable filter
        filter_obj.update_config(enabled=False)
        assert filter_obj.enabled is False
        
        # Same trade should now pass
        should_skip, _ = filter_obj.should_skip_trade(predicted_profit=18.0)
        assert should_skip is False
    
    def test_realistic_xauusd_scenario(self):
        """Test with realistic XAUUSD trading scenario."""
        # XAUUSD: 0.5 pip spread = $10 per trade
        # Set min profit = $15 (spread + 50% buffer)
        filter_obj = MinimumProfitFilter(
            min_expected_profit=15.0,
            spread_pips=0.5,
            pip_value=10.0,
        )
        
        # Realistic signal mix
        signals = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=10, freq="1h"),
            "predicted_profit": [
                8.0,   # Skip: scalping, too small
                12.0,  # Skip: below threshold
                16.0,  # Keep: just above threshold
                25.0,  # Keep: good profit
                10.0,  # Skip: borderline
                30.0,  # Keep: excellent profit
                14.0,  # Skip: just below threshold
                20.0,  # Keep: solid profit
                9.0,   # Skip: too small
                40.0,  # Keep: very good profit
            ],
            "predicted_rr": [1.2, 1.5, 1.8, 2.0, 1.3, 2.5, 1.7, 2.2, 1.1, 3.0],
            "lot_size": [1.0] * 10,
        })
        
        filtered_df, stats = filter_obj.filter_signals(signals)
        
        # Should keep 5 signals: [16, 25, 30, 20, 40]
        assert stats["filtered_signals"] == 5
        assert stats["skipped_signals"] == 5
        assert stats["skip_rate"] == 0.5  # 50% filtered
        
        # Verify correct signals kept
        expected_profits = [16.0, 25.0, 30.0, 20.0, 40.0]
        assert list(filtered_df["predicted_profit"]) == expected_profits
    
    def test_edge_case_exact_threshold(self):
        """Test edge case where predicted profit equals threshold."""
        filter_obj = MinimumProfitFilter(min_expected_profit=15.0)
        
        # Exactly $15 → SKIP (need strictly greater)
        should_skip, _ = filter_obj.should_skip_trade(predicted_profit=15.0)
        assert should_skip is True
        
        # $14.99 → SKIP
        should_skip, _ = filter_obj.should_skip_trade(predicted_profit=14.99)
        assert should_skip is True
        
        # $15.01 → ALLOW (strictly greater than threshold)
        should_skip, _ = filter_obj.should_skip_trade(predicted_profit=15.01)
        assert should_skip is False
    
    def test_negative_predicted_profit(self):
        """Test handling of negative predicted profit."""
        filter_obj = MinimumProfitFilter(min_expected_profit=15.0)
        
        # Negative profit → definitely SKIP
        should_skip, reason = filter_obj.should_skip_trade(predicted_profit=-10.0)
        
        assert should_skip is True
        assert "profit_too_low" in reason
        assert "predicted=$-10.00" in reason


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
