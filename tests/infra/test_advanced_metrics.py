#!/usr/bin/env python3
"""
Tests for advanced_metrics module (Sharpe, Sortino, Calmar, Turnover-Adjusted Return).
"""

import pytest
import pandas as pd
import numpy as np
from xauusd_ai.infra.advanced_metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_daily_returns,
    calculate_all_metrics,
    calculate_turnover_adjusted_return,
)


class TestTurnoverAdjustedReturn:
    """Test suite for turnover-adjusted return calculation."""
    
    def test_basic_turnover_calculation(self):
        """Test basic spread and swap cost calculation."""
        trades = pd.DataFrame({
            'pnl': [50.0, -20.0, 30.0],  # Gross P&L = $60
            'lot_size': [1.0, 1.0, 1.0],
            'holding_bars': [100, 50, 1440],  # Last trade holds 1 day (1440 mins)
            'balance': [10000.0] * 3,
        })
        
        result = calculate_turnover_adjusted_return(
            trades,
            spread_pips=0.5,
            swap_per_lot_per_day=0.15,
            pip_value=10.0,
        )
        
        # Spread cost: 3 trades × 2× (entry+exit) × 0.5 pips × $10 = $30
        # Swap cost: 1 trade × 1 day × 1.0 lot × $0.15 = $0.15
        # Net P&L: $60 - $30 - $0.15 = $29.85
        
        assert result["gross_pnl"] == 60.0
        assert abs(result["spread_cost"] - 30.0) < 0.01
        assert abs(result["swap_cost"] - 0.15) < 0.01
        assert abs(result["net_pnl"] - 29.85) < 0.01
        assert abs(result["turnover_drag"] - 0.5025) < 0.01  # 50.25%
    
    def test_multiple_lot_sizes(self):
        """Test with varying lot sizes."""
        trades = pd.DataFrame({
            'pnl': [100.0, 50.0],
            'lot_size': [2.0, 0.5],  # Different lot sizes
            'holding_bars': [0, 0],  # No overnight holding
            'balance': [10000.0] * 2,
        })
        
        result = calculate_turnover_adjusted_return(
            trades,
            spread_pips=0.5,
            swap_per_lot_per_day=0.0,  # No swap
            pip_value=10.0,
        )
        
        # Spread cost: (2 × 2.0 × 0.5 × 10) + (2 × 0.5 × 0.5 × 10) = 20 + 5 = $25
        assert abs(result["spread_cost"] - 25.0) < 0.01
        assert result["swap_cost"] == 0.0
        assert abs(result["net_pnl"] - 125.0) < 0.01  # 150 - 25
    
    def test_overnight_holding_costs(self):
        """Test swap cost calculation for multi-day holds."""
        trades = pd.DataFrame({
            'pnl': [100.0, 50.0, 30.0],
            'lot_size': [1.0, 1.0, 1.0],
            'holding_bars': [1440, 2880, 4320],  # 1, 2, 3 days
            'balance': [10000.0] * 3,
        })
        
        result = calculate_turnover_adjusted_return(
            trades,
            spread_pips=0.0,  # No spread cost
            swap_per_lot_per_day=0.20,
            pip_value=10.0,
        )
        
        # Swap cost: (1×0.20) + (2×0.20) + (3×0.20) = 0.20 + 0.40 + 0.60 = $1.20
        assert abs(result["swap_cost"] - 1.20) < 0.01
        assert result["spread_cost"] == 0.0
        assert abs(result["net_pnl"] - 178.80) < 0.01  # 180 - 1.20
    
    def test_zero_trades(self):
        """Test with empty DataFrame."""
        trades = pd.DataFrame({
            'pnl': [],
            'lot_size': [],
            'holding_bars': [],
            'balance': [],
        })
        
        result = calculate_turnover_adjusted_return(trades)
        
        assert result["gross_pnl"] == 0.0
        assert result["spread_cost"] == 0.0
        assert result["swap_cost"] == 0.0
        assert result["net_pnl"] == 0.0
        assert result["turnover_drag"] == 0.0
    
    def test_negative_pnl(self):
        """Test with losing trades."""
        trades = pd.DataFrame({
            'pnl': [-50.0, -30.0, -20.0],  # Gross loss = -$100
            'lot_size': [1.0, 1.0, 1.0],
            'holding_bars': [0, 0, 0],
            'balance': [10000.0] * 3,
        })
        
        result = calculate_turnover_adjusted_return(
            trades,
            spread_pips=0.5,
            swap_per_lot_per_day=0.0,
            pip_value=10.0,
        )
        
        # Spread cost: 3 × 2 × 0.5 × 10 = $30
        # Net: -100 - 30 = -$130
        assert result["gross_pnl"] == -100.0
        assert result["spread_cost"] == 30.0
        assert result["net_pnl"] == -130.0
        # Turnover drag is inf for negative gross PnL (set to 0)
        assert result["turnover_drag"] == 0.0
    
    def test_realistic_xauusd_scenario(self):
        """Test with realistic XAUUSD trading scenario."""
        # Simulate 10 trades with 40% win rate
        np.random.seed(42)
        trades = pd.DataFrame({
            'pnl': [50, -30, -20, 60, -25, 40, -15, -30, 45, 55],  # Net $130
            'lot_size': [1.0] * 10,
            'holding_bars': [120, 60, 90, 180, 45, 300, 75, 120, 240, 150],
            'balance': [10000.0] * 10,
        })
        
        result = calculate_turnover_adjusted_return(
            trades,
            spread_pips=0.5,
            swap_per_lot_per_day=0.15,
            pip_value=10.0,
        )
        
        # Spread: 10 trades × 2 × 0.5 × 10 = $100
        # Swap: sum(holding_bars // 1440) × 0.15 = 0 (all < 1 day)
        assert result["gross_pnl"] == 130.0
        assert result["spread_cost"] == 100.0
        assert result["swap_cost"] == 0.0  # No overnight holds
        assert result["net_pnl"] == 30.0
        assert abs(result["turnover_drag"] - 0.7692) < 0.01  # 76.92%
    
    def test_turnover_adjusted_return_pct(self):
        """Test net return percentage calculation."""
        trades = pd.DataFrame({
            'pnl': [500.0],  # Gross $500
            'lot_size': [1.0],
            'holding_bars': [0],
            'balance': [10000.0],  # Starting balance
        })
        
        result = calculate_turnover_adjusted_return(
            trades,
            spread_pips=0.5,
            swap_per_lot_per_day=0.0,
            pip_value=10.0,
        )
        
        # Spread: 2 × 0.5 × 10 = $10
        # Net: 500 - 10 = $490
        # Net return %: 490 / 10000 = 4.9%
        assert abs(result["net_return_pct"] - 0.049) < 0.001


class TestExistingMetrics:
    """Test suite for existing metrics (regression tests)."""
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = [0.01, 0.02, -0.01, 0.015, 0.005]
        sharpe = calculate_sharpe_ratio(returns, periods_per_year=252)
        assert sharpe > 0
        assert isinstance(sharpe, float)
    
    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        # Use more downside returns (need >= 2 for std dev calculation)
        returns = [0.02, 0.03, -0.01, 0.015, 0.005, -0.02, 0.01, -0.015]
        sortino = calculate_sortino_ratio(returns, periods_per_year=252)
        assert sortino > 0
        assert isinstance(sortino, float)
    
    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        calmar = calculate_calmar_ratio(1.5, 0.15, years=1.0)
        assert abs(calmar - 10.0) < 0.01
    
    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        balances = [100, 110, 105, 120, 95, 100]
        dd = calculate_max_drawdown(balances)
        assert 0.20 < dd < 0.21  # ~20.8% from 120 to 95
    
    def test_all_metrics(self):
        """Test comprehensive metrics calculation."""
        balances = [100, 105, 110, 108, 115, 120]
        metrics = calculate_all_metrics(balances, periods_per_year=252)
        assert 'sharpe' in metrics
        assert 'calmar' in metrics
        assert 'sortino' in metrics
        assert 'max_dd' in metrics
        assert 'total_return' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
