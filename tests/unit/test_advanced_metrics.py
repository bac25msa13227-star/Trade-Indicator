#!/usr/bin/env python3
"""Unit tests for advanced performance metrics."""

import pytest
import numpy as np
from src.xauusd_ai.infra.advanced_metrics import (
    calculate_sharpe_ratio,
    calculate_calmar_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_daily_returns,
    calculate_all_metrics,
)


def test_calculate_sharpe_ratio_positive_returns():
    """Test Sharpe ratio with positive returns."""
    returns = [0.01, 0.02, 0.015, 0.008, 0.012]
    sharpe = calculate_sharpe_ratio(returns, periods_per_year=252)
    assert sharpe > 0
    # Mean = 0.013, std ≈ 0.0048, Sharpe ≈ (0.013/0.0048) * sqrt(252) ≈ 43
    assert 30 < sharpe < 50


def test_calculate_sharpe_ratio_mixed_returns():
    """Test Sharpe ratio with mixed returns."""
    returns = [0.02, -0.01, 0.015, -0.005, 0.01]
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe > 0  # Positive mean return
    assert sharpe < 20  # But volatile


def test_calculate_sharpe_ratio_zero_volatility():
    """Test Sharpe with zero volatility."""
    returns = [0.01, 0.01, 0.01, 0.01]
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe == 0.0  # Undefined, return 0


def test_calculate_sharpe_ratio_insufficient_data():
    """Test Sharpe with insufficient data."""
    returns = [0.01]
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe == 0.0


def test_calculate_calmar_ratio():
    """Test Calmar ratio calculation."""
    # 50% return, 10% max DD
    calmar = calculate_calmar_ratio(0.5, 0.10, years=1.0)
    assert calmar == 5.0
    
    # 200% return over 2 years, 20% DD
    # Annualized: (1 + 2.0)^(1/2) - 1 ≈ 0.732 (73.2%)
    calmar = calculate_calmar_ratio(2.0, 0.20, years=2.0)
    assert 3.5 < calmar < 4.0


def test_calculate_calmar_ratio_zero_drawdown():
    """Test Calmar with zero drawdown."""
    calmar = calculate_calmar_ratio(0.5, 0.0)
    assert calmar == 0.0  # Undefined


def test_calculate_calmar_ratio_no_years():
    """Test Calmar without year normalization."""
    calmar = calculate_calmar_ratio(1.5, 0.15, years=None)
    assert calmar == 10.0


def test_calculate_sortino_ratio():
    """Test Sortino ratio calculation."""
    # More upside than downside
    returns = [0.02, 0.03, -0.01, 0.015, 0.005, -0.005]
    sortino = calculate_sortino_ratio(returns)
    assert sortino > 0
    assert sortino > 10  # Less penalty than Sharpe


def test_calculate_sortino_ratio_no_downside():
    """Test Sortino with no downside returns."""
    returns = [0.01, 0.02, 0.015, 0.008]
    sortino = calculate_sortino_ratio(returns)
    assert sortino == 999.0  # Capped at high value


def test_calculate_sortino_ratio_all_losses():
    """Test Sortino with all losses."""
    returns = [-0.01, -0.02, -0.015]
    sortino = calculate_sortino_ratio(returns)
    assert sortino < 0  # Negative mean return


def test_calculate_max_drawdown():
    """Test max drawdown calculation."""
    # Peak at 120, trough at 95, DD = (120-95)/120 = 20.8%
    balances = [100, 110, 105, 120, 95, 100]
    dd = calculate_max_drawdown(balances)
    assert 0.20 < dd < 0.21


def test_calculate_max_drawdown_no_drawdown():
    """Test max DD with monotonically increasing balance."""
    balances = [100, 105, 110, 115, 120]
    dd = calculate_max_drawdown(balances)
    assert dd == 0.0


def test_calculate_max_drawdown_single_value():
    """Test max DD with single value."""
    balances = [100]
    dd = calculate_max_drawdown(balances)
    assert dd == 0.0


def test_calculate_daily_returns():
    """Test daily returns calculation."""
    balances = [100, 102, 101, 105]
    returns = calculate_daily_returns(balances)
    
    assert len(returns) == 3
    assert abs(returns[0] - 0.02) < 0.0001  # (102-100)/100 = 2%
    assert abs(returns[1] - (-0.0098)) < 0.001  # (101-102)/102 ≈ -0.98%
    assert abs(returns[2] - 0.0396) < 0.001  # (105-101)/101 ≈ 3.96%


def test_calculate_daily_returns_insufficient_data():
    """Test returns with insufficient data."""
    balances = [100]
    returns = calculate_daily_returns(balances)
    assert returns == []


def test_calculate_all_metrics():
    """Test comprehensive metrics calculation."""
    # Growing balance with some volatility
    balances = [100, 105, 110, 108, 115, 112, 120, 125, 118, 130]
    
    metrics = calculate_all_metrics(balances, periods_per_year=252)
    
    assert "sharpe" in metrics
    assert "sortino" in metrics
    assert "calmar" in metrics
    assert "max_dd" in metrics
    assert "total_return" in metrics
    
    # Total return: (130-100)/100 = 30%
    assert abs(metrics["total_return"] - 0.30) < 0.01
    
    # Max DD should be positive
    assert metrics["max_dd"] > 0
    
    # Sharpe/Sortino should be reasonable
    assert metrics["sharpe"] > 0
    assert metrics["sortino"] > 0
    
    # Calmar should be positive
    assert metrics["calmar"] > 0


def test_calculate_all_metrics_insufficient_data():
    """Test metrics with insufficient data."""
    balances = [100]
    metrics = calculate_all_metrics(balances)
    
    assert metrics["sharpe"] == 0.0
    assert metrics["calmar"] == 0.0
    assert metrics["total_return"] == 0.0


def test_sharpe_with_risk_free_rate():
    """Test Sharpe ratio with non-zero risk-free rate."""
    returns = [0.05, 0.06, 0.04, 0.055, 0.045]  # 5% avg return
    
    # With 3% risk-free rate
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.03, periods_per_year=12)
    
    # Excess return ≈ 5% - 0.25% = 4.75% monthly
    assert sharpe > 0


def test_metrics_with_monthly_data():
    """Test metrics with monthly periodicity."""
    # Monthly returns for 1 year
    monthly_balances = [100, 102, 105, 104, 107, 110, 108, 112, 115, 113, 118, 120, 122]
    
    metrics = calculate_all_metrics(monthly_balances, periods_per_year=12)
    
    # Total return: 22%
    assert abs(metrics["total_return"] - 0.22) < 0.01
    
    # Metrics should still be reasonable
    assert metrics["sharpe"] > 0
    assert metrics["calmar"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
