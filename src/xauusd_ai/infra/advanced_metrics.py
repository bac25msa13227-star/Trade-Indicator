#!/usr/bin/env python3
"""
Enhanced metrics module with Sharpe, Calmar, and Sortino ratios.
"""

import numpy as np
from typing import List, Optional, Dict, Any
import pandas as pd


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Sharpe ratio measures risk-adjusted return by comparing excess returns
    to volatility. Higher is better (>1.0 is good, >2.0 is excellent).
    
    Args:
        returns: List of periodic returns (as fractions, e.g., 0.02 for 2%)
        risk_free_rate: Annual risk-free rate (default 0%)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
    
    Returns:
        Sharpe ratio (annualized)
    
    Example:
        >>> returns = [0.01, 0.02, -0.01, 0.015, 0.005]
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> sharpe > 0  # Positive Sharpe
        True
    """
    if len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns, dtype=float)
    
    # Calculate periodic risk-free rate
    periodic_rf = risk_free_rate / periods_per_year
    
    # Excess returns
    excess_returns = returns_array - periodic_rf
    
    # Mean excess return
    mean_excess = np.mean(excess_returns)
    
    # Standard deviation of excess returns
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0 or np.isnan(std_excess):
        return 0.0
    
    # Annualize: multiply by sqrt(periods per year)
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    
    return float(sharpe)


def calculate_calmar_ratio(
    total_return: float,
    max_drawdown: float,
    years: Optional[float] = None,
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Calmar ratio measures return relative to worst drawdown. Higher is better
    (>1.0 is good, >3.0 is excellent).
    
    Args:
        total_return: Total return as fraction (e.g., 1.5 for 150% gain)
        max_drawdown: Max drawdown as positive fraction (e.g., 0.15 for 15% DD)
        years: Number of years (for annualization). If None, uses total_return as-is
    
    Returns:
        Calmar ratio
    
    Example:
        >>> calmar = calculate_calmar_ratio(1.5, 0.15, years=1.0)
        >>> calmar == 10.0
        True
    """
    if max_drawdown == 0 or max_drawdown is None:
        return 0.0
    
    # Annualize return if years provided
    if years is not None and years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = total_return
    
    calmar = annualized_return / max_drawdown
    
    return float(calmar)


def calculate_sortino_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sortino ratio (only penalizes downside volatility).
    
    Sortino is similar to Sharpe but only considers downside deviation,
    making it better for strategies with asymmetric returns.
    
    Args:
        returns: List of periodic returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Sortino ratio (annualized)
    
    Example:
        >>> returns = [0.02, 0.03, -0.01, 0.015, 0.005]
        >>> sortino = calculate_sortino_ratio(returns)
        >>> sortino > 0
        True
    """
    if len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns, dtype=float)
    
    # Calculate periodic risk-free rate
    periodic_rf = risk_free_rate / periods_per_year
    
    # Excess returns
    excess_returns = returns_array - periodic_rf
    
    # Mean excess return
    mean_excess = np.mean(excess_returns)
    
    # Only downside returns (below risk-free rate)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        # No downside, infinite Sortino (cap at high value)
        return 999.0
    
    # Downside deviation
    downside_std = np.std(downside_returns, ddof=1)
    
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    
    # Annualize
    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)
    
    return float(sortino)


def calculate_max_drawdown(balance_series: List[float]) -> float:
    """
    Calculate maximum drawdown from balance series.
    
    Args:
        balance_series: List of balance values over time
    
    Returns:
        Max drawdown as positive fraction (e.g., 0.15 for 15% DD)
    
    Example:
        >>> balances = [100, 110, 105, 120, 95, 100]
        >>> dd = calculate_max_drawdown(balances)
        >>> 0.20 < dd < 0.21  # ~20.8% from 120 to 95
        True
    """
    if len(balance_series) < 2:
        return 0.0
    
    balances = np.array(balance_series, dtype=float)
    
    # Running maximum
    running_max = np.maximum.accumulate(balances)
    
    # Drawdown at each point
    drawdowns = (running_max - balances) / running_max
    
    # Max drawdown
    max_dd = np.max(drawdowns)
    
    return float(max_dd)


def calculate_daily_returns(balance_series: List[float]) -> List[float]:
    """
    Calculate daily returns from balance series.
    
    Args:
        balance_series: List of balance values
    
    Returns:
        List of returns as fractions
    
    Example:
        >>> balances = [100, 102, 101, 105]
        >>> returns = calculate_daily_returns(balances)
        >>> len(returns) == 3
        True
    """
    if len(balance_series) < 2:
        return []
    
    balances = np.array(balance_series, dtype=float)
    returns = np.diff(balances) / balances[:-1]
    
    return returns.tolist()


def calculate_turnover_adjusted_return(
    trades: pd.DataFrame,
    spread_pips: float = 0.5,
    swap_per_lot_per_day: float = 0.15,
    pip_value: float = 10.0,
) -> Dict[str, float]:
    """
    Calculate net return after spread and swap costs (turnover-adjusted).
    
    This function quantifies the actual profitability after accounting for
    trading costs that are ignored in raw P&L calculations:
    - Spread cost: bid-ask spread paid on entry + exit (2× per trade)
    - Swap cost: overnight financing fees for positions held multiple days
    
    Args:
        trades: DataFrame with columns [pnl, lot_size, holding_bars, balance]
        spread_pips: Average bid-ask spread in pips (default 0.5 for XAUUSD)
        swap_per_lot_per_day: Overnight financing cost per lot (default 0.15)
        pip_value: Dollar value per pip at 1.0 lot (default 10 for XAUUSD)
    
    Returns:
        Dict with:
            - gross_pnl: Total P&L before costs
            - spread_cost: Total spread costs (entry + exit)
            - swap_cost: Total overnight financing fees
            - net_pnl: P&L after costs (gross - spread - swap)
            - turnover_drag: Fraction of profit lost to costs
            - net_return_pct: Net return as percentage of initial balance
    
    Example:
        >>> trades = pd.DataFrame({
        ...     'pnl': [50, -20, 30],
        ...     'lot_size': [1.0, 1.0, 1.0],
        ...     'holding_bars': [100, 50, 1440],  # 1 day = 1440 mins
        ...     'balance': [10000, 10000, 10000]
        ... })
        >>> result = calculate_turnover_adjusted_return(trades)
        >>> result['gross_pnl'] == 60.0
        True
        >>> result['net_pnl'] < result['gross_pnl']  # Net always less after costs
        True
    """
    if len(trades) == 0:
        return {
            "gross_pnl": 0.0,
            "spread_cost": 0.0,
            "swap_cost": 0.0,
            "net_pnl": 0.0,
            "turnover_drag": 0.0,
            "net_return_pct": 0.0,
        }
    
    gross_pnl = float(trades['pnl'].sum())
    
    # Spread cost: Each trade pays spread on entry AND exit (2×)
    spread_cost = 0.0
    for _, row in trades.iterrows():
        spread_cost += 2 * spread_pips * pip_value * row['lot_size']
    
    # Swap cost: Only for trades held overnight (>= 1440 minutes = 1 day)
    swap_cost = 0.0
    for _, row in trades.iterrows():
        holding_days = int(row['holding_bars']) // 1440
        swap_cost += swap_per_lot_per_day * row['lot_size'] * holding_days
    
    # Net P&L after costs
    net_pnl = gross_pnl - spread_cost - swap_cost
    
    # Turnover drag: fraction of profit lost to transaction costs
    # If gross P&L is negative or zero, set drag to 0 (avoid division issues)
    if gross_pnl > 0:
        turnover_drag = (spread_cost + swap_cost) / gross_pnl
    else:
        turnover_drag = 0.0
    
    # Net return percentage (relative to initial balance)
    initial_balance = float(trades['balance'].iloc[0]) if len(trades) > 0 else 1.0
    net_return_pct = net_pnl / initial_balance
    
    return {
        "gross_pnl": gross_pnl,
        "spread_cost": spread_cost,
        "swap_cost": swap_cost,
        "net_pnl": net_pnl,
        "turnover_drag": turnover_drag,
        "net_return_pct": net_return_pct,
    }


def calculate_all_metrics(
    balance_series: List[float],
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        balance_series: List of balance values over time
        periods_per_year: 252 for daily, 12 for monthly, etc.
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Dict with sharpe, calmar, sortino, max_dd, total_return
    
    Example:
        >>> balances = [100, 105, 110, 108, 115, 120]
        >>> metrics = calculate_all_metrics(balances)
        >>> 'sharpe' in metrics and 'calmar' in metrics
        True
    """
    if len(balance_series) < 2:
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_dd": 0.0,
            "total_return": 0.0,
        }
    
    # Calculate returns
    returns = calculate_daily_returns(balance_series)
    
    # Total return
    total_return = (balance_series[-1] / balance_series[0]) - 1
    
    # Max drawdown
    max_dd = calculate_max_drawdown(balance_series)
    
    # Sharpe ratio
    sharpe = calculate_sharpe_ratio(
        returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )
    
    # Sortino ratio
    sortino = calculate_sortino_ratio(
        returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )
    
    # Calmar ratio
    years = len(balance_series) / periods_per_year
    calmar = calculate_calmar_ratio(total_return, max_dd, years=years)
    
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd": max_dd,
        "total_return": total_return,
    }


if __name__ == "__main__":
    # Example usage
    example_balances = [100, 105, 110, 108, 115, 112, 120, 125, 118, 130]
    metrics = calculate_all_metrics(example_balances, periods_per_year=252)
    
    print("Performance Metrics:")
    print(f"  Sharpe Ratio:    {metrics['sharpe']:.2f}")
    print(f"  Sortino Ratio:   {metrics['sortino']:.2f}")
    print(f"  Calmar Ratio:    {metrics['calmar']:.2f}")
    print(f"  Max Drawdown:    {metrics['max_dd']:.2%}")
    print(f"  Total Return:    {metrics['total_return']:.2%}")
