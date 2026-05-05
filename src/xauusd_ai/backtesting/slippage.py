"""
Dynamic slippage model for realistic backtesting.

Slippage estimation based on:
- ATR (Average True Range) — volatility factor
- Spread — current market spread
- Volume ratio — relative volume (low volume = more slippage)
- Session multiplier — time of day liquidity (Asian vs London/NY)

Typical XAUUSD slippage:
- Normal conditions: 1-2 pips
- High volatility: 2-4 pips
- Asian session + low volume: 3-6 pips
- Extreme conditions (news): 5-10 pips
"""
from __future__ import annotations

from datetime import time
from typing import Optional


def calculate_slippage_pips(
    atr: float,
    atr_mean: float,
    spread_pips: float,
    volume_ratio: float = 1.0,
    session_multiplier: float = 1.0,
) -> float:
    """
    Calculate dynamic slippage in pips based on market conditions.
    
    Args:
        atr: Current ATR value
        atr_mean: Mean ATR over lookback period (baseline)
        spread_pips: Current spread in pips
        volume_ratio: Current volume / mean volume (default 1.0)
        session_multiplier: Session liquidity multiplier (1.0=normal, 1.5=Asian)
    
    Returns:
        Slippage in pips (always >= 0)
    
    Formula:
        slippage = base + atr_component + spread_component + volume_component
        
        base = 1.0 pips (minimum slippage)
        atr_component = max(0, (atr/atr_mean - 1.0)) * 0.5 * base
        spread_component = spread_pips * 0.3
        volume_component = max(0, (1.0 - volume_ratio)) * 0.5 * base
        
        total = (base + atr_component + spread_component + volume_component) * session_multiplier
    
    Examples:
        >>> # Normal conditions
        >>> calculate_slippage_pips(10.0, 10.0, 2.0, 1.0, 1.0)
        1.6  # base(1.0) + atr(0) + spread(0.6) + volume(0) = 1.6 pips
        
        >>> # High volatility (ATR 2x normal)
        >>> calculate_slippage_pips(20.0, 10.0, 2.0, 1.0, 1.0)
        2.1  # base(1.0) + atr(0.5) + spread(0.6) + volume(0) = 2.1 pips
        
        >>> # Asian session + low volume
        >>> calculate_slippage_pips(10.0, 10.0, 3.0, 0.5, 1.5)
        3.525  # (base(1.0) + atr(0) + spread(0.9) + volume(0.25)) * 1.5 = 3.525 pips
    """
    # Base slippage (minimum)
    base_slippage = 1.0
    
    # ATR component: higher volatility = more slippage, lower = less
    # If ATR is 2x normal, add 50% more slippage
    # If ATR is 0.5x normal, reduce by 25% slippage
    if atr_mean > 0:
        atr_factor = (atr / atr_mean - 1.0)  # Can be negative
        atr_component = atr_factor * 0.5 * base_slippage
    else:
        atr_component = 0.0
    
    # Spread component: wider spread = more slippage
    # Each pip of spread adds 30% of that pip as slippage
    spread_component = spread_pips * 0.3
    
    # Volume component: low volume = more slippage
    # If volume is 50% of normal, add 25% more slippage
    volume_factor = max(0.0, (1.0 - volume_ratio))
    volume_component = volume_factor * 0.5 * base_slippage
    
    # Total slippage before session adjustment
    total_slippage = base_slippage + atr_component + spread_component + volume_component
    
    # Apply session multiplier
    final_slippage = total_slippage * session_multiplier
    
    # Ensure minimum slippage of 0.5 pips (can't be too perfect)
    return max(0.5, final_slippage)


def calculate_slippage_rr(
    slippage_pips: float,
    entry_price: float,
    stop_loss_price: float,
) -> float:
    """
    Convert slippage from pips to RR (risk multiples).
    
    Args:
        slippage_pips: Slippage in pips
        entry_price: Entry price
        stop_loss_price: Stop loss price
    
    Returns:
        Slippage as fraction of 1R (e.g., 0.2 = 20% of risk)
    
    Examples:
        >>> # Long position: entry 2000, SL 1990 (10 pips risk)
        >>> # Slippage 2 pips = 0.2 R (20% of risk)
        >>> calculate_slippage_rr(2.0, 2000.0, 1990.0)
        0.2
        
        >>> # Short position: entry 2000, SL 2010 (10 pips risk)
        >>> # Slippage 2 pips = 0.2 R (20% of risk)
        >>> calculate_slippage_rr(2.0, 2000.0, 2010.0)
        0.2
        
        >>> # Tight stop: entry 2000, SL 1995 (5 pips risk)
        >>> # Slippage 2 pips = 0.4 R (40% of risk)
        >>> calculate_slippage_rr(2.0, 2000.0, 1995.0)
        0.4
    """
    # Calculate risk in pips (always positive)
    risk_pips = abs(entry_price - stop_loss_price)
    
    # Avoid division by zero
    if risk_pips < 0.1:  # Less than 0.1 pips = invalid
        return 0.0
    
    # Slippage as fraction of 1R
    slippage_rr = slippage_pips / risk_pips
    
    return slippage_rr


def get_session_multiplier(current_time: time) -> float:
    """
    Get session-based liquidity multiplier.
    
    Args:
        current_time: Current time (UTC)
    
    Returns:
        Session multiplier:
        - 1.5: Asian session (22:00-08:00 UTC) — low liquidity
        - 1.0: London session (08:00-16:00 UTC) — normal liquidity
        - 0.9: NY session (13:00-21:00 UTC) — high liquidity
        - 1.0: London/NY overlap (13:00-16:00 UTC) — best liquidity
    
    Examples:
        >>> get_session_multiplier(time(2, 0))  # 02:00 UTC - Asian
        1.5
        
        >>> get_session_multiplier(time(10, 0))  # 10:00 UTC - London
        1.0
        
        >>> get_session_multiplier(time(15, 0))  # 15:00 UTC - NY
        0.9
    """
    hour = current_time.hour
    
    # Asian session: 22:00-08:00 UTC (low liquidity)
    if hour >= 22 or hour < 8:
        return 1.5
    
    # London session: 08:00-16:00 UTC (normal liquidity)
    elif 8 <= hour < 13:
        return 1.0
    
    # London/NY overlap: 13:00-16:00 UTC (best liquidity)
    elif 13 <= hour < 16:
        return 0.9
    
    # NY session: 16:00-21:00 UTC (good liquidity)
    elif 16 <= hour < 21:
        return 0.9
    
    # Pre-Asian: 21:00-22:00 UTC (transitioning to Asian)
    else:
        return 1.2
