"""
Entry Quality Filter

Requires multiple confirmations before entering a trade to boost win rate.
Scores 5 independent confirmation signals and requires 3/5 to pass (60% threshold).

Target Impact: Win rate 41% → 48% (+7%) = +17% profit
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_entry_quality_score(row: pd.Series | object) -> tuple[float, dict]:
    """
    Calculate entry quality score from 5 independent confirmations.
    
    Args:
        row: Trade signal row with features
    
    Returns:
        (total_score, confirmation_details):
        - total_score: 0.0-1.0 (average of 5 confirmations)
        - confirmation_details: Dict with individual scores
    
    Confirmations (each 0.0-1.0):
        1. ICT Quality: Liquidity sweep + displacement + FVG alignment
        2. Wyckoff Quality: Spring/upthrust + accumulation phase
        3. Volume Quality: Volume > MA, volume spike
        4. Trend Quality: Multi-timeframe trend alignment  
        5. Momentum Quality: RSI + MACD + strength convergence
    """
    confirmations = {}
    
    # 1. ICT Quality (liquidity sweep + displacement + FVG)
    ict_score = 0.0
    liq_sweep = float(getattr(row, "liquidity_sweep", 0.0))
    displacement = abs(float(getattr(row, "price_displacement", 0.0)))
    fvg_present = int(getattr(row, "fvg", 0)) != 0
    
    if liq_sweep > 0:
        ict_score += 0.4  # Liquidity sweep present
    if displacement > 0.5:  # Significant displacement
        ict_score += 0.3
    if fvg_present:
        ict_score += 0.3  # Fair Value Gap present
    
    confirmations["ict"] = min(ict_score, 1.0)
    
    # 2. Wyckoff Quality (spring/upthrust + phase)
    wyckoff_score = 0.0
    wyckoff_phase = int(getattr(row, "wyckoff_phase", 0))
    volume_ratio = float(getattr(row, "tick_volume_ratio", 1.0))
    
    if abs(wyckoff_phase) == 1:
        wyckoff_score += 0.5  # In accumulation/distribution phase
    if volume_ratio > 1.2:  # High volume confirmation
        wyckoff_score += 0.5
    
    confirmations["wyckoff"] = min(wyckoff_score, 1.0)
    
    # 3. Volume Quality (volume > MA, volume spike)
    volume_score = 0.0
    tick_volume_zscore = float(getattr(row, "tick_volume_zscore", 0.0))
    volume_ma_ratio = volume_ratio  # Already have this from wyckoff
    
    if tick_volume_zscore > 1.0:  # Volume spike (>1 std above mean)
        volume_score += 0.6
    elif tick_volume_zscore > 0.5:  # Moderate volume increase
        volume_score += 0.3
    
    if volume_ma_ratio > 1.1:  # Volume above MA
        volume_score += 0.4
    
    confirmations["volume"] = min(volume_score, 1.0)
    
    # 4. Trend Quality (multi-timeframe alignment)
    trend_score = 0.0
    trend_alignment = int(getattr(row, "trend_alignment", 0))
    trend_strength = float(getattr(row, "trend_strength", 0.0))
    daily_bias = float(getattr(row, "daily_bias", 0.0))
    trade_side = str(getattr(row, "trade_side", "")).strip().lower()
    
    if trend_alignment == 1:
        trend_score += 0.5  # All timeframes aligned
    
    if abs(trend_strength) > 0.5:  # Strong trend
        trend_score += 0.3
    
    # Daily bias confirms trade direction
    if trade_side == "buy" and daily_bias > 0:
        trend_score += 0.2
    elif trade_side == "sell" and daily_bias < 0:
        trend_score += 0.2
    
    confirmations["trend"] = min(trend_score, 1.0)
    
    # 5. Momentum Quality (RSI + MACD + strength convergence)
    momentum_score = 0.0
    rsi = float(getattr(row, "rsi", 50.0))
    macd_hist = float(getattr(row, "macd_hist", 0.0))
    momentum_shift = float(getattr(row, "momentum_shift", 0.0))
    
    # RSI in optimal range (not overbought/oversold)
    if trade_side == "buy":
        if 40 <= rsi <= 60:  # RSI bullish but not overbought
            momentum_score += 0.4
        elif rsi > 50:  # At least above 50
            momentum_score += 0.2
    elif trade_side == "sell":
        if 40 <= rsi <= 60:  # RSI bearish but not oversold
            momentum_score += 0.4
        elif rsi < 50:  # At least below 50
            momentum_score += 0.2
    
    # MACD histogram confirms direction
    if trade_side == "buy" and macd_hist > 0:
        momentum_score += 0.3
    elif trade_side == "sell" and macd_hist < 0:
        momentum_score += 0.3
    
    # Momentum shift confirms
    if abs(momentum_shift) > 0.3:
        momentum_score += 0.3
    
    confirmations["momentum"] = min(momentum_score, 1.0)
    
    # Total score: average of 5 confirmations
    total_score = sum(confirmations.values()) / 5.0
    
    return total_score, confirmations


def should_pass_entry_quality_filter(
    row: pd.Series | object,
    min_score: float = 0.60,
    min_confirmations: int = 3,
) -> tuple[bool, str, float, dict]:
    """
    Check if entry passes quality filter.
    
    Args:
        row: Trade signal row with features
        min_score: Minimum total score (0.0-1.0, default 0.60 = 60%)
        min_confirmations: Minimum number of confirmations with score >= 0.6 (default 3/5)
    
    Returns:
        (pass, reason, total_score, confirmations):
        - pass: True if entry quality sufficient
        - reason: Reason for pass/fail
        - total_score: Overall quality score 0.0-1.0
        - confirmations: Dict with individual confirmation scores
    """
    total_score, confirmations = calculate_entry_quality_score(row)
    
    # Count how many confirmations are strong (>= 0.6)
    strong_confirmations = sum(1 for score in confirmations.values() if score >= 0.6)
    
    # Pass if either:
    # 1. Total score >= min_score (overall quality good)
    # 2. At least min_confirmations strong confirmations
    if total_score >= min_score:
        return True, f"quality_score_pass ({total_score:.2f})", total_score, confirmations
    
    if strong_confirmations >= min_confirmations:
        return True, f"confirmations_pass ({strong_confirmations}/{len(confirmations)})", total_score, confirmations
    
    # Fail
    reason = f"quality_too_low (score={total_score:.2f}, strong={strong_confirmations}/{min_confirmations})"
    return False, reason, total_score, confirmations


def get_entry_quality_config(settings) -> dict:
    """
    Get entry quality filter configuration from settings.
    
    Returns:
        Dict with configuration:
        - enabled: Whether entry quality filter is enabled
        - min_score: Minimum total score (0.0-1.0)
        - min_confirmations: Minimum strong confirmations (1-5)
    """
    return {
        "enabled": bool(getattr(settings.risk, "entry_quality_filter_enabled", False)),
        "min_score": float(getattr(settings.risk, "entry_quality_min_score", 0.60)),
        "min_confirmations": int(getattr(settings.risk, "entry_quality_min_confirmations", 3)),
    }
