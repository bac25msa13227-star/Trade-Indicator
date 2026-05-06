"""
Dynamic Risk Management

Scale risk per trade based on signal confidence to maximize profit on high-quality setups.

Target Impact: +15% profit = $1,657 × 1.15 = $1,906/fold
"""
from __future__ import annotations


def calculate_dynamic_risk(
    base_risk: float,
    confidence: float,
    min_risk: float = 0.02,
    max_risk: float = 0.05,
) -> tuple[float, str]:
    """
    Calculate dynamic risk based on signal confidence.
    
    Args:
        base_risk: Base risk per trade (e.g., 0.03 = 3%)
        confidence: Model confidence (0.0-1.0)
        min_risk: Minimum risk threshold (default 2%)
        max_risk: Maximum risk threshold (default 5%)
    
    Returns:
        (risk_fraction, reason):
        - risk_fraction: Dynamic risk to use (between min_risk and max_risk)
        - reason: Explanation of risk adjustment
    
    Logic:
        - Confidence >= 0.85: max_risk (5%) - High confidence, go big
        - Confidence 0.75-0.85: base_risk × 1.33 (4%) - Medium-high
        - Confidence 0.70-0.75: base_risk (3%) - Normal
        - Confidence < 0.70: min_risk (2%) - Low confidence, be cautious
    """
    if confidence >= 0.85:
        # High confidence: max risk
        risk = max_risk
        reason = f"high_conf ({confidence:.2f} >= 0.85)"
    
    elif confidence >= 0.75:
        # Medium-high confidence: scale up 33%
        risk = min(base_risk * 1.33, max_risk)
        reason = f"med_high_conf ({confidence:.2f} >= 0.75)"
    
    elif confidence >= 0.70:
        # Normal confidence: use base risk
        risk = base_risk
        reason = f"normal_conf ({confidence:.2f} >= 0.70)"
    
    else:
        # Low confidence: reduce to min risk
        risk = min_risk
        reason = f"low_conf ({confidence:.2f} < 0.70)"
    
    # Ensure within bounds
    risk = max(min_risk, min(risk, max_risk))
    
    return risk, reason


def get_dynamic_risk_config(settings) -> dict:
    """
    Get dynamic risk configuration from settings.
    
    Returns:
        Dict with configuration:
        - enabled: Whether dynamic risk is enabled
        - min_risk: Minimum risk per trade (0.02 = 2%)
        - max_risk: Maximum risk per trade (0.05 = 5%)
        - base_risk: Base risk per trade (0.03 = 3%)
    """
    return {
        "enabled": bool(getattr(settings.risk, "dynamic_risk_enabled", False)),
        "min_risk": float(getattr(settings.risk, "dynamic_risk_min", 0.02)),
        "max_risk": float(getattr(settings.risk, "dynamic_risk_max", 0.05)),
        "base_risk": float(getattr(settings.risk, "risk_per_trade", 0.03)),
    }
