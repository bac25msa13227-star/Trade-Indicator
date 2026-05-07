"""
Adaptive Trailing Stop Loss

Dynamically adjusts trailing stop distance based on Risk:Reward ratio achieved.
Tighter trails at higher RR to lock in profits, looser trails at lower RR to let winners run.

Target Impact: RR 3.67 → 5.0 (+36% per trade)
"""
from __future__ import annotations


def calculate_adaptive_trail_distance(
    current_rr: float,
    atr: float,
    sl_distance: float,
) -> float:
    """
    Calculate adaptive trailing stop distance based on current RR achieved.
    
    Args:
        current_rr: Current risk-reward ratio (e.g., 2.5 means price is 2.5R favorable)
        atr: Average True Range
        sl_distance: Original stop loss distance (atr * sl_mult)
    
    Returns:
        Trail distance to use (distance behind best price to place trailing SL)
    
    Logic:
        - RR < 1.0:  No trail yet (let activation handle)
        - RR 1.0-2.0: Trail at 0.7×ATR (loose, let it run)
        - RR 2.0-3.5: Trail at 0.4×ATR (medium, protect gains)
        - RR 3.5+:    Trail at 0.25×ATR (tight, lock in big win)
    """
    if current_rr < 1.0:
        # Too early, use sl_distance (will be capped by activation logic)
        return sl_distance
    
    elif current_rr < 2.0:
        # Early profit phase (1-2R): loose trail, let it run
        return 0.7 * atr
    
    elif current_rr < 3.5:
        # Mid profit phase (2-3.5R): medium trail, protect gains
        return 0.4 * atr
    
    else:
        # High profit phase (3.5R+): tight trail, lock in big win
        return 0.25 * atr


def calculate_current_rr(
    entry_price: float,
    current_price: float,
    sl_distance: float,
    direction: int,
) -> float:
    """
    Calculate current Risk:Reward ratio from entry.
    
    Args:
        entry_price: Entry price
        current_price: Current price (best favorable price seen)
        sl_distance: Original stop loss distance
        direction: 1 for long, -1 for short
    
    Returns:
        Current RR ratio (positive if favorable, negative if adverse)
    """
    if sl_distance <= 0:
        return 0.0
    
    price_move = (current_price - entry_price) * direction
    rr = price_move / sl_distance
    return rr


def should_use_adaptive_trailing(
    settings,
) -> bool:
    """
    Check if adaptive trailing SL is enabled in config.
    
    Args:
        settings: Settings object with execution config
    
    Returns:
        True if adaptive trailing should be used
    """
    # Check if explicitly enabled in execution settings
    adaptive_trail = getattr(settings.execution, "adaptive_trailing_sl", False)
    
    # Also check if traditional trailing is enabled (requirement)
    trailing_cfg = getattr(settings.execution, "trailing_sl", None)
    if trailing_cfg:
        trail_enabled = bool(getattr(trailing_cfg, "enabled", False))
        if trail_enabled and adaptive_trail:
            return True
    
    return False


def get_adaptive_trail_config(settings) -> dict:
    """
    Get adaptive trailing stop configuration from settings.
    
    Returns:
        Dict with configuration parameters:
        - enabled: Whether adaptive trailing is enabled
        - breakeven_at_rr: RR level to move to breakeven (e.g., 0.5)
        - activation_rr: RR level to start trailing (e.g., 1.5)
        - trail_multipliers: Dict mapping RR thresholds to ATR multipliers
    """
    trailing_cfg = getattr(settings.execution, "trailing_sl", None)
    if not trailing_cfg:
        return {
            "enabled": False,
            "breakeven_at_rr": 0.5,
            "activation_rr": 1.5,
        }
    
    return {
        "enabled": should_use_adaptive_trailing(settings),
        "breakeven_at_rr": float(getattr(trailing_cfg, "breakeven_at_rr", 0.5)),
        "activation_rr": float(getattr(trailing_cfg, "activation_rr", 1.5)),
        # Adaptive multipliers (not used directly, calculated by calculate_adaptive_trail_distance)
        "trail_multipliers": {
            1.5: 0.5,  # RR 1.5-2.5: 0.5×ATR trail
            2.5: 0.3,  # RR 2.5-4.0: 0.3×ATR trail  
            4.0: 0.2,  # RR 4.0+: 0.2×ATR trail
        }
    }
