from __future__ import annotations

from typing import Literal

TradeRating = Literal["Buy", "Overweight", "Hold", "Underweight", "Sell"]


def rating_from_decision(side: str, confidence: float, should_trade: bool = True) -> TradeRating:
    """Map a directional trade decision to a 5-tier human-readable rating."""
    if not should_trade:
        return "Hold"

    side_norm = str(side).strip().lower()
    conf = max(0.0, min(float(confidence), 1.0))
    if conf < 0.55:
        return "Hold"

    if side_norm == "sell":
        return "Sell" if conf >= 0.70 else "Underweight"
    return "Buy" if conf >= 0.70 else "Overweight"


def rating_risk_multiplier(rating: str) -> float:
    """Return optional lot/risk scaling for the 5-tier rating."""
    if rating in {"Buy", "Sell"}:
        return 1.0
    if rating in {"Overweight", "Underweight"}:
        return 0.60
    return 0.0
