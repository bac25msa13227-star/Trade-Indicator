from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from xauusd_ai.config import Settings
from xauusd_ai.strategies.hybrid import TradeDecision


@dataclass
class OrderPlan:
    symbol: str
    side: str
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reason: str


class RiskManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def risk_fraction(
        self,
        confidence: float,
        volatility_regime: int | None = None,
        strategy_score: float | None = None,
    ) -> float:
        capped_confidence = min(max(confidence, self.settings.risk.min_confidence), 0.95)
        base_fraction = self.settings.risk.risk_per_trade * (capped_confidence / self.settings.risk.min_confidence)
        regime_multiplier = self.settings.risk.normal_risk_multiplier
        if volatility_regime == 0:
            regime_multiplier = self.settings.risk.sideway_risk_multiplier
        elif volatility_regime == 2:
            regime_multiplier = self.settings.risk.strong_volatility_risk_multiplier

        score_multiplier = 1.0
        if strategy_score is not None:
            score_multiplier = min(max(abs(strategy_score), 0.5), 1.0)

        return min(base_fraction * regime_multiplier * score_multiplier, self.settings.risk.max_risk_fraction)

    def build_order_plan(self, decision: TradeDecision, latest_bar: pd.Series) -> OrderPlan:
        volume = self.settings.risk.fixed_lot
        return OrderPlan(
            symbol=self.settings.market.symbol,
            side=decision.side,
            volume=volume,
            entry_price=decision.entry_price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            confidence=decision.confidence,
            reason=decision.reason,
        )
