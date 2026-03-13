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

    def calculate_max_concurrent_positions(self, balance: float | None = None) -> int:
        """
        Dynamically calculate the maximum number of concurrent open positions.

        Logic:
          max_by_budget = floor(max_portfolio_risk_fraction / risk_per_trade)
          e.g. with max_portfolio=3%, risk_per_trade=0.75% → max 4 concurrent trades.
          Apply hard cap max_concurrent_positions_cap as absolute safety valve.
          Balance is accepted but not used in the current fraction-based formula;
          it can be used in future lot-size aware variants.
        """
        _ = balance  # reserved for lot-size-aware variant
        if self.settings.risk.risk_per_trade <= 0:
            return 1
        max_by_budget = int(
            self.settings.risk.max_portfolio_risk_fraction / self.settings.risk.risk_per_trade
        )
        return max(1, min(max_by_budget, self.settings.risk.max_concurrent_positions_cap))

    def can_open_position(
        self,
        balance: float,
        open_positions_count: int,
        total_deployed_risk_fraction: float,
        confidence: float,
        volatility_regime: int | None = None,
        strategy_score: float | None = None,
    ) -> tuple[bool, float]:
        """
        Decide dynamically whether to open a new position.

        Returns (can_open, risk_fraction_for_this_trade).

        Rules:
          1. Number of currently open positions must be below the calculated max.
          2. Adding this trade's risk fraction must not breach max_portfolio_risk_fraction.
          3. Balance is tracked so that max calculation can be made balance-aware
             in the future (e.g. absolute dollar risk caps).
        """
        max_concurrent = self.calculate_max_concurrent_positions(balance)
        if open_positions_count >= max_concurrent:
            return False, 0.0

        risk_frac = self.risk_fraction(confidence, volatility_regime, strategy_score)

        if total_deployed_risk_fraction + risk_frac > self.settings.risk.max_portfolio_risk_fraction:
            return False, 0.0

        return True, risk_frac

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
