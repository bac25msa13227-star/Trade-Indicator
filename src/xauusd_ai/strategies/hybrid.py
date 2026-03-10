from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from xauusd_ai.config import Settings


@dataclass
class TradeDecision:
    should_trade: bool
    side: str
    confidence: float
    reason: str
    entry_price: float
    stop_loss: float
    take_profit: float


class HybridStrategy:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _blocked_by_time(self, timestamp: object) -> tuple[bool, str]:
        if timestamp is None:
            return False, "ok"
        resolved = timestamp
        if not isinstance(resolved, (pd.Timestamp, datetime)):
            resolved = pd.to_datetime(resolved, utc=True, errors="coerce")
        if pd.isna(resolved):
            return False, "ok"

        blocked_hours = set(self.settings.strategy.blocked_hours_utc)
        blocked_weekdays = {value.strip().lower() for value in self.settings.strategy.blocked_weekdays_utc}
        blocked_weekday_hours = {
            weekday.strip().lower(): {int(hour) for hour in hours}
            for weekday, hours in self.settings.strategy.blocked_weekday_hours_utc.items()
        }
        allowed_weekday_hours = {
            weekday.strip().lower(): {int(hour) for hour in hours}
            for weekday, hours in self.settings.strategy.allowed_weekday_hours_utc.items()
        }
        weekday_name = resolved.day_name().lower()
        if allowed_weekday_hours:
            if int(resolved.hour) not in allowed_weekday_hours.get(weekday_name, set()):
                return True, "not_in_allowed_weekday_hour"
        if int(resolved.hour) in blocked_hours:
            return True, "blocked_hour"
        if weekday_name in blocked_weekdays:
            return True, "blocked_weekday"
        if int(resolved.hour) in blocked_weekday_hours.get(weekday_name, set()):
            return True, "blocked_weekday_hour"
        return False, "ok"

    def required_strategy_score(self, volatility_regime: int | float) -> float:
        if int(volatility_regime) == 0:
            return self.settings.strategy.sideway_min_strategy_score
        if int(volatility_regime) == 2:
            return self.settings.strategy.strong_volatility_min_strategy_score
        return self.settings.strategy.min_strategy_score

    def should_allow_row(self, row: pd.Series | object, probability: float) -> tuple[bool, str]:
        strategy_score = abs(float(getattr(row, "strategy_score", 0.0)))
        volatility_regime = int(getattr(row, "volatility_regime", 1))
        trend_alignment = int(getattr(row, "trend_alignment", 1))
        blocked_by_time, blocked_reason = self._blocked_by_time(getattr(row, "time", None))
        if blocked_by_time:
            return False, blocked_reason
        if probability < self.settings.risk.min_confidence:
            return False, "confidence_below_floor"
        if self.settings.strategy.require_trend_alignment and trend_alignment != 1:
            return False, "trend_misaligned"
        if strategy_score < self.required_strategy_score(volatility_regime):
            return False, "strategy_score_too_weak"
        return True, "ok"

    def annotate_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        ict_bias = np.where(
            (frame["liquidity_sweep"] == 1) & (frame["trend_alignment"] == 1),
            np.sign(frame["hourly_bias"]),
            0,
        )
        wyckoff_bias = np.where(frame["wyckoff_phase"] == 1, 1, np.where(frame["wyckoff_phase"] == -1, -1, 0))
        momentum_bias = np.where(
            (frame["rsi"] > self.settings.strategy.rsi_long_threshold) & (frame["macd_hist"] > 0),
            1,
            np.where((frame["rsi"] < self.settings.strategy.rsi_short_threshold) & (frame["macd_hist"] < 0), -1, 0),
        )
        regime_bias = np.where(frame["volatility_regime"] == 0, 0.5, np.where(frame["volatility_regime"] == 2, 1.2, 1.0))
        total_weight = max(
            self.settings.strategy.ict_weight + self.settings.strategy.wyckoff_weight + self.settings.strategy.momentum_weight,
            1e-6,
        )

        result["ict_score"] = ict_bias
        result["wyckoff_score"] = wyckoff_bias
        result["momentum_score"] = momentum_bias
        result["strategy_score"] = (
            (
                ict_bias * self.settings.strategy.ict_weight
                + wyckoff_bias * self.settings.strategy.wyckoff_weight
                + momentum_bias * self.settings.strategy.momentum_weight
            )
            / total_weight
        ) * regime_bias
        return result

    def build_trade_decision(self, frames: dict[str, pd.DataFrame], live_row: pd.Series, model_signal: dict[str, float]) -> TradeDecision:
        confidence = float(model_signal["probability"])
        signal_on = bool(model_signal["prediction"] == 1 and confidence >= self.settings.risk.min_confidence)
        strategy_score = float(live_row["strategy_score"])
        no_news_block = True

        if not signal_on:
            return TradeDecision(False, "flat", confidence, "Model confidence below threshold", float(live_row["close"]), 0.0, 0.0)
        if abs(strategy_score) < self.required_strategy_score(int(live_row["volatility_regime"])):
            return TradeDecision(False, "flat", confidence, "Strategy consensus is weak", float(live_row["close"]), 0.0, 0.0)
        if self.settings.strategy.require_trend_alignment and int(live_row["trend_alignment"]) != 1:
            return TradeDecision(False, "flat", confidence, "Higher timeframe trend is misaligned", float(live_row["close"]), 0.0, 0.0)
        if not no_news_block:
            return TradeDecision(False, "flat", confidence, "News filter blocked trade", float(live_row["close"]), 0.0, 0.0)

        side = "buy" if strategy_score > 0 else "sell"
        atr_value = float(live_row["atr"])
        entry = float(live_row["close"])
        stop_distance = atr_value * self.settings.risk.stop_loss_atr_multiple
        if side == "buy":
            stop_loss = entry - stop_distance
            take_profit = entry + stop_distance * self.settings.risk.take_profit_rr
        else:
            stop_loss = entry + stop_distance
            take_profit = entry - stop_distance * self.settings.risk.take_profit_rr

        return TradeDecision(True, side, confidence, "Hybrid strategy aligned", entry, stop_loss, take_profit)
