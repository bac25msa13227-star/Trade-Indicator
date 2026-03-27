from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from xauusd_ai.config import Settings

LOGGER = logging.getLogger(__name__)


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

    @staticmethod
    def _as_float(value: object) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

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

    def _regime_shutdown(self, row: pd.Series | object, probability: float, side: str | None = None) -> tuple[bool, str]:
        rules = getattr(self.settings.strategy, "regime_shutdown_rules", [])
        if not rules:
            return False, "ok"

        timestamp = getattr(row, "time", None)
        resolved = timestamp
        if timestamp is not None and not isinstance(timestamp, (pd.Timestamp, datetime)):
            resolved = pd.to_datetime(timestamp, utc=True, errors="coerce")

        weekday_name = resolved.day_name().lower() if isinstance(resolved, (pd.Timestamp, datetime)) and not pd.isna(resolved) else ""
        hour = int(resolved.hour) if isinstance(resolved, (pd.Timestamp, datetime)) and not pd.isna(resolved) else None
        row_side = (side or str(getattr(row, "trade_side", ""))).strip().lower()

        for rule in rules:
            if not getattr(rule, "enabled", True):
                continue
            if rule.weekdays_utc and weekday_name not in {value.strip().lower() for value in rule.weekdays_utc}:
                continue
            if rule.hours_utc and hour not in {int(value) for value in rule.hours_utc}:
                continue
            if rule.sides and row_side not in {value.strip().lower() for value in rule.sides}:
                continue

            volatility_regime = int(getattr(row, "volatility_regime", 1))
            if rule.volatility_regimes and volatility_regime not in {int(value) for value in rule.volatility_regimes}:
                continue

            trend_alignment = int(getattr(row, "trend_alignment", 1))
            if rule.trend_alignment_values and trend_alignment not in {int(value) for value in rule.trend_alignment_values}:
                continue

            if rule.probability_min is not None and probability < float(rule.probability_min):
                continue
            if rule.probability_max is not None and probability > float(rule.probability_max):
                continue

            adx_value = self._as_float(getattr(row, "adx", None))
            if rule.adx_min is not None and (adx_value is None or adx_value < float(rule.adx_min)):
                continue
            if rule.adx_max is not None and (adx_value is None or adx_value > float(rule.adx_max)):
                continue

            trend_strength = self._as_float(getattr(row, "trend_strength_score", None))
            if rule.trend_strength_min is not None and (trend_strength is None or trend_strength < float(rule.trend_strength_min)):
                continue
            if rule.trend_strength_max is not None and (trend_strength is None or trend_strength > float(rule.trend_strength_max)):
                continue

            pullback_quality = self._as_float(getattr(row, "pullback_quality", None))
            if rule.pullback_quality_min is not None and (pullback_quality is None or pullback_quality < float(rule.pullback_quality_min)):
                continue
            if rule.pullback_quality_max is not None and (pullback_quality is None or pullback_quality > float(rule.pullback_quality_max)):
                continue

            execution_quality = self._as_float(getattr(row, "execution_quality", None))
            if rule.execution_quality_min is not None and (execution_quality is None or execution_quality < float(rule.execution_quality_min)):
                continue
            if rule.execution_quality_max is not None and (execution_quality is None or execution_quality > float(rule.execution_quality_max)):
                continue

            strategy_score = self._as_float(getattr(row, "strategy_score", None))
            strategy_abs = abs(strategy_score) if strategy_score is not None else None
            if rule.strategy_score_min is not None and (strategy_abs is None or strategy_abs < float(rule.strategy_score_min)):
                continue
            if rule.strategy_score_max is not None and (strategy_abs is None or strategy_abs > float(rule.strategy_score_max)):
                continue

            name = rule.name.strip() or "regime_shutdown"
            return True, name

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

        # Regime-specific confidence thresholds
        if volatility_regime == 0:
            min_conf = self.settings.strategy.sideway_min_confidence
        elif volatility_regime == 2:
            min_conf = self.settings.strategy.volatile_min_confidence
        else:
            min_conf = self.settings.risk.min_confidence

        # Silver Bullet boost: increase effective probability during high-probability windows
        effective_prob = probability
        if self.settings.strategy.silver_bullet_enabled:
            timestamp = getattr(row, "time", None)
            if timestamp is not None:
                resolved = timestamp
                if not isinstance(resolved, (pd.Timestamp, datetime)):
                    resolved = pd.to_datetime(resolved, utc=True, errors="coerce")
                if not pd.isna(resolved):
                    hour = int(resolved.hour)
                    for window in self.settings.strategy.silver_bullet_windows_utc:
                        if len(window) == 2 and window[0] <= hour <= window[1]:
                            effective_prob += self.settings.strategy.silver_bullet_confidence_boost
                            break

        shutdown, shutdown_reason = self._regime_shutdown(row, effective_prob)
        if shutdown:
            return False, f"regime_shutdown ({shutdown_reason})"

        if effective_prob < min_conf:
            return False, f"confidence_below_floor ({effective_prob:.3f} < {min_conf:.3f})"
        if self.settings.strategy.require_trend_alignment and trend_alignment != 1:
            return False, "trend_misaligned"
        if strategy_score < self.required_strategy_score(volatility_regime):
            return False, "strategy_score_too_weak"

        # ADX gate: only trade when there's sufficient trend strength
        if self.settings.strategy.adx_gate_enabled:
            adx_val = float(getattr(row, "adx", 0.0))
            if adx_val < self.settings.strategy.adx_min_trend:
                return False, f"adx_too_low ({adx_val:.1f} < {self.settings.strategy.adx_min_trend})"

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

    def _regime_sl_rr(self, volatility_regime: int) -> tuple[float, float]:
        """
        Trả về (sl_atr_multiple, take_profit_rr) phù hợp với market regime.

        Sideways  (0): SL chặt hơn, TP gần hơn — thị trường ít room
        Normal    (1): giá trị mặc định từ config
        Volatile  (2): SL rộng hơn, TP xa hơn — cho giá chạy đủ xa
        """
        r = self.settings.risk
        if volatility_regime == 0:
            return r.sideway_sl_atr_multiple, r.sideway_take_profit_rr
        elif volatility_regime == 2:
            return r.volatile_sl_atr_multiple, r.volatile_take_profit_rr
        return r.stop_loss_atr_multiple, r.take_profit_rr

    def build_trade_decision(self, frames: dict[str, pd.DataFrame], live_row: pd.Series, model_signal: dict[str, float]) -> TradeDecision:
        confidence = float(model_signal["probability"])
        signal_on = bool(model_signal["prediction"] == 1 and confidence >= self.settings.risk.min_confidence)
        strategy_score = float(live_row["strategy_score"])
        volatility_regime = int(live_row.get("volatility_regime", 1))

        # ── News blocking: kiểm tra news_is_blackout thực tế ──────────────────
        # news_is_blackout=1 → đang trong cửa sổ ±1h của High-impact news
        news_is_blackout = int(live_row.get("news_is_blackout", 0))
        news_impact_ahead = int(live_row.get("news_impact_ahead", 0))
        # Block nếu đang trong blackout HOẶC news high-impact sắp xảy ra (trong ~15 phút)
        news_minutes_ahead = float(live_row.get("news_hours_ahead", 48.0)) * 60.0
        block_news_minutes = float(self.settings.strategy.news_block_minutes)
        no_news_block = not (
            news_is_blackout == 1
            or (news_impact_ahead == 2 and block_news_minutes > 0 and news_minutes_ahead <= block_news_minutes)
        )

        # ── Tính ATR-based SL/TP với multiplier theo regime ───────────────────
        sl_mult, rr = self._regime_sl_rr(volatility_regime)
        atr_value = float(live_row.get("atr", 0))
        entry = float(live_row["close"])
        stop_distance = atr_value * sl_mult if atr_value > 0 else 0.0

        hyp_side = "buy" if strategy_score >= 0 else "sell"
        if hyp_side == "buy":
            hyp_sl = entry - stop_distance if stop_distance > 0 else 0.0
            hyp_tp = entry + stop_distance * rr if stop_distance > 0 else 0.0
        else:
            hyp_sl = entry + stop_distance if stop_distance > 0 else 0.0
            hyp_tp = entry - stop_distance * rr if stop_distance > 0 else 0.0

        # force_trade: bỏ qua bộ lọc strategy, nhưng vẫn giữ news blackout
        if self.settings.strategy.force_trade:
            LOGGER.warning("FORCE TRADE active — bypassing strategy filters (news blackout still enforced)")
            if not no_news_block:
                return TradeDecision(False, hyp_side, confidence, "FORCE: blocked by news blackout", entry, hyp_sl, hyp_tp)
            return TradeDecision(True, hyp_side, confidence, "FORCE TRADE (news-safe)", entry, hyp_sl, hyp_tp)

        if not signal_on:
            return TradeDecision(False, hyp_side, confidence, "Model confidence below threshold", entry, hyp_sl, hyp_tp)
        shutdown, shutdown_reason = self._regime_shutdown(live_row, confidence, hyp_side)
        if shutdown:
            return TradeDecision(False, hyp_side, confidence, f"Regime shutdown: {shutdown_reason}", entry, hyp_sl, hyp_tp)
        if abs(strategy_score) < self.required_strategy_score(int(live_row["volatility_regime"])):
            return TradeDecision(False, hyp_side, confidence, "Strategy consensus is weak", entry, hyp_sl, hyp_tp)
        if self.settings.strategy.require_trend_alignment and int(live_row["trend_alignment"]) != 1:
            return TradeDecision(False, hyp_side, confidence, "Higher timeframe trend is misaligned", entry, hyp_sl, hyp_tp)
        if not no_news_block:
            return TradeDecision(False, hyp_side, confidence, "News filter blocked trade", entry, hyp_sl, hyp_tp)

        stop_loss = hyp_sl
        take_profit = hyp_tp
        regime_label = {0: "SIDEWAYS", 1: "NORMAL", 2: "VOLATILE"}.get(volatility_regime, "NORMAL")
        reason = f"Hybrid aligned | {regime_label} sl×{sl_mult:.1f} RR{rr:.1f}"
        return TradeDecision(True, hyp_side, confidence, reason, entry, stop_loss, take_profit)
