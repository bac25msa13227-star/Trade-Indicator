from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from xauusd_ai.config import Settings
from xauusd_ai.strategies.hybrid import TradeDecision

LOGGER = logging.getLogger(__name__)

# XAUUSD contract spec: 1 lot = 100 oz.
# PnL per lot for $1 price move = $100. For 0.01 lot = $1.
_XAUUSD_CONTRACT_OZ = 100.0
_MIN_LOT = 0.01
_LOT_STEP = 0.01


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
        # Use account-specific state files to avoid acc1/acc2 sharing the same file
        _acct = os.environ.get("TRADING_ACCOUNT", "")
        _suffix = f"_{_acct}" if _acct else ""
        self._PEAK_FILE = Path(f"outputs/risk_peak_balance{_suffix}.json")
        self._DAILY_STATE_FILE = Path(f"outputs/risk_daily_state{_suffix}.json")
        self._peak_balance: float = self._load_peak_balance()
        # Circuit breaker state
        self._consecutive_losses: int = 0
        self._cooldown_bars_remaining: int = 0
        self._daily_loss: float = 0.0
        self._daily_date: str = ""
        self._killed: bool = False  # max drawdown kill switch
        self._load_daily_state()

    @staticmethod
    def _as_float(value: object) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _row_value(row: object, field: str, default: object = None) -> object:
        if isinstance(row, dict):
            return row.get(field, default)
        if isinstance(row, pd.Series):
            return row.get(field, default)
        return getattr(row, field, default)

    # ------------------------------------------------------------------
    # Peak balance persistence
    # ------------------------------------------------------------------
    def _load_peak_balance(self) -> float:
        try:
            if self._PEAK_FILE.exists():
                data = json.loads(self._PEAK_FILE.read_text(encoding="utf-8"))
                val = float(data.get("peak_balance", 0.0))
                if val > 0:
                    LOGGER.info("RiskManager: loaded peak_balance=%.2f", val)
                return val
        except Exception:
            pass
        return 0.0

    def _save_peak_balance(self) -> None:
        try:
            self._PEAK_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._PEAK_FILE.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"peak_balance": round(self._peak_balance, 2)}),
                encoding="utf-8",
            )
            tmp.replace(self._PEAK_FILE)
        except Exception as exc:
            LOGGER.warning("RiskManager: failed to save peak_balance: %s", exc)

    # ------------------------------------------------------------------
    # Daily state persistence (circuit breaker)
    # ------------------------------------------------------------------
    def _load_daily_state(self) -> None:
        try:
            if self._DAILY_STATE_FILE.exists():
                data = json.loads(self._DAILY_STATE_FILE.read_text(encoding="utf-8"))
                self._daily_date = str(data.get("date", ""))
                self._daily_loss = float(data.get("daily_loss", 0.0))
                self._consecutive_losses = int(data.get("consecutive_losses", 0))
                self._cooldown_bars_remaining = int(data.get("cooldown_bars", 0))
                self._killed = bool(data.get("killed", False))
        except Exception:
            pass

    def _save_daily_state(self) -> None:
        try:
            self._DAILY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._DAILY_STATE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "date": self._daily_date,
                "daily_loss": round(self._daily_loss, 4),
                "consecutive_losses": self._consecutive_losses,
                "cooldown_bars": self._cooldown_bars_remaining,
                "killed": self._killed,
            }), encoding="utf-8")
            tmp.replace(self._DAILY_STATE_FILE)
        except Exception as exc:
            LOGGER.warning("RiskManager: failed to save daily state: %s", exc)

    def record_trade_result(self, pnl: float, balance: float) -> None:
        """Call after each trade closes to update circuit breaker state."""
        import datetime as _dt
        today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_date = today
            self._daily_loss = 0.0

        if pnl < 0:
            self._daily_loss += abs(pnl)
            self._consecutive_losses += 1
            # Check consecutive loss cooldown
            pause_count = self.settings.risk.consecutive_loss_pause_count
            if pause_count > 0 and self._consecutive_losses >= pause_count:
                self._cooldown_bars_remaining = self.settings.risk.consecutive_loss_cooldown_bars
                LOGGER.warning(
                    "CircuitBreaker: %d consecutive losses → cooldown %d bars",
                    self._consecutive_losses, self._cooldown_bars_remaining,
                )
        else:
            self._consecutive_losses = 0  # Reset on win

        # Check max drawdown kill switch
        if self._peak_balance > 0:
            dd = (self._peak_balance - balance) / self._peak_balance
            if dd >= self.settings.risk.max_drawdown_kill_pct:
                self._killed = True
                LOGGER.critical(
                    "KILL SWITCH: drawdown %.1f%% exceeds %.1f%% limit — TRADING HALTED",
                    dd * 100, self.settings.risk.max_drawdown_kill_pct * 100,
                )

        self._save_daily_state()

    def tick_cooldown(self) -> None:
        """Call once per bar to decrement cooldown timer."""
        if self._cooldown_bars_remaining > 0:
            self._cooldown_bars_remaining -= 1
            self._save_daily_state()

    def is_circuit_breaker_active(self, balance: float) -> tuple[bool, str]:
        """Check if any circuit breaker is active. Returns (blocked, reason)."""
        # Master bypass — all circuit breakers disabled when kill_switch_enabled=False
        if not self.settings.risk.kill_switch_enabled:
            return False, "ok"

        # Kill switch
        if self._killed:
            return True, "KILL_SWITCH: max drawdown exceeded — manual restart required"

        # Reset daily loss at UTC midnight (in case no trade closed at day boundary)
        import datetime as _dt
        today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_date = today
            self._daily_loss = 0.0
            self._save_daily_state()

        # Daily loss limit
        limit_pct = self.settings.risk.daily_loss_limit_pct
        if limit_pct > 0 and self._peak_balance > 0:
            limit_amount = self._peak_balance * limit_pct
            if self._daily_loss >= limit_amount:
                return True, f"DAILY_LOSS_LIMIT: lost ${self._daily_loss:.2f} today (limit ${limit_amount:.2f})"

        # Consecutive loss cooldown
        if self._cooldown_bars_remaining > 0:
            return True, f"COOLDOWN: {self._cooldown_bars_remaining} bars remaining after {self._consecutive_losses} consecutive losses"

        return False, "ok"

    def get_anti_martingale_factor(self) -> float:
        """Returns risk multiplier based on consecutive losses (anti-martingale)."""
        factor = self.settings.risk.anti_martingale_factor
        max_reductions = self.settings.risk.anti_martingale_max_reductions
        if factor >= 1.0 or self._consecutive_losses == 0:
            return 1.0
        n = min(self._consecutive_losses, max_reductions)
        return factor ** n

    def check_total_exposure(
        self, balance: float, current_risk_amount: float, existing_risk_total: float
    ) -> tuple[bool, str]:
        """Check if adding a new trade would exceed total exposure cap."""
        cap = self.settings.risk.max_total_exposure_pct
        if cap <= 0:
            return True, "ok"
        max_risk = balance * cap
        if existing_risk_total + current_risk_amount > max_risk:
            return False, (
                f"EXPOSURE_CAP: total risk ${existing_risk_total + current_risk_amount:.2f} "
                f"would exceed {cap*100:.0f}% cap (${max_risk:.2f})"
            )
        return True, "ok"

    # ------------------------------------------------------------------
    # Dynamic position limit — phụ thuộc balance + market regime
    # ------------------------------------------------------------------
    def get_dynamic_max_positions(self, balance: float, volatility_regime: int = 1) -> int:
        """
        Tính số lệnh tối đa được phép mở dựa trên số dư tài khoản.

        Tier map (theo balance USD):
            < $200   → 2 lệnh   (tài khoản micro, giảm exposure)
            < $500   → 3 lệnh   (vốn nhỏ, max 18% exposure)
            < $2 000 → 5 lệnh
            < $10 000→ 8 lệnh
            < $50 000→ 10 lệnh
            >= $50 000→ 15 lệnh

        Market‑regime điều chỉnh thêm:
            sideway  (0) → giảm 50%   (ít cơ hội, ít rủi ro)
            volatile (2) → giảm 30%   (biến động mạnh, thu nhỏ exposure)
        """
        config_ceiling = self.settings.risk.max_open_positions

        if balance < 200:
            tier_max = 2          # < $200: tối đa 2 lệnh, giảm rủi ro vốn micro
        elif balance < 500:
            tier_max = 3          # $200-$499: 3 lệnh, max 18% exposure
        elif balance < 2_000:
            tier_max = 5
        elif balance < 10_000:
            tier_max = 8
        elif balance < 50_000:
            tier_max = 10
        else:
            tier_max = 15

        base_max = min(config_ceiling, tier_max)

        # Chỉ giảm theo regime khi base_max > 3.
        # Với tài khoản nhỏ (base_max <= 3), giảm thêm không hợp lý:
        #   base_max=3 → sideways 0.67× → 2, mất đi vị trí đã config.
        if base_max > 3:
            if volatility_regime == 0:      # sideways
                adjusted = max(1, round(base_max * 0.67))
            elif volatility_regime == 2:    # strong volatility
                adjusted = max(1, round(base_max * 0.7))
            else:
                adjusted = base_max
        else:
            adjusted = base_max

        return adjusted

    # ------------------------------------------------------------------
    # Dynamic lot size — phụ thuộc balance + stop distance
    # ------------------------------------------------------------------
    def calculate_dynamic_lot(
        self,
        balance: float,
        stop_distance: float,
        risk_fraction: float | None = None,
    ) -> float:
        """
        Tính lot size để risk đúng % balance mỗi lệnh.

        Công thức:
            risk_amount   = balance × risk_fraction
            lot           = risk_amount / (contract_oz × stop_distance)

        Ví dụ: balance=$100, risk=1%, stop=$3.60
            lot = (100 × 0.01) / (100 × 3.60) = 1 / 360 ≈ 0.01 lot (minimum)

        Ví dụ: balance=$10 000, risk=1%, stop=$3.60
            lot = (10000 × 0.01) / (100 × 3.60) = 100 / 360 ≈ 0.28 lot
        """
        if risk_fraction is None:
            risk_fraction = self.settings.risk.risk_per_trade

        risk_amount = balance * risk_fraction

        if stop_distance <= 0:
            return _MIN_LOT

        raw_lot = risk_amount / (_XAUUSD_CONTRACT_OZ * stop_distance)

        # Làm tròn đến lot step
        lot = max(_MIN_LOT, round(raw_lot / _LOT_STEP) * _LOT_STEP)

        # Hard cap: không vượt max_risk_fraction trong bất kỳ tình huống nào
        max_risk_amount = balance * self.settings.risk.max_risk_fraction
        max_lot = max(_MIN_LOT, round(
            (max_risk_amount / (_XAUUSD_CONTRACT_OZ * stop_distance)) / _LOT_STEP
        ) * _LOT_STEP)
        lot = min(lot, max_lot)

        LOGGER.debug(
            "lot_size: balance=%.2f stop=%.4f risk_pct=%.4f → lot=%.2f",
            balance, stop_distance, risk_fraction, lot,
        )
        return lot

    # ------------------------------------------------------------------
    # Gate: có được phép mở lệnh mới không?
    # ------------------------------------------------------------------
    def can_open_position(
        self,
        balance: float,
        current_open_positions: int,
        volatility_regime: int = 1,
    ) -> tuple[bool, str]:
        """
        Trả về (True, 'ok') khi được phép mở lệnh.
        Trả về (False, lý_do) khi bị chặn.
        """
        # Circuit breaker check (daily loss, kill switch, cooldown)
        cb_blocked, cb_reason = self.is_circuit_breaker_active(balance)
        if cb_blocked:
            return False, cb_reason

        if balance < 0:
            return False, f"Balance quá thấp (${balance:.2f}), cần tối thiểu $0"

        max_pos = self.get_dynamic_max_positions(balance, volatility_regime)
        if current_open_positions >= max_pos:
            return (
                False,
                f"Đã đủ lệnh: {current_open_positions}/{max_pos} "
                f"(balance=${balance:.0f}, regime={volatility_regime})",
            )

        return True, "ok"

    # ------------------------------------------------------------------
    # Risk fraction (unchanged logic, kept for backtest engine)
    # ------------------------------------------------------------------
    def risk_fraction(
        self,
        confidence: float,
        volatility_regime: int | None = None,
        strategy_score: float | None = None,
        current_balance: float | None = None,
        market_row: object | None = None,
        side: str | None = None,
    ) -> float:
        # Dynamic risk tier: scale between floor and ceiling based on drawdown.
        rf_base  = self.settings.risk.risk_per_trade
        rf_floor = float(getattr(self.settings.risk, 'risk_tier_floor', 0.0))
        if rf_floor > 0.0 and current_balance is not None and current_balance > 0:
            # Update peak balance
            if current_balance > self._peak_balance:
                self._peak_balance = current_balance
                self._save_peak_balance()
            if self._peak_balance > 0:
                dd = (self._peak_balance - current_balance) / self._peak_balance
                if dd >= 0.10:
                    rf_base = rf_floor
                elif dd >= 0.03:
                    ratio = (dd - 0.03) / 0.07
                    rf_base = self.settings.risk.risk_per_trade - ratio * (self.settings.risk.risk_per_trade - rf_floor)
                # else: < 3% drawdown — full risk (rf_base unchanged)

        # Dynamic risk based on confidence (if enabled)
        dynamic_risk_enabled = bool(getattr(self.settings.risk, "dynamic_risk_enabled", False))
        if dynamic_risk_enabled:
            from xauusd_ai.strategies.dynamic_risk import calculate_dynamic_risk
            
            min_risk = float(getattr(self.settings.risk, "dynamic_risk_min", 0.02))
            max_risk = float(getattr(self.settings.risk, "dynamic_risk_max", 0.05))
            
            dynamic_risk, risk_reason = calculate_dynamic_risk(
                base_risk=rf_base,
                confidence=confidence,
                min_risk=min_risk,
                max_risk=max_risk,
            )
            base_fraction = dynamic_risk
        else:
            # Legacy: scale by confidence linearly
            capped_confidence = min(max(confidence, self.settings.risk.min_confidence), 0.95)
            base_fraction = rf_base * (capped_confidence / self.settings.risk.min_confidence)
        
        regime_multiplier = self.settings.risk.normal_risk_multiplier
        if volatility_regime == 0:
            regime_multiplier = self.settings.risk.sideway_risk_multiplier
        elif volatility_regime == 2:
            regime_multiplier = self.settings.risk.strong_volatility_risk_multiplier

        # score_multiplier: 1.0 when strategy_score >= 0 (aligned signals get full risk)
        # Scale 0.7–1.0 based on strength; never penalise high-confidence aligned signals.
        score_multiplier = 1.0
        if strategy_score is not None:
            abs_score = abs(strategy_score)
            if abs_score >= 0.5:
                score_multiplier = 1.0
            else:
                # Linearly scale 0.7 (score=0) to 1.0 (score=0.5)
                score_multiplier = 0.7 + 0.6 * abs_score

        # Anti-martingale: reduce risk after consecutive losses
        anti_mart = self.get_anti_martingale_factor()

        raw_fraction = base_fraction * regime_multiplier * score_multiplier * anti_mart
        throttled_fraction, _, _ = self.apply_risk_throttle(
            raw_fraction,
            market_row,
            side=side,
            probability=confidence,
        )
        return min(throttled_fraction, self.settings.risk.max_risk_fraction)

    def risk_throttle_multiplier(
        self,
        row: object | None,
        side: str | None = None,
        probability: float | None = None,
    ) -> tuple[float, str]:
        rules = getattr(self.settings.risk, "risk_throttle_rules", [])
        if row is None or not rules:
            return 1.0, "ok"

        timestamp = self._row_value(row, "time", None)
        resolved = timestamp
        if timestamp is not None and not isinstance(timestamp, pd.Timestamp):
            resolved = pd.to_datetime(timestamp, utc=True, errors="coerce")

        weekday_name = resolved.day_name().lower() if isinstance(resolved, pd.Timestamp) and not pd.isna(resolved) else ""
        hour = int(resolved.hour) if isinstance(resolved, pd.Timestamp) and not pd.isna(resolved) else None
        row_side = (side or str(self._row_value(row, "trade_side", self._row_value(row, "side", "")))).strip().lower()

        matched_names: list[str] = []
        min_multiplier = 1.0
        for rule in rules:
            if not getattr(rule, "enabled", True):
                continue
            rule_multiplier = float(getattr(rule, "risk_multiplier", 1.0))
            if rule_multiplier >= 1.0:
                continue

            if rule.weekdays_utc and weekday_name not in {value.strip().lower() for value in rule.weekdays_utc}:
                continue
            if rule.hours_utc and hour not in {int(value) for value in rule.hours_utc}:
                continue
            if rule.sides and row_side not in {value.strip().lower() for value in rule.sides}:
                continue

            volatility_regime = int(self._row_value(row, "volatility_regime", 1))
            if rule.volatility_regimes and volatility_regime not in {int(value) for value in rule.volatility_regimes}:
                continue

            trend_alignment = int(self._row_value(row, "trend_alignment", 1))
            if rule.trend_alignment_values and trend_alignment not in {int(value) for value in rule.trend_alignment_values}:
                continue

            if probability is not None:
                if rule.probability_min is not None and float(probability) < float(rule.probability_min):
                    continue
                if rule.probability_max is not None and float(probability) > float(rule.probability_max):
                    continue

            adx_value = self._as_float(self._row_value(row, "adx", None))
            if rule.adx_min is not None and (adx_value is None or adx_value < float(rule.adx_min)):
                continue
            if rule.adx_max is not None and (adx_value is None or adx_value > float(rule.adx_max)):
                continue

            trend_strength = self._as_float(self._row_value(row, "trend_strength_score", None))
            if rule.trend_strength_min is not None and (trend_strength is None or trend_strength < float(rule.trend_strength_min)):
                continue
            if rule.trend_strength_max is not None and (trend_strength is None or trend_strength > float(rule.trend_strength_max)):
                continue

            pullback_quality = self._as_float(self._row_value(row, "pullback_quality", None))
            if rule.pullback_quality_min is not None and (pullback_quality is None or pullback_quality < float(rule.pullback_quality_min)):
                continue
            if rule.pullback_quality_max is not None and (pullback_quality is None or pullback_quality > float(rule.pullback_quality_max)):
                continue

            execution_quality = self._as_float(self._row_value(row, "execution_quality", None))
            if rule.execution_quality_min is not None and (execution_quality is None or execution_quality < float(rule.execution_quality_min)):
                continue
            if rule.execution_quality_max is not None and (execution_quality is None or execution_quality > float(rule.execution_quality_max)):
                continue

            strategy_score = self._as_float(self._row_value(row, "strategy_score", None))
            strategy_abs = abs(strategy_score) if strategy_score is not None else None
            if rule.strategy_score_min is not None and (strategy_abs is None or strategy_abs < float(rule.strategy_score_min)):
                continue
            if rule.strategy_score_max is not None and (strategy_abs is None or strategy_abs > float(rule.strategy_score_max)):
                continue

            min_multiplier = min(min_multiplier, max(0.0, rule_multiplier))
            matched_names.append(rule.name.strip() or "risk_throttle")

        if min_multiplier >= 1.0:
            return 1.0, "ok"
        return min_multiplier, ",".join(matched_names)

    def apply_risk_throttle(
        self,
        base_fraction: float,
        row: object | None,
        side: str | None = None,
        probability: float | None = None,
    ) -> tuple[float, float, str]:
        multiplier, reason = self.risk_throttle_multiplier(row, side=side, probability=probability)
        throttled_fraction = min(base_fraction * multiplier, self.settings.risk.max_risk_fraction)
        return throttled_fraction, multiplier, reason

    # ------------------------------------------------------------------
    # Build order plan with dynamic sizing
    # ------------------------------------------------------------------
    def build_order_plan(
        self,
        decision: TradeDecision,
        latest_bar: pd.Series,
        account_balance: float | None = None,
        current_open_positions: int = 0,
        volatility_regime: int = 1,
    ) -> OrderPlan:
        """
        Tạo kế hoạch lệnh với lot size phù hợp tài khoản.
        Khi account_balance được cung cấp → dùng dynamic sizing.
        Ngược lại → fallback fixed_lot từ config.
        """
        throttle_mult, throttle_reason = self.risk_throttle_multiplier(
            latest_bar,
            side=decision.side,
            probability=decision.confidence,
        )

        if account_balance is not None and account_balance > 0:
            stop_distance = abs(decision.entry_price - decision.stop_loss)
            rf = self.risk_fraction(decision.confidence, volatility_regime,
                                    strategy_score=float(getattr(latest_bar, "strategy_score", 0.0)),
                                    current_balance=account_balance,
                                    market_row=latest_bar,
                                    side=decision.side)
            rf_throttled = rf * throttle_mult
            volume = self.calculate_dynamic_lot(account_balance, stop_distance, rf_throttled)
        else:
            volume = self.settings.risk.fixed_lot * throttle_mult

        reason = decision.reason
        if throttle_mult < 1.0:
            reason = f"{reason} | risk_throttle({throttle_reason}) x{throttle_mult:.2f}"

        return OrderPlan(
            symbol=self.settings.market.symbol,
            side=decision.side,
            volume=volume,
            entry_price=decision.entry_price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            confidence=decision.confidence,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Trailing Stop Loss
    # ------------------------------------------------------------------
    def compute_trailing_sl(
        self,
        position: dict,
        atr: float,
    ) -> float | None:
        """
        Tính SL mới nếu cần dịch trailing stop.

        position dict cần có: side, open_price, sl, current_price
        atr: ATR hiện tại tính bằng giá (USD per bar)

        Returns:
            float  → new_sl nếu cần cập nhật
            None   → giữ nguyên SL
        """
        cfg = self.settings.execution.trailing_sl
        if not cfg.enabled or atr <= 0:
            return None

        side = position.get("side", "")
        open_price = float(position.get("open_price", 0))
        current_sl = float(position.get("sl", 0))
        current_price = float(position.get("current_price", 0))

        # Ước tính initial_risk từ ATR × SL multiple
        initial_risk = atr * self.settings.risk.stop_loss_atr_multiple
        if initial_risk <= 0:
            return None

        # Profit tính bằng R
        if side == "buy":
            profit_distance = current_price - open_price
        else:
            profit_distance = open_price - current_price

        profit_r = profit_distance / initial_risk

        # Step 1: Breakeven — dịch SL về entry khi đạt breakeven_at_rr
        if profit_r < cfg.breakeven_at_rr:
            return None  # Chưa đủ lợi nhuận để dịch

        # Step 2: Trail — dịch SL theo giá khi đạt activation_rr
        if profit_r >= cfg.activation_rr:
            trail_distance = atr * cfg.trail_atr_multiple
            if side == "buy":
                new_sl = current_price - trail_distance
                # Chỉ dịch lên, không bao giờ dịch xuống
                if new_sl > current_sl:
                    return round(new_sl, 5)
            else:
                new_sl = current_price + trail_distance
                # Chỉ dịch xuống, không bao giờ dịch lên
                if new_sl < current_sl or current_sl == 0:
                    return round(new_sl, 5)
            return None

        # Step 3: Chỉ breakeven (chưa đủ để trail)
        if side == "buy":
            if open_price > current_sl:
                return round(open_price, 5)  # Move to breakeven
        else:
            if current_sl == 0 or open_price < current_sl:
                return round(open_price, 5)  # Move to breakeven

        return None

    # ------------------------------------------------------------------
    # DCA — Dollar Cost Averaging
    # ------------------------------------------------------------------
    def should_dca(
        self,
        position: dict,
        atr: float,
        dca_count: int,
        account_balance: float,
        total_open_lots: float = 0.0,
    ) -> bool:
        """
        Quyết định có nên DCA không.

        Điều kiện:
          1. DCA chưa đạt max_dca_count
          2. Giá đi ngược trigger_atr_multiple × ATR
          3. Tổng risk (open lots) chưa vượt max_total_risk_pct
        """
        cfg = self.settings.execution.dca
        if not cfg.enabled:
            return False
        if dca_count >= cfg.max_dca_count:
            return False
        if atr <= 0 or account_balance <= 0:
            return False

        side = position.get("side", "")
        open_price = float(position.get("open_price", 0))
        current_price = float(position.get("current_price", 0))

        # Giá đi ngược bao nhiêu?
        if side == "buy":
            loss_distance = open_price - current_price
        else:
            loss_distance = current_price - open_price

        if loss_distance < cfg.trigger_atr_multiple * atr:
            return False  # Chưa đủ sâu để DCA

        # Kiểm tra tổng risk không vượt giới hạn
        if total_open_lots > 0 and account_balance > 0:
            # Ước tính risk tổng: lots × stop_cost trung bình
            estimated_total_risk = (total_open_lots * _XAUUSD_CONTRACT_OZ * atr * self.settings.risk.stop_loss_atr_multiple)
            risk_pct = estimated_total_risk / account_balance
            if risk_pct >= cfg.max_total_risk_pct:
                LOGGER.info(
                    "DCA blocked: total_risk=%.2f%% >= max=%.2f%%",
                    risk_pct * 100, cfg.max_total_risk_pct * 100,
                )
                return False

        return True

    def build_dca_plan(
        self,
        position: dict,
        atr: float,
        dca_count: int,
        account_balance: float,
    ) -> OrderPlan:
        """
        Tạo kế hoạch lệnh DCA.
        Lot = lot_gốc × lot_multiplier^dca_count (tăng dần mỗi lần DCA).
        SL và TP tính lại từ giá hiện tại.
        """
        cfg = self.settings.execution.dca
        base_lot = float(position.get("volume", _MIN_LOT))
        dca_lot = max(_MIN_LOT, round(
            base_lot * (cfg.lot_multiplier ** dca_count) / _LOT_STEP
        ) * _LOT_STEP)

        side = position.get("side", "buy")
        current_price = float(position.get("current_price", 0))
        sl_dist = atr * self.settings.risk.stop_loss_atr_multiple
        tp_dist = sl_dist * self.settings.risk.take_profit_rr

        if side == "buy":
            stop_loss = round(current_price - sl_dist, 5)
            take_profit = round(current_price + tp_dist, 5)
        else:
            stop_loss = round(current_price + sl_dist, 5)
            take_profit = round(current_price - tp_dist, 5)

        LOGGER.info(
            "DCA#%d plan: side=%s lot=%.2f entry=%.2f sl=%.2f tp=%.2f",
            dca_count + 1, side, dca_lot, current_price, stop_loss, take_profit,
        )
        return OrderPlan(
            symbol=self.settings.market.symbol,
            side=side,
            volume=dca_lot,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.0,
            reason=f"DCA#{dca_count + 1} ticket={position.get('ticket', '?')}",
        )

    # ------------------------------------------------------------------
    # Lot size table by regime (dùng cho dashboard hiển thị)
    # ------------------------------------------------------------------
    def get_lot_table_by_regime(self, balance: float, atr: float) -> dict:
        """
        Trả về bảng lot size dự kiến theo từng regime.
        Dùng để hiển thị trong dashboard.
        """
        sl_distance = atr * self.settings.risk.stop_loss_atr_multiple
        if sl_distance <= 0:
            sl_distance = 5.0  # Fallback nếu ATR không có

        regimes = {
            "sideway": self.settings.risk.sideway_risk_multiplier,
            "normal":  self.settings.risk.normal_risk_multiplier,
            "volatile": self.settings.risk.strong_volatility_risk_multiplier,
        }
        result = {}
        for label, multiplier in regimes.items():
            rf = self.settings.risk.risk_per_trade * multiplier
            lot = self.calculate_dynamic_lot(balance, sl_distance, rf)
            result[label] = {
                "lot": lot,
                "risk_pct": round(rf * 100, 3),
                "risk_usd": round(balance * rf, 2),
            }
        return result
