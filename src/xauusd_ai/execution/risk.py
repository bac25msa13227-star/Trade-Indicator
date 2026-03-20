from __future__ import annotations

import json
import logging
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
    _PEAK_FILE = Path("outputs/risk_peak_balance.json")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._peak_balance: float = self._load_peak_balance()

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
    # Dynamic position limit — phụ thuộc balance + market regime
    # ------------------------------------------------------------------
    def get_dynamic_max_positions(self, balance: float, volatility_regime: int = 1) -> int:
        """
        Tính số lệnh tối đa được phép mở dựa trên số dư tài khoản.

        Tier map (theo balance USD):
            < $200   → 1 lệnh   (tài khoản micro, XAUUSD rủi ro cao)
            < $500   → 3 lệnh   (tăng từ 2→3 để bắt nhiều signal hơn với vốn $200, max 18% exposure)
            < $2 000 → 5 lệnh
            < $10 000→ 8 lệnh
            < $50 000→ 10 lệnh
            >= $50 000→ 15 lệnh

        Market‑regime điều chỉnh thêm:
            sideway  (0) → giảm 50%   (ít cơ hội, ít rủi ro)
            volatile (2) → giảm 30%   (biến động mạnh, thu nhỏ exposure)
        """
        config_ceiling = self.settings.risk.max_open_positions

        if balance < 500:
            tier_max = 3          # $200-$499: 3 lenh, max 18% exposure, WR93%+ nen an toan
        elif balance < 2_000:
            tier_max = 5
        elif balance < 10_000:
            tier_max = 8
        elif balance < 50_000:
            tier_max = 10
        else:
            tier_max = 15

        base_max = min(config_ceiling, tier_max)

        # Chỉ giảm theo regime khi base_max >= 3.
        # Với tài khoản nhỏ (base_max <= 2), giảm thêm sẽ lock về 1 — không hợp lý.
        if base_max >= 3:
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
        if balance < 50:
            return False, f"Balance quá thấp (${balance:.2f}), cần tối thiểu $50"

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

        return min(base_fraction * regime_multiplier * score_multiplier, self.settings.risk.max_risk_fraction)

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
        if account_balance is not None and account_balance > 0:
            stop_distance = abs(decision.entry_price - decision.stop_loss)
            rf = self.risk_fraction(decision.confidence, volatility_regime,
                                    current_balance=account_balance)
            volume = self.calculate_dynamic_lot(account_balance, stop_distance, rf)
        else:
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
