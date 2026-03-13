from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import requests

from xauusd_ai.config import Settings
from xauusd_ai.execution.risk import OrderPlan
from xauusd_ai.strategies.hybrid import TradeDecision

LOGGER = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _post(self, token: str, chat_id: str, text: str) -> None:
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Telegram send failed: %s", exc)

    def _get_credentials(self) -> tuple[str, str] | tuple[None, None]:
        token = os.getenv(self.settings.integrations.telegram.token_env)
        chat_id = os.getenv(self.settings.integrations.telegram.chat_id_env)
        if not token or not chat_id:
            return None, None
        return token, chat_id

    def send_signal(self, decision: TradeDecision, order_plan: OrderPlan) -> None:
        if not self.settings.notifications.telegram_enabled:
            return
        token, chat_id = self._get_credentials()
        if not token:
            return

        if not decision.should_trade:
            return  # Don't spam with non-trade signals

        balance = self.settings.risk.account_balance
        risk_pct = self.settings.risk.risk_per_trade * 100
        risk_usd = balance * self.settings.risk.risk_per_trade
        rr_ratio = self.settings.risk.take_profit_rr

        # XAUUSD lot calculation:
        # 1 standard lot = 100 troy oz
        # Each $1 price movement = $100 P&L per lot
        # → lot = risk_usd / (sl_distance_usd × 100)
        XAUUSD_VALUE_PER_POINT = 100.0  # USD per lot per $1 gold price move
        sl_pts = abs(order_plan.entry_price - order_plan.stop_loss)
        if sl_pts > 0:
            raw_lot = risk_usd / (sl_pts * XAUUSD_VALUE_PER_POINT)
            lot_hint = round(
                max(self.settings.risk.min_lot_size, min(raw_lot, self.settings.risk.max_lot_size)),
                2,
            )
        else:
            lot_hint = self.settings.risk.min_lot_size

        # Actual risk/reward based on final lot (may differ from target when clipped to min lot)
        actual_risk_usd = lot_hint * sl_pts * XAUUSD_VALUE_PER_POINT
        actual_reward_usd = actual_risk_usd * rr_ratio
        actual_risk_pct = (actual_risk_usd / balance) * 100

        side_emoji = "🟢 BUY" if decision.side == "buy" else "🔴 SELL"
        conf_bar = "█" * int(decision.confidence * 10) + "░" * (10 - int(decision.confidence * 10))
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        message = (
            f"⚡ <b>XAUUSD AI SIGNAL</b> ⚡\n"
            f"🕐 {now}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{side_emoji}  |  Confidence: {decision.confidence:.0%}\n"
            f"{conf_bar}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 <b>Entry Price :</b>  <code>{order_plan.entry_price:.2f}</code>\n"
            f"🛑 <b>Stop Loss   :</b>  <code>{order_plan.stop_loss:.2f}</code>  (${sl_pts:.2f}/oz)\n"
            f"🎯 <b>Take Profit :</b>  <code>{order_plan.take_profit:.2f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>Risk / Reward</b>\n"
            f"   Lot gợi ý : <code>{lot_hint:.2f}</code>\n"
            f"   Rủi ro   : ~${actual_risk_usd:.2f} ({actual_risk_pct:.1f}% vốn)\n"
            f"   Lợi nhuận: ~${actual_reward_usd:.2f}  (R:R = 1:{rr_ratio})\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Lý do:</b> {decision.reason}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ <i>Đây là tín hiệu AI, hãy tự xác nhận trước khi vào lệnh!</i>"
        )
        self._post(token, chat_id, message)

    def send_text(self, text: str) -> None:
        """Send a plain text notification (e.g. startup, errors, news events)."""
        if not self.settings.notifications.telegram_enabled:
            return
        token, chat_id = self._get_credentials()
        if not token:
            return
        self._post(token, chat_id, text)

    def send_startup(self) -> None:
        balance = self.settings.risk.account_balance
        risk_pct = self.settings.risk.risk_per_trade * 100
        self.send_text(
            f"🤖 <b>XAUUSD AI Bot đã khởi động</b>\n"
            f"💵 Vốn: ${balance:.0f}  |  Risk: {risk_pct:.0f}%/lệnh\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"📡 Đang theo dõi thị trường và gửi tín hiệu về đây..."
        )
