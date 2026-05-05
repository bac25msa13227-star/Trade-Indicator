from __future__ import annotations

import os

import requests

from xauusd_ai.config import Settings
from xauusd_ai.execution.risk import OrderPlan
from xauusd_ai.strategies.hybrid import TradeDecision


class TelegramNotifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Derive account tag for message prefix (e.g. "acc1", "acc2")
        from pathlib import Path as _Path
        _stem = _Path(settings.app.live_closed_trades_path).stem  # live_closed_trades_acc1
        self._acct = _stem.replace("live_closed_trades_", "").strip("_") or ""

    def send_signal(self, decision: TradeDecision, order_plan: OrderPlan) -> None:
        if not self.settings.notifications.telegram_enabled:
            return

        token = os.getenv(self.settings.integrations.telegram.token_env)
        chat_id = os.getenv(self.settings.integrations.telegram.chat_id_env)
        if not token or not chat_id:
            return

        _side = str(decision.side or "").strip().lower()
        if _side == "buy":
            _side_label = "BUY (Mua)"
        elif _side == "sell":
            _side_label = "SELL (Ban)"
        else:
            _side_label = _side.upper() if _side else "N/A"

        acct_prefix = f"[<b>{self._acct.upper()}</b>] " if self._acct else ""
        message = (
            f"{acct_prefix}<b>XAUUSD AI Signal / Tín hiệu XAUUSD AI</b>\n"
            f"Side / Hướng lệnh: {_side_label}\n"
            f"Confidence / Độ tin cậy: {decision.confidence:.2%}\n"
            f"Entry / Giá vào: {order_plan.entry_price:.2f}\n"
            f"SL / Cắt lỗ: {order_plan.stop_loss:.2f}\n"
            f"TP / Chốt lời: {order_plan.take_profit:.2f}\n"
            f"Reason / Lý do: {decision.reason}"
        )
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )

    def send_message(self, text: str) -> None:
        """Gửi tin nhắn tùy ý qua Telegram (dùng cho cảnh báo, loss analysis, v.v.)."""
        import logging as _log
        _logger = _log.getLogger(__name__)
        if not self.settings.notifications.telegram_enabled:
            return
        token = os.getenv(self.settings.integrations.telegram.token_env)
        chat_id = os.getenv(self.settings.integrations.telegram.chat_id_env)
        if not token or not chat_id:
            _logger.warning("Telegram send_message: token/chat_id missing (env=%s/%s)",
                            self.settings.integrations.telegram.token_env,
                            self.settings.integrations.telegram.chat_id_env)
            return
        # Prepend account tag so user can distinguish ACC1 vs ACC2 messages
        if self._acct:
            text = f"[<b>{self._acct.upper()}</b>] {text}"
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            if resp.ok:
                _logger.info("Telegram message sent OK (msg_id=%s)", resp.json().get("result", {}).get("message_id", "?"))
            else:
                # HTML parse_mode failed — retry without parse_mode
                _logger.warning("Telegram send_message HTML failed (%s): %s — retrying as plain text",
                                resp.status_code, resp.text[:200])
                resp2 = requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": text},
                    timeout=10,
                )
                if resp2.ok:
                    _logger.info("Telegram message sent OK plain-text (msg_id=%s)", resp2.json().get("result", {}).get("message_id", "?"))
                else:
                    _logger.error("Telegram send_message plain text also failed (%s): %s",
                                  resp2.status_code, resp2.text[:200])
        except Exception as e:
            _logger.error("Telegram send_message network error: %s", e)
