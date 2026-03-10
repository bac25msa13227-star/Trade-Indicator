from __future__ import annotations

import os

import requests

from xauusd_ai.config import Settings
from xauusd_ai.execution.risk import OrderPlan
from xauusd_ai.strategies.hybrid import TradeDecision


class TelegramNotifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def send_signal(self, decision: TradeDecision, order_plan: OrderPlan) -> None:
        if not self.settings.notifications.telegram_enabled:
            return

        token = os.getenv(self.settings.integrations.telegram.token_env)
        chat_id = os.getenv(self.settings.integrations.telegram.chat_id_env)
        if not token or not chat_id:
            return

        message = (
            f"XAUUSD AI Signal\n"
            f"Side: {decision.side}\n"
            f"Confidence: {decision.confidence:.2%}\n"
            f"Entry: {order_plan.entry_price:.2f}\n"
            f"SL: {order_plan.stop_loss:.2f}\n"
            f"TP: {order_plan.take_profit:.2f}\n"
            f"Reason: {decision.reason}"
        )
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=10,
        )
