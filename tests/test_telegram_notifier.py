from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from xauusd_ai.config import Settings
from xauusd_ai.notifications.telegram import TelegramNotifier


class TelegramNotifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings()
        self.settings.notifications.telegram_enabled = True
        self.settings.app.live_closed_trades_path = "outputs/live_closed_trades_acc1.csv"
        self.notifier = TelegramNotifier(self.settings)

    def test_send_message_skips_when_env_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("xauusd_ai.notifications.telegram.requests.post") as post:
                self.notifier.send_message("hello")
                post.assert_not_called()

    def test_send_message_sends_html_and_retries_plain_on_failure(self) -> None:
        bad_resp = Mock(ok=False, status_code=400, text="bad")
        good_resp = Mock(ok=True, status_code=200, text="ok")
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "token", "TELEGRAM_CHAT_ID": "chat"},
            clear=True,
        ):
            with patch(
                "xauusd_ai.notifications.telegram.requests.post",
                side_effect=[bad_resp, good_resp],
            ) as post:
                self.notifier.send_message("test message")
                self.assertEqual(post.call_count, 2)
                first_payload = post.call_args_list[0].kwargs["json"]
                self.assertEqual(first_payload["parse_mode"], "HTML")
                self.assertIn("[<b>ACC1</b>]", first_payload["text"])


if __name__ == "__main__":
    unittest.main()
