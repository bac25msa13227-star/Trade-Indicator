"""
Bot Healthwatch — monitors live trading bots and sends Telegram alerts.

Runs as a standalone Docker service. Every 60s it checks whether each
account's live_status file has been updated recently. If the file is
stale (> MAX_STALE_SECS) or missing, a Telegram alert is fired.
When the bot comes back online, a recovery notification is sent.

Environment variables (loaded from .env via docker-compose):
  TELEGRAM_BOT_TOKEN_ACC1  or  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID_ACC1    or  TELEGRAM_CHAT_ID
  TELEGRAM_BOT_TOKEN_ACC2
  TELEGRAM_CHAT_ID_ACC2
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | healthwatch | %(message)s",
)
LOGGER = logging.getLogger(__name__)

OUTPUTS = Path(os.getenv("OUTPUTS_PATH", "outputs"))
POLL_INTERVAL   = int(os.getenv("HEALTHWATCH_INTERVAL", "60"))   # seconds between checks
MAX_STALE_SECS  = int(os.getenv("HEALTHWATCH_MAX_STALE", "300")) # 5 min = stale

ACCOUNTS = {
    "acc1": {
        "status_file": "live_status_acc1.json",
        "label":       "ACC1 (Aggressive)",
        "token_env":   os.getenv("TELEGRAM_BOT_TOKEN_ACC1") or os.getenv("TELEGRAM_BOT_TOKEN"),
        "chat_id_env": os.getenv("TELEGRAM_CHAT_ID_ACC1")   or os.getenv("TELEGRAM_CHAT_ID"),
    },
    "acc2": {
        "status_file": "live_status_acc2.json",
        "label":       "ACC2",
        "token_env":   os.getenv("TELEGRAM_BOT_TOKEN_ACC2") or os.getenv("TELEGRAM_BOT_TOKEN"),
        "chat_id_env": os.getenv("TELEGRAM_CHAT_ID_ACC2")   or os.getenv("TELEGRAM_CHAT_ID"),
    },
}


def _send_telegram(token: str, chat_id: str, text: str) -> None:
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
        if not resp.ok:
            # Retry without HTML parse mode
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=15,
            )
    except Exception as exc:
        LOGGER.error("Telegram send failed: %s", exc)


def _check_status(acct_id: str, cfg: dict, down_state: dict[str, bool]) -> None:
    token   = cfg["token_env"]
    chat_id = cfg["chat_id_env"]
    label   = cfg["label"]
    path    = OUTPUTS / cfg["status_file"]

    if not token or not chat_id:
        LOGGER.warning("%s: Telegram credentials not set — skipping alerts", acct_id)
        return

    now = time.time()
    is_down = down_state.get(acct_id, False)

    if not path.exists():
        stale_secs = MAX_STALE_SECS + 1
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ts_str = data.get("ts", "")
            ts = datetime.fromisoformat(ts_str).astimezone(timezone.utc)
            stale_secs = now - ts.timestamp()
        except Exception as exc:
            LOGGER.warning("%s: Could not parse status file: %s", acct_id, exc)
            stale_secs = MAX_STALE_SECS + 1

    if stale_secs > MAX_STALE_SECS:
        minutes_stale = int(stale_secs // 60)
        if not is_down:
            # First detection — send alert
            msg = (
                f"🔴 <b>⚠️ Bot OFFLINE — {label}</b>\n"
                f"├ Status file stale: {minutes_stale} phút\n"
                f"├ File: {cfg['status_file']}\n"
                f"└ Kiểm tra container: docker ps -a"
            )
            LOGGER.warning("%s: OFFLINE — stale %ds", acct_id, int(stale_secs))
            _send_telegram(token, chat_id, msg)
            down_state[acct_id] = True
        else:
            LOGGER.info("%s: Still offline — stale %ds", acct_id, int(stale_secs))
    else:
        if is_down:
            # Recovery — was down, now back
            bal = data.get("account_balance", 0) if path.exists() else 0
            msg = (
                f"🟢 <b>Bot ONLINE — {label}</b>\n"
                f"├ Balance: ${bal:.2f}\n"
                f"└ Status file age: {int(stale_secs)}s"
            )
            LOGGER.info("%s: RECOVERED — stale %ds", acct_id, int(stale_secs))
            _send_telegram(token, chat_id, msg)
            down_state[acct_id] = False
        else:
            LOGGER.debug("%s: OK — stale %ds", acct_id, int(stale_secs))


def main() -> None:
    LOGGER.info("Healthwatch started — interval=%ds  max_stale=%ds", POLL_INTERVAL, MAX_STALE_SECS)
    down_state: dict[str, bool] = {acct_id: False for acct_id in ACCOUNTS}

    # Send startup notification
    for acct_id, cfg in ACCOUNTS.items():
        if cfg["token_env"] and cfg["chat_id_env"]:
            _send_telegram(
                cfg["token_env"], cfg["chat_id_env"],
                f"👁️ <b>Healthwatch bắt đầu giám sát</b> — {cfg['label']}\n"
                f"├ Max stale: {MAX_STALE_SECS // 60} phút\n"
                f"└ Poll interval: {POLL_INTERVAL}s",
            )

    while True:
        for acct_id, cfg in ACCOUNTS.items():
            try:
                _check_status(acct_id, cfg, down_state)
            except Exception as exc:
                LOGGER.error("Error checking %s: %s", acct_id, exc)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
