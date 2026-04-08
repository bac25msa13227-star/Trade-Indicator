"""
Bot Healthwatch — monitors live trading bots and sends Telegram alerts.

Runs as a standalone Docker service. Every 60s it checks whether each
account's live_status file has been updated recently. If the file is
stale (> MAX_STALE_SECS) or missing, a Telegram alert is fired.
When the bot comes back online, a recovery notification is sent.

Also detects market open/close by reading market_is_open from the bot's
live status JSON (which queries MT5 directly) — handles all holidays
automatically without any hardcoded schedule.

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
from datetime import datetime, timedelta, timezone
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


# ── Market open/close detection ─────────────────────────────────────────────
# Reads market_next_open_utc / market_next_close_utc from the bot's live status
# JSON (which queries MT5 directly — handles all holidays automatically).
# Fires alerts at T-1440min (24h ahead) and T-5min before each event.

_MARKET_ALERT_SLOTS_MIN = [1440, 5]  # 24h and 5min before
_MARKET_SLOT_TOLERANCE_MIN = 3.0     # ±3 min window so 60s poll never misses


def _read_status_data() -> dict | None:
    """Return parsed JSON from the first available (fresh) status file."""
    for cfg in ACCOUNTS.values():
        p = OUTPUTS / cfg["status_file"]
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(data["ts"]).astimezone(timezone.utc)
            if (datetime.now(timezone.utc) - ts).total_seconds() > 600:
                continue  # stale
            return data
        except Exception:
            continue
    return None


def _check_market_state(state: dict) -> None:
    """
    Two behaviours in one function:

    1. Advance alerts — fires at T-1440min (24h) and T-5min before the next
       open or close, using the exact UTC time reported by MT5.

    2. Transition alert — fires once when market_is_open flips, confirming
       the actual open/close moment.
    """
    data = _read_status_data()
    if data is None:
        return

    now = datetime.now(timezone.utc)

    # ── 1. Advance alerts ────────────────────────────────────────────────────
    for ev_type, key_utc in (("open", "market_next_open_utc"), ("close", "market_next_close_utc")):
        raw = data.get(key_utc)
        if not raw:
            continue
        try:
            ev_time = datetime.fromisoformat(raw).astimezone(timezone.utc)
        except Exception:
            continue

        minutes_to = (ev_time - now).total_seconds() / 60.0
        if minutes_to < 0:
            continue

        for slot in _MARKET_ALERT_SLOTS_MIN:
            if not (slot - _MARKET_SLOT_TOLERANCE_MIN <= minutes_to < slot + _MARKET_SLOT_TOLERANCE_MIN):
                continue
            sent_key = f"{ev_time.strftime('%Y-%m-%dT%H:%M')}_{ev_type}_{slot}"
            if sent_key in state:
                break
            state[sent_key] = True

            ev_vn = ev_time + timedelta(hours=7)
            if ev_type == "open":
                if slot >= 1440:
                    msg = (
                        f"📅 <b>Nhắc trước 24h — Sàn XAUUSD MỞ CỬA</b>\n"
                        f"├ Thời gian: {ev_time.strftime('%d/%m/%Y %H:%M UTC')} "
                        f"({ev_vn.strftime('%d/%m %H:%M +07')})\n"
                        f"└ Chuẩn bị chiến lược cho phiên tới!"
                    )
                else:
                    msg = (
                        f"🟢 <b>Sàn XAUUSD sắp MỞ CỬA (~{slot} phút)</b>\n"
                        f"├ Thời gian: {ev_time.strftime('%H:%M UTC')} "
                        f"({ev_vn.strftime('%H:%M +07')})\n"
                        f"└ Chuẩn bị vào lệnh!"
                    )
            else:
                if slot >= 1440:
                    msg = (
                        f"📅 <b>Nhắc trước 24h — Sàn XAUUSD ĐÓNG CỬA</b>\n"
                        f"├ Thời gian: {ev_time.strftime('%d/%m/%Y %H:%M UTC')} "
                        f"({ev_vn.strftime('%d/%m %H:%M +07')})\n"
                        f"└ Lên kế hoạch đóng vị thế trước khi sàn đóng!"
                    )
                else:
                    msg = (
                        f"🔴 <b>Sàn XAUUSD sắp ĐÓNG CỬA (~{slot} phút)</b>\n"
                        f"├ Thời gian: {ev_time.strftime('%H:%M UTC')} "
                        f"({ev_vn.strftime('%H:%M +07')})\n"
                        f"└ Hãy kiểm tra và đóng vị thế nếu cần!"
                    )

            for cfg in ACCOUNTS.values():
                if cfg["token_env"] and cfg["chat_id_env"]:
                    _send_telegram(cfg["token_env"], cfg["chat_id_env"], msg)
            LOGGER.info("Market advance alert: %s T-%dmin", ev_type, slot)
            break

    # ── 2. Transition alert ──────────────────────────────────────────────────
    market_is_open: bool = bool(data.get("market_is_open", False))
    prev = state.get("_is_open")
    state["_is_open"] = market_is_open

    if prev is None or market_is_open == prev:
        return  # first run or no change

    now_vn = now + timedelta(hours=7)
    if market_is_open:
        msg = (
            f"🟢 <b>Sàn XAUUSD đã MỞ CỬA</b>\n"
            f"├ Thời gian: {now.strftime('%H:%M UTC')} ({now_vn.strftime('%H:%M +07')})\n"
            f"└ Thị trường đang hoạt động!"
        )
        LOGGER.info("Market OPENED at %s UTC", now.strftime("%H:%M"))
    else:
        msg = (
            f"🔴 <b>Sàn XAUUSD đã ĐÓNG CỬA</b>\n"
            f"├ Thời gian: {now.strftime('%H:%M UTC')} ({now_vn.strftime('%H:%M +07')})\n"
            f"└ Thị trường tạm dừng giao dịch."
        )
        LOGGER.info("Market CLOSED at %s UTC", now.strftime("%H:%M"))

    for cfg in ACCOUNTS.values():
        if cfg["token_env"] and cfg["chat_id_env"]:
            _send_telegram(cfg["token_env"], cfg["chat_id_env"], msg)


def main() -> None:
    LOGGER.info("Healthwatch started — interval=%ds  max_stale=%ds", POLL_INTERVAL, MAX_STALE_SECS)
    down_state: dict[str, bool] = {acct_id: False for acct_id in ACCOUNTS}
    market_state: dict = {}  # shared state for advance alerts + transition tracking

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
        try:
            _check_market_state(market_state)
        except Exception as exc:
            LOGGER.error("Error in market state check: %s", exc)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
