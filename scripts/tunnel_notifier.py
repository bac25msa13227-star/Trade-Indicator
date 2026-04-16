"""
tunnel_notifier.py — Watcher for Cloudflare tunnel URL → sends to Telegram.

Usage (run once when tunnel starts, or as a background watcher):
    python scripts/tunnel_notifier.py [--watch] [--log outputs/tunnel_err.txt]

Environment variables required:
    TELEGRAM_BOT_TOKEN  — Bot token from @BotFather
    TELEGRAM_CHAT_ID    — Chat / channel ID to send to

Options:
    --watch          Keep watching the log file for new URLs (poll every 5s).
                     Without this flag, script reads once and exits.
    --log PATH       Path to the tunnel log file (default: outputs/tunnel_err.txt).
    --message TEXT   Custom prefix message (optional).
    --dry-run        Print message instead of sending to Telegram.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_URL_PATTERN = re.compile(r"https://[a-z0-9\-]+\.trycloudflare\.com")
_CANDIDATE_LOGS = [
    ROOT / "outputs" / "tunnel_err.txt",
    ROOT / "outputs" / "cloudflared.log",
    ROOT / "outputs" / "tunnel.log",
]


def _find_url_in_file(path: Path) -> str:
    """Return the last matching tunnel URL found in the file."""
    if not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        matches = _URL_PATTERN.findall(content)
        return matches[-1] if matches else ""
    except Exception:
        return ""


def _find_tunnel_url(log_path: Path | None) -> str:
    """Search log_path first, then fall back to candidate locations."""
    if log_path is not None:
        url = _find_url_in_file(log_path)
        if url:
            return url
    for candidate in _CANDIDATE_LOGS:
        url = _find_url_in_file(candidate)
        if url:
            return url
    return ""


def _send_telegram(token: str, chat_id: str, text: str) -> bool:
    try:
        import urllib.request, urllib.parse, json as _json
        payload = _json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        print(f"[tunnel_notifier] Telegram error: {exc}", file=sys.stderr)
        return False


def _build_message(url: str, custom_prefix: str = "") -> str:
    import datetime as dt
    now_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    if custom_prefix:
        lines.append(custom_prefix)
    lines += [
        "🚀 <b>XAUUSD AI Dashboard — New Tunnel URL</b>",
        f"🌐 <a href='{url}'>{url}</a>",
        f"⏰ {now_str}",
        "",
        "<i>URL này sẽ thay đổi mỗi lần khởi động lại cloudflared.</i>",
    ]
    return "\n".join(lines)


def run_once(log_path: Path | None, dry_run: bool, custom_prefix: str) -> int:
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    url = _find_tunnel_url(log_path)
    if not url:
        print("[tunnel_notifier] No tunnel URL found in log files.", file=sys.stderr)
        return 1

    msg = _build_message(url, custom_prefix)
    print(f"[tunnel_notifier] URL found: {url}")

    if dry_run:
        print(f"[tunnel_notifier] DRY RUN — message:\n{msg}")
        return 0

    if not token or not chat_id:
        print("[tunnel_notifier] ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.", file=sys.stderr)
        return 1

    ok = _send_telegram(token, chat_id, msg)
    if ok:
        print(f"[tunnel_notifier] ✅ Sent to Telegram chat {chat_id}")
        return 0
    else:
        print("[tunnel_notifier] ❌ Failed to send Telegram message.", file=sys.stderr)
        return 1


def run_watch(log_path: Path | None, dry_run: bool, custom_prefix: str, interval: int = 5) -> None:
    """Poll for URL changes and notify Telegram when URL changes."""
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not dry_run and (not token or not chat_id):
        print("[tunnel_notifier] ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.", file=sys.stderr)
        sys.exit(1)

    print(f"[tunnel_notifier] Watching for tunnel URL (poll every {interval}s)... Ctrl+C to stop.")
    last_url = ""

    while True:
        try:
            url = _find_tunnel_url(log_path)
            if url and url != last_url:
                last_url = url
                print(f"[tunnel_notifier] New URL detected: {url}")
                msg = _build_message(url, custom_prefix)
                if dry_run:
                    print(f"[tunnel_notifier] DRY RUN — would send:\n{msg}")
                else:
                    ok = _send_telegram(token, chat_id, msg)
                    if ok:
                        print(f"[tunnel_notifier] ✅ Sent to Telegram")
                    else:
                        print(f"[tunnel_notifier] ❌ Send failed")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[tunnel_notifier] Stopped.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Send Cloudflare tunnel URL to Telegram.")
    parser.add_argument("--watch", action="store_true", help="Keep watching for new URLs (poll loop)")
    parser.add_argument("--log", type=str, default=None, help="Path to tunnel log file")
    parser.add_argument("--message", type=str, default="", help="Custom prefix for the message")
    parser.add_argument("--interval", type=int, default=5, help="Watch poll interval in seconds (default 5)")
    parser.add_argument("--dry-run", action="store_true", help="Print message, do not send")
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else None

    if args.watch:
        run_watch(log_path, args.dry_run, args.message, args.interval)
    else:
        sys.exit(run_once(log_path, args.dry_run, args.message))


if __name__ == "__main__":
    main()
