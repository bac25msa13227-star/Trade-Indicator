#!/usr/bin/env python3
"""
Realtime Bar Appender — append closed MT5 bars into local CSV files.

Chạy mỗi phút qua crontab (hoặc loop). Mỗi lần chạy:
  - Gọi bridge /bars cho tất cả TF: M1, M5, M15, M30, H1, H4, D1
  - Với mỗi bar mới ĐÃ ĐÓNG → append vào CSV đúng TF
  - Tính tick_volume_delta và volume_imbalance (cần cho features)
  - Ghi atomic (.tmp → rename) để tránh corrupt file

CSV format (phải khớp với WF dataset loader):
  time,open,high,low,close,tick_volume,spread_points,tick_volume_delta,volume_imbalance

Biến môi trường:
  BRIDGE_HOST        default: localhost
  BRIDGE_PORT        default: 5600
  MT5_SYMBOL         default: XAUUSDm
  DATA_DIR           default: src/xauusd_ai/real_data
  RUN_LOOP           nếu set=1, chạy loop vô tận (mỗi 55s). Mặc định: chạy 1 lần rồi thoát.
  LOG_LEVEL          DEBUG / INFO (default INFO)

Ví dụ crontab (chạy mỗi phút):
  * * * * * cd /path/to/Trade-Indicator && PYTHONPATH=src python3 scripts/realtime_bar_appender.py >> /tmp/bar_appender.log 2>&1

Ví dụ chạy loop nền:
  RUN_LOOP=1 python3 scripts/realtime_bar_appender.py &
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

# ── Config từ env ────────────────────────────────────────────────────────────
BRIDGE_HOST = os.getenv("BRIDGE_HOST", "localhost")
BRIDGE_PORT  = int(os.getenv("BRIDGE_PORT", "5600"))
SYMBOL       = os.getenv("MT5_SYMBOL", "XAUUSDm")
RUN_LOOP     = os.getenv("RUN_LOOP", "0") == "1"
LOOP_SLEEP   = 55  # seconds between iterations

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(_REPO_ROOT / "src" / "xauusd_ai" / "real_data")))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [bar_appender] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("bar_appender")

# ── Timeframe definitions ────────────────────────────────────────────────────
# (name, bar_duration_seconds, bridge_count_to_fetch)
_TIMEFRAMES: list[tuple[str, int, int]] = [
    ("M1",  1 * 60,   5),
    ("M5",  5 * 60,   5),
    ("M15", 15 * 60,  4),
    ("M30", 30 * 60,  4),
    ("H1",  1 * 3600, 3),
    ("H4",  4 * 3600, 3),
    ("D1",  24 * 3600, 2),
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _csv_path(tf: str) -> Path:
    return DATA_DIR / f"XAUUSDm_{tf}.csv"


def _bar_is_closed(bar_open_ts: int, bar_dur_secs: int, now_ts: float) -> bool:
    """True if bar has fully closed (bar_open + duration <= now)."""
    return (bar_open_ts + bar_dur_secs) <= now_ts


def _volume_imbalance(open_: float, high: float, low: float, close: float) -> float:
    """Body pressure: (close-low)/(high-low). 1.0=closed at top, 0.0=at bottom."""
    rng = high - low
    if rng < 1e-9:
        return 0.5
    return max(0.0, min(1.0, (close - low) / rng))


def _fetch_bars(tf: str, count: int) -> list[dict]:
    """GET /bars from bridge. Returns list of bar dicts or raises."""
    url = (
        f"http://{BRIDGE_HOST}:{BRIDGE_PORT}/bars"
        f"?symbol={SYMBOL}&timeframe={tf}&count={count}"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode())
    # Bridge may wrap in {"bars": [...]} or return list directly
    if isinstance(data, dict):
        data = data.get("bars", data.get("data", []))
    return data  # type: ignore[return-value]


def _load_csv_last_tv(csv_path: Path) -> tuple[int, float]:
    """
    Read last row of CSV to get (last_bar_unix_ts, last_tick_volume).
    Returns (-1, 0.0) if file missing or empty.
    """
    if not csv_path.exists():
        return -1, 0.0
    try:
        # Read last non-empty line efficiently
        with open(csv_path, "rb") as f:
            # Seek near end
            f.seek(0, 2)
            size = f.tell()
            if size < 50:
                return -1, 0.0
            f.seek(max(0, size - 512))
            tail = f.read().decode(errors="replace")
        lines = [l for l in tail.splitlines() if l.strip() and not l.startswith("time")]
        if not lines:
            return -1, 0.0
        last = lines[-1].split(",")
        # columns: time,open,high,low,close,tick_volume,spread_points,tick_volume_delta,volume_imbalance
        raw_time = last[0].strip()
        tv = float(last[5]) if len(last) > 5 else 0.0
        # Parse ISO time → unix
        ts = dt.datetime.fromisoformat(raw_time)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        return int(ts.timestamp()), tv
    except Exception as exc:
        log.debug("Could not read last CSV row (%s): %s", csv_path.name, exc)
        return -1, 0.0


def _append_bar_to_csv(
    csv_path: Path,
    bar: dict,
    prev_tick_volume: float,
) -> None:
    """Append one bar row to CSV (atomic write)."""
    bar_ts   = int(bar["time"])
    open_    = float(bar["open"])
    high     = float(bar["high"])
    low      = float(bar["low"])
    close    = float(bar["close"])
    tv       = float(bar.get("tick_volume", 0.0))
    spread   = float(bar.get("spread", bar.get("spread_points", 0.0)))

    tick_volume_delta = tv - prev_tick_volume
    vi = _volume_imbalance(open_, high, low, close)

    # UTC ISO format matching existing CSV rows
    bar_dt = dt.datetime.fromtimestamp(bar_ts, tz=dt.timezone.utc)
    time_str = bar_dt.strftime("%Y-%m-%d %H:%M:%S+00:00")

    row = f"{time_str},{open_},{high},{low},{close},{tv},{spread},{tick_volume_delta:.10f},{vi:.16f}\n"

    # Ensure dir exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # If file doesn't exist, write header first
    if not csv_path.exists():
        header = "time,open,high,low,close,tick_volume,spread_points,tick_volume_delta,volume_imbalance\n"
        tmp = csv_path.with_suffix(".tmp")
        tmp.write_text(header + row, encoding="utf-8")
        tmp.rename(csv_path)
        return

    # Atomic append: read existing + append + write to .tmp then rename
    # For large files, use direct append (much faster)
    tmp = csv_path.with_suffix(".tmp_append")
    try:
        with open(csv_path, "rb") as src, open(tmp, "wb") as dst:
            # Stream copy existing content
            while True:
                chunk = src.read(65536)
                if not chunk:
                    break
                dst.write(chunk)
            dst.write(row.encode("utf-8"))
        tmp.rename(csv_path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def process_timeframe(tf: str, bar_dur_secs: int, count: int, now_ts: float) -> int:
    """
    Fetch bars for one TF and append any new closed bars.
    Returns number of rows appended.
    """
    csv_path = _csv_path(tf)
    appended = 0

    try:
        bars = _fetch_bars(tf, count)
    except Exception as exc:
        log.warning("Bridge unreachable for %s (%s: %s) — skipping", tf, type(exc).__name__, exc)
        return 0

    if not bars:
        log.debug("No bars returned for %s", tf)
        return 0

    # Sort ascending by time
    bars = sorted(bars, key=lambda b: int(b.get("time", 0)))

    last_csv_ts, last_csv_tv = _load_csv_last_tv(csv_path)

    prev_tv = last_csv_tv  # tick_volume of the row just before the first new bar

    for bar in bars:
        bar_ts = int(bar.get("time", 0))

        # Skip bars already in CSV
        if bar_ts <= last_csv_ts:
            # Update prev_tv tracking within existing bars
            prev_tv = float(bar.get("tick_volume", prev_tv))
            continue

        # Skip bar that's still open (not yet closed)
        if not _bar_is_closed(bar_ts, bar_dur_secs, now_ts):
            log.debug("%s bar %d still open — skip", tf, bar_ts)
            prev_tv = float(bar.get("tick_volume", prev_tv))
            continue

        # Append this new closed bar
        try:
            _append_bar_to_csv(csv_path, bar, prev_tv)
            bar_dt = dt.datetime.fromtimestamp(bar_ts, tz=dt.timezone.utc)
            log.info("Appended %s bar %s  tv=%.4f  vi=%.3f",
                     tf, bar_dt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                     float(bar.get("tick_volume", 0)),
                     _volume_imbalance(float(bar["open"]), float(bar["high"]),
                                       float(bar["low"]), float(bar["close"])))
            appended += 1
            last_csv_ts = bar_ts
        except Exception as exc:
            log.error("Failed to append %s bar %d: %s", tf, bar_ts, exc)

        prev_tv = float(bar.get("tick_volume", prev_tv))

    return appended


def run_once() -> None:
    now_ts = time.time()
    total = 0
    for tf, dur, count in _TIMEFRAMES:
        n = process_timeframe(tf, dur, count, now_ts)
        total += n
    if total == 0:
        log.debug("No new bars this cycle")
    else:
        log.info("Cycle done — %d bar(s) appended across all TFs", total)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Bar appender started — bridge=%s:%d  symbol=%s  data=%s  loop=%s",
             BRIDGE_HOST, BRIDGE_PORT, SYMBOL, DATA_DIR, RUN_LOOP)

    if RUN_LOOP:
        while True:
            try:
                run_once()
            except Exception as exc:
                log.error("Unexpected error in cycle: %s", exc)
            time.sleep(LOOP_SLEEP)
    else:
        run_once()


if __name__ == "__main__":
    main()
