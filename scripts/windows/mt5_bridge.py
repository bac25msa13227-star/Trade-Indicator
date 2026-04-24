#!/usr/bin/env python3
"""
MT5 Bridge Service — runs on Windows host, proxies MT5 trade operations over HTTP.

The live-bot Docker containers (Linux) cannot import MetaTrader5 directly.
This bridge runs natively on Windows and exposes all MT5 operations as JSON
endpoints so the containers can call them via http://host.docker.internal:5600.

Usage (run once on Windows host, keep running in background):
    python scripts/windows/mt5_bridge.py

Or with explicit credentials (overrides env vars):
    set MT5_LOGIN=270832477
    set MT5_PASSWORD=yourpassword
    set MT5_SERVER=Exness-MT5Trial17
    set MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
    python scripts/windows/mt5_bridge.py
"""

from __future__ import annotations

import datetime as _dt
import ipaddress
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

# IPs allowed to call the bridge (localhost + Docker private subnets).
# External internet traffic is rejected with 403.
_ALLOWED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "../../outputs/mt5_bridge.log"),
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("mt5_bridge")

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 not installed.")
    print("Run:  pip install MetaTrader5")
    sys.exit(1)

import argparse as _argparse
_ap = _argparse.ArgumentParser(description="MT5 HTTP bridge")
_ap.add_argument("--port", type=int, default=int(os.getenv("MT5_BRIDGE_PORT", "5600")))
_args, _unknown = _ap.parse_known_args()
PORT = _args.port

# ── MT5 lifecycle ──────────────────────────────────────────────────────────────

def _init_mt5() -> bool:
    """Initialize MT5 connection using env vars or .env-like env."""
    login    = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server   = os.getenv("MT5_SERVER")
    path     = os.getenv("MT5_TERMINAL_PATH")

    kwargs: dict = {}
    if login:    kwargs["login"]    = int(login)
    if password: kwargs["password"] = password
    if server:   kwargs["server"]   = server
    if path:     kwargs["path"]     = path

    ok = mt5.initialize(**kwargs) if kwargs else mt5.initialize()
    if not ok:
        log.error("MT5 initialize failed: %s", mt5.last_error())
        return False

    info = mt5.account_info()
    log.info("MT5 connected: account=%s server=%s balance=%s",
             info.login if info else "?",
             info.server if info else "?",
             info.balance if info else "?")

    # Verify the connected account matches the expected login
    expected_login = int(login) if login else None
    if expected_login and info and info.login != expected_login:
        log.error(
            "MT5 connected to WRONG account! expected=%s but got=%s (%s). "
            "Open the MT5 terminal manually and log in to the correct account, "
            "then restart this bridge.",
            expected_login, info.login, info.server
        )
        mt5.shutdown()
        return False

    return True


def _ensure() -> None:
    """Re-initialize if connection was dropped or account switched mid-session."""
    terminal_gone = mt5.terminal_info() is None
    if not terminal_gone:
        expected_login = int(os.getenv("MT5_LOGIN", "0"))
        if expected_login:
            info = mt5.account_info()
            account_wrong = info is not None and info.login != expected_login
            if account_wrong:
                log.warning("Account switched mid-session! expected=%s got=%s. Re-initializing...",
                            expected_login, info.login)
                mt5.shutdown()
                terminal_gone = True
    if terminal_gone:
        if not _init_mt5():
            raise RuntimeError(f"MT5 not connected: {mt5.last_error()}")


# ── Operations ─────────────────────────────────────────────────────────────────

def op_place_order(body: dict) -> dict:
    _ensure()
    symbol    = str(body["symbol"])
    side      = str(body["side"])
    volume    = float(body["volume"])
    stop_loss = float(body.get("stop_loss") or 0)
    take_profit = float(body.get("take_profit") or 0)
    entry_price = float(body.get("entry_price") or 0)
    deviation = int(body.get("deviation", 20))
    magic     = int(body.get("magic", 0))
    comment   = str(body.get("comment", "mt5-bridge"))

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick data for {symbol}: {mt5.last_error()}")

    sinfo  = mt5.symbol_info(symbol)
    digits = sinfo.digits if sinfo is not None else 5

    order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
    price = tick.ask if side == "buy" else tick.bid

    # Re-anchor SL/TP to current execution price using original signal distances.
    # This prevents retcode=10016 when price moves between signal generation and execution.
    if stop_loss and take_profit:
        ref = entry_price if entry_price else (stop_loss + take_profit) / 2  # fallback
        sl_dist = abs(ref - stop_loss)
        tp_dist = abs(take_profit - ref)
        if side == "buy":
            sl = round(price - sl_dist, digits)
            tp = round(price + tp_dist, digits)
        else:
            sl = round(price + sl_dist, digits)
            tp = round(price - tp_dist, digits)
    else:
        sl = round(stop_loss, digits)
        tp = round(take_profit, digits)
    log.info("Order params: side=%s price=%.3f sl=%.3f tp=%.3f (orig_entry=%.3f sl_dist=%.3f)",
             side, price, sl, tp, entry_price or 0, abs(price - sl))

    last_err = None
    for filling in [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]:
        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       volume,
            "type":         order_type,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "deviation":    deviation,
            "magic":        magic,
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        result = mt5.order_send(req)
        if result is None:
            last_err = f"order_send None: {mt5.last_error()}"
            continue
        r = result._asdict()
        rc = r.get("retcode")
        if rc == 10009:
            log.info("Order placed: side=%s vol=%.2f sym=%s ticket=%s", side, volume, symbol, r.get("order"))
            return {k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in r.items()}
        if rc == 10027:
            raise RuntimeError("AutoTrading disabled — enable Algo Trading button in MT5. retcode=10027")
        if rc != 10030:
            raise RuntimeError(f"Order failed: retcode={rc} comment={r.get('comment')}")
        last_err = f"INVALID_FILL filling={filling}"

    raise RuntimeError(f"All filling modes failed: {last_err}")


def op_get_positions(symbol: str, magic: int | None = None) -> list:
    _ensure()
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return []
    result = []
    for p in positions:
        if magic is not None and p.magic != magic:
            continue
        d = p._asdict()
        result.append({
            "ticket":        int(d.get("ticket", 0)),
            "side":          "buy" if d.get("type", 0) == 0 else "sell",
            "volume":        float(d.get("volume", 0)),
            "open_price":    float(d.get("price_open", 0)),
            "current_price": float(d.get("price_current", 0)),
            "sl":            float(d.get("sl", 0)),
            "tp":            float(d.get("tp", 0)),
            "profit":        float(d.get("profit", 0)),
            "swap":          float(d.get("swap", 0)),
            "magic":         int(d.get("magic", 0)),
            "comment":       str(d.get("comment", "")),
            "open_time":     int(d.get("time", 0)),
            "symbol":        str(d.get("symbol", symbol)),
        })
    return result


def op_close_position(ticket: int, volume: float, deviation: int = 20, magic: int = 0) -> dict:
    _ensure()
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        raise RuntimeError(f"Position {ticket} not found")
    pos = positions[0]
    close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        raise RuntimeError(f"No tick for {pos.symbol}")
    price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

    last_err = None
    for filling in [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]:
        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       volume,
            "type":         close_type,
            "position":     ticket,
            "price":        price,
            "deviation":    deviation,
            "magic":        magic,
            "comment":      "close-bridge",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        result = mt5.order_send(req)
        if result is None:
            last_err = f"order_send None: {mt5.last_error()}"
            continue
        r = result._asdict()
        if r.get("retcode") == 10009:
            log.info("Position %s closed vol=%.2f", ticket, volume)
            return {k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in r.items()}
        if r.get("retcode") != 10030:
            raise RuntimeError(f"close_position failed: retcode={r.get('retcode')} comment={r.get('comment')}")
        last_err = f"INVALID_FILL filling={filling}"
    raise RuntimeError(f"close_position all modes failed: {last_err}")


def op_modify_sl(ticket: int, new_sl: float) -> dict:
    _ensure()
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        raise RuntimeError(f"Position {ticket} not found")
    pos = positions[0]
    req = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   pos.symbol,
        "position": ticket,
        "sl":       new_sl,
        "tp":       pos.tp,
    }
    result = mt5.order_send(req)
    if result is None:
        raise RuntimeError(f"modify_sl failed: {mt5.last_error()}")
    r = result._asdict()
    if r.get("retcode") != 10009:
        raise RuntimeError(f"modify_sl retcode={r.get('retcode')} comment={r.get('comment')}")
    return {k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in r.items()}


def op_get_account() -> dict:
    _ensure()
    info = mt5.account_info()
    if info is None:
        return {"connected": False, "balance": 0.0, "equity": 0.0, "margin": 0.0, "free_margin": 0.0}
    d = info._asdict()
    return {
        "connected":   True,
        "balance":     float(d.get("balance", 0)),
        "equity":      float(d.get("equity", 0)),
        "margin":      float(d.get("margin", 0)),
        "free_margin": float(d.get("margin_free", 0)),
        "login":       int(d.get("login", 0)),
        "server":      str(d.get("server", "")),
    }


def op_get_tick(symbol: str) -> dict:
    """Return current real-time ask/bid price for symbol."""
    _ensure()
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick data for {symbol}: {mt5.last_error()}")
    return {
        "symbol": symbol,
        "ask":    float(tick.ask),
        "bid":    float(tick.bid),
        "last":   float(tick.last),
        "time":   int(tick.time),
    }


def op_get_rates(symbol: str, timeframe: str, bars: int) -> list:
    """Return OHLCV bars for symbol/timeframe directly from MT5 (realtime)."""
    _ensure()
    tf_map = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }
    if timeframe not in tf_map:
        raise ValueError(f"Unknown timeframe: {timeframe}. Valid: {list(tf_map)}")
    mt5.symbol_select(symbol, True)
    import time as _time
    rates = None
    for _attempt in range(6):
        rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)
        if rates is not None and len(rates) > 0:
            break
        _time.sleep(min(2 * (1.5 ** _attempt), 10))
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No rates returned for {symbol} {timeframe}: {mt5.last_error()}")
    result = []
    for r in rates:
        result.append({
            "time":        int(r["time"]),
            "open":        float(r["open"]),
            "high":        float(r["high"]),
            "low":         float(r["low"]),
            "close":       float(r["close"]),
            "tick_volume": int(r["tick_volume"]),
            "spread":      int(r["spread"]),
        })
    return result


def _collect_session_windows(symbol: str, now_utc: _dt.datetime, days_ahead: int = 8) -> list[tuple[_dt.datetime, _dt.datetime]]:
    """Collect MT5 trade sessions as UTC windows [open, close)."""
    fn = getattr(mt5, "symbol_info_session_trade", None)
    if fn is None:
        return []
    windows: list[tuple[_dt.datetime, _dt.datetime]] = []
    for day_offset in range(max(1, days_ahead + 1)):
        day_dt = now_utc + _dt.timedelta(days=day_offset - 1)
        day_start = _dt.datetime.combine(day_dt.date(), _dt.time.min, tzinfo=_dt.timezone.utc)
        dow = day_dt.weekday()
        for session_idx in range(12):
            session = fn(symbol, dow, session_idx)
            if session is None:
                break
            try:
                open_sec = int(session[0])
                close_sec = int(session[1])
            except Exception:
                continue
            if open_sec == close_sec:
                continue
            open_dt = day_start + _dt.timedelta(seconds=open_sec)
            close_dt = day_start + _dt.timedelta(seconds=close_sec)
            if close_sec <= open_sec:
                close_dt += _dt.timedelta(days=1)
            windows.append((open_dt, close_dt))

    windows.sort(key=lambda x: x[0])
    dedup: list[tuple[_dt.datetime, _dt.datetime]] = []
    seen: set[tuple[str, str]] = set()
    for open_dt, close_dt in windows:
        key = (open_dt.isoformat(), close_dt.isoformat())
        if key in seen:
            continue
        seen.add(key)
        dedup.append((open_dt, close_dt))
    return dedup


def op_market_state(symbol: str, stale_seconds: int = 300) -> dict:
    """Best-effort broker market state for XAUUSD via MT5 server sessions + tick freshness."""
    _ensure()
    now_utc = _dt.datetime.now(_dt.timezone.utc)
    stale_seconds = max(int(stale_seconds or 300), 30)

    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info failed for {symbol}: {mt5.last_error()}")
    tick = mt5.symbol_info_tick(symbol)

    trade_mode = int(getattr(info, "trade_mode", -1))
    trade_enabled = trade_mode not in (0, 3)  # disabled / close-only

    tick_time_utc: _dt.datetime | None = None
    tick_age_sec: float | None = None
    if tick is not None and getattr(tick, "time", 0):
        tick_time_utc = _dt.datetime.fromtimestamp(int(tick.time), tz=_dt.timezone.utc)
        tick_age_sec = max(0.0, (now_utc - tick_time_utc).total_seconds())
    tick_is_fresh = tick_age_sec is not None and tick_age_sec <= stale_seconds

    sessions = _collect_session_windows(symbol, now_utc, days_ahead=8)
    schedule_is_open: bool | None = None
    next_open_utc: _dt.datetime | None = None
    next_close_utc: _dt.datetime | None = None
    if sessions:
        schedule_is_open = False
        for open_dt, close_dt in sessions:
            if open_dt <= now_utc < close_dt:
                schedule_is_open = True
                next_close_utc = close_dt
                break
            if open_dt > now_utc:
                next_open_utc = open_dt
                break

    is_open = bool(trade_enabled and tick_is_fresh and (schedule_is_open is not False))
    if not trade_enabled:
        reason = "TRADE_DISABLED"
    elif schedule_is_open is False:
        reason = "OUTSIDE_SESSION"
    elif not tick_is_fresh:
        reason = "STALE_TICK"
    else:
        reason = "OPEN"

    mins_to_open = None
    if next_open_utc is not None:
        mins_to_open = int((next_open_utc - now_utc).total_seconds() // 60)
    mins_to_close = None
    if next_close_utc is not None:
        mins_to_close = int((next_close_utc - now_utc).total_seconds() // 60)

    return {
        "symbol": symbol,
        "server_time_utc": now_utc.isoformat(),
        "is_open": is_open,
        "reason": reason,
        "trade_mode": trade_mode,
        "trade_enabled": trade_enabled,
        "tick_time_utc": tick_time_utc.isoformat() if tick_time_utc else None,
        "tick_age_sec": round(float(tick_age_sec), 2) if tick_age_sec is not None else None,
        "tick_is_fresh": bool(tick_is_fresh),
        "stale_seconds": stale_seconds,
        "schedule_available": bool(sessions),
        "schedule_is_open": schedule_is_open,
        "next_open_utc": next_open_utc.isoformat() if next_open_utc else None,
        "next_close_utc": next_close_utc.isoformat() if next_close_utc else None,
        "minutes_to_next_open": mins_to_open,
        "minutes_to_next_close": mins_to_close,
    }


# ── News cache (ForexFactory schedule + TradingView actuals via Windows PowerShell) ──
import threading as _threading
_news_ff_cache: list = []
_news_ff_cache_ts: float = 0.0
_news_ff_fetch_lock = _threading.Lock()
_tv_actuals_cache: list = []          # last successful TradingView actuals (preserved on TV failure)
_tv_actuals_cache_ts: float = 0.0
_TV_ACTUALS_MAX_AGE = 7200.0          # reuse TV actuals up to 2 hours old when TV is unavailable
# ForexFactory JSON — event names, forecast, previous, impact (no actual values — schedule-only feed)
_NEWS_FF_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_thismonth.json",
]
# TradingView economic calendar — has real-time actual values after release
_TV_CAL_URL = "https://economic-calendar.tradingview.com/events"
_TV_CURRENCIES = ["US", "EU", "GB", "JP", "AU", "NZ", "CA", "CH", "CN"]
# category keywords for matching FF events to TV events (by currency + time + category)
_TV_MATCH_KW = [
    "manufacturing", "services", "composite", "cpi", "ppi", "gdp", "nfp",
    "payroll", "employment", "unemployment", "jobless", "retail", "trade",
    "confidence", "sentiment", "michigan", "ism", "housing", "building",
    "durable", "current account", "interest rate", "fomc", "fed", "ecb", "boe",
    "rba", "rbnz", "boc", "snb", "boj", "inflation", "balance",
]
_NEWS_FF_CACHE_TTL = 120.0  # 2 min — aggressive refresh so actuals appear quickly after release


def _fetch_ps_get(url: str) -> object | None:
    """GET a URL via PowerShell (Windows Schannel TLS — bypasses Cloudflare fingerprinting)."""
    import subprocess as _sp, json as _json
    ps_script = (
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
        f"$h=@{{'User-Agent'='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36';"
        f"'Accept'='application/json, */*';'Referer'='https://www.forexfactory.com/';'Accept-Language'='en-US,en;q=0.9'}};"
        f"$r=Invoke-WebRequest '{url}' -Headers $h -UseBasicParsing -TimeoutSec 12;"
        f"Write-Output $r.Content"
    )
    try:
        result = _sp.run(
            ["powershell", "-NonInteractive", "-NoProfile", "-Command", ps_script],
            capture_output=True, timeout=20, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0 or not result.stdout.strip():
            log.debug("PS GET %s failed (rc=%s): %s", url, result.returncode, result.stderr[:200])
            return None
        return _json.loads(result.stdout.strip())
    except Exception as exc:
        log.debug("PS GET %s error: %s", url, exc)
        return None


def _fetch_tv_actuals() -> list:
    """Fetch TradingView economic calendar (POST) — returns events with actual values.
    TV importance: 1=High, 0=Medium, -1=Low/None
    """
    import subprocess as _sp, json as _json, datetime as _dt
    now_utc = _dt.datetime.utcnow()
    from_s  = (now_utc - _dt.timedelta(days=3)).strftime("%Y-%m-%dT00:00:00.000Z")
    to_s    = (now_utc + _dt.timedelta(days=5)).strftime("%Y-%m-%dT23:59:59.000Z")
    countries = _json.dumps(_TV_CURRENCIES)
    body_json = f'{{"from":"{from_s}","to":"{to_s}","countries":{countries}}}'
    ps_script = (
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
        f"$b=[System.Text.Encoding]::UTF8.GetBytes('{body_json}');"
        f"$r=Invoke-WebRequest '{_TV_CAL_URL}' -Method POST -Body $b"
        f" -ContentType 'application/json'"
        f" -Headers @{{'Origin'='https://www.tradingview.com';'Referer'='https://www.tradingview.com/'}}"
        f" -UseBasicParsing -TimeoutSec 12;"
        f"Write-Output $r.Content"
    )
    try:
        result = _sp.run(
            ["powershell", "-NonInteractive", "-NoProfile", "-Command", ps_script],
            capture_output=True, timeout=22, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0 or not result.stdout.strip():
            log.debug("TV actuals fetch failed (rc=%s): %s", result.returncode, result.stderr[:200])
            return []
        data = _json.loads(result.stdout.strip())
        return data.get("result", []) if isinstance(data, dict) else []
    except Exception as exc:
        log.debug("TV actuals fetch error: %s", exc)
        return []


def _merge_tv_actuals(ff_events: list, tv_events: list) -> None:
    """Inject TradingView actual values into ForexFactory events in-place.
    Matches by: same currency + event time within ±5 min + shared category keyword.
    """
    import datetime as _dt, re as _re

    def _parse_utc(s: str) -> "_dt.datetime | None":
        if not s:
            return None
        try:
            ts = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            return ts.astimezone(_dt.timezone.utc)
        except Exception:
            return None

    # Build TV lookup: (currency_upper, 5-min-bucket) → list of (title_lower, actual_str)
    tv_lookup: dict = {}
    for ev in tv_events:
        actual_raw = ev.get("actual")
        if actual_raw is None:
            continue
        actual_str = str(actual_raw).strip()
        if not actual_str:
            continue
        ccy = str(ev.get("currency") or "").upper()
        dt  = _parse_utc(str(ev.get("date") or ""))
        if not ccy or dt is None:
            continue
        # 5-min bucket: key = (currency, doy_min // 5) where doy_min = total minutes from day start
        bucket = (ccy, (dt.hour * 60 + dt.minute) // 5)  # 5-min slot
        tv_lookup.setdefault(bucket, []).append((str(ev.get("title") or "").lower(), actual_str))

    for ff_ev in ff_events:
        if ff_ev.get("actual"):
            continue  # already has actual
        ccy = str(ff_ev.get("country") or ff_ev.get("currency") or "").upper()
        dt  = _parse_utc(str(ff_ev.get("date") or ""))
        if not ccy or dt is None:
            continue
        ff_title_lower = str(ff_ev.get("title") or "").lower()
        # Check ±5 min buckets (one 5-min window on each side)
        center_bucket = dt.hour * 60 + dt.minute
        for offset in (0, -5, 5):
            b = (ccy, (center_bucket + offset) // 5)
            candidates = tv_lookup.get(b, [])
            for tv_title_lower, actual_str in candidates:
                # both must share at least one category keyword
                if any(kw in ff_title_lower and kw in tv_title_lower for kw in _TV_MATCH_KW):
                    ff_ev["actual"] = actual_str
                    break
            if ff_ev.get("actual"):
                break


def op_fetch_news() -> dict:
    """Fetch economic calendar: ForexFactory JSON (schedule/forecast) + TradingView (actuals).
    ForexFactory provides event names familiar to traders.
    TradingView fills in real-time actual values after each event releases.
    Returns: {"events": [...], "source": str, "cached": bool, "ts": float}
    """
    global _news_ff_cache, _news_ff_cache_ts
    now = time.time()
    if _news_ff_cache and now - _news_ff_cache_ts < _NEWS_FF_CACHE_TTL:
        return {"events": _news_ff_cache, "source": "cached", "cached": True, "ts": now}

    with _news_ff_fetch_lock:
        now = time.time()
        if _news_ff_cache and now - _news_ff_cache_ts < _NEWS_FF_CACHE_TTL:
            return {"events": _news_ff_cache, "source": "cached", "cached": True, "ts": now}

        # Step 1: Fetch ForexFactory schedule (event names, forecast, previous, impact)
        ff_events: list = []
        for ff_url in _NEWS_FF_URLS:
            raw = _fetch_ps_get(ff_url)
            if isinstance(raw, list) and raw:
                for ev in raw:
                    ev.setdefault("actual", "")
                ff_events = raw
                log.info("ForexFactory schedule fetched: %d events from %s", len(raw), ff_url)
                break

        if not ff_events:
            # FF completely unavailable — return stale or empty
            if _news_ff_cache:
                log.warning("ForexFactory unavailable — returning stale cache (%d events)", len(_news_ff_cache))
                return {"events": _news_ff_cache, "source": "stale", "cached": True, "ts": _news_ff_cache_ts}
            log.error("ForexFactory fetch failed and no cache available")
            return {"events": [], "source": "error", "cached": False, "ts": now}

        # Step 2: Fetch TradingView actuals (use stale TV cache if TV temporarily unavailable)
        global _tv_actuals_cache, _tv_actuals_cache_ts
        tv_events = _fetch_tv_actuals()
        if tv_events:
            _tv_actuals_cache = tv_events
            _tv_actuals_cache_ts = now
            log.debug("TV actuals fetched: %d events", len(tv_events))
        elif _tv_actuals_cache and now - _tv_actuals_cache_ts < _TV_ACTUALS_MAX_AGE:
            tv_events = _tv_actuals_cache
            log.info("TradingView unavailable — reusing cached TV actuals (%d events, age %.0fs)",
                     len(tv_events), now - _tv_actuals_cache_ts)
        else:
            log.warning("TradingView actuals unavailable — FF events have no actual values this cycle")

        if tv_events:
            _merge_tv_actuals(ff_events, tv_events)
            n_actual = sum(1 for e in ff_events if e.get("actual"))
            log.info("TradingView actuals merged: %d/%d FF events now have actual", n_actual, len(ff_events))

        _news_ff_cache = ff_events
        _news_ff_cache_ts = now
        return {"events": ff_events, "source": "forexfactory+tv", "cached": False, "ts": now}



def op_get_history_deals(symbol: str, since_epoch: float, magic: int | None = None) -> list:
    """Return all OUT deals for symbol since since_epoch (Unix timestamp)."""
    import datetime as _dt
    _ensure()
    from_dt = _dt.datetime.fromtimestamp(since_epoch, tz=_dt.timezone.utc)
    to_dt   = _dt.datetime.now(_dt.timezone.utc)
    deals = mt5.history_deals_get(from_dt, to_dt)
    if not deals:
        return []
    result = []
    for d in deals:
        row = d._asdict()
        if row.get("symbol") != symbol:
            continue
        if magic is not None and row.get("magic") != magic:
            continue
        if row.get("entry", -1) != 1:   # DEAL_ENTRY_OUT == 1
            continue
        close_price = float(row.get("price", 0))
        open_price  = close_price
        try:
            pos_id = int(row.get("position_id", 0))
            if pos_id:
                pos_deals = mt5.history_deals_get(position=pos_id)
                if pos_deals:
                    in_deal = next((pd for pd in pos_deals if pd.entry == 0), None)
                    if in_deal:
                        open_price = float(in_deal.price)
        except Exception:
            pass
        result.append({
            "ticket":      int(row.get("position_id", 0)),
            "deal_ticket": int(row.get("ticket", 0)),
            "side":        "buy" if row.get("type", 1) == 1 else "sell",
            "volume":      float(row.get("volume", 0)),
            "open_price":  open_price,
            "close_price": close_price,
            "profit":      float(row.get("profit", 0)),
            "swap":        float(row.get("swap", 0)),
            "commission":  float(row.get("commission", 0)),
            "open_time":   float(row.get("time_msc", 0)) / 1000,
            "close_time":  float(row.get("time_msc", 0)) / 1000,
            "comment":     str(row.get("comment", "")),
            "reason":      int(row.get("reason", 0)),
        })
    log.info("history_deals symbol=%s since=%.0f magic=%s -> %d deals", symbol, since_epoch, magic, len(result))
    return result


# ── HTTP handler ───────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # silence default access log
        pass

    def _is_allowed(self) -> bool:
        """Return True if the client IP is on an allowed (private/local) network."""
        try:
            addr = ipaddress.ip_address(self.client_address[0])
            return any(addr in net for net in _ALLOWED_NETWORKS)
        except ValueError:
            return False

    def _send_json(self, code: int, obj) -> None:
        data = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        """Read and parse JSON body; return empty dict on missing/invalid body."""
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc

    def do_GET(self):
        if not self._is_allowed():
            log.warning("Blocked request from external IP: %s", self.client_address[0])
            self._send_json(403, {"error": "forbidden"})
            return
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        path = parsed.path
        try:
            if path == "/health":
                self._send_json(200, {"status": "ok", "mt5_connected": mt5.terminal_info() is not None})
            elif path == "/account":
                self._send_json(200, op_get_account())
            elif path == "/positions":
                symbol = qs.get("symbol", ["XAUUSD"])[0]
                magic  = int(qs["magic"][0]) if "magic" in qs else None
                self._send_json(200, op_get_positions(symbol, magic))
            elif path == "/positions/count":
                symbol = qs.get("symbol", ["XAUUSD"])[0]
                magic  = int(qs["magic"][0]) if "magic" in qs else None
                self._send_json(200, {"count": len(op_get_positions(symbol, magic))})
            elif path == "/history/closed":
                symbol = qs.get("symbol", ["XAUUSD"])[0]
                since  = float(qs["since"][0]) if "since" in qs else 0.0
                magic  = int(qs["magic"][0]) if "magic" in qs else None
                self._send_json(200, op_get_history_deals(symbol, since, magic))
            elif path == "/tick":
                symbol = qs.get("symbol", ["XAUUSD"])[0]
                self._send_json(200, op_get_tick(symbol))
            elif path == "/rates":
                symbol    = qs.get("symbol",    ["XAUUSD"])[0]
                timeframe = qs.get("timeframe", ["M15"])[0]
                bars      = int(qs.get("bars",  ["300"])[0])
                self._send_json(200, op_get_rates(symbol, timeframe, bars))
            elif path == "/market/state":
                symbol = qs.get("symbol", ["XAUUSD"])[0]
                stale_seconds = int(qs.get("stale_seconds", ["300"])[0])
                self._send_json(200, op_market_state(symbol, stale_seconds))
            elif path == "/news":
                # Proxy ForexFactory JSON via Windows host IP (avoids Docker NAT rate-limit)
                self._send_json(200, op_fetch_news())
            else:
                self._send_json(404, {"error": "not found"})
        except Exception as exc:
            log.error("GET %s error: %s", path, exc)
            self._send_json(500, {"error": str(exc)})

    def do_POST(self):
        if not self._is_allowed():
            log.warning("Blocked request from external IP: %s", self.client_address[0])
            self._send_json(403, {"error": "forbidden"})
            return
        parsed = urlparse(self.path)
        path   = parsed.path
        try:
            body = self._body()  # moved inside try so JSONDecodeError is caught
            if path == "/order":
                self._send_json(200, op_place_order(body))
            elif "/positions/" in path and path.endswith("/close"):
                ticket = int(path.split("/")[2])
                self._send_json(200, op_close_position(
                    ticket,
                    float(body["volume"]),
                    int(body.get("deviation", 20)),
                    int(body.get("magic", 0)),
                ))
            elif "/positions/" in path and path.endswith("/sl"):
                ticket = int(path.split("/")[2])
                self._send_json(200, op_modify_sl(ticket, float(body["new_sl"])))
            else:
                self._send_json(404, {"error": "not found"})
        except Exception as exc:
            log.error("POST %s error: %s", path, exc)
            self._send_json(500, {"error": str(exc)})


# ── Entry point ────────────────────────────────────────────────────────────────

class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each HTTP request in a separate thread so one blocked MT5 call
    (e.g. copy_rates_from_pos retry loop) cannot freeze the entire server."""
    daemon_threads = True


def _watchdog(interval: int = 60) -> None:
    """Background thread: reconnects MT5 if the terminal disconnects.
    Runs every `interval` seconds. Does not stop the bridge if reconnect fails;
    the next operator request will surface the error via _ensure()."""
    while True:
        time.sleep(interval)
        try:
            if mt5.terminal_info() is None:
                log.warning("Watchdog: MT5 terminal_info() is None — attempting reconnect...")
                if _init_mt5():
                    log.info("Watchdog: MT5 reconnected OK")
                else:
                    log.error("Watchdog: MT5 reconnect FAILED — will retry next cycle")
            else:
                # Also check that connected account matches expected
                expected = int(os.getenv("MT5_LOGIN", "0"))
                if expected:
                    info = mt5.account_info()
                    if info and info.login != expected:
                        log.warning(
                            "Watchdog: account mismatch expected=%s got=%s — reconnecting",
                            expected, info.login
                        )
                        mt5.shutdown()
                        _init_mt5()
        except Exception as exc:
            log.error("Watchdog error: %s", exc)


if __name__ == "__main__":
    print(f"MT5 Bridge starting on port {PORT} ...")
    if not _init_mt5():
        print("WARNING: MT5 init failed at startup — will retry on first request.")
    # Start background MT5 watchdog
    _wt = threading.Thread(target=_watchdog, kwargs={"interval": 60}, daemon=True, name="mt5-watchdog")
    _wt.start()
    # Bind to all interfaces so Docker containers on host.docker.internal can reach us.
    # External traffic is blocked by the _is_allowed() IP-allowlist in the handler.
    server = _ThreadedHTTPServer(("0.0.0.0", PORT), _Handler)
    print(f"MT5 Bridge ready -> http://localhost:{PORT}/health")
    print("Keep this window open while live bots are running.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMT5 Bridge stopped.")
        mt5.shutdown()
