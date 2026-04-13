#!/usr/bin/env python3
r"""
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
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

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

PORT = int(os.getenv("MT5_BRIDGE_PORT", "5600"))

_TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

# ── Symbol remapping ───────────────────────────────────────────────────────────
# Nếu bot request XAUUSDm nhưng terminal chỉ có XAUUSD (Exness Trial), tự remap.
_SYMBOL_REMAP: dict[str, str] = {}

def _resolve_symbol(symbol: str) -> str:
    """Return the actual symbol name available in this MT5 terminal."""
    if symbol in _SYMBOL_REMAP:
        return _SYMBOL_REMAP[symbol]
    # Try as-is first
    if mt5.symbol_select(symbol, True):
        info = mt5.symbol_info(symbol)
        if info is not None:
            _SYMBOL_REMAP[symbol] = symbol
            return symbol
    # Try common variants
    for candidate in (symbol.replace("m", ""), symbol + "m", symbol.replace(".", "")):
        if candidate == symbol:
            continue
        if mt5.symbol_select(candidate, True):
            info = mt5.symbol_info(candidate)
            if info is not None:
                log.info("Symbol remap: %s -> %s", symbol, candidate)
                _SYMBOL_REMAP[symbol] = candidate
                return candidate
    # Return original and let caller handle the error
    return symbol


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
    return True


def _ensure() -> None:
    """Re-initialize if connection was dropped."""
    if mt5.terminal_info() is None:
        if not _init_mt5():
            raise RuntimeError(f"MT5 not connected: {mt5.last_error()}")


# ── Operations ─────────────────────────────────────────────────────────────────

def op_place_order(body: dict) -> dict:
    _ensure()
    symbol    = _resolve_symbol(str(body["symbol"]))
    side      = str(body["side"])
    volume    = float(body["volume"])
    stop_loss = float(body.get("stop_loss") or 0)
    take_profit = float(body.get("take_profit") or 0)
    entry_price = float(body.get("entry_price") or 0)
    deviation = int(body.get("deviation", 20))
    magic     = int(body.get("magic", 0))
    comment   = str(body.get("comment", "mt5-bridge"))[:29]  # MT5 practical limit 29 chars

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
    symbol = _resolve_symbol(symbol)
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
    symbol = _resolve_symbol(symbol)
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


def op_get_bars(symbol: str, timeframe: str, count: int, start_pos: int = 0) -> list[dict]:
    """Return OHLCV bars directly from MT5 terminal history."""
    _ensure()
    symbol = _resolve_symbol(symbol)
    tf_name = str(timeframe or "M1").upper()
    if tf_name not in _TIMEFRAME_MAP:
        raise RuntimeError(f"Unsupported timeframe: {timeframe}")

    count = max(1, min(int(count or 0), 200_000))
    start_pos = max(0, int(start_pos or 0))

    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"symbol_select failed for {symbol}: {mt5.last_error()}")

    rates = None
    for _attempt in range(8):
        rates = mt5.copy_rates_from_pos(symbol, _TIMEFRAME_MAP[tf_name], start_pos, count)
        if rates is not None and len(rates) > 0:
            break
        import time as _time
        _time.sleep(min(2 * (1.5 ** _attempt), 10))

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No rates returned for {symbol} {tf_name}")

    names = rates.dtype.names or ()
    out: list[dict] = []
    for row in rates:
        out.append(
            {
                "time": int(row["time"]) if "time" in names else 0,
                "open": float(row["open"]) if "open" in names else 0.0,
                "high": float(row["high"]) if "high" in names else 0.0,
                "low": float(row["low"]) if "low" in names else 0.0,
                "close": float(row["close"]) if "close" in names else 0.0,
                "tick_volume": float(row["tick_volume"]) if "tick_volume" in names else 0.0,
                "spread": float(row["spread"]) if "spread" in names else 0.0,
                "real_volume": float(row["real_volume"]) if "real_volume" in names else 0.0,
            }
        )
    return out


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
    symbol = _resolve_symbol(symbol)
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


def op_get_history_deals(symbol: str, since_epoch: float, magic: int | None = None) -> list:
    """Return all OUT deals for symbol since since_epoch (Unix timestamp)."""
    import datetime as _dt
    _ensure()
    symbol = _resolve_symbol(symbol)
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

    def _send_json(self, code: int, obj) -> None:
        data = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_GET(self):
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
            elif path == "/bars":
                symbol = qs.get("symbol", ["XAUUSD"])[0]
                timeframe = qs.get("timeframe", ["M1"])[0]
                count = int(qs.get("count", ["500"])[0])
                start_pos = int(qs.get("start_pos", ["0"])[0])
                self._send_json(200, op_get_bars(symbol, timeframe, count, start_pos))
            elif path == "/market/state":
                symbol = qs.get("symbol", ["XAUUSD"])[0]
                stale_seconds = int(qs.get("stale_seconds", ["300"])[0])
                self._send_json(200, op_market_state(symbol, stale_seconds))
            else:
                self._send_json(404, {"error": "not found"})
        except Exception as exc:
            log.error("GET %s error: %s", path, exc)
            self._send_json(500, {"error": str(exc)})

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        body   = self._body()
        try:
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

if __name__ == "__main__":
    print(f"MT5 Bridge starting on port {PORT} ...")
    if not _init_mt5():
        print("WARNING: MT5 init failed at startup — will retry on first request.")
    server = HTTPServer(("0.0.0.0", PORT), _Handler)
    print(f"MT5 Bridge ready -> http://localhost:{PORT}/health")
    print("Keep this window open while live bots are running.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMT5 Bridge stopped.")
        mt5.shutdown()
