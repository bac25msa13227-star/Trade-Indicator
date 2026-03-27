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
