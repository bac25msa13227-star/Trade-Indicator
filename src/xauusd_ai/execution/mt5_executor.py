from __future__ import annotations

import datetime as _dt
import json as _json
import os
import urllib.error as _urlerr
import urllib.request as _urllib
from typing import Any

from xauusd_ai.config import Settings
from xauusd_ai.execution.risk import OrderPlan

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# When running inside a Linux Docker container MetaTrader5 is unavailable.
# Set MT5_BRIDGE_URL=http://host.docker.internal:5600 to delegate all MT5 calls
# to the bridge process running natively on the Windows host.
_BRIDGE_URL: str = os.getenv("MT5_BRIDGE_URL", "").rstrip("/")


def _bridge_call(method: str, path: str, body: dict | None = None) -> Any:
    """Send a JSON request to the MT5 bridge and return the parsed response."""
    url = _BRIDGE_URL + path
    data = _json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = _urllib.Request(url, data=data, method=method, headers=headers)
    try:
        with _urllib.urlopen(req, timeout=10) as resp:
            result = _json.loads(resp.read())
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(f"MT5 bridge error: {result['error']}")
            return result
    except _urlerr.URLError as exc:
        raise RuntimeError(f"MT5 bridge unreachable at {url}: {exc}") from exc


class MT5Executor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _ensure_connection(self) -> None:
        if mt5 is None:
            if _BRIDGE_URL:
                return  # all calls delegated to bridge; no local connection needed
            raise RuntimeError("MetaTrader5 package is not installed in this environment")

        if self.settings.integrations.mt5.enabled:
            mt5cfg = self.settings.integrations.mt5
            login = str(mt5cfg.login) if mt5cfg.login else os.getenv(mt5cfg.login_env)
            password = mt5cfg.password or os.getenv(mt5cfg.password_env)
            server = mt5cfg.server or os.getenv(mt5cfg.server_env)
            path = mt5cfg.terminal_path or None
            if login and password and server:
                kwargs: dict = dict(login=int(login), password=password, server=server)
                if path:
                    kwargs["path"] = path
                if not mt5.initialize(**kwargs):
                    raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
                return

        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    def connection_status(self) -> dict[str, Any]:
        self._ensure_connection()
        symbol = self.settings.market.symbol
        account_info = mt5.account_info() if mt5 is not None else None
        terminal_info = mt5.terminal_info() if mt5 is not None else None
        symbol_info = mt5.symbol_info(symbol) if mt5 is not None else None
        tick_info = mt5.symbol_info_tick(symbol) if mt5 is not None else None
        if symbol_info is None:
            raise RuntimeError(f"MT5 symbol_info failed for {symbol}: {mt5.last_error()}")
        return {
            "connected": True,
            "symbol": symbol,
            "account": account_info._asdict() if account_info is not None else None,
            "terminal": terminal_info._asdict() if terminal_info is not None else None,
            "symbol_info": symbol_info._asdict(),
            "tick": tick_info._asdict() if tick_info is not None else None,
        }

    def get_account_info(self) -> dict[str, Any]:
        """Lấy thông tin tài khoản: balance, equity, margin, free_margin."""
        try:
            self._ensure_connection()
        except Exception:
            return {"balance": 0.0, "equity": 0.0, "margin": 0.0, "free_margin": 0.0, "connected": False}
        if mt5 is None:
            if _BRIDGE_URL:
                return _bridge_call("GET", "/account")
            return {"balance": 0.0, "equity": 0.0, "margin": 0.0, "free_margin": 0.0, "connected": False}
        info = mt5.account_info()
        if info is None:
            return {"balance": 0.0, "equity": 0.0, "margin": 0.0, "free_margin": 0.0, "connected": False}
        d = info._asdict()
        return {
            "balance": float(d.get("balance", 0)),
            "equity": float(d.get("equity", 0)),
            "margin": float(d.get("margin", 0)),
            "free_margin": float(d.get("margin_free", 0)),
            "leverage": int(d.get("leverage", 100)),
            "currency": str(d.get("currency", "USD")),
            "name": str(d.get("name", "")),
            "connected": True,
        }

    def get_tick(self, symbol: str | None = None) -> dict[str, Any]:
        """Return latest tick in normalized shape for monitor/diagnostics."""
        symbol = symbol or self.settings.market.symbol
        try:
            self._ensure_connection()
        except Exception as exc:
            return {
                "symbol": symbol,
                "bid": 0.0,
                "ask": 0.0,
                "time": None,
                "source": "none",
                "error": str(exc),
            }
        if mt5 is None:
            if _BRIDGE_URL:
                try:
                    raw = _bridge_call("GET", f"/tick?symbol={symbol}")
                    return {
                        "symbol": symbol,
                        "bid": float(raw.get("bid", 0.0) or 0.0),
                        "ask": float(raw.get("ask", 0.0) or 0.0),
                        "time": raw.get("time"),
                        "source": "bridge",
                        "error": "",
                    }
                except Exception as exc:
                    return {
                        "symbol": symbol,
                        "bid": 0.0,
                        "ask": 0.0,
                        "time": None,
                        "source": "bridge",
                        "error": str(exc),
                    }
            return {
                "symbol": symbol,
                "bid": 0.0,
                "ask": 0.0,
                "time": None,
                "source": "none",
                "error": "MT5_UNAVAILABLE",
            }
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {
                "symbol": symbol,
                "bid": 0.0,
                "ask": 0.0,
                "time": None,
                "source": "mt5",
                "error": str(mt5.last_error()),
            }
        return {
            "symbol": symbol,
            "bid": float(getattr(tick, "bid", 0.0) or 0.0),
            "ask": float(getattr(tick, "ask", 0.0) or 0.0),
            "time": int(getattr(tick, "time", 0) or 0),
            "source": "mt5",
            "error": "",
        }

    def get_current_price(self, symbol: str, side: str) -> float:
        """
        Lấy giá thị trường real-time từ MT5: ask cho BUY, bid cho SELL.
        Dùng để rebase entry/SL/TP từ giá bar close sang giá tick hiện tại
        trước khi gửi Telegram notification và đặt lệnh.
        """
        try:
            self._ensure_connection()
        except Exception:
            return 0.0
        if mt5 is None:
            if _BRIDGE_URL:
                try:
                    resp = _bridge_call("GET", f"/tick?symbol={symbol}")
                    return float(resp.get("ask" if side == "buy" else "bid", 0))
                except Exception:
                    return 0.0
            return 0.0
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return 0.0
        return float(tick.ask) if side == "buy" else float(tick.bid)

    def _collect_session_windows(
        self,
        symbol: str,
        now_utc: _dt.datetime,
        days_ahead: int = 8,
    ) -> list[tuple[_dt.datetime, _dt.datetime]]:
        fn = getattr(mt5, "symbol_info_session_trade", None)
        if fn is None:
            return []
        windows: list[tuple[_dt.datetime, _dt.datetime]] = []
        for day_offset in range(max(1, days_ahead + 1)):
            day_dt = now_utc + _dt.timedelta(days=day_offset - 1)
            day_start = _dt.datetime.combine(day_dt.date(), _dt.time.min, tzinfo=_dt.timezone.utc)
            dow = day_dt.weekday()
            for idx in range(12):
                session = fn(symbol, dow, idx)
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

    def get_market_state(self, symbol: str | None = None, stale_seconds: int = 300) -> dict[str, Any]:
        """Return broker-aware market open/closed state.

        Uses MT5 bridge `/market/state` when available; otherwise uses local MT5.
        """
        symbol = symbol or self.settings.market.symbol
        stale_seconds = max(int(stale_seconds or 300), 30)

        try:
            self._ensure_connection()
        except Exception as exc:
            return {
                "symbol": symbol,
                "is_open": False,
                "reason": f"CONNECTION_ERROR:{exc}",
                "stale_seconds": stale_seconds,
            }

        if mt5 is None:
            if _BRIDGE_URL:
                try:
                    return _bridge_call(
                        "GET",
                        f"/market/state?symbol={symbol}&stale_seconds={stale_seconds}",
                    )
                except Exception as exc:
                    return {
                        "symbol": symbol,
                        "is_open": False,
                        "reason": f"BRIDGE_ERROR:{exc}",
                        "stale_seconds": stale_seconds,
                    }
            return {
                "symbol": symbol,
                "is_open": False,
                "reason": "MT5_UNAVAILABLE",
                "stale_seconds": stale_seconds,
            }

        now_utc = _dt.datetime.now(_dt.timezone.utc)
        info = mt5.symbol_info(symbol)
        if info is None:
            return {
                "symbol": symbol,
                "is_open": False,
                "reason": f"SYMBOL_INFO_ERROR:{mt5.last_error()}",
                "stale_seconds": stale_seconds,
            }
        tick = mt5.symbol_info_tick(symbol)

        trade_mode = int(getattr(info, "trade_mode", -1))
        trade_enabled = trade_mode not in (0, 3)  # disabled / close-only

        tick_time_utc: _dt.datetime | None = None
        tick_age_sec: float | None = None
        if tick is not None and getattr(tick, "time", 0):
            tick_time_utc = _dt.datetime.fromtimestamp(int(tick.time), tz=_dt.timezone.utc)
            tick_age_sec = max(0.0, (now_utc - tick_time_utc).total_seconds())
        tick_is_fresh = tick_age_sec is not None and tick_age_sec <= stale_seconds

        sessions = self._collect_session_windows(symbol, now_utc, days_ahead=8)
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

    def get_open_positions_count(self, magic_number: int | None = None) -> int:
        """Đếm số lệnh đang mở (theo magic number nếu có)."""
        try:
            self._ensure_connection()
        except Exception:
            return 0
        if mt5 is None:
            if _BRIDGE_URL:
                symbol = self.settings.market.symbol
                qs = f"/positions/count?symbol={symbol}"
                if magic_number is not None:
                    qs += f"&magic={magic_number}"
                return int(_bridge_call("GET", qs).get("count", 0))
            return 0
        if magic_number is not None:
            positions = mt5.positions_get(symbol=self.settings.market.symbol)
            if positions is None:
                return 0
            return sum(1 for p in positions if p.magic == magic_number)
        positions = mt5.positions_get(symbol=self.settings.market.symbol)
        return 0 if positions is None else len(positions)

    def get_open_positions(self, magic_number: int | None = None) -> list[dict[str, Any]]:
        """
        Lấy danh sách đầy đủ các lệnh đang mở.
        Mỗi lệnh trả về dict gồm: ticket, side, volume, open_price,
        current_price, sl, tp, profit, swap, magic, comment, open_time.
        """
        try:
            self._ensure_connection()
        except Exception:
            return []
        if mt5 is None:
            if _BRIDGE_URL:
                symbol = self.settings.market.symbol
                qs = f"/positions?symbol={symbol}"
                if magic_number is not None:
                    qs += f"&magic={magic_number}"
                return _bridge_call("GET", qs)
            return []
        positions = mt5.positions_get(symbol=self.settings.market.symbol)
        if not positions:
            return []
        result = []
        for p in positions:
            if magic_number is not None and p.magic != magic_number:
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
                "open_time":     d.get("time", 0),
            })
        return result

    def close_position(self, ticket: int, volume: float) -> dict[str, Any]:
        """
        Đóng một lệnh đang mở theo ticket bằng cách gửi lệnh ngược chiều.
        Dùng TRADE_ACTION_DEAL với position=ticket.
        """
        self._ensure_connection()
        if mt5 is None:
            if _BRIDGE_URL:
                return _bridge_call("POST", f"/positions/{ticket}/close", {
                    "volume": volume,
                    "deviation": self.settings.execution.deviation,
                    "magic": self.settings.execution.magic_number,
                })
            raise RuntimeError("MetaTrader5 not available")
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            raise RuntimeError(f"Position {ticket} not found")
        pos = positions[0]
        # Lệnh đóng: ngược chiều với lệnh đang mở
        # BUY (type=0) → đóng bằng SELL; SELL (type=1) → đóng bằng BUY
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            raise RuntimeError(f"No tick data for {pos.symbol}")
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        filling_modes = [
            mt5.ORDER_FILLING_RETURN,
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_FOK,
        ]
        last_error = None
        for filling in filling_modes:
            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       pos.symbol,
                "volume":       volume,
                "type":         close_type,
                "position":     ticket,
                "price":        price,
                "deviation":    self.settings.execution.deviation,
                "magic":        self.settings.execution.magic_number,
                "comment":      "close-opposite",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            result = mt5.order_send(request)
            if result is None:
                last_error = f"order_send None: {mt5.last_error()}"
                continue
            r = result._asdict()
            if r.get("retcode") == 10009:
                return r
            if r.get("retcode") == 10030:
                last_error = f"INVALID_FILL filling={filling}"
                continue
            raise RuntimeError(
                f"close_position failed: retcode={r.get('retcode')} "
                f"comment={r.get('comment')} ticket={ticket}"
            )
        raise RuntimeError(f"close_position failed all filling modes: {last_error}")

    def modify_position_sl(self, ticket: int, new_sl: float) -> dict[str, Any]:
        """
        Sửa Stop Loss của một lệnh đang mở.
        Giữ nguyên TP. Dùng TRADE_ACTION_SLTP.
        """
        self._ensure_connection()
        if mt5 is None:
            if _BRIDGE_URL:
                return _bridge_call("POST", f"/positions/{ticket}/sl", {"new_sl": new_sl})
            raise RuntimeError("MetaTrader5 not available")
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            raise RuntimeError(f"Position {ticket} not found")
        pos = positions[0]
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   pos.symbol,
            "position": ticket,
            "sl":       new_sl,
            "tp":       pos.tp,
        }
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"modify_position_sl failed: {mt5.last_error()}")
        r = result._asdict()
        if r.get("retcode") != 10009:
            raise RuntimeError(
                f"modify_position_sl: retcode={r.get('retcode')} "
                f"comment={r.get('comment')} ticket={ticket} new_sl={new_sl}"
            )
        return r

    def place_order(self, plan: OrderPlan) -> dict[str, Any]:
        self._ensure_connection()
        if mt5 is None:
            if _BRIDGE_URL:
                return _bridge_call("POST", "/order", {
                    "symbol":      plan.symbol,
                    "side":        plan.side,
                    "volume":      plan.volume,
                    "stop_loss":   plan.stop_loss,
                    "take_profit": plan.take_profit,
                    "entry_price": plan.entry_price,
                    "deviation":   self.settings.execution.deviation,
                    "magic":       self.settings.execution.magic_number,
                    "comment":     self.settings.execution.comment,
                })
            raise RuntimeError("MetaTrader5 not available")
        # Log trạng thái AutoTrading để debug
        _tinfo = mt5.terminal_info()
        if _tinfo is not None:
            import logging as _log
            _log.getLogger(__name__).info(
                "MT5 terminal: trade_allowed=%s path=%s account=%s",
                _tinfo.trade_allowed,
                getattr(_tinfo, "path", "?"),
                mt5.account_info().login if mt5.account_info() else "?"
            )
        tick = mt5.symbol_info_tick(plan.symbol)
        if tick is None:
            raise RuntimeError(f"No tick data for {plan.symbol}")

        # P1b: Spread gate — reject order if spread too wide
        _spread_pts = tick.ask - tick.bid
        _max_spread = float(getattr(self.settings.risk, 'max_spread_points', 0))
        if _max_spread > 0 and _spread_pts > _max_spread:
            import logging as _log
            _log.getLogger(__name__).warning(
                "Spread %.1f > max %.1f — order rejected for %s",
                _spread_pts, _max_spread, plan.symbol,
            )
            return None

        # Round to symbol's decimal precision
        _sinfo = mt5.symbol_info(plan.symbol)
        _digits = _sinfo.digits if _sinfo is not None else 5

        order_type = mt5.ORDER_TYPE_BUY if plan.side == "buy" else mt5.ORDER_TYPE_SELL
        price = tick.ask if plan.side == "buy" else tick.bid

        # Re-anchor SL/TP về giá fill thực tế (giống mt5_bridge)
        # Tránh sai lệch khi entry_price từ bar close lệch xa giá fill thực tế
        if plan.entry_price and plan.entry_price > 0 and plan.stop_loss and plan.take_profit:
            _sl_dist = abs(plan.entry_price - plan.stop_loss)
            _tp_dist = abs(plan.take_profit - plan.entry_price)
            if plan.side == "buy":
                _sl = round(price - _sl_dist, _digits)
                _tp = round(price + _tp_dist, _digits)
            else:
                _sl = round(price + _sl_dist, _digits)
                _tp = round(price - _tp_dist, _digits)
        else:
            _sl = round(plan.stop_loss, _digits) if plan.stop_loss else 0.0
            _tp = round(plan.take_profit, _digits) if plan.take_profit else 0.0

        # Thử các filling mode theo thứ tự ưu tiên (Exness thường dùng RETURN)
        filling_modes = [
            mt5.ORDER_FILLING_RETURN,
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_FOK,
        ]

        last_error = None
        for filling in filling_modes:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": plan.symbol,
                "volume": plan.volume,
                "type": order_type,
                "price": price,
                "sl": _sl,
                "tp": _tp,
                "deviation": self.settings.execution.deviation,
                "magic": self.settings.execution.magic_number,
                "comment": self.settings.execution.comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            result = mt5.order_send(request)
            if result is None:
                last_error = f"order_send returned None: {mt5.last_error()}"
                continue
            r = result._asdict()
            # retcode 10009 = TRADE_RETCODE_DONE (success)
            # retcode 10030 = TRADE_RETCODE_INVALID_FILL → thử filling khác
            if r.get("retcode") == 10009:
                return r
            if r.get("retcode") == 10027:
                raise RuntimeError(
                    "MT5 AutoTrading bị tắt! Vào MT5 → bật nút 'Algo Trading' trên toolbar "
                    "(icon robot). retcode=10027"
                )
            if r.get("retcode") == 10030:
                last_error = f"retcode={r.get('retcode')} INVALID_FILL with filling={filling}, trying next"
                continue
            # Lỗi khác → raise ngay
            raise RuntimeError(
                f"MT5 order_send failed: retcode={r.get('retcode')} comment={r.get('comment')} "
                f"volume={plan.volume} symbol={plan.symbol}"
            )

        raise RuntimeError(f"MT5 order_send failed after all filling modes: {last_error}")

    def get_recently_closed_positions(
        self,
        since_epoch: float,
        magic_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Lấy danh sách các lệnh đã đóng kể từ `since_epoch` (Unix timestamp).
        Trả về list dict gồm: ticket, side, volume, open_price, close_price,
        profit, swap, commission, open_time, close_time, comment.
        Lệnh thua: profit + swap + commission < 0.
        """
        try:
            self._ensure_connection()
        except Exception:
            return []
        if mt5 is None:
            if _BRIDGE_URL:
                symbol = self.settings.market.symbol
                qs = f"/history/closed?symbol={symbol}&since={since_epoch}"
                if magic_number is not None:
                    qs += f"&magic={magic_number}"
                try:
                    return _bridge_call("GET", qs)
                except Exception:
                    return []
            return []

        import datetime as _dt
        from_dt = _dt.datetime.fromtimestamp(since_epoch, tz=_dt.timezone.utc)
        to_dt = _dt.datetime.now(_dt.timezone.utc)

        deals = mt5.history_deals_get(from_dt, to_dt)
        if not deals:
            return []

        symbol = self.settings.market.symbol
        result: list[dict[str, Any]] = []
        for d in deals:
            row = d._asdict()
            if row.get("symbol") != symbol:
                continue
            if magic_number is not None and row.get("magic") != magic_number:
                continue
            # Chỉ lấy deal OUT (đóng lệnh), bỏ qua IN (mở lệnh) và BALANCE
            entry = row.get("entry", -1)
            if entry != 1:   # mt5.DEAL_ENTRY_OUT == 1
                continue
            # Fetch open price from the corresponding IN deal for this position
            close_price = float(row.get("price", 0))
            open_price = close_price  # fallback
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
                "side":        "buy" if row.get("type", 1) == 1 else "sell",  # DEAL_TYPE_BUY=0, SELL=1 → reversed for close
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
        return result
