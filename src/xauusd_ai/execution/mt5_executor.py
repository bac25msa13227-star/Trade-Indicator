from __future__ import annotations

import os
from typing import Any

from xauusd_ai.config import Settings
from xauusd_ai.execution.risk import OrderPlan

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


class MT5Executor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _ensure_connection(self) -> None:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package is not installed in this environment")

        if self.settings.integrations.mt5.enabled:
            login = os.getenv(self.settings.integrations.mt5.login_env)
            password = os.getenv(self.settings.integrations.mt5.password_env)
            server = os.getenv(self.settings.integrations.mt5.server_env)
            if login and password and server:
                if not mt5.initialize(login=int(login), password=password, server=server):
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

    def get_open_positions_count(self, magic_number: int | None = None) -> int:
        """Đếm số lệnh đang mở (theo magic number nếu có)."""
        try:
            self._ensure_connection()
        except Exception:
            return 0
        if mt5 is None:
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

    def modify_position_sl(self, ticket: int, new_sl: float) -> dict[str, Any]:
        """
        Sửa Stop Loss của một lệnh đang mở.
        Giữ nguyên TP. Dùng TRADE_ACTION_SLTP.
        """
        self._ensure_connection()
        if mt5 is None:
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
        tick = mt5.symbol_info_tick(plan.symbol)
        if tick is None:
            raise RuntimeError(f"No tick data for {plan.symbol}")

        order_type = mt5.ORDER_TYPE_BUY if plan.side == "buy" else mt5.ORDER_TYPE_SELL
        price = tick.ask if plan.side == "buy" else tick.bid

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
                "sl": plan.stop_loss,
                "tp": plan.take_profit,
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
            result.append({
                "ticket":      int(row.get("position_id", 0)),
                "deal_ticket": int(row.get("ticket", 0)),
                "side":        "buy" if row.get("type", 1) == 1 else "sell",  # DEAL_TYPE_BUY=0, SELL=1 → reversed for close
                "volume":      float(row.get("volume", 0)),
                "open_price":  float(row.get("price", 0)),  # close deal price; open_price from history_orders
                "close_price": float(row.get("price", 0)),
                "profit":      float(row.get("profit", 0)),
                "swap":        float(row.get("swap", 0)),
                "commission":  float(row.get("commission", 0)),
                "open_time":   float(row.get("time_msc", 0)) / 1000,
                "close_time":  float(row.get("time_msc", 0)) / 1000,
                "comment":     str(row.get("comment", "")),
                "reason":      int(row.get("reason", 0)),
            })
        return result
