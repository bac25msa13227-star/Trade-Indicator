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
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

        if self.settings.integrations.mt5.enabled:
            login = os.getenv(self.settings.integrations.mt5.login_env)
            password = os.getenv(self.settings.integrations.mt5.password_env)
            server = os.getenv(self.settings.integrations.mt5.server_env)
            if login and password and server:
                if not mt5.login(login=int(login), password=password, server=server):
                    raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

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

    def place_order(self, plan: OrderPlan) -> dict[str, Any]:
        self._ensure_connection()
        tick = mt5.symbol_info_tick(plan.symbol)
        if tick is None:
            raise RuntimeError(f"No tick data for {plan.symbol}")

        order_type = mt5.ORDER_TYPE_BUY if plan.side == "buy" else mt5.ORDER_TYPE_SELL
        price = tick.ask if plan.side == "buy" else tick.bid
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
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"MT5 order_send failed: {mt5.last_error()}")
        return result._asdict()
