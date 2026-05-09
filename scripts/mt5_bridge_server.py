# MT5 Bridge Server (Run this in Windows VM)
# File: mt5_bridge_server.py
"""
MT5 Bridge Server - Runs in Windows VM
Receives commands from macOS via socket, executes on MT5, returns results
"""
import json
import socket
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import MetaTrader5 as mt5


class MT5BridgeServer:
    """
    Socket server running in Windows VM
    Receives commands from macOS Python bot, executes on MT5
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 7777):
        """
        Args:
            host: Bind to all interfaces (0.0.0.0 for VM access)
            port: Socket port (7777)
        """
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return False
        
        print("✅ MT5 initialized successfully")
        
        # Print account info
        account_info = mt5.account_info()
        if account_info:
            print(f"   Account: {account_info.login}")
            print(f"   Balance: ${account_info.balance:.2f}")
            print(f"   Server: {account_info.server}")
        
        return True
    
    def shutdown_mt5(self):
        """Shutdown MT5 connection"""
        mt5.shutdown()
        print("🔌 MT5 connection closed")
    
    def start(self):
        """Start socket server"""
        # Initialize MT5 first
        if not self.initialize_mt5():
            return
        
        try:
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            
            self.running = True
            print(f"🚀 MT5 Bridge Server started on {self.host}:{self.port}")
            print("   Waiting for connections from macOS...")
            
            while self.running:
                try:
                    client_socket, address = self.socket.accept()
                    print(f"✅ Client connected: {address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.start()
                    
                except KeyboardInterrupt:
                    print("\n⚠️  Shutting down server...")
                    break
                except Exception as e:
                    print(f"❌ Accept error: {e}")
                    
        finally:
            self.stop()
    
    def stop(self):
        """Stop server"""
        self.running = False
        if self.socket:
            self.socket.close()
        self.shutdown_mt5()
        print("👋 Server stopped")
    
    def handle_client(self, client_socket: socket.socket, address):
        """Handle client connection"""
        try:
            while True:
                # Receive data
                data = client_socket.recv(65536)
                if not data:
                    break
                
                # Parse request
                request_str = data.decode('utf-8').strip()
                request = json.loads(request_str)
                
                print(f"📨 Command: {request.get('command')} from {address}")
                
                # Execute command
                response = self.execute_command(request)
                
                # Send response
                response_str = json.dumps(response) + "\n"
                client_socket.sendall(response_str.encode('utf-8'))
                
        except Exception as e:
            print(f"❌ Client error: {e}")
        finally:
            client_socket.close()
            print(f"🔌 Client disconnected: {address}")
    
    def execute_command(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MT5 command and return result"""
        command = request.get("command")
        params = request.get("params", {})
        
        try:
            if command == "get_data":
                return self._get_historical_data(params)
            
            elif command == "place_order":
                return self._place_order(params)
            
            elif command == "get_account":
                return self._get_account_info()
            
            elif command == "get_positions":
                return self._get_positions()
            
            elif command == "close_position":
                return self._close_position(params)
            
            else:
                return {"success": False, "error": f"Unknown command: {command}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_historical_data(self, params: Dict) -> Dict:
        """Fetch historical data from MT5"""
        symbol = params.get("symbol", "XAUUSD")
        timeframe_str = params.get("timeframe", "M1")
        count = params.get("count", 1000)
        
        # Map timeframe string to MT5 constant
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        timeframe = timeframe_map.get(timeframe_str, mt5.TIMEFRAME_M1)
        
        # Fetch bars
        bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        
        if bars is None:
            return {"success": False, "error": f"Failed to fetch data: {mt5.last_error()}"}
        
        # Convert to list of dicts
        data = []
        for bar in bars:
            data.append({
                "time": datetime.fromtimestamp(bar[0]).isoformat(),
                "open": float(bar[1]),
                "high": float(bar[2]),
                "low": float(bar[3]),
                "close": float(bar[4]),
                "volume": int(bar[5]),
            })
        
        return {"success": True, "data": data}
    
    def _place_order(self, params: Dict) -> Dict:
        """Place market order"""
        symbol = params.get("symbol", "XAUUSD")
        order_type_str = params.get("type", "BUY")
        volume = params.get("volume", 0.01)
        price = params.get("price")
        sl = params.get("sl", 0.0)
        tp = params.get("tp", 0.0)
        comment = params.get("comment", "")
        
        # Map order type
        order_type = mt5.ORDER_TYPE_BUY if order_type_str == "BUY" else mt5.ORDER_TYPE_SELL
        
        # Get current price if not provided
        if not price:
            tick = mt5.symbol_info_tick(symbol)
            price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "success": False,
                "error": f"Order failed: {result.comment}",
                "retcode": result.retcode
            }
        
        return {
            "success": True,
            "data": {
                "ticket": result.order,
                "volume": result.volume,
                "price": result.price,
                "comment": result.comment
            }
        }
    
    def _get_account_info(self) -> Dict:
        """Get account info"""
        account = mt5.account_info()
        if not account:
            return {"success": False, "error": "Failed to get account info"}
        
        return {
            "success": True,
            "data": {
                "login": account.login,
                "balance": account.balance,
                "equity": account.equity,
                "margin": account.margin,
                "margin_free": account.margin_free,
                "profit": account.profit,
                "server": account.server,
            }
        }
    
    def _get_positions(self) -> Dict:
        """Get open positions"""
        positions = mt5.positions_get()
        if positions is None:
            return {"success": False, "error": "Failed to get positions"}
        
        data = []
        for pos in positions:
            data.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": pos.volume,
                "price": pos.price_open,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
                "comment": pos.comment,
            })
        
        return {"success": True, "data": data}
    
    def _close_position(self, params: Dict) -> Dict:
        """Close position by ticket"""
        ticket = params.get("ticket")
        
        # Get position info
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"success": False, "error": f"Position {ticket} not found"}
        
        position = positions[0]
        
        # Build close request
        close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": ticket,
            "deviation": 20,
            "magic": 234000,
            "comment": "Close by bridge",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(close_request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "error": f"Close failed: {result.comment}"}
        
        return {"success": True, "data": {"closed_ticket": ticket}}


# ============================================
# Run Server
# ============================================
if __name__ == '__main__':
    print("="*70)
    print("🌉 MT5 Bridge Server for Windows VM")
    print("="*70)
    print()
    
    server = MT5BridgeServer(host="0.0.0.0", port=7777)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        server.stop()
