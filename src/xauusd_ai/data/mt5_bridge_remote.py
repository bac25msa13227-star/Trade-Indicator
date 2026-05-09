#!/usr/bin/env python3
"""
MT5 Bridge - Connect from macOS to MT5 running in Windows VM
Uses socket connection to send/receive data
"""
import json
import socket
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd


class MT5RemoteBridge:
    """
    Bridge to connect from macOS host to MT5 in Windows VM via socket
    
    Architecture:
    macOS (Python bot) --socket--> Windows VM (MT5 bridge server) --API--> MT5 Terminal
    """
    
    def __init__(self, host: str = "localhost", port: int = 7777):
        """
        Args:
            host: Windows VM IP (use 'localhost' if port forwarded, or VM IP like '10.211.55.3')
            port: Socket port (default 7777)
        """
        self.host = host
        self.port = port
        self.socket = None
        
    def connect(self) -> bool:
        """Connect to MT5 bridge server in Windows VM"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"✅ Connected to MT5 bridge at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close connection"""
        if self.socket:
            self.socket.close()
            print("🔌 Disconnected from MT5 bridge")
    
    def _send_command(self, command: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Send command to MT5 bridge and get response"""
        try:
            # Build request
            request = {
                "command": command,
                "params": params,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send JSON
            message = json.dumps(request) + "\n"
            self.socket.sendall(message.encode('utf-8'))
            
            # Receive response
            response_data = self.socket.recv(65536).decode('utf-8')
            response = json.loads(response_data)
            
            return response
            
        except Exception as e:
            print(f"❌ Command failed: {e}")
            return None
    
    def get_historical_data(
        self, 
        symbol: str = "XAUUSD",
        timeframe: str = "M1",
        start_date: datetime = None,
        end_date: datetime = None,
        count: int = 10000
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from MT5
        
        Args:
            symbol: Trading symbol
            timeframe: M1, M5, M15, H1, H4, D1
            start_date: Start datetime
            end_date: End datetime
            count: Number of bars if dates not specified
            
        Returns:
            DataFrame with OHLCV data
        """
        params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "count": count
        }
        
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        
        response = self._send_command("get_data", params)
        
        if response and response.get("success"):
            data = response.get("data", [])
            if data:
                df = pd.DataFrame(data)
                df['time'] = pd.to_datetime(df['time'])
                return df
        
        return None
    
    def send_order(
        self,
        symbol: str,
        order_type: str,  # "BUY" or "SELL"
        volume: float,
        price: float,
        sl: float,
        tp: float,
        comment: str = ""
    ) -> Optional[Dict]:
        """
        Send market order to MT5
        
        Returns:
            Order result dict with ticket, price, etc.
        """
        params = {
            "symbol": symbol,
            "type": order_type,
            "volume": volume,
            "price": price,
            "sl": sl,
            "tp": tp,
            "comment": comment
        }
        
        response = self._send_command("place_order", params)
        return response
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account balance, equity, margin, etc."""
        response = self._send_command("get_account", {})
        if response and response.get("success"):
            return response.get("data")
        return None
    
    def get_open_positions(self) -> Optional[list]:
        """Get all open positions"""
        response = self._send_command("get_positions", {})
        if response and response.get("success"):
            return response.get("data")
        return None
    
    def close_position(self, ticket: int) -> bool:
        """Close position by ticket"""
        response = self._send_command("close_position", {"ticket": ticket})
        return response and response.get("success")


# ============================================
# Example Usage
# ============================================
if __name__ == '__main__':
    # Connect to MT5 bridge in Windows VM
    bridge = MT5RemoteBridge(host="localhost", port=7777)
    
    if bridge.connect():
        # 1. Get account info
        print("\n📊 Account Info:")
        account = bridge.get_account_info()
        if account:
            print(f"   Balance: ${account['balance']:.2f}")
            print(f"   Equity: ${account['equity']:.2f}")
            print(f"   Margin: ${account['margin']:.2f}")
        
        # 2. Get historical data
        print("\n📈 Fetching XAUUSD M1 data...")
        df = bridge.get_historical_data("XAUUSD", "M1", count=100)
        if df is not None:
            print(f"   ✅ {len(df)} bars loaded")
            print(df.tail())
        
        # 3. Get open positions
        print("\n📊 Open Positions:")
        positions = bridge.get_open_positions()
        if positions:
            for pos in positions:
                print(f"   {pos['ticket']}: {pos['type']} {pos['volume']} @ {pos['price']}")
        
        # 4. Example: Place order (commented out for safety)
        # order = bridge.send_order(
        #     symbol="XAUUSD",
        #     order_type="BUY",
        #     volume=0.01,
        #     price=2350.0,
        #     sl=2340.0,
        #     tp=2360.0,
        #     comment="Test from macOS"
        # )
        
        bridge.disconnect()
    else:
        print("❌ Failed to connect to MT5 bridge")
        print("\n💡 Make sure:")
        print("   1. Windows VM is running")
        print("   2. MT5 bridge server is running in VM")
        print("   3. Port 7777 is forwarded (Parallels settings)")
