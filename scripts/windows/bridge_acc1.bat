@echo off
title MT5 Bridge ACC1 (463222227 / Exness-MT5Trial17) - Port 5600
set MT5_BRIDGE_PORT=5600
set MT5_LOGIN=463222227
set MT5_PASSWORD=07032001bB@
set MT5_SERVER=Exness-MT5Trial17
set MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
cd /d F:\Trading_BOT_AUTO\Trade-Indicator
echo Starting ACC1 bridge on port 5600...
python scripts\windows\mt5_bridge.py
echo Bridge stopped. Press any key to exit.
pause
