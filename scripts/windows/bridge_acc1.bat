@echo off
title MT5 Bridge ACC1 (103613837 / Exness-MT5Real15) - Port 5600
set MT5_BRIDGE_PORT=5600
set MT5_LOGIN=103613837
set MT5_PASSWORD=07032001bB@
set MT5_SERVER=Exness-MT5Real15
set MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
cd /d C:\Users\Administrator\Documents\Trade-Indicator
echo Starting ACC1 bridge on port 5600...
python scripts\windows\mt5_bridge.py
echo Bridge stopped. Press any key to exit.
pause
