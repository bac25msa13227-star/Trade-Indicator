@echo off
title MT5 Bridge ACC2 (415523600 / Exness-MT5Trial14) - Port 5601
set MT5_BRIDGE_PORT=5601
set MT5_LOGIN=415523600
set MT5_PASSWORD=07032001bB@@
set MT5_SERVER=Exness-MT5Trial14
set MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe
cd /d F:\Trading_BOT_AUTO\Trade-Indicator
echo Starting ACC2 bridge on port 5601...
python scripts\windows\mt5_bridge.py
echo Bridge stopped. Press any key to exit.
pause
