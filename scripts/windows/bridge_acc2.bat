@echo off
title MT5 Bridge ACC2 (433326057 / Exness-MT5Trial7) - Port 5601
set MT5_BRIDGE_PORT=5601
set MT5_LOGIN=433326057
set MT5_PASSWORD=07032001bB@
set MT5_SERVER=Exness-MT5Trial7
set MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe
cd /d C:\Users\Administrator\Documents\Trade-Indicator

:restart
echo [%date% %time%] Starting ACC2 bridge on port 5601...
python scripts\windows\mt5_bridge.py
echo [%date% %time%] Bridge exited (code %ERRORLEVEL%). Restarting in 15 seconds...
timeout /t 15 /nobreak > nul
goto restart
