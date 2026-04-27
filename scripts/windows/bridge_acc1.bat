@echo off
title MT5 Bridge ACC1 (270832477 / Exness-MT5Trial17) - Port 5600
set MT5_BRIDGE_PORT=5600
set MT5_LOGIN=270832477
set MT5_PASSWORD=07032001bB@
set MT5_SERVER=Exness-MT5Trial17
set MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
cd /d C:\Users\Administrator\Documents\Trade-Indicator

:restart
echo [%date% %time%] Starting ACC1 bridge on port 5600...
python scripts\windows\mt5_bridge.py
echo [%date% %time%] Bridge exited (code %ERRORLEVEL%). Restarting in 15 seconds...
timeout /t 15 /nobreak > nul
goto restart
