@echo off
title MT5 Bridge ACC1 - Port 5600
set MT5_BRIDGE_PORT=5600
set MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
cd /d "%~dp0..\.."

if exist ".env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" set "%%A=%%B"
  )
)

if "%MT5_LOGIN%"=="" (
  echo ERROR: MT5_LOGIN is missing. Set rotated ACC1 credentials in .env.
  exit /b 1
)
if "%MT5_PASSWORD%"=="" (
  echo ERROR: MT5_PASSWORD is missing. Set rotated ACC1 credentials in .env.
  exit /b 1
)
if "%MT5_SERVER%"=="" (
  echo ERROR: MT5_SERVER is missing. Set ACC1 server in .env.
  exit /b 1
)

:restart
echo [%date% %time%] Starting ACC1 bridge on port 5600...
python scripts\windows\mt5_bridge.py
echo [%date% %time%] Bridge exited (code %ERRORLEVEL%). Restarting in 15 seconds...
timeout /t 15 /nobreak > nul
goto restart
