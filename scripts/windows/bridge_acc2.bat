@echo off
title MT5 Bridge ACC2 - Port 5601
set MT5_BRIDGE_PORT=5601
set MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe
cd /d "%~dp0..\.."

if exist ".env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" set "%%A=%%B"
  )
)

set "MT5_LOGIN=%MT5_LOGIN_ACC2%"
set "MT5_PASSWORD=%MT5_PASSWORD_ACC2%"
set "MT5_SERVER=%MT5_SERVER_ACC2%"

if "%MT5_LOGIN%"=="" (
  echo ERROR: MT5_LOGIN_ACC2 is missing. Set rotated ACC2 credentials in .env.
  exit /b 1
)
if "%MT5_PASSWORD%"=="" (
  echo ERROR: MT5_PASSWORD_ACC2 is missing. Set rotated ACC2 credentials in .env.
  exit /b 1
)
if "%MT5_SERVER%"=="" (
  echo ERROR: MT5_SERVER_ACC2 is missing. Set ACC2 server in .env.
  exit /b 1
)

:restart
echo [%date% %time%] Starting ACC2 bridge on port 5601...
python scripts\windows\mt5_bridge.py
echo [%date% %time%] Bridge exited (code %ERRORLEVEL%). Restarting in 15 seconds...
timeout /t 15 /nobreak > nul
goto restart
