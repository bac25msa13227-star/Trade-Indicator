@echo off
title XAUUSD Bot ACC1 - acc1_breakthrough_net66590_dd3953
cd /d C:\Users\Administrator\Documents\Trade-Indicator

REM Load .env vars
for /f "usebackq tokens=1,* delims==" %%A in (`findstr /v "^#" .env`) do (
    if not "%%A"=="" set "%%A=%%B"
)

REM Override for direct host run (not Docker)
set PYTHONPATH=src
set MT5_BRIDGE_URL=http://localhost:5600
set TELEGRAM_BOT_TOKEN=%TELEGRAM_BOT_TOKEN_ACC1%
set TELEGRAM_CHAT_ID=%TELEGRAM_CHAT_ID_ACC1%
set PYTHONUNBUFFERED=1
set MLFLOW_TRACKING_URI=file:///C:/Temp/mlflowruns

echo [ACC1] Starting live bot...
echo [ACC1] MT5_BRIDGE_URL=%MT5_BRIDGE_URL%
echo [ACC1] Config: configs/live_acc1.yaml
echo [ACC1] Model: outputs/acc1_breakthrough_net66590_dd3953_model.pkl

python -m xauusd_ai.main live --config configs/live_acc1.yaml >> logs\bot_acc1.log 2>&1
echo [ACC1] Process exited with code %ERRORLEVEL% >> logs\bot_acc1.log
