@echo off
title XAUUSD Bot ACC2 - acc2_breakthrough_net21k_dd2333
cd /d C:\Users\Administrator\Documents\Trade-Indicator

REM Load .env vars
for /f "usebackq tokens=1,* delims==" %%A in (`findstr /v "^#" .env`) do (
    if not "%%A"=="" set "%%A=%%B"
)

REM Override for direct host run (not Docker)
set PYTHONPATH=src
set MT5_BRIDGE_URL=http://localhost:5601
set TELEGRAM_BOT_TOKEN=%TELEGRAM_BOT_TOKEN_ACC2%
set TELEGRAM_CHAT_ID=%TELEGRAM_CHAT_ID_ACC2%
set PYTHONUNBUFFERED=1
set MLFLOW_TRACKING_URI=file:///C:/Temp/mlflowruns

echo [ACC2] Starting live bot...
echo [ACC2] MT5_BRIDGE_URL=%MT5_BRIDGE_URL%
echo [ACC2] Config: configs/live_acc2.yaml
echo [ACC2] Model: outputs/acc2_breakthrough_net21k_dd2333_model.pkl

python -m xauusd_ai.main live --config configs/live_acc2.yaml >> logs\bot_acc2.log 2>&1
echo [ACC2] Process exited with code %ERRORLEVEL% >> logs\bot_acc2.log
