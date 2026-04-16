# ============================================================
# deploy_new_machine.ps1
# One-shot deploy script for XAUUSD AI Trading System (V14++)
# Chạy: Set-ExecutionPolicy -Scope Process Bypass; .\scripts\windows\deploy_new_machine.ps1
# ============================================================
#Requires -Version 5.1
$ErrorActionPreference = "Continue"

$ROOT  = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$LOG   = "$ROOT\outputs\deploy_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$null  = New-Item -ItemType Directory -Path "$ROOT\outputs" -Force

function Log($msg, $color = "White") {
    $ts = Get-Date -Format "HH:mm:ss"
    $line = "$ts  $msg"
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $LOG -Value $line -Encoding UTF8
}

function LogOk($msg)   { Log "  [OK]   $msg" "Green" }
function LogWarn($msg) { Log "  [WARN] $msg" "Yellow" }
function LogErr($msg)  { Log "  [FAIL] $msg" "Red" }
function LogStep($msg) { Log "" ; Log ">>> $msg" "Cyan" }

Set-Location $ROOT
Log "============================================================" "Cyan"
Log "  XAUUSD AI — Deploy New Machine                           " "Cyan"
Log "  Date  : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')        " "Cyan"
Log "  Root  : $ROOT                                            " "Cyan"
Log "  Log   : $LOG                                             " "Cyan"
Log "============================================================" "Cyan"

# ── Detect Python ─────────────────────────────────────────────────────────────
LogStep "STEP 1/9 — Detect Python"
$python = $null
foreach ($candidate in @("python", "py", "python3")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "Python 3\.(1[1-9]|[2-9]\d)") {
            $python = $candidate
            LogOk "Found: $ver  ($candidate)"
            break
        }
    } catch { }
}
if (-not $python) {
    LogErr "Python 3.11+ not found. Install from https://www.python.org/downloads/"
    LogErr "Then re-run this script."
    exit 1
}

# ── Virtual environment ───────────────────────────────────────────────────────
LogStep "STEP 2/9 — Virtual environment"
$VENV = "$ROOT\.venv"
if (-not (Test-Path "$VENV\Scripts\python.exe")) {
    Log "Creating .venv..."
    & $python -m venv $VENV
    if ($LASTEXITCODE -ne 0) { LogErr "venv creation failed"; exit 1 }
    LogOk "Created: $VENV"
} else {
    LogOk "Already exists: $VENV"
}
$py = "$VENV\Scripts\python.exe"

# ── Install dependencies ─────────────────────────────────────────────────────
LogStep "STEP 3/9 — Install dependencies (requirements-windows.txt)"
if (-not (Test-Path "$ROOT\requirements-windows.txt")) {
    LogErr "requirements-windows.txt not found in $ROOT"
    exit 1
}
Log "Running pip install (may take 2-5 min first time)..."
& $py -m pip install --upgrade pip --quiet
& $py -m pip install -r "$ROOT\requirements-windows.txt" --quiet
if ($LASTEXITCODE -ne 0) {
    LogErr "pip install failed — check internet connection"
    exit 1
}
LogOk "Dependencies installed"

# ── Verify model artifacts ────────────────────────────────────────────────────
LogStep "STEP 4/9 — Verify model artifacts"
$artifacts = @(
    "outputs\acc1_v14pp_model.pkl",
    "outputs\acc1_v14pp_scaler.pkl",
    "outputs\acc1_v14pp_model_meta.json",
    "outputs\acc2_v14pp_model.pkl",
    "outputs\acc2_v14pp_scaler.pkl",
    "outputs\acc2_v14pp_model_meta.json"
)
$missingArtifacts = 0
foreach ($f in $artifacts) {
    if (Test-Path "$ROOT\$f") {
        LogOk $f
    } else {
        LogErr "MISSING: $f"
        $missingArtifacts++
    }
}
if ($missingArtifacts -gt 0) {
    LogErr "$missingArtifacts model file(s) missing. Copy from source machine to outputs\ and re-run."
    exit 1
}

# ── Verify config binding ─────────────────────────────────────────────────────
LogStep "STEP 5/9 — Verify config + model binding"
$bindScript = @"
import sys
sys.path.insert(0, 'src')
from pathlib import Path
from xauusd_ai.config import load_settings
errors = 0
for cfg, expected_model in [
    ('configs/live_acc1.yaml', 'outputs/acc1_v14pp_model.pkl'),
    ('configs/live_acc2.yaml', 'outputs/acc2_v14pp_model.pkl'),
]:
    s = load_settings(Path(cfg))
    model_ok = str(s.app.model_path) == expected_model
    print(f'  {cfg}: model={s.app.model_path}  thr={s.strategy.signal_threshold}  auto_trade={s.execution.auto_trade}  binding={"OK" if model_ok else "MISMATCH"}')
    if not model_ok:
        errors += 1
sys.exit(errors)
"@
$bindResult = & $py -c $bindScript 2>&1
$bindResult | ForEach-Object { Log "  $_" }
if ($LASTEXITCODE -ne 0) {
    LogErr "Model binding mismatch — check configs/live_acc1.yaml and configs/live_acc2.yaml"
    exit 1
}
LogOk "Config binding OK"

# ── Load .env ─────────────────────────────────────────────────────────────────
LogStep "STEP 6/9 — Load .env"
$envFile = "$ROOT\.env"
if (-not (Test-Path $envFile)) {
    if (Test-Path "$ROOT\.env.example") {
        Copy-Item "$ROOT\.env.example" $envFile
        LogWarn ".env not found — created from .env.example. EDIT IT NOW with real credentials, then re-run."
        notepad $envFile
        exit 0
    } else {
        LogErr ".env not found and .env.example also missing"
        exit 1
    }
}
Get-Content $envFile | Where-Object { $_ -notmatch '^\s*#' -and $_ -match '=' } | ForEach-Object {
    $parts = $_ -split '=', 2
    if ($parts.Count -eq 2) {
        $k = $parts[0].Trim(); $v = $parts[1].Trim()
        [System.Environment]::SetEnvironmentVariable($k, $v, 'Process')
    }
}
LogOk ".env loaded"

$requiredEnvVars = @(
    "TELEGRAM_BOT_TOKEN_ACC1", "TELEGRAM_CHAT_ID_ACC1",
    "TELEGRAM_BOT_TOKEN_ACC2", "TELEGRAM_CHAT_ID_ACC2"
)
$missingEnv = 0
foreach ($var in $requiredEnvVars) {
    $val = [System.Environment]::GetEnvironmentVariable($var, 'Process')
    if (-not $val -or $val -like "*your_*" -or $val -like "<*>") {
        LogWarn "Not configured: $var  (Telegram will be disabled)"
        $missingEnv++
    }
}
if ($missingEnv -eq 0) { LogOk "All Telegram env vars configured" }

# ── Start MT5 Bridges ─────────────────────────────────────────────────────────
LogStep "STEP 7/9 — Start MT5 Bridges (background)"
$env:PYTHONPATH = "src"

function Start-Bridge($acct, $port, $loginVar, $passVar, $serverVar) {
    $login  = [System.Environment]::GetEnvironmentVariable($loginVar,  'Process')
    $pass   = [System.Environment]::GetEnvironmentVariable($passVar,   'Process')
    $server = [System.Environment]::GetEnvironmentVariable($serverVar, 'Process')

    if (-not $login -or $login -like "*your_*" -or $login -like "<*>") {
        LogWarn "Bridge $acct (port $port): MT5 credentials not set in .env — skipping bridge"
        return
    }

    # Kill existing process on that port
    $existing = netstat -ano 2>$null | Select-String ":$port " | Select-Object -First 1
    if ($existing) {
        $pid_ = ($existing -split '\s+')[-1]
        if ($pid_ -match '^\d+$') {
            Stop-Process -Id $pid_ -Force -ErrorAction SilentlyContinue
            Start-Sleep 1
        }
    }

    $env:MT5_LOGIN    = $login
    $env:MT5_PASSWORD = $pass
    $env:MT5_SERVER   = $server

    $proc = Start-Process -FilePath $py `
        -ArgumentList "scripts/windows/mt5_bridge.py --port $port" `
        -WorkingDirectory $ROOT -WindowStyle Hidden -PassThru
    Log "  Bridge $acct started: PID $($proc.Id)  port $port"
}

Start-Bridge "ACC1" 5600 "MT5_LOGIN_ACC1" "MT5_PASSWORD_ACC1" "MT5_SERVER_ACC1"
Start-Bridge "ACC2" 5601 "MT5_LOGIN_ACC2" "MT5_PASSWORD_ACC2" "MT5_SERVER_ACC2"

Log "  Waiting 8s for bridges to initialise..."
Start-Sleep 8

# Check bridge health
foreach ($port in @(5600, 5601)) {
    try {
        $resp = Invoke-RestMethod "http://localhost:$port/ping" -TimeoutSec 3 -ErrorAction Stop
        LogOk "Bridge port $port alive — status: $($resp.status)"
    } catch {
        LogWarn "Bridge port $port not responding yet (may still be initialising)"
    }
}

# ── Start Live Bots ────────────────────────────────────────────────────────────
LogStep "STEP 8/9 — Start Live Bots (background)"

function Start-Bot($acct, $config, $bridgeUrl, $botTokenVar, $chatIdVar) {
    $env:MT5_BRIDGE_URL         = $bridgeUrl
    $env:TELEGRAM_BOT_TOKEN     = [System.Environment]::GetEnvironmentVariable($botTokenVar, 'Process')
    $env:TELEGRAM_CHAT_ID       = [System.Environment]::GetEnvironmentVariable($chatIdVar,   'Process')
    $env:OUTPUTS_PATH           = "$ROOT\outputs"

    $proc = Start-Process -FilePath $py `
        -ArgumentList "-m xauusd_ai.app live --config $config" `
        -WorkingDirectory $ROOT -WindowStyle Normal -PassThru
    Log "  Bot $acct started: PID $($proc.Id)  config=$config"
    return $proc
}

$proc1 = Start-Bot "ACC1" "configs/live_acc1.yaml" "http://localhost:5600" `
    "TELEGRAM_BOT_TOKEN_ACC1" "TELEGRAM_CHAT_ID_ACC1"
Start-Sleep 3
$proc2 = Start-Bot "ACC2" "configs/live_acc2.yaml" "http://localhost:5601" `
    "TELEGRAM_BOT_TOKEN_ACC2" "TELEGRAM_CHAT_ID_ACC2"

Log "  Waiting 10s for bots to initialise..."
Start-Sleep 10

foreach ($p in @($proc1, $proc2)) {
    if ($p -and (Get-Process -Id $p.Id -ErrorAction SilentlyContinue)) {
        LogOk "Bot PID $($p.Id) is still running"
    } else {
        LogWarn "Bot PID $($p.Id) may have exited — check logs in outputs\"
    }
}

# ── Start FastAPI Dashboard ────────────────────────────────────────────────────
LogStep "STEP 9/9 — Start FastAPI Dashboard (port 8000)"

$existing8000 = netstat -ano 2>$null | Select-String ":8000 " | Where-Object { $_ -match "LISTENING" } | Select-Object -First 1
if ($existing8000) {
    $pid8000 = ($existing8000 -split '\s+')[-1]
    if ($pid8000 -match '^\d+$') {
        Log "  Port 8000 already in use by PID $pid8000 — stopping it..."
        Stop-Process -Id $pid8000 -Force -ErrorAction SilentlyContinue
        Start-Sleep 2
    }
}

$env:MLFLOW_TRACKING_URI = "file:///$ROOT/outputs/mlflowruns"
$env:OUTPUTS_PATH        = "$ROOT\outputs"

$apiProc = Start-Process -FilePath $py `
    -ArgumentList "-m uvicorn xauusd_ai.api.main:app --host 0.0.0.0 --port 8000" `
    -WorkingDirectory $ROOT -WindowStyle Normal -PassThru
Log "  FastAPI started: PID $($apiProc.Id)"
Log "  Waiting 8s for API to come up..."
Start-Sleep 8

# ── CHECKLIST REPORT ──────────────────────────────────────────────────────────
Log ""
Log "============================================================" "Cyan"
Log "  CHECKLIST REPORT                                         " "Cyan"
Log "============================================================" "Cyan"

$checks = @()
$pass = 0; $fail = 0; $warn = 0

function Check($name, $ok, $detail = "", $isWarn = $false) {
    if ($ok) {
        $status = "[PASS]"; $color = "Green"; $script:pass++
    } elseif ($isWarn) {
        $status = "[WARN]"; $color = "Yellow"; $script:warn++
    } else {
        $status = "[FAIL]"; $color = "Red"; $script:fail++
    }
    $line = "  $status  $name"
    if ($detail) { $line += "  →  $detail" }
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $LOG -Value $line -Encoding UTF8
}

# 1. Model files
foreach ($f in $artifacts) {
    Check "Model file: $f" (Test-Path "$ROOT\$f")
}

# 2. Bridges
foreach ($item in @(@{port=5600;acct="ACC1"}, @{port=5601;acct="ACC2"})) {
    try {
        $r = Invoke-RestMethod "http://localhost:$($item.port)/ping" -TimeoutSec 3 -ErrorAction Stop
        Check "MT5 Bridge $($item.acct) (port $($item.port))" $true "status=$($r.status)"
    } catch {
        Check "MT5 Bridge $($item.acct) (port $($item.port))" $false "not responding" $true
    }
}

# 3. FastAPI health
try {
    $health = Invoke-RestMethod "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction Stop
    Check "FastAPI /health" $true "status=$($health.status)"
} catch {
    Check "FastAPI /health" $false "not responding"
}

# 4. Dashboard payload
try {
    $dash = Invoke-RestMethod "http://localhost:8000/api/v1/dashboard" -TimeoutSec 10 -ErrorAction Stop
    $acc1Model = $dash.accounts.acc1.status.model_path
    $acc2Model = $dash.accounts.acc2.status.model_path
    Check "Dashboard ACC1 model binding" ($acc1Model -like "*acc1_v14pp*") "model=$acc1Model"
    Check "Dashboard ACC2 model binding" ($acc2Model -like "*acc2_v14pp*") "model=$acc2Model"

    $acc1AT  = $dash.accounts.acc1.auto_trade_enabled
    $acc2AT  = $dash.accounts.acc2.auto_trade_enabled
    Check "Auto trade ACC1" ($acc1AT -eq $true) "auto_trade_enabled=$acc1AT" ($acc1AT -ne $true)
    Check "Auto trade ACC2" ($acc2AT -eq $true) "auto_trade_enabled=$acc2AT" ($acc2AT -ne $true)

    $killed1 = $dash.accounts.acc1.daily.killed
    $killed2 = $dash.accounts.acc2.daily.killed
    Check "Kill switch ACC1 not triggered" (-not $killed1) "killed=$killed1" $killed1
    Check "Kill switch ACC2 not triggered" (-not $killed2) "killed=$killed2" $killed2
} catch {
    Check "Dashboard payload fetch" $false "$_"
}

# 5. Kill switch state files
foreach ($f in @("outputs\risk_daily_state_acc1.json", "outputs\risk_daily_state_acc2.json")) {
    $fp = "$ROOT\$f"
    if (Test-Path $fp) {
        $state = Get-Content $fp | ConvertFrom-Json
        Check "Kill switch state $f" (-not $state.killed) "killed=$($state.killed)  consec=$($state.consecutive_losses)"
    } else {
        Check "Kill switch state $f" $true "(not yet created — OK on first run)"
    }
}

# 6. Telegram env vars
foreach ($var in $requiredEnvVars) {
    $val = [System.Environment]::GetEnvironmentVariable($var, 'Process')
    $configured = $val -and $val -notlike "*your_*" -and $val -notlike "<*>"
    Check "Env var: $var" $configured $(if (-not $configured) { "not set — Telegram disabled" } else { "configured" }) (-not $configured)
}

# ── Summary ─────────────────────────────────────────────────────────────────
Log ""
Log "============================================================" "Cyan"
Log "  SUMMARY" "Cyan"
Log "  PASS: $pass    FAIL: $fail    WARN: $warn" $(if ($fail -gt 0) { "Red" } elseif ($warn -gt 0) { "Yellow" } else { "Green" })
Log "============================================================" "Cyan"

if ($fail -eq 0) {
    Log ""
    Log "  All checks passed! System is LIVE." "Green"
    Log "  Dashboard : http://localhost:8000/dashboard" "Green"
    Log "  API Docs  : http://localhost:8000/docs" "Green"
    Log ""
    Start-Process "http://localhost:8000/dashboard"
} else {
    Log ""
    Log "  $fail check(s) failed. Review log: $LOG" "Red"
    Log "  Fix issues above then re-run this script." "Red"
    Log ""
}

Log "  Full deploy log: $LOG" "Cyan"
Log ""
