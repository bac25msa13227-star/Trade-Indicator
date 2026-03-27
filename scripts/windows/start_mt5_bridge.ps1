<#
.SYNOPSIS
    Start the MT5 Bridge service on the Windows host.
    Keep this window open while live-bot Docker containers are running.

.DESCRIPTION
    The live-bot containers cannot import MetaTrader5 (Linux only).
    This bridge runs natively on Windows and exposes MT5 operations via HTTP
    so the containers can call http://host.docker.internal:5600.

    Credentials are loaded from configs\settings.yaml or env vars.
    Edit the variables below if needed.
#>

# ── Configuration — ACC1 (port 5600) ──────────────────────────────────────────
# For ACC2, use scripts\windows\start_acc2_bridge.ps1 (port 5601)
$env:MT5_BRIDGE_PORT    = "5600"
$env:MT5_LOGIN          = "270832477"
$env:MT5_PASSWORD       = "07032001bB@"
$env:MT5_SERVER         = "Exness-MT5Trial17"
$env:MT5_TERMINAL_PATH  = "C:\Program Files\MetaTrader 5\terminal64.exe"

# ── Prompt for password if not set ────────────────────────────────────────────
if (-not $env:MT5_PASSWORD) {
    $pw = Read-Host "Enter MT5 password for login $($env:MT5_LOGIN)" -AsSecureString
    $env:MT5_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($pw)
    )
}

# ── Confirm MT5 terminal is open ───────────────────────────────────────────────
Write-Host ""
Write-Host "=== MT5 Bridge Startup ===" -ForegroundColor Cyan
Write-Host "Account : $($env:MT5_LOGIN) @ $($env:MT5_SERVER)"
Write-Host "Port    : $($env:MT5_BRIDGE_PORT)"
Write-Host ""
Write-Host "Make sure MetaTrader 5 terminal is OPEN and LOGGED IN before continuing."
Write-Host "Also make sure 'Algo Trading' button (top toolbar) is GREEN/enabled."
Write-Host ""

# ── Check Python + MetaTrader5 package ────────────────────────────────────────
$mt5Check = python -c "import MetaTrader5; print('OK')" 2>&1
if ($mt5Check -ne "OK") {
    Write-Host "ERROR: MetaTrader5 Python package not found." -ForegroundColor Red
    Write-Host "Install it with:  pip install MetaTrader5"
    exit 1
}

# ── Change to workspace root ───────────────────────────────────────────────────
Set-Location "$PSScriptRoot\..\.."

# ── Start bridge ───────────────────────────────────────────────────────────────
Write-Host "Starting MT5 Bridge..." -ForegroundColor Green
python scripts\windows\mt5_bridge.py
