<#
.SYNOPSIS
    Start the MT5 Bridge service for ACC2 on port 5601 (Windows host).
    Keep this window open while live-bot Docker containers are running.

.DESCRIPTION
    ACC2 uses login 433326057 on Exness-MT5Trial7 via the MetaTrader 5 EXNESS terminal.
    The live-acc2 container connects to http://host.docker.internal:5601.

    Run BEFORE starting docker compose for the live (ACC2) service.
#>

# ── Configuration ──────────────────────────────────────────────────────────────
$env:MT5_BRIDGE_PORT    = "5601"
$env:MT5_LOGIN          = "433326057"
$env:MT5_PASSWORD       = "07032001bB@"
$env:MT5_SERVER         = "Exness-MT5Trial7"
$env:MT5_TERMINAL_PATH  = "C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe"

# ── Confirm MT5 terminal is open ───────────────────────────────────────────────
Write-Host ""
Write-Host "=== MT5 Bridge Startup — ACC2 ===" -ForegroundColor Cyan
Write-Host "Account : $($env:MT5_LOGIN) @ $($env:MT5_SERVER)"
Write-Host "Port    : $($env:MT5_BRIDGE_PORT)"
Write-Host ""
Write-Host "Make sure MetaTrader 5 EXNESS terminal is OPEN and LOGGED IN before continuing."
Write-Host ""

# ── Check Python + MetaTrader5 package ────────────────────────────────────────
$mt5Check = python -c "import MetaTrader5; print('OK')" 2>&1
if ($mt5Check -ne "OK") {
    Write-Host "ERROR: MetaTrader5 Python package not found." -ForegroundColor Red
    Write-Host "Install it with:  pip install MetaTrader5"
    exit 1
}

# ── Change to workspace root ──────────────────────────────────────────────────
Set-Location "$PSScriptRoot\..\.."

# ── Start bridge ──────────────────────────────────────────────────────────────
Write-Host "Starting MT5 Bridge for ACC2..." -ForegroundColor Green
python scripts\windows\mt5_bridge.py
