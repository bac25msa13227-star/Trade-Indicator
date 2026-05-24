<#
.SYNOPSIS
    Start the ACC1 MT5 bridge service on the Windows host.

.DESCRIPTION
    Credentials are loaded from .env or existing environment variables:
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER.
#>

Set-Location "$PSScriptRoot\..\.."

if (Test-Path ".env") {
    Get-Content ".env" | Where-Object { $_ -and $_ -notmatch '^\s*#' -and $_ -match '=' } | ForEach-Object {
        $k, $v = $_ -split '=', 2
        if ($k -and -not [Environment]::GetEnvironmentVariable($k.Trim(), 'Process')) {
            [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), 'Process')
        }
    }
}

$env:MT5_BRIDGE_PORT = "5600"
if (-not $env:MT5_TERMINAL_PATH) {
    $env:MT5_TERMINAL_PATH = "C:\Program Files\MetaTrader 5\terminal64.exe"
}

foreach ($name in @("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER")) {
    if (-not [Environment]::GetEnvironmentVariable($name, 'Process')) {
        Write-Host "ERROR: $name is missing. Set rotated ACC1 credentials in .env." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "=== MT5 Bridge Startup - ACC1 ===" -ForegroundColor Cyan
Write-Host "Account : loaded from .env"
Write-Host "Port    : $($env:MT5_BRIDGE_PORT)"
Write-Host ""
Write-Host "Make sure MetaTrader 5 terminal is OPEN and LOGGED IN."
Write-Host "Also make sure 'Algo Trading' is enabled."
Write-Host ""

$mt5Check = python -c "import MetaTrader5; print('OK')" 2>&1
if ($mt5Check -ne "OK") {
    Write-Host "ERROR: MetaTrader5 Python package not found." -ForegroundColor Red
    Write-Host "Install it with: pip install MetaTrader5"
    exit 1
}

Write-Host "Starting MT5 Bridge for ACC1..." -ForegroundColor Green
python scripts\windows\mt5_bridge.py
