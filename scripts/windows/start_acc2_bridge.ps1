<#
.SYNOPSIS
    Start the ACC2 MT5 bridge service on the Windows host.

.DESCRIPTION
    Credentials are loaded from .env or existing environment variables:
    MT5_LOGIN_ACC2, MT5_PASSWORD_ACC2, MT5_SERVER_ACC2.
#>

param(
    [int]$Port = 5601,
    [string]$TerminalPath = "C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe"
)

Set-Location "$PSScriptRoot\..\.."

if (Test-Path ".env") {
    Get-Content ".env" | Where-Object { $_ -and $_ -notmatch '^\s*#' -and $_ -match '=' } | ForEach-Object {
        $k, $v = $_ -split '=', 2
        if ($k -and -not [Environment]::GetEnvironmentVariable($k.Trim(), 'Process')) {
            [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), 'Process')
        }
    }
}

$env:MT5_BRIDGE_PORT = "$Port"
$env:MT5_LOGIN = $env:MT5_LOGIN_ACC2
$env:MT5_PASSWORD = $env:MT5_PASSWORD_ACC2
$env:MT5_SERVER = $env:MT5_SERVER_ACC2
if ($TerminalPath) {
    $env:MT5_TERMINAL_PATH = $TerminalPath
} elseif (-not $env:MT5_TERMINAL_PATH) {
    $env:MT5_TERMINAL_PATH = "C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe"
}

foreach ($name in @("MT5_LOGIN_ACC2", "MT5_PASSWORD_ACC2", "MT5_SERVER_ACC2")) {
    if (-not [Environment]::GetEnvironmentVariable($name, 'Process')) {
        Write-Host "ERROR: $name is missing. Set rotated ACC2 credentials in .env." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "=== MT5 Bridge Startup - ACC2 ===" -ForegroundColor Cyan
Write-Host "Account : loaded from .env"
Write-Host "Port    : $($env:MT5_BRIDGE_PORT)"
Write-Host "Terminal: $($env:MT5_TERMINAL_PATH)"
Write-Host ""
Write-Host "Make sure MetaTrader 5 EXNESS terminal is OPEN and LOGGED IN."
Write-Host ""

$mt5Check = python -c "import MetaTrader5; print('OK')" 2>&1
if ($mt5Check -ne "OK") {
    Write-Host "ERROR: MetaTrader5 Python package not found." -ForegroundColor Red
    Write-Host "Install it with: pip install MetaTrader5"
    exit 1
}

Write-Host "Starting MT5 Bridge for ACC2..." -ForegroundColor Green
python scripts\windows\mt5_bridge.py
