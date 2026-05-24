<#
.SYNOPSIS
    Start the rolling_v3 MT5 bridge service on the Windows host.

.DESCRIPTION
    Uses MT5_LOGIN, MT5_PASSWORD, MT5_SERVER from .env by default.
    This is the live path for rolling_v3 model signals. EA/MetaEditor remains
    WF/Strategy Tester only.
#>

param(
    [int]$Port = 5601,
    [string]$LoginEnv = "MT5_LOGIN",
    [string]$PasswordEnv = "MT5_PASSWORD",
    [string]$ServerEnv = "MT5_SERVER",
    [string]$TerminalPath = ""
)

$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\.."

if (Test-Path ".env") {
    Get-Content ".env" | Where-Object { $_ -and $_ -notmatch '^\s*#' -and $_ -match '=' } | ForEach-Object {
        $k, $v = $_ -split '=', 2
        if ($k -and -not [Environment]::GetEnvironmentVariable($k.Trim(), 'Process')) {
            [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), 'Process')
        }
    }
}

$login = [Environment]::GetEnvironmentVariable($LoginEnv, 'Process')
$password = [Environment]::GetEnvironmentVariable($PasswordEnv, 'Process')
$server = [Environment]::GetEnvironmentVariable($ServerEnv, 'Process')

foreach ($item in @(
    @{ Name = $LoginEnv; Value = $login },
    @{ Name = $PasswordEnv; Value = $password },
    @{ Name = $ServerEnv; Value = $server }
)) {
    if (-not $item.Value) {
        Write-Host "ERROR: $($item.Name) is missing in .env/process env." -ForegroundColor Red
        exit 1
    }
}

$env:MT5_BRIDGE_PORT = "$Port"
$env:MT5_LOGIN = $login
$env:MT5_PASSWORD = $password
$env:MT5_SERVER = $server
if (-not [string]::IsNullOrWhiteSpace($TerminalPath)) {
    $env:MT5_TERMINAL_PATH = $TerminalPath
} elseif (-not $env:MT5_TERMINAL_PATH) {
    $exnessPath = "C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe"
    $defaultPath = "C:\Program Files\MetaTrader 5\terminal64.exe"
    if (Test-Path $exnessPath) {
        $env:MT5_TERMINAL_PATH = $exnessPath
    } else {
        $env:MT5_TERMINAL_PATH = $defaultPath
    }
}

Write-Host ""
Write-Host "=== MT5 Bridge Startup - rolling_v3 ===" -ForegroundColor Cyan
Write-Host "Account : loaded from $LoginEnv"
Write-Host "Server  : loaded from $ServerEnv"
Write-Host "Port    : $($env:MT5_BRIDGE_PORT)"
Write-Host "Terminal: $($env:MT5_TERMINAL_PATH)"
Write-Host ""
Write-Host "Make sure the MT5 terminal is open and logged in to this same account."
Write-Host ""

$mt5Check = python -c "import MetaTrader5; print('OK')" 2>&1
if ($mt5Check -ne "OK") {
    Write-Host "ERROR: MetaTrader5 Python package not found." -ForegroundColor Red
    Write-Host "Install it with: pip install MetaTrader5"
    exit 1
}

python scripts\windows\mt5_bridge.py
