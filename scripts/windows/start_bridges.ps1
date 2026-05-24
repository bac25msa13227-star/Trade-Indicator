# start_bridges.ps1 - Kill old bridges and restart both MT5 bridges.
# Credentials are loaded from .env. Do not hardcode login/password/server here.
# ACC1 -> port 5600 -> C:\Program Files\MetaTrader 5\terminal64.exe
# ACC2 -> port 5601 -> C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe

$ROOT = "C:\Users\Administrator\Documents\Trade-Indicator"
$ENV_FILE = Join-Path $ROOT ".env"

function Load-DotEnv([string]$path) {
    if (-not (Test-Path $path)) {
        throw ".env not found at $path"
    }

    Get-Content $path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }

        $parts = $line -split "=", 2
        if ($parts.Count -ne 2) { return }

        $key = $parts[0].Trim()
        $value = $parts[1].Trim()
        [Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

function Require-Env([string]$name) {
    $value = [Environment]::GetEnvironmentVariable($name, "Process")
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Missing required env var: $name"
    }
    return $value
}

function Kill-Port([int]$port) {
    $lines = netstat -ano | Select-String "0\.0\.0\.0:$port\s"
    foreach ($line in $lines) {
        $pidText = ($line -split '\s+' | Where-Object { $_ -match '^\d+$' } | Select-Object -Last 1)
        if ($pidText -and [int]$pidText -gt 4) {
            Write-Host "  Killing PID $pidText on port $port"
            Stop-Process -Id ([int]$pidText) -Force -ErrorAction SilentlyContinue
        }
    }
}

Load-DotEnv $ENV_FILE
$acc1Login = Require-Env "MT5_LOGIN"
$acc2Login = Require-Env "MT5_LOGIN_ACC2"
Require-Env "MT5_PASSWORD" | Out-Null
Require-Env "MT5_PASSWORD_ACC2" | Out-Null
Require-Env "MT5_SERVER" | Out-Null
Require-Env "MT5_SERVER_ACC2" | Out-Null

Write-Host "=== Killing existing bridges ==="
Kill-Port 5600
Kill-Port 5601
Start-Sleep 3

Write-Host "=== Starting ACC1 bridge (port 5600) ==="
$proc1 = Start-Process cmd -ArgumentList "/c `"$ROOT\scripts\windows\bridge_acc1.bat`"" `
    -WorkingDirectory $ROOT -WindowStyle Hidden -PassThru
Write-Host "  ACC1 bridge PID=$($proc1.Id)"

Start-Sleep 3

Write-Host "=== Starting ACC2 bridge (port 5601) ==="
$proc2 = Start-Process cmd -ArgumentList "/c `"$ROOT\scripts\windows\bridge_acc2.bat`"" `
    -WorkingDirectory $ROOT -WindowStyle Hidden -PassThru
Write-Host "  ACC2 bridge PID=$($proc2.Id)"

Start-Sleep 5

Write-Host "=== Verifying bridges ==="
try {
    $a1 = (Invoke-WebRequest "http://localhost:5600/account" -UseBasicParsing -TimeoutSec 8).Content | ConvertFrom-Json
    Write-Host "PORT 5600: balance=$($a1.balance)"
    if ([string]$a1.login -eq [string]$acc1Login) {
        Write-Host "  [OK] ACC1 correct"
    } else {
        Write-Host "  [WRONG] ACC1 connected to an unexpected login"
    }
} catch {
    Write-Host "PORT 5600 error: $_"
}

try {
    $a2 = (Invoke-WebRequest "http://localhost:5601/account" -UseBasicParsing -TimeoutSec 8).Content | ConvertFrom-Json
    Write-Host "PORT 5601: balance=$($a2.balance)"
    if ([string]$a2.login -eq [string]$acc2Login) {
        Write-Host "  [OK] ACC2 correct"
    } else {
        Write-Host "  [WRONG] ACC2 connected to an unexpected login"
    }
} catch {
    Write-Host "PORT 5601 error: $_"
}
