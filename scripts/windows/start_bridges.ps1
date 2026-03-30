# start_bridges.ps1 — Kill old bridges and restart both with correct MT5 terminal paths
# ACC1: 103613837  / Exness-MT5Real15   -> port 5600  -> C:\Program Files\MetaTrader 5\terminal64.exe
# ACC2: 433326057  / Exness-MT5Trial7   -> port 5601  -> C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe

$ROOT   = "C:\Users\Administrator\Documents\Trade-Indicator"
$BRIDGE = "$ROOT\scripts\windows\mt5_bridge.py"

function Kill-Port([int]$port) {
    $lines = netstat -ano | Select-String "0\.0\.0\.0:$port\s"
    foreach ($l in $lines) {
        $pid_ = ($l -split '\s+' | Where-Object { $_ -match '^\d+$' } | Select-Object -Last 1)
        if ($pid_ -and [int]$pid_ -gt 4) {
            Write-Host "  Killing PID $pid_ on port $port"
            Stop-Process -Id ([int]$pid_) -Force -ErrorAction SilentlyContinue
        }
    }
}


Write-Host "=== Killing existing bridges ==="
Kill-Port 5600
Kill-Port 5601
Start-Sleep 3

Write-Host "=== Starting ACC1 bridge (port 5600 -> 103613837 / Exness-MT5Real15) ==="
$proc1 = Start-Process cmd -ArgumentList "/c `"$ROOT\scripts\windows\bridge_acc1.bat`"" `
    -WorkingDirectory $ROOT -WindowStyle Normal -PassThru
Write-Host "  ACC1 bridge PID=$($proc1.Id)"

Start-Sleep 3

Write-Host "=== Starting ACC2 bridge (port 5601 -> 433326057 / Exness-MT5Trial7) ==="
$proc2 = Start-Process cmd -ArgumentList "/c `"$ROOT\scripts\windows\bridge_acc2.bat`"" `
    -WorkingDirectory $ROOT -WindowStyle Normal -PassThru
Write-Host "  ACC2 bridge PID=$($proc2.Id)"

Start-Sleep 5

Write-Host "=== Verifying bridges ==="
try {
    $a1 = (Invoke-WebRequest 'http://localhost:5600/account' -UseBasicParsing -TimeoutSec 8).Content | ConvertFrom-Json
    Write-Host "PORT 5600: login=$($a1.login) server=$($a1.server) balance=$($a1.balance)"
    if ([string]$a1.login -eq "103613837") { Write-Host "  [OK] ACC1 correct" }
    else { Write-Host "  [WRONG] Expected 103613837, got $($a1.login)!" }
} catch { Write-Host "PORT 5600 error: $_" }

try {
    $a2 = (Invoke-WebRequest 'http://localhost:5601/account' -UseBasicParsing -TimeoutSec 8).Content | ConvertFrom-Json
    Write-Host "PORT 5601: login=$($a2.login) server=$($a2.server) balance=$($a2.balance)"
    if ([string]$a2.login -eq "433326057") { Write-Host "  [OK] ACC2 correct" }
    else { Write-Host "  [WRONG] Expected 433326057, got $($a2.login)!" }
} catch { Write-Host "PORT 5601 error: $_" }
