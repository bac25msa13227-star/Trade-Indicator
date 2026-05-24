# launch_bot_acc1.ps1 — Start live bot ACC1 (MT5_LOGIN / port 5600)
$ROOT = "C:\Users\Administrator\Documents\Trade-Indicator"
Set-Location $ROOT

# Load .env
Get-Content "$ROOT\.env" | Where-Object { $_ -notmatch '^\s*#' -and $_ -match '=' } | ForEach-Object {
    $parts = $_ -split '=', 2
    if ($parts.Count -eq 2) {
        $k = $parts[0].Trim(); $v = $parts[1].Trim()
        [System.Environment]::SetEnvironmentVariable($k, $v, 'Process')
    }
}
$env:PYTHONPATH = "src"
# Bridge URL for direct (non-Docker) run
$env:MT5_BRIDGE_URL = "http://localhost:5600"
# Ensure ACC1 telegram vars are set (config uses TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)
$env:TELEGRAM_BOT_TOKEN = $env:TELEGRAM_BOT_TOKEN_ACC1
$env:TELEGRAM_CHAT_ID   = $env:TELEGRAM_CHAT_ID_ACC1
Write-Host "[ACC1] MT5_BRIDGE_URL=$($env:MT5_BRIDGE_URL)"
Write-Host "[ACC1] Config: configs/live_acc1.yaml"
python -m xauusd_ai.app live --config configs/live_acc1.yaml
