$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $projectRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install Python 3.11+ for Windows first."
}

if (-not (Test-Path .venv)) {
    py -3.11 -m venv .venv
}

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

& $python -m pip install --upgrade pip
& $python -m pip install -r requirements-windows.txt

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
}

Write-Host "Windows bootstrap complete." -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Fill .env with MT5_LOGIN, MT5_PASSWORD, MT5_SERVER"
Write-Host "2. Open MT5 terminal and log into your broker account"
Write-Host "3. Run scripts/windows/mt5-check.ps1"
Write-Host "4. If successful, run scripts/windows/live.ps1"