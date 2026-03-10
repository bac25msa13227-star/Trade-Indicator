$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
& $python -m xauusd_ai.main live --config configs/live_mt5.yaml