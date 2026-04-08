# launch_api.ps1 — Start FastAPI/uvicorn dashboard on port 8000
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
$env:MLFLOW_TRACKING_URI = "file:///C:/Temp/mlflowruns"
$env:MINIO_ENDPOINT = "localhost:9000"
Write-Host "[API] Starting uvicorn on port 8000..."
python -m uvicorn xauusd_ai.api.main:app --host 0.0.0.0 --port 8000
