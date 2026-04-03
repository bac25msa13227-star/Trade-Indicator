#!/usr/bin/env pwsh
# restart_lean.ps1 — Stop all, clear pycache, start only essential services
# Services started: postgres, live-acc1, live (ACC2), frontend, api, nginx, cloudflared, healthwatch
# Services skipped: mlflow, airflow, minio, prometheus, grafana

Set-Location 'C:\Users\Administrator\Documents\Trade-Indicator'

Write-Host "=== [1/5] STOPPING ALL CONTAINERS ===" -ForegroundColor Cyan
docker compose down --remove-orphans 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "compose down had issues, force-stopping all..." -ForegroundColor Yellow
    $ids = docker ps -q
    if ($ids) { docker stop $ids 2>&1 | Out-Null }
    $ids = docker ps -aq
    if ($ids) { docker rm -f $ids 2>&1 | Out-Null }
}
Write-Host "All containers stopped." -ForegroundColor Green

Write-Host ""
Write-Host "=== [2/5] CLEARING PYTHON __pycache__ ===" -ForegroundColor Cyan
Get-ChildItem -Path src -Recurse -Filter "__pycache__" -Directory | ForEach-Object {
    Remove-Item $_.FullName -Recurse -Force
    Write-Host "  Removed: $($_.FullName)"
}
Write-Host "Pycache cleared." -ForegroundColor Green

Write-Host ""
Write-Host "=== [3/5] PRUNING UNUSED DOCKER RESOURCES (free memory) ===" -ForegroundColor Cyan
docker system prune -f 2>&1
Write-Host "Docker prune done." -ForegroundColor Green

Write-Host ""
Write-Host "=== [4/5] STARTING ESSENTIAL SERVICES ===" -ForegroundColor Cyan
Write-Host "Starting: postgres (DB)..."
docker compose up -d postgres 2>&1
Start-Sleep 8

Write-Host "Waiting for postgres healthy..."
$retries = 10
for ($i = 0; $i -lt $retries; $i++) {
    $health = docker inspect xauusd-postgres --format "{{.State.Health.Status}}" 2>&1
    if ($health -match "healthy") {
        Write-Host "  postgres: healthy" -ForegroundColor Green
        break
    }
    Write-Host "  postgres not ready yet ($health) — waiting 5s..."
    Start-Sleep 5
}

Write-Host "Starting: ACC1 bot (live-acc1)..."
docker compose up -d live-acc1 2>&1

Write-Host "Starting: ACC2 bot (live)..."
docker compose up -d live 2>&1

Write-Host "Starting: API..."
docker compose up -d api 2>&1
Start-Sleep 5

Write-Host "Starting: Frontend dashboard..."
docker compose up -d frontend 2>&1

Write-Host "Starting: Nginx reverse proxy..."
docker compose up -d nginx 2>&1

Write-Host "Starting: Cloudflared tunnel..."
docker compose up -d cloudflared 2>&1

Write-Host "Starting: Healthwatch (Telegram + market alerts)..."
docker compose up -d healthwatch 2>&1

Write-Host ""
Write-Host "=== [5/5] STATUS CHECK ===" -ForegroundColor Cyan
Start-Sleep 10
docker ps --format "table {{.Names}}`t{{.Status}}`t{{.Ports}}"

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "Services running: postgres, live-acc1, live(ACC2), api, frontend, nginx, cloudflared, healthwatch"
Write-Host "Skipped: mlflow, airflow, minio, prometheus, grafana"
