# ============================================================
# stop_all.ps1 — Dung bot, dashboard, tunnel
# ============================================================

Write-Host "Dang dung tat ca..." -ForegroundColor Yellow

Get-Process -Name cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process -Name python      -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host "Da dung: bot, dashboard, tunnel." -ForegroundColor Green
