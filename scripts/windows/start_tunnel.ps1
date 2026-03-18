# ============================================================
# start_tunnel.ps1 — Tạo public URL cho Dashboard qua Cloudflare Tunnel
# Không cần tài khoản, không cần cài đặt thêm gì.
# ============================================================

$Root     = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$CFExe    = Join-Path $Root "cloudflared.exe"
$UrlFile  = Join-Path $Root "outputs\tunnel_url.txt"

if (-not (Test-Path $CFExe)) {
    Write-Host "[ERROR] Khong tim thay cloudflared.exe tai: $CFExe" -ForegroundColor Red
    Write-Host "Chay script nay de tai ve:" -ForegroundColor Yellow
    Write-Host '  Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile cloudflared.exe'
    exit 1
}

# Đảm bảo outputs/ tồn tại
$null = New-Item -ItemType Directory -Path (Join-Path $Root "outputs") -Force

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Cloudflare Tunnel -> http://localhost:8501" -ForegroundColor Cyan
Write-Host "  Doi URL xuat hien o dong 'trycloudflare.com'..." -ForegroundColor Cyan
Write-Host "  Nhan Ctrl+C de dung tunnel" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Chạy tunnel với --protocol http2 (tránh QUIC bị block bởi firewall)
& $CFExe tunnel --url http://localhost:8501 --protocol http2 2>&1 | ForEach-Object {
    $line = $_
    Write-Host $line
    if ($line -match "https://[a-z0-9\-]+\.trycloudflare\.com") {
        $url = $Matches[0]
        Write-Host ""
        Write-Host "============================================" -ForegroundColor Green
        Write-Host "  PUBLIC URL: $url" -ForegroundColor Green  
        Write-Host "  Da luu vao: outputs\tunnel_url.txt" -ForegroundColor Green
        Write-Host "============================================" -ForegroundColor Green
        Write-Host ""
        Set-Content -Path $UrlFile -Value $url -Encoding UTF8
        # Mo browser
        Start-Process $url
    }
}
