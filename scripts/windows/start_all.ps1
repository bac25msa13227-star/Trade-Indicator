# ============================================================
# start_all.ps1 — Khoi dong tat ca: Bot + Dashboard + Tunnel
# Chay script nay mot lan de co moi thu hoat dong
# ============================================================

$Root     = "f:\Trading_BOT_AUTO\Trade-Indicator"
$Python   = Join-Path $Root ".venv\Scripts\python.exe"
$CFExe    = Join-Path $Root "cloudflared.exe"
$OutDir   = Join-Path $Root "outputs"
$UrlFile  = Join-Path $OutDir "tunnel_url.txt"

Set-Location $Root
$null = New-Item -ItemType Directory -Path $OutDir -Force

# ── Kiem tra .env ──────────────────────────────────────────
if (-not (Test-Path "$Root\.env")) {
    Write-Host "[WARN] Khong tim thay .env - dam bao ban da tao file .env" -ForegroundColor Yellow
}

# ── Ham kiem tra port ─────────────────────────────────────
function Test-Port($port) {
    $conn = New-Object System.Net.Sockets.TcpClient
    try { $conn.Connect("127.0.0.1", $port); $conn.Close(); return $true }
    catch { return $false }
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  XAUUSD AI Bot - Khoi dong he thong" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# ── 1. Kill tiến trình cũ nếu có ─────────────────────────
Write-Host "`n[1/3] Don dep tien trinh cu..." -ForegroundColor Yellow
Get-Process -Name python  -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process -Name cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Write-Host "      Done." -ForegroundColor Green

# ── 2. Khởi động Live Bot ─────────────────────────────────
Write-Host "`n[2/3] Khoi dong Live Bot (PID se hien ra sau)..." -ForegroundColor Yellow
$botArgs = "-m xauusd_ai.main live --config configs/live_mt5.yaml"
$botProc = Start-Process -FilePath $Python -ArgumentList $botArgs -WorkingDirectory $Root -NoNewWindow -PassThru
Start-Sleep -Seconds 3
if (Get-Process -Id $botProc.Id -ErrorAction SilentlyContinue) {
    Write-Host "      Live Bot dang chay: PID $($botProc.Id)" -ForegroundColor Green
} else {
    Write-Host "      [WARN] Bot co the da crash. Kiem tra log." -ForegroundColor Red
}

# ── 3. Khởi động Streamlit Dashboard ─────────────────────
Write-Host "`n[3/3] Khoi dong Dashboard Streamlit (port 8501)..." -ForegroundColor Yellow
$dashArgs = "-m streamlit run src/xauusd_ai/dashboard/app.py --server.port 8501 --server.headless true"
$dashProc = Start-Process -FilePath $Python -ArgumentList $dashArgs -WorkingDirectory $Root -NoNewWindow -PassThru
# Doi streamlit san sang
$maxWait = 20
$waited  = 0
Write-Host "      Dang doi Streamlit khoi dong..." -ForegroundColor Gray
while (-not (Test-Port 8501) -and $waited -lt $maxWait) {
    Start-Sleep -Seconds 1
    $waited++
}
if (Test-Port 8501) {
    Write-Host "      Dashboard dang chay: http://localhost:8501  (PID $($dashProc.Id))" -ForegroundColor Green
} else {
    Write-Host "      [WARN] Streamlit chua san sang sau ${maxWait}s." -ForegroundColor Red
}

# ── 4. Tạo Cloudflare Tunnel (public URL) ─────────────────
Write-Host "`n[4/4] Tao Cloudflare Tunnel public URL..." -ForegroundColor Yellow
if (-not (Test-Path $CFExe)) {
    Write-Host "      [WARN] Khong tim thay cloudflared.exe - bo qua tunnel" -ForegroundColor Red
    Write-Host "      URL noi bo: http://localhost:8501" -ForegroundColor Cyan
} else {
    # Chạy cloudflared trong background, capture output vào file tạm
    $logFile = Join-Path $OutDir "tunnel.log"
    $null = Start-Process -FilePath $CFExe `
        -ArgumentList "tunnel --url http://localhost:8501" `
        -WorkingDirectory $Root `
        -RedirectStandardError $logFile `
        -NoNewWindow -PassThru

    # Chờ URL xuất hiện trong log (tối đa 30s)
    Write-Host "      Dang doi tunnel URL..." -ForegroundColor Gray
    $found = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Path $logFile) {
            $content = Get-Content $logFile -Raw -ErrorAction SilentlyContinue
            if ($content -match "https://[a-z0-9\-]+\.trycloudflare\.com") {
                $pubUrl = $Matches[0]
                Set-Content -Path $UrlFile -Value $pubUrl -Encoding UTF8
                $found = $true
                break
            }
        }
    }

    if ($found) {
        Write-Host ""
        Write-Host "================================================" -ForegroundColor Green
        Write-Host "  PUBLIC URL (xem o moi noi):" -ForegroundColor Green
        Write-Host "  $pubUrl" -ForegroundColor White
        Write-Host "  Da luu vao: outputs\tunnel_url.txt" -ForegroundColor Gray
        Write-Host "================================================" -ForegroundColor Green
        Start-Process $pubUrl
    } else {
        Write-Host "      Khong lay duoc URL. Kiem tra outputs\tunnel.log" -ForegroundColor Red
        Write-Host "      Hoac chay thu cong: .\cloudflared.exe tunnel --url http://localhost:8501" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  TOM TAT:" -ForegroundColor Cyan
Write-Host "  Bot PID     : $($botProc.Id)" -ForegroundColor White
Write-Host "  Dashboard   : http://localhost:8501" -ForegroundColor White
if ($found) {
    Write-Host "  Public URL  : $pubUrl" -ForegroundColor Green
}
Write-Host ""
Write-Host "  De dung tat ca: Get-Process python,cloudflared | Stop-Process" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
