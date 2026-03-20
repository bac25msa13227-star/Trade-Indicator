# ============================================================
# start_tunnel.ps1 — Tạo public URL cho Dashboard qua Cloudflare Tunnel
# Chạy cloudflared ngầm + tự đọc URL từ log file.
# ============================================================

$Root     = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$CFExe    = Join-Path $Root "cloudflared.exe"
$UrlFile  = Join-Path $Root "outputs\tunnel_url.txt"
$LogFile  = Join-Path $Root "outputs\tunnel_log.txt"

# Đảm bảo outputs/ tồn tại
$null = New-Item -ItemType Directory -Path (Join-Path $Root "outputs") -Force

# ── Tự động tải cloudflared.exe nếu chưa có ──────────────────────────────
if (-not (Test-Path $CFExe)) {
    Write-Host "[INFO] cloudflared.exe chua co. Dang tai tu GitHub..." -ForegroundColor Yellow
    $DownloadUrl = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $DownloadUrl -OutFile $CFExe -UseBasicParsing
        $ProgressPreference = 'Continue'
        Write-Host "[OK] Da tai cloudflared.exe thanh cong!" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Khong tai duoc cloudflared.exe: $_" -ForegroundColor Red
        Write-Host "  Tai thu cong tai: https://github.com/cloudflare/cloudflared/releases/latest"
        exit 1
    }
}

# Kill process cloudflared cu neu dang chay
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 500

# Xoa file cu
Remove-Item -Path $UrlFile  -ErrorAction SilentlyContinue
Remove-Item -Path $LogFile  -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Cloudflare Tunnel -> http://localhost:8501" -ForegroundColor Cyan
Write-Host "  Dang khoi dong tunnel ngam..." -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# Chay cloudflared NHAM (redirect ca stdout+stderr vao log file)
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName        = $CFExe
$psi.Arguments       = "tunnel --url http://localhost:8501 --protocol http2"
$psi.WorkingDirectory = $Root
$psi.UseShellExecute  = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$psi.CreateNoWindow   = $true

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi

# Su kien bat output/error -> ghi vao log
$logLock = [System.Object]::new()
$sbLog   = [System.Text.StringBuilder]::new()

$outputAction = {
    param($sender, $e)
    if ($e.Data) {
        $null = $sbLog.AppendLine($e.Data)
        Add-Content -Path $LogFile -Value $e.Data -Encoding UTF8 -ErrorAction SilentlyContinue
    }
}
Register-ObjectEvent -InputObject $proc -EventName OutputDataReceived -Action $outputAction | Out-Null
Register-ObjectEvent -InputObject $proc -EventName ErrorDataReceived  -Action $outputAction | Out-Null

$proc.Start() | Out-Null
$proc.BeginOutputReadLine()
$proc.BeginErrorReadLine()

Write-Host "[INFO] cloudflared PID=$($proc.Id) dang chay..." -ForegroundColor Yellow

# Cho toi 30 giay de URL xuat hien trong log
$deadline = [DateTime]::UtcNow.AddSeconds(30)
$foundUrl = ""
while ([DateTime]::UtcNow -lt $deadline -and $foundUrl -eq "") {
    Start-Sleep -Milliseconds 800
    $logContent = $sbLog.ToString()
    if ($logContent -match "https://[a-z0-9\-]+\.(trycloudflare|cfargotunnel)\.com") {
        $foundUrl = $Matches[0]
    }
}

if ($foundUrl -ne "") {
    Set-Content -Path $UrlFile -Value $foundUrl -Encoding UTF8
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "  PUBLIC URL: $foundUrl"                     -ForegroundColor Green
    Write-Host "  Luu tai   : outputs\tunnel_url.txt"        -ForegroundColor Green
    Write-Host "  Dashboard : http://localhost:8501"         -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Green

    # Gui Telegram thong bao URL moi
    $PyExe = "C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe"
    $EnvFile = Join-Path $Root ".env"
    $SendScript = @"
import os, pathlib, sys
try:
    ef = pathlib.Path(r'$EnvFile')
    if ef.exists():
        for line in ef.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())
    import requests
    token   = os.getenv('TELEGRAM_BOT_TOKEN_ACC2') or os.getenv('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.getenv('TELEGRAM_CHAT_ID_ACC2')   or os.getenv('TELEGRAM_CHAT_ID', '')
    url     = r'$foundUrl'
    if token and chat_id and url:
        r = requests.post(
            f'https://api.telegram.org/bot{token}/sendMessage',
            json={'chat_id': chat_id, 'text': f'🌐 Dashboard URL mới:\n{url}', 'parse_mode': 'HTML'},
            timeout=10
        )
        print('Telegram sent:', r.status_code)
    else:
        print('Telegram skip: token/chat_id empty')
except Exception as e:
    print('Telegram error:', e)
"@
    try {
        $result = & $PyExe -c $SendScript 2>&1
        Write-Host "[Telegram] $result" -ForegroundColor Cyan
    } catch {
        Write-Host "[Telegram] Loi gui thong bao: $_" -ForegroundColor Yellow
    }

    # Mo browser
    Start-Process $foundUrl
} else {
    Write-Host "[WARN] Khong tim duoc URL sau 30 giay." -ForegroundColor Yellow
    Write-Host "  Xem log: $LogFile"
    # In 20 dong cuoi cua log de debug
    if (Test-Path $LogFile) {
        Write-Host "--- Tail log ---" -ForegroundColor Gray
        Get-Content $LogFile -Tail 20
    }
}
