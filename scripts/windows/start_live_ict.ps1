# ─────────────────────────────────────────────────────────────────────────────
# start_live_ict.ps1 — Khởi động Live Trading Bot (ICT + Wyckoff Model)
# Chạy: .\scripts\windows\start_live_ict.ps1
# ─────────────────────────────────────────────────────────────────────────────
Set-Location "$PSScriptRoot\..\..\"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  XAUUSD AI LIVE BOT — ICT + Wyckoff (28 features)" -ForegroundColor Cyan
Write-Host "  Config: configs/live_ict_wyckoff.yaml" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# ── Kiểm tra model artifacts ──────────────────────────────────────────────────
$modelOk     = Test-Path "outputs\model_ict_wyckoff.pkl"
$scalerOk    = Test-Path "outputs\scaler_ict_wyckoff.pkl"
$metaOk      = Test-Path "outputs\model_meta_ict_wyckoff.json"

if (-not ($modelOk -and $scalerOk -and $metaOk)) {
    Write-Host "[ERROR] Model artifacts chưa tồn tại! Cần train trước:" -ForegroundColor Red
    Write-Host "  python scripts\train_ict_wyckoff.py"  -ForegroundColor Yellow
    exit 1
}

# Đọc threshold từ meta
$meta = Get-Content "outputs\model_meta_ict_wyckoff.json" | ConvertFrom-Json
Write-Host "[OK] Model loaded:" -ForegroundColor Green
Write-Host "     model   : outputs/model_ict_wyckoff.pkl"
Write-Host "     scaler  : outputs/scaler_ict_wyckoff.pkl"
Write-Host "     threshold: $($meta.decision_threshold)"
Write-Host ""

# ── Kiểm tra .env ─────────────────────────────────────────────────────────────
if (-not (Test-Path ".env")) {
    Write-Host "[ERROR] File .env không tìm thấy!" -ForegroundColor Red
    Write-Host "  Tạo .env với nội dung:" -ForegroundColor Yellow
    Write-Host "  MT5_LOGIN=<login>"
    Write-Host "  MT5_PASSWORD=<password>"
    Write-Host "  MT5_SERVER=Exness-MT5Trial17"
    Write-Host "  TELEGRAM_BOT_TOKEN=<token>"
    Write-Host "  TELEGRAM_CHAT_ID=<chat_id>"
    exit 1
}

Write-Host "[OK] .env found"

# ── Load env vars từ .env ─────────────────────────────────────────────────────
Get-Content ".env" | Where-Object { $_ -notmatch "^#" -and $_ -match "=" } | ForEach-Object {
    $k, $v = $_ -split "=", 2
    [System.Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process")
}

Write-Host "[OK] MT5 Login: $env:MT5_LOGIN @ $env:MT5_SERVER"
Write-Host ""

# ── Set encoding ──────────────────────────────────────────────────────────────
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# ── Kiểm tra MT5 connection trước khi bắt đầu ────────────────────────────────
Write-Host "[ ] Kiểm tra kết nối MT5..." -ForegroundColor Yellow
python -m xauusd_ai.main mt5-check --config configs/live_ict_wyckoff.yaml 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] MT5 check trả về lỗi — kiểm tra MT5 Terminal đang chạy không?" -ForegroundColor Yellow
}
Write-Host ""

# ── Start live bot ────────────────────────────────────────────────────────────
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  STARTING LIVE BOT..." -ForegroundColor Green
Write-Host "  Nhấn Ctrl+C để dừng" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

python -m xauusd_ai.main live --config configs/live_ict_wyckoff.yaml
