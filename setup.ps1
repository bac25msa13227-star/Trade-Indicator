# =============================================================================
#  setup.ps1 — Trade Indicator: bootstrap trên Windows (PowerShell)
#  Chạy: .\setup.ps1 [-Bot acc2|acc1|both] [-DataSrc "user@host"]
# =============================================================================
param(
    [string]$Bot      = "acc2",
    [string]$DataSrc  = ""
)

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoDir

function ok   { param($m) Write-Host "✅  $m" -ForegroundColor Green }
function warn { param($m) Write-Host "⚠️   $m" -ForegroundColor Yellow }
function info { param($m) Write-Host "➤  $m"  -ForegroundColor Cyan }
function die  { param($m) Write-Host "❌  $m" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "   Trade Indicator — Setup Bootstrap (Windows)             " -ForegroundColor White
Write-Host "   Bot target: $Bot  |  Dir: $RepoDir" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Prerequisites
# ─────────────────────────────────────────────────────────────────────────────
info "STEP 1 — Checking prerequisites"

try { $dv = (docker --version); ok "Docker: $dv" }
catch { die "Docker not found. Install: https://docs.docker.com/get-docker/" }

$DockerCompose = "docker compose"
try { docker compose version | Out-Null; ok "docker compose: OK" }
catch {
    try { docker-compose --version | Out-Null; $DockerCompose = "docker-compose"; ok "docker-compose: OK" }
    catch { die "docker compose / docker-compose not found." }
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Git pull
# ─────────────────────────────────────────────────────────────────────────────
info "STEP 2 — Git pull"

$Branch = git rev-parse --abbrev-ref HEAD
info "Branch: $Branch"
git pull
$Commit = git rev-parse --short HEAD
ok "HEAD: $Commit"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — .env file
# ─────────────────────────────────────────────────────────────────────────────
info "STEP 3 — Environment file"

if (Test-Path ".env") {
    ok ".env already exists — skipping"
} elseif (Test-Path ".env.example") {
    Copy-Item ".env.example" ".env"
    ok ".env created from .env.example"
    warn "Kiểm tra .env và sửa nếu cần: notepad .env"
} else {
    die ".env.example not found"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Directories
# ─────────────────────────────────────────────────────────────────────────────
info "STEP 4 — Directories"

New-Item -ItemType Directory -Force -Path "outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "src\xauusd_ai\real_data" | Out-Null
ok "outputs\ và real_data\ ready"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — CSV market data
# ─────────────────────────────────────────────────────────────────────────────
info "STEP 5 — Market data CSV"

$RequiredFiles = @("XAUUSDm_D1.csv", "XAUUSDm_H4.csv", "XAUUSDm_H1.csv")
if ($Bot -eq "acc2" -or $Bot -eq "both") { $RequiredFiles += "XAUUSDm_M1.csv" }
if ($Bot -eq "acc1" -or $Bot -eq "both") { $RequiredFiles += "XAUUSDm_M5.csv" }

$Missing = @()
foreach ($f in $RequiredFiles) {
    if (-not (Test-Path "src\xauusd_ai\real_data\$f")) { $Missing += $f }
}

if ($Missing.Count -eq 0) {
    ok "Tất cả CSV data đã có"
} else {
    warn "Thiếu $($Missing.Count) file(s): $($Missing -join ', ')"

    if ($DataSrc -ne "") {
        foreach ($f in $Missing) {
            info "  scp: $f"
            $SrcPath = "${DataSrc}:/path/to/Trade-Indicator/src/xauusd_ai/real_data/$f"
            scp $SrcPath "src\xauusd_ai\real_data\"
        }
    } else {
        Write-Host ""
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
        Write-Host "  Copy CSV từ máy Mac (chạy trên Mac):" -ForegroundColor Yellow
        Write-Host ""
        $TargetDir = (Join-Path $RepoDir "src\xauusd_ai\real_data\")
        $MacSrc = "/Users/dodoannang/Documents/Thac si MSE/Trade Indicator/src/xauusd_ai/real_data"
        foreach ($f in $Missing) {
            Write-Host "  scp `"$MacSrc/$f`" user@WINDOWS_IP:`"$TargetDir`"" -ForegroundColor Cyan
        }
        Write-Host ""
        Write-Host "  Hoặc: .\setup.ps1 -Bot $Bot -DataSrc user@MAC_IP" -ForegroundColor Cyan
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
        Write-Host ""
        $Cont = Read-Host "Tiếp tục mà không có CSV? (bot sẽ fail khi chạy) [y/N]"
        if ($Cont -notmatch "^[Yy]$") {
            warn "Setup dừng — copy CSV rồi chạy lại"
            exit 0
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Model artifacts
# ─────────────────────────────────────────────────────────────────────────────
info "STEP 6 — Model artifacts"

foreach ($f in @("outputs\acc2_scalp_m1_model.pkl",
                 "outputs\acc2_scalp_m1_scaler.pkl",
                 "outputs\acc2_scalp_m1_model_meta.json")) {
    if (Test-Path $f) { ok "  $f" }
    else              { warn "  Thiếu: $f (sẽ tự train khi start bot)" }
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Build Docker image
# ─────────────────────────────────────────────────────────────────────────────
info "STEP 7 — Build Docker image"

switch ($Bot) {
    "acc2" { $Services = @("live-scalp-acc2") }
    "acc1" { $Services = @("live-acc1") }
    "both" { $Services = @("live-scalp-acc2", "live-acc1") }
    default { die "Unknown -Bot value: $Bot (acc2 | acc1 | both)" }
}

Invoke-Expression "$DockerCompose build $($Services -join ' ')"
ok "Build xong"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Start bots
# ─────────────────────────────────────────────────────────────────────────────
info "STEP 8 — Start bots"

Invoke-Expression "$DockerCompose up -d $($Services -join ' ')"
ok "Containers started"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "  ✅  SETUP HOÀN TẤT — commit: $Commit" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host ""
Write-Host "  Commands hữu ích:"
Write-Host ""
Write-Host "  # Xem live logs" -ForegroundColor Cyan
Write-Host "  $DockerCompose logs live-scalp-acc2 -f"
Write-Host ""
Write-Host "  # Trạng thái bot" -ForegroundColor Cyan
Write-Host "  Get-Content outputs\live_status_acc2.json"
Write-Host ""
Write-Host "  # Dừng bot" -ForegroundColor Cyan
Write-Host "  $DockerCompose stop live-scalp-acc2"
Write-Host ""
