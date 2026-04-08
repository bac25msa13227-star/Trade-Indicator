#Requires -Version 5.1
<#
.SYNOPSIS
    Trade Indicator — Automated Setup (Windows PowerShell)

.DESCRIPTION
    Automates: Docker check, git pull, .env creation, Docker build, start bot.

.PARAMETER Bot
    Which bot to run: acc2 (default), acc1, or both

.EXAMPLE
    .\setup.ps1 -Bot acc2
    .\setup.ps1 -Bot both
#>
param(
    [ValidateSet("acc2","acc1","both")]
    [string]$Bot = "acc2"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Ok   { param($msg) Write-Host "[OK]    $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Err  { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

$Repo = $PSScriptRoot

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Trade Indicator — Automated Setup (Windows)           ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Set-Location $Repo

# ── 1. Check Docker ──────────────────────────────────────────────────────────
Write-Host "[1/7] Checking Docker..."
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Err "Docker is not running. Start Docker Desktop first." }
    Write-Ok "Docker running"
} catch {
    Write-Err "Docker not found. Install from https://docs.docker.com/get-docker/"
}

# ── 2. git pull + lfs ────────────────────────────────────────────────────────
Write-Host "[2/7] Pulling latest code..."
try {
    git pull 2>&1 | Out-Null
    Write-Ok "git pull done"
} catch {
    Write-Warn "git pull failed — continuing with current code"
}
try {
    git lfs pull 2>&1 | Out-Null
    Write-Ok "git lfs pull done (.pkl model files downloaded)"
} catch {
    Write-Warn "git lfs pull failed — model files may be missing (will auto-train)"
}

# ── 3. .env setup ────────────────────────────────────────────────────────────
Write-Host "[3/7] Checking .env..."
$envFile = Join-Path $Repo ".env"
$envExample = Join-Path $Repo ".env.example"
if (-not (Test-Path $envFile)) {
    if (Test-Path $envExample) {
        Copy-Item $envExample $envFile
        Write-Warn ".env created from .env.example"
        Write-Host "      Edit .env with your MT5 credentials before continuing." -ForegroundColor Yellow
        Write-Host "      Required: MT5_LOGIN_ACC2, MT5_PASSWORD_ACC2, MT5_SERVER_ACC2" -ForegroundColor Yellow
        Write-Host "      Required: TELEGRAM_BOT_TOKEN_ACC2, TELEGRAM_CHAT_ID_ACC2" -ForegroundColor Yellow
        notepad.exe $envFile
        Read-Host "      Press ENTER after saving .env to continue"
    } else {
        Write-Err ".env.example not found. Cannot create .env."
    }
} else {
    Write-Ok ".env exists"
}

# ── 4. outputs/ and logs/ directories ────────────────────────────────────────
Write-Host "[4/7] Checking directories..."
$null = New-Item -ItemType Directory -Force -Path (Join-Path $Repo "outputs")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $Repo "logs")
Write-Ok "outputs/ and logs/ directories ready"

# ── 5. Check model .pkl ──────────────────────────────────────────────────────
Write-Host "[5/7] Checking model artifacts..."
function Test-Pkl {
    param([string]$Path)
    if (Test-Path $Path) {
        $size = (Get-Item $Path).Length
        if ($size -gt 10000) {
            $kb = [math]::Round($size / 1KB)
            Write-Ok "$Path ($($kb)KB)"
        } else {
            Write-Warn "$Path looks like an LFS pointer stub — run: git lfs pull"
        }
    } else {
        Write-Warn "$Path not found — will auto-train on first start (~4 min)"
    }
}

if ($Bot -in @("acc2","both")) { Test-Pkl "outputs\acc2_scalp_m1_model.pkl" }
if ($Bot -in @("acc1","both")) { Test-Pkl "outputs\acc1_expand_net127313_dd3215_model.pkl" }

# ── 6. Build Docker image ────────────────────────────────────────────────────
Write-Host "[6/7] Building Docker image(s)..."
$services = switch ($Bot) {
    "acc2" { @("live-scalp-acc2") }
    "acc1" { @("live-acc1") }
    "both" { @("live-scalp-acc2","live-acc1") }
}
foreach ($svc in $services) {
    Write-Host "      Building $svc..." -ForegroundColor Gray
    docker compose build $svc
    if ($LASTEXITCODE -ne 0) { Write-Err "docker compose build $svc failed" }
}
Write-Ok "Docker image(s) built"

# ── 7. Start bot ─────────────────────────────────────────────────────────────
Write-Host "[7/7] Starting bot(s)..."
foreach ($svc in $services) {
    docker compose up -d $svc
    if ($LASTEXITCODE -ne 0) { Write-Err "docker compose up $svc failed" }
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Ok " Setup complete!"
Write-Host "║"
Write-Host "║  Logs:    docker compose logs live-scalp-acc2 -f"
Write-Host "║  Status:  Get-Content outputs\live_status_acc2.json"
Write-Host "║  Stop:    docker compose stop live-scalp-acc2"
Write-Host "║"
Write-Host "║  ⚠️  Make sure MT5 Bridge is running on host port 5601" -ForegroundColor Yellow
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
