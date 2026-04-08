# =============================================================================
#  setup.ps1 - Trade Indicator: bootstrap Windows (PowerShell)
#  Run: .\setup.ps1 [-Bot acc2|acc1|both]
# =============================================================================
param(
    [string]$Bot = "acc2"
)

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoDir

function ok   { param($m) Write-Host "OK   $m" -ForegroundColor Green }
function warn { param($m) Write-Host "WARN $m" -ForegroundColor Yellow }
function info { param($m) Write-Host ">>   $m" -ForegroundColor Cyan }
function die  { param($m) Write-Host "ERR  $m" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "===================================================" -ForegroundColor White
Write-Host "   Trade Indicator -- Setup Bootstrap (Windows)"    -ForegroundColor White
Write-Host "   Bot: $Bot" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor White
Write-Host ""

# STEP 1 -- Prerequisites
info "STEP 1 -- Prerequisites"
try { $dv = (docker --version); ok "Docker: $dv" }
catch { die "Docker not found. Install: https://docs.docker.com/get-docker/" }
$DockerCompose = "docker compose"
try { docker compose version | Out-Null; ok "docker compose: OK" }
catch {
    try { docker-compose --version | Out-Null; $DockerCompose = "docker-compose"; ok "docker-compose: OK" }
    catch { die "docker compose not found." }
}

# STEP 2 -- Git pull
info "STEP 2 -- Git pull"
$Branch = git rev-parse --abbrev-ref HEAD
info "Branch: $Branch"
git pull
$Commit = git rev-parse --short HEAD
ok "HEAD: $Commit"

# STEP 3 -- .env file
info "STEP 3 -- Environment file"
if (Test-Path ".env") {
    ok ".env exists -- skipping"
} elseif (Test-Path ".env.example") {
    Copy-Item ".env.example" ".env"
    ok ".env created from .env.example"
    warn "Edit .env if credentials differ: notepad .env"
} else {
    die ".env.example not found"
}

# STEP 4 -- Directories
info "STEP 4 -- Directories"
New-Item -ItemType Directory -Force -Path "outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "src\xauusd_ai\real_data" | Out-Null
ok "outputs/ and real_data/ ready"

# STEP 5 -- Model artifacts (already in git -- no CSV needed)
# - retrain_on_startup: false  => bot does NOT retrain on start
# - pkl is committed in git at outputs/acc2_scalp_m1_model.pkl
# - live bars come from MT5 Bridge (host.docker.internal:5601)
# - CSV only needed to retrain from scratch: scripts/acc2_scalp_m1_save_model.py
info "STEP 5 -- Model artifacts (in git)"
foreach ($f in @("outputs\acc2_scalp_m1_model.pkl",
                 "outputs\acc2_scalp_m1_scaler.pkl",
                 "outputs\acc2_scalp_m1_model_meta.json")) {
    if (Test-Path $f) { ok "  $f" }
    else              { warn "  Missing: $f" }
}

# STEP 6 -- Build Docker image
info "STEP 6 -- Build Docker image"
switch ($Bot) {
    "acc2" { $Services = @("live-scalp-acc2") }
    "acc1" { $Services = @("live-acc1") }
    "both" { $Services = @("live-scalp-acc2", "live-acc1") }
    default { die "Unknown -Bot: $Bot  (acc2 | acc1 | both)" }
}
Invoke-Expression "$DockerCompose build $($Services -join ' ')"
ok "Build done"

# STEP 7 -- Start bots
info "STEP 7 -- Start bots"
Invoke-Expression "$DockerCompose up -d $($Services -join ' ')"
ok "Containers started"

Write-Host ""
Write-Host "===================================================" -ForegroundColor Green
Write-Host "  SETUP DONE -- commit: $Commit" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Useful commands:"
Write-Host "  $DockerCompose logs live-scalp-acc2 -f"
Write-Host "  Get-Content outputs\live_status_acc2.json"
Write-Host "  $DockerCompose stop live-scalp-acc2"
Write-Host ""
