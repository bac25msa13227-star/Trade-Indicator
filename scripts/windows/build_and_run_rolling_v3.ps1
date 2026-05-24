param(
    [string]$OutDir = "outputs/target1200_live_rolling_v3"
)

# Rolling v3:
# - fold 01 bootstrap map: row1970 post-select exclude hour 12 specialist
# - fold 02 bootstrap: original adaptive combo
# - fold 03 bootstrap map: row24 excluded-hours r3.1 specialist
# - fold 04..28: selected from prior MT5 feedback only

$ErrorActionPreference = "Stop"
$root = "f:\Trading_BOT_AUTO\Trade-Indicator"
Set-Location $root

$base = "outputs/target1200_mt5full_ict_adaptive_combo_mt5v2"
$fold1 = "outputs/target1200_row1970_postexcl12_r175_mp2_28fold"
$fold3 = "outputs/target1200_row24_excl_1_5_12_16_19_23_r31_mp1_28fold"
$row1970 = "outputs/target1200_row1970_r175_mp2_28fold"
$row24 = "outputs/target1200_row24_r35_mp1_28fold"

$required = @($base, $fold1, $fold3, $row1970, $row24)
foreach ($dir in $required) {
    if (-not (Test-Path (Join-Path $dir "manifest.json"))) {
        throw "Missing manifest: $dir"
    }
    if (-not (Test-Path (Join-Path $dir "mt5_wf_results.csv"))) {
        throw "Missing MT5 results: $dir"
    }
}

Write-Output "[$(Get-Date -Format 'HH:mm:ss')] Building rolling selector v3..."
& python scripts/export_mt5_feedback_rolling_selector.py `
    --candidate-dir $base `
    --candidate-dir $fold1 `
    --candidate-dir $fold3 `
    --candidate-dir $row1970 `
    --candidate-dir $row24 `
    --out-dir $OutDir `
    --bootstrap-candidate-dir $base `
    --bootstrap-map "1=$fold1" `
    --bootstrap-map "3=$fold3" `
    --min-history-folds 3

if ($LASTEXITCODE -ne 0) {
    throw "Rolling selector v3 export failed"
}

Write-Output "[$(Get-Date -Format 'HH:mm:ss')] Running MT5 WF for rolling selector v3..."
$logFile = Join-Path $OutDir "wf_run.log"
$cmd = ".\scripts\run_mt5_wf_folds.ps1 -Manifest '$OutDir\manifest.json' -MaxDDKillPct 20 -Deposit 200 -TrailingEnabled 0 -MaxDailyLoss 100 *> '$logFile'"
$proc = Start-Process powershell -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-Command",$cmd -PassThru -WorkingDirectory $root -Wait
if ($proc.ExitCode -ne 0) {
    throw "MT5 WF failed with exit code $($proc.ExitCode)"
}

$maxRisk = (& python -c "import pandas as pd; r=pd.read_csv('$OutDir/mt5_wf_results.csv'); print(max(r['risk_pct']))").Trim()
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] Running strict preflight..."
& python scripts/live_protocol_preflight.py `
    --manifest "$OutDir/manifest.json" `
    --results "$OutDir/mt5_wf_results.csv" `
    --out "$OutDir/live_preflight_report_strict.json" `
    --deposit 200 `
    --target-balance 1200 `
    --min-dd-pct -20 `
    --max-target-fails 0 `
    --max-dd-fails 0 `
    --max-loss-folds 0 `
    --max-risk-pct $maxRisk `
    --max-exposure-pct 7 `
    --max-positions 2 `
    --max-loaded-signal-gap 0 `
    --no-fail-exit

Write-Output "[$(Get-Date -Format 'HH:mm:ss')] Running bootstrap-approved preflight..."
& python scripts/live_protocol_preflight.py `
    --manifest "$OutDir/manifest.json" `
    --results "$OutDir/mt5_wf_results.csv" `
    --out "$OutDir/live_preflight_report.json" `
    --deposit 200 `
    --target-balance 1200 `
    --min-dd-pct -20 `
    --max-target-fails 0 `
    --max-dd-fails 0 `
    --max-loss-folds 0 `
    --max-risk-pct $maxRisk `
    --max-exposure-pct 7 `
    --max-positions 2 `
    --max-loaded-signal-gap 0 `
    --allow-bootstrap `
    --no-fail-exit

Write-Output "DONE: $OutDir"
