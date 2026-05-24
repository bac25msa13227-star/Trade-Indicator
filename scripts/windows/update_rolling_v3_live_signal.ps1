param(
    [string]$BaseRates = "outputs\mt5_rates_export_202306_20260513_merged.csv",
    [string]$LiveRates = "outputs\mt5_rates_export_202306_20260513_live_merged.csv",
    [string]$LiveFeatures = "outputs\mt5_full_ict_wyckoff_features_live_current.csv",
    # WF-validated MT5 feedback/regime-family selector:
    # 30/30 target pass, 30/30 DD pass, loaded_signal_max_abs_gap=0 on 2026-05-14.
    [string]$Fold29Source = "outputs\target1200_live_rolling_v3_xauusdm_risk_sweep\f29_k250_r60",
    [string]$Fold30Source = "outputs\target1200_live_rolling_v3_xauusdm_risk_sweep\f30_r70",
    # Fail-closed full MT5 validation. If this exists, guard uses it instead of only the latest 2 folds.
    [string]$CleanValidationSource = "outputs\target1200_regime_rule_family_selector_v1_20260514",
    [string]$SelectedOut = "outputs\target1200_live_rolling_v3_current_selected",
    [string]$BridgeUrl = "http://localhost:5601",
    [int]$Bars = 20000,
    [double]$StressExtraRoundtripPoints = 100.0,
    [double]$StressThinHourExtraPoints = 100.0,
    [double]$StressFridayExtraPoints = 150.0,
    [double]$StressMaxDDPct = 20.0,
    [switch]$RegenerateCurrentFold,
    [switch]$SkipRealisticGate,
    [switch]$CopySignalToMt5ForTester,
    [switch]$CompileLiveEA
)

$ErrorActionPreference = "Stop"
$root = "f:\Trading_BOT_AUTO\Trade-Indicator"
Set-Location $root
$env:PYTHONWARNINGS = "ignore::FutureWarning"
$FeatureToDate = (Get-Date).Date.AddDays(1).ToString("yyyy-MM-dd")

Write-Output "[1/8] Updating live MT5 M5 rates from Python API..."
python scripts\update_mt5_rates_from_live.py `
    --old $BaseRates `
    --out $LiveRates `
    --symbol XAUUSD `
    --timeframe M5 `
    --bars $Bars `
    --bridge-url $BridgeUrl

Write-Output "[2/8] Rebuilding MT5 ICT/Wyckoff features..."
python scripts\build_mt5_full_features.py `
    --rates $LiveRates `
    --out $LiveFeatures `
    --from-date 2023-06-01 `
    --to-date $FeatureToDate

if ($RegenerateCurrentFold) {
    Write-Output "[3/8] Regenerating current fold-30 signal source (WARNING: requires MT5 WF re-run after)..."
    $regenDir = "outputs\target1200_live_rolling_v3_current_tests\tp25_h12_p50_k150_r7_live"
    python scripts\export_rolling_v3_current_campaign.py `
        --features $LiveFeatures `
        --base-manifest outputs\target1200_live_rolling_v3\manifest.json `
        --source-manifest outputs\target1200_mt5full_ict_adaptive_combo_mt5v2\manifest.json `
        --out-dir $regenDir `
        --tp-rr 2.5 `
        --sl-mult 1.0 `
        --horizon-bars 12 `
        --min-probability 0.50 `
        --top-k 150 `
        --risk-pct 7.0 `
        --max-risk-pct 7.0 `
        --max-exposure-pct 7.0 `
        --max-positions 1
    Write-Output "WARNING: Regenerated signals written to $regenDir - NOT deployed. Re-run MT5 WF then update Fold30Source."
}

Write-Output "[4/8] Refreshing selected MT5 WF validation rows..."
New-Item -ItemType Directory -Path $SelectedOut -Force | Out-Null
$selectedResults = Join-Path $SelectedOut "mt5_wf_results.csv"
$selectedJson = Join-Path $SelectedOut "mt5_wf_results.json"
$validationRows = @()
$sourceDirs = @($Fold29Source, $Fold30Source)
$onlineSourceManifest = Join-Path $Fold30Source "manifest.json"
$cleanSelectorManifest = $null
if (-not [string]::IsNullOrWhiteSpace($CleanValidationSource)) {
    $cleanResults = Join-Path $CleanValidationSource "mt5_wf_results.csv"
    if (Test-Path $cleanResults) {
        Write-Output "Using clean full MT5 validation source: $cleanResults"
        $sourceDirs = @($CleanValidationSource)
        $cleanManifest = Join-Path $CleanValidationSource "manifest.json"
        if (Test-Path $cleanManifest) {
            $onlineSourceManifest = $cleanManifest
            $cleanSelectorManifest = $cleanManifest
        }
    }
}
if ([string]::IsNullOrWhiteSpace($cleanSelectorManifest) -or -not (Test-Path $cleanSelectorManifest)) {
    throw "Clean selector manifest is required for live refresh. Refusing to use adaptive/current-test source."
}
foreach ($sourceDir in $sourceDirs) {
    $sourceResults = Join-Path $sourceDir "mt5_wf_results.csv"
    if (-not (Test-Path $sourceResults)) {
        throw "Missing source MT5 result: $sourceResults"
    }
    foreach ($row in (Import-Csv $sourceResults)) {
        if (
            [string]::IsNullOrWhiteSpace($row.final_balance) -or
            [string]::IsNullOrWhiteSpace($row.loaded_signals) -or
            [string]::IsNullOrWhiteSpace($row.max_dd_pct)
        ) {
            throw "Invalid or empty MT5 result in $sourceResults"
        }
        $validationRows += $row
    }
}
$validationRows = $validationRows | Sort-Object { [int]$_.fold }
$validationRows | Export-Csv -NoTypeInformation -Path $selectedResults
$validationRows | ConvertTo-Json -Depth 6 | Set-Content -Path $selectedJson -Encoding UTF8

if (-not $SkipRealisticGate) {
    Write-Output "[5/8] Rebuilding MT5 trade feedback and realistic stress gate..."
    $feedbackOut = Join-Path $SelectedOut "mt5_trade_feedback_detailed.csv"
    $realisticReport = Join-Path $SelectedOut "realistic_gate_report.json"
    $realisticFolds = Join-Path $SelectedOut "realistic_gate_folds.csv"
    python scripts\export_mt5_trade_feedback.py `
        --manifest $cleanSelectorManifest `
        --results $selectedResults `
        --out $feedbackOut
    if ($LASTEXITCODE -ne 0) {
        throw "MT5 trade feedback export failed."
    }
    python scripts\mt5_realistic_stress_gate.py `
        --feedback $feedbackOut `
        --out $realisticReport `
        --fold-report-out $realisticFolds `
        --max-dd-pct $StressMaxDDPct `
        --extra-roundtrip-points $StressExtraRoundtripPoints `
        --thin-hour-extra-points $StressThinHourExtraPoints `
        --friday-extra-points $StressFridayExtraPoints
    if ($LASTEXITCODE -ne 0) {
        throw "Realistic stress gate generation failed."
    }
    $gate = Get-Content $realisticReport -Raw | ConvertFrom-Json
    if (-not $gate.summary.realistic_gate_pass) {
        Write-Warning (
            "Realistic MT5 stress gate failed; execute will be blocked. " +
            "target_pass=$($gate.summary.target_pass_folds)/$($gate.summary.folds), " +
            "dd_pass=$($gate.summary.dd_pass_folds)/$($gate.summary.folds), " +
            "loss_folds=$($gate.summary.loss_folds)"
        )
    }
} else {
    Write-Output "[5/8] Skipping realistic stress gate."
}

Write-Output "[6/8] Generating online per-M5 decision/signal stream with clean MT5-feedback selector..."
python scripts\generate_mt5_feedback_online_signal.py `
    --features $LiveFeatures `
    --out-dir $SelectedOut `
    --bridge-url $BridgeUrl `
    --selector-manifest $cleanSelectorManifest `
    --selector-results $selectedResults `
    --paper-risk-pct 1.0 `
    --paper-max-exposure-pct 1.0
if ($LASTEXITCODE -ne 0) {
    throw "Online signal generation failed."
}

Write-Output "[7/8] Running guard for bridge live..."
$prepareArgs = @(
    "scripts\windows\prepare_rolling_v3_canary.ps1",
    "-Manifest", (Join-Path $SelectedOut "manifest.json"),
    "-PreflightReport", (Join-Path $SelectedOut "live_preflight_report.json"),
    "-Results", (Join-Path $SelectedOut "mt5_wf_results.csv"),
    "-Features", $LiveFeatures,
    "-OutReport", (Join-Path $SelectedOut "live_canary_guard_report.json"),
    "-MaxSignalFileAgeMinutes", "240"
)
if ($CopySignalToMt5ForTester) {
    $prepareArgs += "-CopySignalToMt5"
}
if ($CompileLiveEA) {
    $prepareArgs += "-CompileLiveEA"
}
& powershell.exe -ExecutionPolicy Bypass -File @prepareArgs
if ($LASTEXITCODE -ne 0) {
    throw "Rolling v3 canary guard blocked live start."
}

Write-Output "[8/8] Bridge live signal update complete. EA/MetaEditor copy is disabled unless -CopySignalToMt5ForTester is used."
