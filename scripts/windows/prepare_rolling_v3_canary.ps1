param(
    [string]$Manifest = "outputs/target1200_live_rolling_v3_current_selected/manifest.json",
    [string]$PreflightReport = "outputs/target1200_live_rolling_v3_current_selected/live_preflight_report.json",
    [string]$Results = "outputs/target1200_live_rolling_v3_current_selected/mt5_wf_results.csv",
    [string]$Features = "",
    [string]$OutReport = "outputs/target1200_live_rolling_v3_current_selected/live_canary_guard_report.json",
    [double]$MaxSignalFileAgeMinutes = 240.0,
    [switch]$CopySignalToMt5,
    [switch]$CompileLiveEA
)

$ErrorActionPreference = "Stop"
$root = "f:\Trading_BOT_AUTO\Trade-Indicator"
Set-Location $root

if ([string]::IsNullOrWhiteSpace($Features)) {
    $latestFeatures = Get-ChildItem "outputs\mt5_full_ict_wyckoff_features_*.csv" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $latestFeatures) {
        throw "No mt5_full_ict_wyckoff_features_*.csv file found under outputs."
    }
    $Features = $latestFeatures.FullName
}

$guardArgs = @(
    "scripts/rolling_live_canary_guard.py",
    "--manifest", $Manifest,
    "--preflight-report", $PreflightReport,
    "--results", $Results,
    "--features", $Features,
    "--out", $OutReport,
    "--deposit", "200",
    "--target-balance", "1200",
    "--min-dd-pct", "-20",
    "--max-risk-pct", "1",
    "--max-exposure-pct", "1",
    "--max-positions", "1",
    "--max-loaded-signal-gap", "0",
    "--max-signal-file-age-minutes", "$MaxSignalFileAgeMinutes",
    "--max-signal-horizon-lag-minutes", "15"
)

python @guardArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "Rolling v3 canary guard blocked live start. See $OutReport"
    exit $LASTEXITCODE
}

if ($CopySignalToMt5) {
    $report = Get-Content $OutReport -Raw | ConvertFrom-Json
    $signalPath = $report.current_signal.path
    if (-not $signalPath -or -not (Test-Path $signalPath)) {
        Write-Error "Guard passed but current signal file is missing: $signalPath"
        exit 2
    }

    $terminalFiles = "C:\Users\doanbacremote\AppData\Roaming\MetaQuotes\Terminal\53785E099C927DB68A545C249CDBCE06\MQL5\Files"
    $commonFiles = "C:\Users\doanbacremote\AppData\Roaming\MetaQuotes\Terminal\Common\Files"
    New-Item -ItemType Directory -Path $terminalFiles -Force | Out-Null
    New-Item -ItemType Directory -Path $commonFiles -Force | Out-Null
    Copy-Item $signalPath (Join-Path $terminalFiles "signals_for_mt5.csv") -Force
    Copy-Item $signalPath (Join-Path $commonFiles "signals_for_mt5.csv") -Force
    Write-Output "Copied current canary signal to MT5 Files/Common Files."
}

if ($CompileLiveEA) {
    $metaEditor = "C:\Program Files\MetaTrader 5 EXNESS\MetaEditor64.exe"
    $terminalId = "53785E099C927DB68A545C249CDBCE06"
    $expertsDir = "C:\Users\doanbacremote\AppData\Roaming\MetaQuotes\Terminal\$terminalId\MQL5\Experts"
    $sourceMq5 = Join-Path $root "scripts\mt5\AI_Signal_Reader_Live.mq5"
    $destMq5 = Join-Path $expertsDir "AI_Signal_Reader_Live.mq5"
    $compileLog = "C:\Users\doanbacremote\tradingview-mcp\compile_ai_signal_reader_live_log.txt"
    if (-not (Test-Path $metaEditor)) { throw "MetaEditor not found: $metaEditor" }
    if (-not (Test-Path $sourceMq5)) { throw "Live EA source missing: $sourceMq5" }
    New-Item -ItemType Directory -Path $expertsDir -Force | Out-Null
    Copy-Item $sourceMq5 $destMq5 -Force
    if (Test-Path $compileLog) { Remove-Item $compileLog -Force }
    $compileArgs = "/compile:`"$destMq5`" /log:`"$compileLog`""
    Start-Process -FilePath $metaEditor -ArgumentList $compileArgs -Wait -WindowStyle Hidden
    Start-Sleep -Seconds 3
    $destEx5 = Join-Path $expertsDir "AI_Signal_Reader_Live.ex5"
    if (-not (Test-Path $destEx5)) {
        if (Test-Path $compileLog) { Get-Content $compileLog -Encoding Unicode -ErrorAction SilentlyContinue }
        throw "Compile failed: $destEx5 not found"
    }
    Write-Output "Compiled live EA: $destEx5"

    $report = Get-Content $OutReport -Raw | ConvertFrom-Json
    $fold = $report.current_fold
    $setOut = Join-Path (Split-Path $OutReport -Parent) "AI_Signal_Reader_Live_current.set"
    $setContent = "; rolling_v3 current selected live inputs`r`n"
    $setContent += "InpCSVFile=signals_for_mt5.csv`r`n"
    $setContent += "InpReloadCsvOnNewBar=true`r`n"
    $setContent += "InpSkipHistoricalSignalsOnInit=true`r`n"
    $setContent += "InpRiskPct=$($fold.risk_pct)`r`n"
    $setContent += "InpMaxPositions=$($fold.max_positions)`r`n"
    $setContent += "InpMaxDailyLoss=100`r`n"
    $setContent += "InpMaxRiskPct=$($fold.max_risk_pct)`r`n"
    $setContent += "InpMaxExposurePct=$($fold.max_exposure_pct)`r`n"
    $setContent += "InpMaxDDKillPct=20`r`n"
    $setContent += "InpMaxPeakDDKillPct=0`r`n"
    $setContent += "InpTargetBalanceStop=0`r`n"
    $setContent += "InpMinEntryDriftR=-999`r`n"
    $setContent += "InpMaxEntryDriftR=999`r`n"
    $setContent += "InpTrailingEnabled=false`r`n"
    $setContent += "InpBreakevenRR=0.5`r`n"
    $setContent += "InpTrailActivationRR=1.0`r`n"
    $setContent += "InpTrailAtrMultiple=1.0`r`n"
    $setContent += "InpMaxHoldBars=0`r`n"
    $setContent += "InpCloseOnStop=true`r`n"
    $setContent += "InpMagic=13001`r`n"
    $setContent += "InpVerbose=false`r`n"
    [System.IO.File]::WriteAllText($setOut, $setContent)
    Write-Output "Wrote live EA set file: $setOut"
}

Write-Output "Rolling v3 canary guard passed: $OutReport"
