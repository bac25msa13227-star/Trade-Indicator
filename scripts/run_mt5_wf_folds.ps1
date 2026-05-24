param(
    [string]$Manifest = "outputs\mt5_wf_r2_5\manifest.json",
    [string]$Runner = "C:\Users\doanbacremote\tradingview-mcp\Run-AISignalTest.ps1",
    [string]$RunnerWorkDir = "C:\Users\doanbacremote\tradingview-mcp",
    [double]$RiskPct = 2.0,
    [int]$MaxPositions = 3,
    [double]$MaxDailyLoss = 5.0,
    [double]$MaxRiskPct = 2.0,
    [double]$MaxExposurePct = 6.0,
    [double]$MaxDDKillPct = 15.0,
    [double]$MaxPeakDDKillPct = 0.0,
    [double]$TargetBalanceStop = 0.0,
    [double]$MinEntryDriftR = -999.0,
    [double]$MaxEntryDriftR = 999.0,
    [int]$TrailingEnabled = 1,
    [double]$BreakevenRR = 0.5,
    [double]$TrailActivationRR = 1.0,
    [double]$TrailAtrMultiple = 1.0,
    [int]$MaxHoldBars = 0,
    [switch]$ForceCliRisk,
    [int]$Deposit = 200,
    [string]$Symbol = "XAUUSDm",
    [string]$Period = "M5",
    [int]$Model = 0,
    [int]$WaitSeconds = 300,
    [int]$StartFold = 1,
    [int]$EndFold = 999,
    [string]$PortableRoot = "",
    [switch]$KillExistingMT5
)

$ErrorActionPreference = "Stop"

function Convert-DateForMt5([string]$DateText) {
    return $DateText.Replace("-", ".")
}

function Parse-RunOutput([string[]]$Lines) {
    $text = $Lines -join "`n"
    $finalBalance = $null
    $netPnl = $null
    $netPct = $null
    $trades = $null
    $tpHits = $null
    $slHits = $null
    $winRate = $null
    $report = $null

    if ($text -match "Report will be:\s*(.+?\.html)") {
        $report = $Matches[1].Trim()
    }
    if ($text -match "Final Bal\s+:\s*(-?[0-9.]+)\s+USD") {
        $finalBalance = [double]$Matches[1]
    }
    if ($text -match "Net PnL\s+:\s*([+-]?\$[0-9.]+)\s+\(([+-]?[0-9.]+)%\)") {
        $netPnl = $Matches[1]
        $netPct = [double]$Matches[2]
    }
    if ($text -match "Trades\s+:\s*([0-9]+)") {
        $trades = [int]$Matches[1]
    }
    if ($text -match "TP hits\s+:\s*([0-9]+)\s+\|\s+SL hits:\s*([0-9]+)") {
        $tpHits = [int]$Matches[1]
        $slHits = [int]$Matches[2]
    }
    if ($text -match "Win rate\s+:\s*([0-9.]+)%") {
        $winRate = [double]$Matches[1]
    }
    if ($finalBalance -ne $null -and $netPct -eq $null) {
        $netPct = [math]::Round((($finalBalance - 200.0) / 200.0) * 100.0, 2)
    }

    return [pscustomobject]@{
        final_balance = $finalBalance
        net_pnl_text = $netPnl
        net_pct = $netPct
        trades = $trades
        tp_hits = $tpHits
        sl_hits = $slHits
        win_rate_pct = $winRate
        report = $report
    }
}

function Parse-RawTesterLog([string]$Path, [double]$StartingBalance) {
    if (-not (Test-Path $Path)) {
        return [pscustomobject]@{
            loaded_signals = $null
            max_dd_pct = $null
        }
    }
    $rawLines = Get-Content $Path -Encoding Unicode
    $loadedSignals = $null
    $balances = New-Object System.Collections.Generic.List[double]
    foreach ($line in $rawLines) {
        if ($line -match "Loaded\s+([0-9]+)\s+signals") {
            $loadedSignals = [int]$Matches[1]
        }
        if ($line -match "BALANCE_UPDATE\s+balance=(-?[0-9.]+)\s+equity=(-?[0-9.]+)") {
            $balancePoint = [double]$Matches[1]
            $equityPoint = [double]$Matches[2]
            $balances.Add([math]::Min($balancePoint, $equityPoint))
        }
    }
    $peak = $StartingBalance
    $maxDd = 0.0
    foreach ($bal in $balances) {
        if ($bal -gt $peak) { $peak = $bal }
        if ($peak -gt 0) {
            $dd = ($bal - $peak) / $peak * 100.0
            if ($dd -lt $maxDd) { $maxDd = $dd }
        }
    }
    return [pscustomobject]@{
        loaded_signals = $loadedSignals
        max_dd_pct = [math]::Round($maxDd, 2)
    }
}

$root = (Resolve-Path ".").Path
$manifestPath = Resolve-Path $Manifest
$portableRootResolved = ""
if (-not [string]::IsNullOrWhiteSpace($PortableRoot)) {
    $portableRootResolved = (Resolve-Path $PortableRoot).Path
}
$manifestObj = Get-Content $manifestPath -Raw | ConvertFrom-Json
$outDir = Split-Path $manifestPath -Parent
$logDir = Join-Path $outDir "logs"
$rawLogDir = Join-Path $outDir "tester_logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
New-Item -ItemType Directory -Path $rawLogDir -Force | Out-Null

$resultsPath = Join-Path $outDir "mt5_wf_results.csv"
$jsonPath = Join-Path $outDir "mt5_wf_results.json"
$allResults = @()

foreach ($fold in $manifestObj.folds) {
    $foldId = [int]$fold.fold
    if ($foldId -lt $StartFold -or $foldId -gt $EndFold) {
        continue
    }

    $signalCsv = Resolve-Path $fold.csv
    $fromDate = Convert-DateForMt5 $fold.test_start
    $toDate = Convert-DateForMt5 $fold.test_end
    $foldLog = Join-Path $logDir ("fold_{0:D2}.log" -f $foldId)
    $foldRiskPct = $RiskPct
    $foldMaxRiskPct = $MaxRiskPct
    $foldMaxExposurePct = $MaxExposurePct
    $foldMaxHoldBars = $MaxHoldBars
    $foldMaxPositions = $MaxPositions
    if (-not $ForceCliRisk -and $fold.PSObject.Properties.Name -contains "risk_pct" -and $fold.risk_pct -ne $null) {
        $foldRiskPct = [double]$fold.risk_pct
    }
    if (-not $ForceCliRisk -and $fold.PSObject.Properties.Name -contains "max_risk_pct" -and $fold.max_risk_pct -ne $null) {
        $foldMaxRiskPct = [double]$fold.max_risk_pct
    }
    if (-not $ForceCliRisk -and $fold.PSObject.Properties.Name -contains "max_exposure_pct" -and $fold.max_exposure_pct -ne $null) {
        $foldMaxExposurePct = [double]$fold.max_exposure_pct
    }
    if (-not $ForceCliRisk -and $fold.PSObject.Properties.Name -contains "max_positions" -and $fold.max_positions -ne $null) {
        $foldMaxPositions = [int]$fold.max_positions
    } elseif (-not $ForceCliRisk -and $fold.PSObject.Properties.Name -contains "max_positions_hint" -and $fold.max_positions_hint -ne $null) {
        $foldMaxPositions = [int]$fold.max_positions_hint
    }
    if ($MaxHoldBars -lt 0 -and $fold.PSObject.Properties.Name -contains "selected_horizon_bars" -and $fold.selected_horizon_bars -ne $null) {
        $foldMaxHoldBars = [int]$fold.selected_horizon_bars
    }
    if ($foldMaxHoldBars -lt 0) { $foldMaxHoldBars = 0 }

    Write-Host ""
    Write-Host ("=== MT5 WF fold {0:D2}: {1} -> {2}, signals={3}, risk={4}%, maxPos={5}, maxHoldBars={6} ===" -f $foldId, $fromDate, $toDate, $fold.signals, $foldRiskPct, $foldMaxPositions, $foldMaxHoldBars)

    Push-Location $RunnerWorkDir
    try {
        $runnerArgs = @(
            "-ExecutionPolicy", "Bypass",
            "-File", $Runner,
            "-SignalCSV", $signalCsv,
            "-RiskPct", $foldRiskPct,
            "-MaxPositions", $foldMaxPositions,
            "-MaxDailyLoss", $MaxDailyLoss,
            "-MaxRiskPct", $foldMaxRiskPct,
            "-MaxExposurePct", $foldMaxExposurePct,
            "-MaxDDKillPct", $MaxDDKillPct,
            "-MaxPeakDDKillPct", $MaxPeakDDKillPct,
            "-TargetBalanceStop", $TargetBalanceStop,
            "-MinEntryDriftR", $MinEntryDriftR,
            "-MaxEntryDriftR", $MaxEntryDriftR,
            "-TrailingEnabled", $TrailingEnabled,
            "-BreakevenRR", $BreakevenRR,
            "-TrailActivationRR", $TrailActivationRR,
            "-TrailAtrMultiple", $TrailAtrMultiple,
            "-MaxHoldBars", $foldMaxHoldBars,
            "-Deposit", $Deposit,
            "-FromDate", $fromDate,
            "-ToDate", $toDate,
            "-Symbol", $Symbol,
            "-Period", $Period,
            "-Model", $Model,
            "-WaitSeconds", $WaitSeconds
        )
        if (-not [string]::IsNullOrWhiteSpace($portableRootResolved)) {
            $runnerArgs += @("-PortableRoot", $portableRootResolved)
        }
        if ($KillExistingMT5) {
            $runnerArgs += "-KillExistingMT5"
        }
        $lines = & powershell.exe @runnerArgs 2>&1
    }
    finally {
        Pop-Location
    }

    $lines | Set-Content -Path $foldLog -Encoding UTF8
    $testerLog = $null
    $logDate = (Get-Date -Format "yyyyMMdd") + ".log"
    if (-not [string]::IsNullOrWhiteSpace($portableRootResolved)) {
        $portableTesterRoot = Join-Path $portableRootResolved "Tester"
        if (Test-Path $portableTesterRoot) {
            $latestLog = Get-ChildItem $portableTesterRoot -Recurse -Filter $logDate -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -match "Agent-127\.0\.0\.1-3000\\logs" } |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1
            if ($latestLog) { $testerLog = $latestLog.FullName }
            if (-not $testerLog) {
                $portableRootLog = Join-Path (Join-Path $portableTesterRoot "logs") $logDate
                if (Test-Path $portableRootLog) { $testerLog = $portableRootLog }
            }
        }
    } else {
        $testerLogDir = "C:\Users\doanbacremote\AppData\Roaming\MetaQuotes\Tester\53785E099C927DB68A545C249CDBCE06\Agent-127.0.0.1-3000\logs"
        $testerLog = Join-Path $testerLogDir $logDate
    }
    $rawFoldLog = Join-Path $rawLogDir ("fold_{0:D2}_tester.log" -f $foldId)
    if ($testerLog -and (Test-Path $testerLog)) {
        Copy-Item $testerLog $rawFoldLog -Force
    }
    $parsed = Parse-RunOutput $lines
    $rawParsed = Parse-RawTesterLog $rawFoldLog $Deposit
    if ($parsed.final_balance -eq $null -and [int]$fold.signals -eq 0) {
        $parsed.final_balance = [double]$Deposit
        $parsed.net_pct = 0.0
        if ($parsed.trades -eq $null) { $parsed.trades = 0 }
    }
    $signalsCount = [int]$fold.signals
    $loadedMismatch = (
        $signalsCount -gt 0 -and
        $rawParsed.loaded_signals -ne $null -and
        [int]$rawParsed.loaded_signals -ne $signalsCount
    )
    if (
        $signalsCount -gt 0 -and
        (
            $parsed.final_balance -eq $null -or
            $parsed.trades -eq $null -or
            $rawParsed.loaded_signals -eq $null -or
            $loadedMismatch
        )
    ) {
        Add-Content -Path $foldLog -Encoding UTF8 -Value ""
        Add-Content -Path $foldLog -Encoding UTF8 -Value "ERROR: MT5 tester produced no trustworthy parseable result for a non-empty signal fold."
        if ($loadedMismatch) {
            Add-Content -Path $foldLog -Encoding UTF8 -Value ("ERROR: loaded_signals={0} expected_signals={1}" -f $rawParsed.loaded_signals, $signalsCount)
        }
        $terminalLogDir = "C:\Users\doanbacremote\AppData\Roaming\MetaQuotes\Terminal\53785E099C927DB68A545C249CDBCE06\logs"
        if (Test-Path $terminalLogDir) {
            $latestTerminalLog = Get-ChildItem $terminalLogDir -Filter "*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($latestTerminalLog) {
                Add-Content -Path $foldLog -Encoding UTF8 -Value ""
                Add-Content -Path $foldLog -Encoding UTF8 -Value ("Latest terminal log tail: {0}" -f $latestTerminalLog.FullName)
                Get-Content $latestTerminalLog.FullName -Encoding Unicode -Tail 80 | Add-Content -Path $foldLog -Encoding UTF8
            }
        }
        throw ("MT5 WF fold {0:D2} failed: no parseable final balance/trades/loaded signals. Results CSV was not updated for this failed fold." -f $foldId)
    }
    $row = [pscustomobject]@{
        fold = $foldId
        test_start = $fold.test_start
        test_end = $fold.test_end
        signals = [int]$fold.signals
        deposit = $Deposit
        risk_pct = $foldRiskPct
        max_positions = $foldMaxPositions
        max_daily_loss_pct = $MaxDailyLoss
        max_risk_pct = $foldMaxRiskPct
        max_exposure_pct = $foldMaxExposurePct
        max_dd_kill_pct = $MaxDDKillPct
        max_peak_dd_kill_pct = $MaxPeakDDKillPct
        target_balance_stop = $TargetBalanceStop
        min_entry_drift_r = $MinEntryDriftR
        max_entry_drift_r = $MaxEntryDriftR
        trailing_enabled = $TrailingEnabled
        breakeven_rr = $BreakevenRR
        trail_activation_rr = $TrailActivationRR
        trail_atr_multiple = $TrailAtrMultiple
        max_hold_bars = $foldMaxHoldBars
        model = $Model
        final_balance = $parsed.final_balance
        net_pct = $parsed.net_pct
        max_dd_pct = $rawParsed.max_dd_pct
        trades = $parsed.trades
        tp_hits = $parsed.tp_hits
        sl_hits = $parsed.sl_hits
        win_rate_pct = $parsed.win_rate_pct
        loaded_signals = $rawParsed.loaded_signals
        report = $parsed.report
        log = $foldLog
        tester_log = $rawFoldLog
    }
    $allResults += $row
    $allResults | Export-Csv -NoTypeInformation -Path $resultsPath
    $allResults | ConvertTo-Json -Depth 6 | Set-Content -Path $jsonPath -Encoding UTF8

    Write-Host (
        "fold {0:D2} done: loaded={1} final={2} net={3}% maxDD={4}% trades={5} WR={6}%" -f
        $foldId, $row.loaded_signals, $row.final_balance, $row.net_pct, $row.max_dd_pct, $row.trades, $row.win_rate_pct
    )
}

Write-Host ""
Write-Host "Results CSV: $resultsPath"
Write-Host "Results JSON: $jsonPath"
