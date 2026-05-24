param(
    [int]$IntervalSeconds = 300,
    [string]$LogPath = "logs\rolling_v3_bridge_loop.log",
    [string]$BridgeUrl = "http://localhost:5601",
    [string]$Manifest = "outputs\target1200_live_rolling_v3_current_selected\manifest.json",
    [string]$GuardReport = "outputs\target1200_live_rolling_v3_current_selected\live_canary_guard_report.json",
    [string]$CircuitGuardReport = "outputs\target1200_live_rolling_v3_current_selected\equity_target_guard_report.json",
    [string]$PreflightReport = "outputs\target1200_live_rolling_v3_current_selected\live_preflight_report.json",
    [string]$PaperReport = "outputs\target1200_live_rolling_v3_current_selected\paper_30day_report.json",
    [string]$PaperEventLog = "outputs\target1200_live_rolling_v3_current_selected\paper_bridge_events.jsonl",
    [string]$PaperTradesOut = "outputs\target1200_live_rolling_v3_current_selected\paper_trades.csv",
    [string]$RealisticGateReport = "",
    [string]$State = "outputs\target1200_live_rolling_v3_current_selected\bridge_live_state.json",
    [string]$OutReport = "outputs\target1200_live_rolling_v3_current_selected\bridge_live_last_report.json",
    [string]$EquityGuardOut = "outputs\target1200_live_rolling_v3_current_selected\equity_target_guard_report.json",
    [string]$EquityGuardState = "outputs\target1200_live_rolling_v3_current_selected\equity_target_guard_state.json",
    [string]$LiveStatusOut = "outputs\live_status_acc1.json",
    [string]$SignalsCsvOut = "outputs\paper_trade_signals_acc1.csv",
    [string]$SignalLogOut = "outputs\bridge_signal_log.json",
    [string]$OnlineDecision = "outputs\target1200_live_rolling_v3_current_selected\online_decision.json",
    [string]$ExpectedLoginEnv = "MT5_LOGIN",
    [string]$BaseRates = "outputs\mt5_rates_export_202306_20260513_merged.csv",
    [string]$LiveRates = "outputs\mt5_rates_export_202306_20260513_live_merged.csv",
    [string]$LiveFeatures = "outputs\mt5_full_ict_wyckoff_features_live_current.csv",
    [string]$SelectedOut = "outputs\target1200_live_rolling_v3_current_selected",
    [string]$Symbol = "XAUUSD",
    [int]$Magic = 13001,
    [double]$InitialBalance = 200.0,
    [double]$MinBalance = 0.0,
    [double]$MaxDDKillPct = 20.0,
    [double]$TargetBalanceStop = 1200.0,
    [double]$MaxEntryLagSeconds = 90.0,
    [double]$MaxSpreadPoints = 120.0,
    [double]$EquityPollSeconds = 0.25,
    [int]$EntryDelaySeconds = 5,
    [switch]$Execute,
    [switch]$DryRun,
    [switch]$AllowFirstRunTrade,
    [switch]$CanaryMode,
    [switch]$AllowExecuteWithoutPaper,
    [switch]$NoPaperGate,
    [switch]$NoCanaryGate,
    [switch]$NoRealisticGate,
    [switch]$SkipDashboardAdapter,
    [double]$LiveRiskCapPct = 1.0,
    [double]$LiveExposureCapPct = 1.0,
    [switch]$DisableEquityGuard,
    [switch]$DisableM5Alignment,
    [switch]$RegenerateCurrentFold   # Regenerate fold signals each cycle from live model inference
)

$ErrorActionPreference = "Continue"
$root = "f:\Trading_BOT_AUTO\Trade-Indicator"
Set-Location $root

$logFullPath = Join-Path $root $LogPath
New-Item -ItemType Directory -Force -Path (Split-Path $logFullPath -Parent) | Out-Null
$EffectiveDryRun = ($DryRun -or -not $Execute)
if ([string]::IsNullOrWhiteSpace($RealisticGateReport)) {
    $RealisticGateReport = Join-Path $SelectedOut "realistic_gate_report.json"
}

function Write-LoopLog {
    param([string]$Message)
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    "$stamp $Message" | Tee-Object -FilePath $logFullPath -Append
}

Write-LoopLog "rolling_v3 bridge loop starting interval_seconds=$IntervalSeconds dry_run=$EffectiveDryRun execute=$Execute bridge_url=$BridgeUrl"

if (-not $DisableEquityGuard -and $TargetBalanceStop -gt 0) {
    $existingGuard = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -like "*rolling_v3_equity_target_guard.py*" -and
        $_.CommandLine -like "*$BridgeUrl*" -and
        $_.CommandLine -like "*--magic*" -and
        $_.CommandLine -like "*$Magic*"
    } | Select-Object -First 1
    if ($existingGuard) {
        Write-LoopLog "equity_target_guard already_running pid=$($existingGuard.ProcessId)"
    } else {
        $guardStdout = Join-Path $root "logs\rolling_v3_equity_target_guard.log"
        $guardStderr = Join-Path $root "logs\rolling_v3_equity_target_guard_err.log"
        $guardArgs = @(
            "scripts\rolling_v3_equity_target_guard.py",
            "--bridge-url", $BridgeUrl,
            "--symbol", $Symbol,
            "--magic", "$Magic",
            "--initial-balance", "$InitialBalance",
            "--min-balance", "$MinBalance",
            "--max-dd-kill-pct", "$MaxDDKillPct",
            "--target-balance-stop", "$TargetBalanceStop",
            "--poll-seconds", "$EquityPollSeconds",
            "--out", $EquityGuardOut,
            "--state", $EquityGuardState
        )
        if (-not $EffectiveDryRun) {
            $guardArgs += "--execute"
        }
        $guardProcess = Start-Process -FilePath "python" -ArgumentList $guardArgs -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $guardStdout -RedirectStandardError $guardStderr -PassThru
        Write-LoopLog "equity_target_guard started pid=$($guardProcess.Id) target=$TargetBalanceStop poll_seconds=$EquityPollSeconds execute=$(-not $EffectiveDryRun)"
    }
}

while ($true) {
    try {
        Write-LoopLog "cycle_start"

        $updateArgs = @(
            "-ExecutionPolicy", "Bypass",
            "-File", "scripts\windows\update_rolling_v3_live_signal.ps1",
            "-BridgeUrl", $BridgeUrl,
            "-BaseRates", $BaseRates,
            "-LiveRates", $LiveRates,
            "-LiveFeatures", $LiveFeatures,
            "-SelectedOut", $SelectedOut
        )
        if ($RegenerateCurrentFold) { $updateArgs += "-RegenerateCurrentFold" }
        & powershell.exe @updateArgs *>&1 | Tee-Object -FilePath $logFullPath -Append
        $updateCode = $LASTEXITCODE
        if ($updateCode -ne 0) {
            Write-LoopLog "cycle_guard_refresh_failed exit_code=$updateCode"
        } else {
            $args = @(
                "-ExecutionPolicy", "Bypass",
                "-File", "scripts\windows\run_rolling_v3_bridge_once.ps1",
                "-SkipRefreshSignal",
                "-BridgeUrl", $BridgeUrl,
                "-Manifest", $Manifest,
                "-GuardReport", $GuardReport,
                "-CircuitGuardReport", $CircuitGuardReport,
                "-PreflightReport", $PreflightReport,
                "-PaperReport", $PaperReport,
                "-PaperEventLog", $PaperEventLog,
                "-RealisticGateReport", $RealisticGateReport,
                "-State", $State,
                "-OutReport", $OutReport,
                "-Symbol", $Symbol,
                "-Magic", "$Magic",
                "-InitialBalance", "$InitialBalance",
                "-MinBalance", "$MinBalance",
                "-MaxDDKillPct", "$MaxDDKillPct",
                "-TargetBalanceStop", "$TargetBalanceStop",
                "-MaxEntryLagSeconds", "$MaxEntryLagSeconds",
                "-MaxSpreadPoints", "$MaxSpreadPoints",
                "-LiveRiskCapPct", "$LiveRiskCapPct",
                "-LiveExposureCapPct", "$LiveExposureCapPct",
                "-ExpectedLoginEnv", $ExpectedLoginEnv
            )
            if (-not $EffectiveDryRun) {
                $args += "-Execute"
            }
            if ($CanaryMode) {
                $args += "-CanaryMode"
            }
            if ($AllowFirstRunTrade) {
                $args += "-AllowFirstRunTrade"
            }
            if ($AllowExecuteWithoutPaper) {
                $args += "-AllowExecuteWithoutPaper"
            }
            if ($NoPaperGate) {
                $args += "-NoPaperGate"
            }
            if ($NoCanaryGate) {
                $args += "-NoCanaryGate"
            }
            if ($NoRealisticGate) {
                $args += "-NoRealisticGate"
            }

            & powershell.exe @args *>&1 | Tee-Object -FilePath $logFullPath -Append
            $execCode = $LASTEXITCODE
            Write-LoopLog "cycle_executor_exit exit_code=$execCode"
        }

        # Sync bridge state → dashboard files (live_status_acc1.json + paper_trade_signals_acc1.csv)
        if (-not $SkipDashboardAdapter) {
            python scripts\bridge_status_adapter.py `
                --report $OutReport `
                --state $State `
                --target-guard-report $EquityGuardOut `
                --live-status $LiveStatusOut `
                --signals-csv $SignalsCsvOut `
                --signal-log $SignalLogOut `
                --online-decision $OnlineDecision *>&1 | Tee-Object -FilePath $logFullPath -Append
        }
        if ($EffectiveDryRun) {
            python scripts\update_paper_trade_report.py `
                --events $PaperEventLog `
                --rates $LiveRates `
                --trades-out $PaperTradesOut `
                --out $PaperReport *>&1 | Tee-Object -FilePath $logFullPath -Append
        }

    } catch {
        Write-LoopLog ("cycle_exception " + $_.Exception.Message)
    }

    if ($DisableM5Alignment) {
        Write-LoopLog "cycle_sleep seconds=$IntervalSeconds"
        Start-Sleep -Seconds $IntervalSeconds
    } else {
        $nowUtc = (Get-Date).ToUniversalTime()
        $secondsIntoM5 = (($nowUtc.Minute % 5) * 60) + $nowUtc.Second
        $sleepSeconds = 300 - $secondsIntoM5 + $EntryDelaySeconds
        if ($sleepSeconds -lt 5) { $sleepSeconds += 300 }
        Write-LoopLog "cycle_sleep_m5_aligned seconds=$sleepSeconds entry_delay_seconds=$EntryDelaySeconds"
        Start-Sleep -Seconds $sleepSeconds
    }
}
