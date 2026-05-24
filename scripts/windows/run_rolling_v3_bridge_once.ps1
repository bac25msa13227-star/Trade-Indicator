param(
    [string]$BridgeUrl = "http://localhost:5601",
    [string]$Manifest = "outputs\target1200_live_rolling_v3_current_selected\manifest.json",
    [string]$GuardReport = "outputs\target1200_live_rolling_v3_current_selected\live_canary_guard_report.json",
    [string]$CircuitGuardReport = "outputs\target1200_live_rolling_v3_current_selected\equity_target_guard_report.json",
    [string]$PreflightReport = "outputs\target1200_live_rolling_v3_current_selected\live_preflight_report.json",
    [string]$PaperReport = "outputs\target1200_live_rolling_v3_current_selected\paper_30day_report.json",
    [string]$PaperEventLog = "outputs\target1200_live_rolling_v3_current_selected\paper_bridge_events.jsonl",
    [string]$RealisticGateReport = "outputs\target1200_live_rolling_v3_current_selected\realistic_gate_report.json",
    [string]$State = "outputs\target1200_live_rolling_v3_current_selected\bridge_live_state.json",
    [string]$OutReport = "outputs\target1200_live_rolling_v3_current_selected\bridge_live_last_report.json",
    [string]$Symbol = "XAUUSD",
    [int]$Magic = 13001,
    [double]$InitialBalance = 200.0,
    [double]$MinBalance = 0.0,
    [double]$MaxDDKillPct = 20.0,
    [double]$TargetBalanceStop = 1200.0,
    [double]$MaxEntryLagSeconds = 90.0,
    [double]$MaxSpreadPoints = 120.0,
    [double]$LiveRiskCapPct = 1.0,
    [double]$LiveExposureCapPct = 1.0,
    [string]$ExpectedLoginEnv = "MT5_LOGIN",
    [switch]$Execute,
    [switch]$SkipRefreshSignal,
    [switch]$AllowFirstRunTrade,
    [switch]$CanaryMode,
    [switch]$AllowExecuteWithoutPaper,
    [switch]$NoPaperGate,
    [switch]$NoCanaryGate,
    [switch]$NoRealisticGate
)

$ErrorActionPreference = "Stop"
$root = "f:\Trading_BOT_AUTO\Trade-Indicator"
Set-Location $root

if (-not $SkipRefreshSignal) {
    Write-Output "[1/2] Refreshing live MT5 bars/features/signals for rolling_v3..."
    & powershell.exe -ExecutionPolicy Bypass -File "scripts\windows\update_rolling_v3_live_signal.ps1" -BridgeUrl $BridgeUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Signal refresh failed."
    }
} else {
    Write-Output "[1/2] Skipping signal refresh."
}

$expectedLogin = ""
if (Test-Path ".env") {
    $pattern = '^\s*' + [regex]::Escape($ExpectedLoginEnv) + '\s*='
    $line = Get-Content ".env" | Where-Object { $_ -match $pattern } | Select-Object -First 1
    if ($line) {
        $expectedLogin = (($line -split '=', 2)[1]).Trim()
    }
}

Write-Output "[2/2] Running rolling_v3 bridge executor..."
$args = @(
    "scripts\rolling_v3_bridge_executor.py",
    "--bridge-url", $BridgeUrl,
    "--manifest", $Manifest,
    "--guard-report", $GuardReport,
    "--circuit-guard-report", $CircuitGuardReport,
    "--preflight-report", $PreflightReport,
    "--paper-report", $PaperReport,
    "--paper-event-log", $PaperEventLog,
    "--realistic-gate-report", $RealisticGateReport,
    "--state", $State,
    "--out", $OutReport,
    "--symbol", $Symbol,
    "--magic", "$Magic",
    "--initial-balance", "$InitialBalance",
    "--min-balance", "$MinBalance",
    "--max-dd-kill-pct", "$MaxDDKillPct",
    "--target-balance-stop", "$TargetBalanceStop",
    "--max-entry-lag-seconds", "$MaxEntryLagSeconds",
    "--max-spread-points", "$MaxSpreadPoints",
    "--live-risk-cap-pct", "$LiveRiskCapPct",
    "--live-exposure-cap-pct", "$LiveExposureCapPct"
)
if ($expectedLogin) {
    $args += @("--expected-login", $expectedLogin)
}
if ($Execute) {
    $args += "--execute"
}
if ($CanaryMode) {
    $args += "--canary-mode"
}
if ($AllowExecuteWithoutPaper) {
    $args += "--allow-execute-without-paper"
}
if ($NoPaperGate) {
    $args += "--no-paper-gate"
}
if ($NoCanaryGate) {
    $args += "--no-canary-gate"
}
if ($NoRealisticGate) {
    $args += "--no-realistic-gate"
}
if ($AllowFirstRunTrade) {
    $args += "--allow-first-run-trade"
}

python @args
exit $LASTEXITCODE
