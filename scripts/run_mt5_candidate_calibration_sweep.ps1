param(
    [string]$CandidateRoot = "outputs\mt5_rule_library_calibration_20260514",
    [double]$Deposit = 200,
    [string]$Symbol = "XAUUSDm",
    [string]$Period = "M5",
    [int]$WaitSeconds = 300,
    [double]$MaxDailyLoss = 100,
    [double]$MaxDDKillPct = 20,
    [int]$TrailingEnabled = 0
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$RootPath = Resolve-Path -LiteralPath $CandidateRoot
$Runner = Join-Path $RepoRoot "scripts\run_mt5_wf_folds.ps1"
$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$SummaryLog = Join-Path $LogDir "mt5_rule_calibration_sweep_20260514.log"
$ErrorLog = Join-Path $LogDir "mt5_rule_calibration_sweep_20260514_err.log"

"=== candidate calibration sweep start $(Get-Date -Format o) ===" | Out-File -FilePath $SummaryLog -Encoding utf8
"" | Out-File -FilePath $ErrorLog -Encoding utf8

$Candidates = Get-ChildItem -LiteralPath $RootPath -Directory | Sort-Object Name
foreach ($Candidate in $Candidates) {
    $Manifest = Join-Path $Candidate.FullName "manifest.json"
    $Results = Join-Path $Candidate.FullName "mt5_wf_results.csv"
    $RunLog = Join-Path $Candidate.FullName "wf_run.log"

    if (-not (Test-Path -LiteralPath $Manifest)) {
        "SKIP $($Candidate.Name): missing manifest.json" | Tee-Object -FilePath $SummaryLog -Append
        continue
    }

    "=== START $($Candidate.Name) $(Get-Date -Format o) ===" | Tee-Object -FilePath $SummaryLog -Append

    try {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Runner `
            -Manifest $Manifest `
            -Deposit $Deposit `
            -Symbol $Symbol `
            -Period $Period `
            -WaitSeconds $WaitSeconds `
            -MaxDailyLoss $MaxDailyLoss `
            -MaxDDKillPct $MaxDDKillPct `
            -TrailingEnabled $TrailingEnabled *>&1 |
            Tee-Object -FilePath $RunLog

        if (Test-Path -LiteralPath $Results) {
            $Rows = (Import-Csv -LiteralPath $Results).Count
            "DONE $($Candidate.Name): rows=$Rows $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append
        }
        else {
            "FAIL $($Candidate.Name): mt5_wf_results.csv missing $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append
        }
    }
    catch {
        "ERROR $($Candidate.Name): $($_.Exception.Message)" | Tee-Object -FilePath $SummaryLog -Append
        "ERROR $($Candidate.Name): $($_ | Out-String)" | Out-File -FilePath $ErrorLog -Append -Encoding utf8
    }
}

"=== candidate calibration sweep end $(Get-Date -Format o) ===" | Tee-Object -FilePath $SummaryLog -Append
