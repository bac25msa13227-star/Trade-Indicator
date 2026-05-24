param(
    [string]$CandidateRoot = "outputs\fixed_library_candidates_20260514",
    [string[]]$CandidateNames = @("wide_t15_m240_r1p8", "core_t20_m240_r2p0"),
    [double]$Deposit = 200,
    [string]$Symbol = "XAUUSDm",
    [string]$Period = "M5",
    [int]$WaitSeconds = 300,
    [double]$MaxDailyLoss = 100,
    [double]$MaxDDKillPct = 20,
    [double]$TargetBalanceStop = 1200,
    [int]$MaxHoldBars = 0,
    [int]$TrailingEnabled = 0
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Runner = Join-Path $RepoRoot "scripts\run_mt5_wf_folds.ps1"
$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$SummaryLog = Join-Path $LogDir "mt5_fixed_library_sweep_20260514.log"
$ErrorLog = Join-Path $LogDir "mt5_fixed_library_sweep_20260514_err.log"

"=== fixed-library sweep start $(Get-Date -Format o) ===" | Out-File -FilePath $SummaryLog -Encoding utf8
"" | Out-File -FilePath $ErrorLog -Encoding utf8

foreach ($Name in $CandidateNames) {
    $Dir = Join-Path $CandidateRoot $Name
    $Manifest = Join-Path $Dir "manifest.json"
    $Results = Join-Path $Dir "mt5_wf_results.csv"
    $RunLog = Join-Path $Dir "wf_run.log"

    if (-not (Test-Path -LiteralPath $Manifest)) {
        "SKIP ${Name}: missing manifest.json" | Tee-Object -FilePath $SummaryLog -Append
        continue
    }

    "=== START ${Name} $(Get-Date -Format o) ===" | Tee-Object -FilePath $SummaryLog -Append
    try {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Runner `
            -Manifest $Manifest `
            -Deposit $Deposit `
            -Symbol $Symbol `
            -Period $Period `
            -WaitSeconds $WaitSeconds `
            -MaxDailyLoss $MaxDailyLoss `
            -MaxDDKillPct $MaxDDKillPct `
            -TargetBalanceStop $TargetBalanceStop `
            -MaxHoldBars $MaxHoldBars `
            -TrailingEnabled $TrailingEnabled *>&1 |
            Tee-Object -FilePath $RunLog

        if (Test-Path -LiteralPath $Results) {
            $Rows = (Import-Csv -LiteralPath $Results).Count
            "DONE ${Name}: rows=$Rows $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append
        }
        else {
            "FAIL ${Name}: mt5_wf_results.csv missing $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append
        }
    }
    catch {
        "ERROR ${Name}: $($_.Exception.Message)" | Tee-Object -FilePath $SummaryLog -Append
        "ERROR ${Name}: $($_ | Out-String)" | Out-File -FilePath $ErrorLog -Append -Encoding utf8
    }
}

"=== fixed-library sweep end $(Get-Date -Format o) ===" | Tee-Object -FilePath $SummaryLog -Append
