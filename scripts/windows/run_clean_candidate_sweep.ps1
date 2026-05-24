param(
    [string]$Root = "outputs\mt5_candidate_clean_20260513",
    [string]$Log = "logs\mt5_candidate_clean_sweep_20260513.log",
    [int]$WaitSeconds = 240
)

$ErrorActionPreference = "Stop"
$repo = "f:\Trading_BOT_AUTO\Trade-Indicator"
Set-Location $repo

New-Item -ItemType Directory -Path (Split-Path $Log -Parent) -Force | Out-Null
"sweep_start=$(Get-Date -Format s) root=$Root" | Set-Content -Path $Log -Encoding UTF8

$dirs = Get-ChildItem $Root -Directory | Sort-Object Name
$index = 0
foreach ($dir in $dirs) {
    $index += 1
    $manifest = Join-Path $dir.FullName "manifest.json"
    $results = Join-Path $dir.FullName "mt5_wf_results.csv"
    if (Test-Path $results) {
        $existing = Import-Csv $results
        if ($existing.Count -ge 1 -and -not [string]::IsNullOrWhiteSpace($existing[0].final_balance) -and -not [string]::IsNullOrWhiteSpace($existing[0].loaded_signals)) {
            "[$index/$($dirs.Count)] skip_valid $($dir.Name)" | Add-Content -Path $Log -Encoding UTF8
            continue
        }
    }

    "[$index/$($dirs.Count)] run $($dir.Name)" | Add-Content -Path $Log -Encoding UTF8
    try {
        & powershell.exe -ExecutionPolicy Bypass -File scripts\run_mt5_wf_folds.ps1 `
            -Manifest $manifest `
            -Symbol XAUUSDm `
            -Deposit 200 `
            -MaxDailyLoss 100 `
            -MaxDDKillPct 20 `
            -MaxPeakDDKillPct 0 `
            -TargetBalanceStop 0 `
            -TrailingEnabled 0 `
            -WaitSeconds $WaitSeconds 2>&1 |
            Add-Content -Path $Log -Encoding UTF8
        "[$index/$($dirs.Count)] done $($dir.Name) exit=$LASTEXITCODE" | Add-Content -Path $Log -Encoding UTF8
    }
    catch {
        "[$index/$($dirs.Count)] failed $($dir.Name): $($_.Exception.Message)" | Add-Content -Path $Log -Encoding UTF8
    }
}

"sweep_end=$(Get-Date -Format s)" | Add-Content -Path $Log -Encoding UTF8
