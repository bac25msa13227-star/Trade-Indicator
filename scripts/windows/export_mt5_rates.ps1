param(
    [string]$FromDate = "2026.04.01",
    [string]$ToDate = "2026.05.13",
    [string]$OutFile = "outputs/mt5_rates_export_202604_current.csv",
    [string]$Symbol = "XAUUSD",
    [string]$Period = "M5",
    [int]$WaitSeconds = 300,
    [switch]$ForceCloseTerminal
)

$ErrorActionPreference = "Stop"
$root = "f:\Trading_BOT_AUTO\Trade-Indicator"
Set-Location $root

$MetaEditor = "C:\Program Files\MetaTrader 5 EXNESS\MetaEditor64.exe"
$Terminal = "C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe"
$TerminalId = "53785E099C927DB68A545C249CDBCE06"
$ExpertsDir = "C:\Users\doanbacremote\AppData\Roaming\MetaQuotes\Terminal\$TerminalId\MQL5\Experts"
$CommonFilesDir = "C:\Users\doanbacremote\AppData\Roaming\MetaQuotes\Terminal\Common\Files"
$ReportDir = "C:\Users\doanbacremote\tradingview-mcp\MT5Reports"
$TesterIni = "C:\Users\doanbacremote\tradingview-mcp\tester_rate_export.ini"
$EAName = "Rate_Exporter"
$CommonExportName = "mt5_rates_export_current.csv"

if (-not (Test-Path $MetaEditor)) { throw "MetaEditor not found: $MetaEditor" }
if (-not (Test-Path $Terminal)) { throw "MT5 terminal not found: $Terminal" }
New-Item -ItemType Directory -Path $ExpertsDir -Force | Out-Null
New-Item -ItemType Directory -Path $CommonFilesDir -Force | Out-Null
New-Item -ItemType Directory -Path $ReportDir -Force | Out-Null

$running = Get-Process -Name "terminal64" -ErrorAction SilentlyContinue
if ($running -and -not $ForceCloseTerminal) {
    throw "terminal64 is running. Stop it yourself or rerun with -ForceCloseTerminal."
}
if ($running -and $ForceCloseTerminal) {
    $running | Stop-Process -Force
    Start-Sleep -Seconds 4
}

$sourceMq5 = Join-Path $root "scripts\mt5\Rate_Exporter.mq5"
$destMq5 = Join-Path $ExpertsDir "$EAName.mq5"
Copy-Item $sourceMq5 $destMq5 -Force

$compileLog = "C:\Users\doanbacremote\tradingview-mcp\compile_rate_export_log.txt"
if (Test-Path $compileLog) { Remove-Item $compileLog -Force }
$compileArgs = "/compile:`"$destMq5`" /log:`"$compileLog`""
Start-Process -FilePath $MetaEditor -ArgumentList $compileArgs -Wait -WindowStyle Hidden
Start-Sleep -Seconds 3

$destEx5 = Join-Path $ExpertsDir "$EAName.ex5"
if (-not (Test-Path $destEx5)) {
    if (Test-Path $compileLog) { Get-Content $compileLog -Encoding Unicode -ErrorAction SilentlyContinue }
    throw "Compile failed: $destEx5 not found"
}

$commonExport = Join-Path $CommonFilesDir $CommonExportName
if (Test-Path $commonExport) { Remove-Item $commonExport -Force }

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportBase = Join-Path $ReportDir "Report_RateExporter_$timestamp"
$ini = "[Tester]`r`n"
$ini += "Expert=$EAName`r`n"
$ini += "Symbol=$Symbol`r`n"
$ini += "Period=$Period`r`n"
$ini += "Deposit=200`r`n"
$ini += "Currency=USD`r`n"
$ini += "Leverage=100`r`n"
$ini += "ExecutionMode=0`r`n"
$ini += "Model=0`r`n"
$ini += "FromDate=$FromDate`r`n"
$ini += "ToDate=$ToDate`r`n"
$ini += "ShutdownTerminal=1`r`n"
$ini += "VisualMode=0`r`n"
$ini += "Report=$reportBase`r`n"
$ini += "Optimization=0`r`n"
$ini += "`r`n[TesterInputs]`r`n"
$ini += "InpOutFile=$CommonExportName`r`n"
[System.IO.File]::WriteAllText($TesterIni, $ini)

$setFile = "C:\Users\doanbacremote\AppData\Roaming\MetaQuotes\Terminal\$TerminalId\MQL5\Profiles\Tester\$EAName.set"
$setContent = "; saved automatically by export_mt5_rates.ps1`r`n"
$setContent += "InpOutFile=$CommonExportName`r`n"
[System.IO.File]::WriteAllText($setFile, $setContent, [System.Text.Encoding]::Unicode)

Write-Output "Running MT5 Rate_Exporter: $FromDate -> $ToDate"
$proc = Start-Process -FilePath $Terminal -ArgumentList "/config:`"$TesterIni`"" -PassThru -WindowStyle Minimized
$elapsed = 0
while (-not $proc.HasExited -and $elapsed -lt $WaitSeconds) {
    Start-Sleep -Seconds 5
    $elapsed += 5
}
if (-not $proc.HasExited) {
    $proc | Stop-Process -Force
    throw "Rate export timed out after ${WaitSeconds}s"
}
Start-Sleep -Seconds 3

if (-not (Test-Path $commonExport)) {
    throw "Rate export file not found: $commonExport"
}

$target = Join-Path $root $OutFile
New-Item -ItemType Directory -Path (Split-Path $target -Parent) -Force | Out-Null
Copy-Item $commonExport $target -Force

Write-Output "Exported rates: $target"
