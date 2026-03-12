# setup_env.ps1 — Tạo file .env từ .env.example
# Chạy 1 lần khi setup máy mới:
#   .\scripts\windows\setup_env.ps1

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$envFile     = Join-Path $projectRoot ".env"
$exampleFile = Join-Path $projectRoot ".env.example"

if (-not (Test-Path $exampleFile)) {
    Write-Error ".env.example không tồn tại tại $exampleFile"
    exit 1
}

if (Test-Path $envFile) {
    $overwrite = Read-Host ".env đã tồn tại. Ghi đè? (y/N)"
    if ($overwrite -notmatch '^[Yy]$') {
        Write-Host "Bỏ qua. File .env không thay đổi." -ForegroundColor Yellow
        exit 0
    }
}

Write-Host ""
Write-Host "=== SETUP CREDENTIALS ===" -ForegroundColor Cyan
Write-Host "Nhập từng giá trị (Enter = giữ giá trị mặc định từ .env.example):"
Write-Host ""

$lines = Get-Content $exampleFile
$output = @()

foreach ($line in $lines) {
    # Bỏ qua comment và dòng trống
    if ($line -match '^\s*#' -or $line.Trim() -eq '') {
        $output += $line
        continue
    }

    # Tách KEY=value
    $parts = $line -split '=', 2
    if ($parts.Count -lt 2) {
        $output += $line
        continue
    }

    $key     = $parts[0].Trim()
    $default = $parts[1].Trim()

    $prompt = "  $key"
    if ($default -and $default -notmatch '^your_') {
        $prompt += " [$default]"
    }
    $prompt += ": "

    $value = Read-Host $prompt
    if ([string]::IsNullOrWhiteSpace($value)) {
        $value = $default
    }

    $output += "$key=$value"
}

$output | Set-Content -Path $envFile -Encoding UTF8
Write-Host ""
Write-Host "✅ Đã tạo .env tại $envFile" -ForegroundColor Green
Write-Host "   (File này KHÔNG push lên git — đã có trong .gitignore)" -ForegroundColor DarkGray
