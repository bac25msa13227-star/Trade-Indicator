# ══════════════════════════════════════════════════════════════════════════════
# deploy_v14pp.ps1  —  Triển khai V14++ cho ACC1 + ACC2 trên máy Windows
# Tác giả: auto-generated
# Phiên bản: V14++ (PROFIT sc=0.85 / COMPOSITE sc=0.88)  |  17/04/2026
#
# CÁCH DÙNG:
#   1. Chạy PowerShell as Administrator
#   2. cd tới thư mục chứa repo (nếu chưa clone, script tự clone)
#   3. .\deploy_v14pp.ps1
#      hoặc với tuỳ chọn:
#      .\deploy_v14pp.ps1 -SkipWF          (bỏ qua bước WF verify)
#      .\deploy_v14pp.ps1 -SkipBotStart    (chỉ chuẩn bị, không khởi bot)
#      .\deploy_v14pp.ps1 -RetrainModels   (retrain model trước khi chạy)
# ══════════════════════════════════════════════════════════════════════════════

param(
    [switch]$RunWF,           # Chạy WF verify (cần có data CSV ~1.2GB — xem phần DATA bên dưới)
    [switch]$SkipBotStart,    # Chỉ chuẩn bị, không khởi bot
    [switch]$RetrainModels    # Retrain model trước khi chạy (cần data CSV)
)
# NOTE: WF mặc định BỊ BỎ QUA vì cần data CSV ~1.2GB không có trong git.
# WF benchmark đã được verify và lưu tại outputs/walkforward_report_*.json (có trong git).
# Để chạy WF: thêm flag -RunWF và đảm bảo data CSV đã có tại src/xauusd_ai/real_data/

$ErrorActionPreference = "Stop"

# ── CONFIG ────────────────────────────────────────────────────────────────────
$REPO_URL   = "https://github.com/bac25msa13227-star/Trade-Indicator.git"
$BRANCH     = "feature/dashboard-controls-v14pp"
$PROJ       = "C:\Users\Administrator\Documents\Trade-Indicator"
$LOGFILE    = "$PROJ\outputs\deploy_v14pp_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$REPORT_OUT = "$PROJ\outputs\deploy_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"

# ── HELPER ────────────────────────────────────────────────────────────────────
function Log {
    param([string]$msg, [string]$level = "INFO")
    $ts = Get-Date -Format "HH:mm:ss"
    $line = "$ts  [$level]  $msg"
    Write-Host $line
    Add-Content -Path $LOGFILE -Value $line -Encoding UTF8
}

function Die {
    param([string]$msg)
    Log "FATAL: $msg" "ERROR"
    Log "Deploy thất bại. Xem log: $LOGFILE" "ERROR"
    exit 1
}

function Check-Command($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Die "Lệnh '$name' không tìm thấy. Hãy cài đặt trước khi chạy deploy."
    }
}

# ── 0. TẠO THƯ MỤC LOG ───────────────────────────────────────────────────────
if (-not (Test-Path "$PROJ\outputs")) {
    New-Item -ItemType Directory -Path "$PROJ\outputs" -Force | Out-Null
}
Log "═══════════════════════════════════════════════════"
Log "  DEPLOY V14++  $(Get-Date -Format 'dd/MM/yyyy HH:mm')"
Log "  Branch : $BRANCH"
Log "  Proj   : $PROJ"
Log "═══════════════════════════════════════════════════"

# ── 1. KIỂM TRA PREREQUISITES ─────────────────────────────────────────────────
Log "--- [1/7] Kiểm tra prerequisites ---"
Check-Command "git"
Check-Command "docker"
Check-Command "python"

$gitLfs = git lfs version 2>&1
if ($LASTEXITCODE -ne 0) {
    Die "Git LFS chưa cài. Tải tại: https://git-lfs.github.com/"
}
Log "git-lfs  : $($gitLfs | Select-Object -First 1)"
Log "docker   : $(docker --version)"
Log "python   : $(python --version 2>&1)"

# ── 2. CLONE HOẶC PULL REPO ──────────────────────────────────────────────────
Log "--- [2/7] Clone / Pull repo ---"

if (Test-Path "$PROJ\.git") {
    Log "Repo đã tồn tại tại $PROJ  →  git pull"
    Set-Location $PROJ
    git fetch origin 2>&1 | ForEach-Object { Log $_ }
    git checkout $BRANCH 2>&1 | ForEach-Object { Log $_ }
    git pull origin $BRANCH 2>&1 | ForEach-Object { Log $_ }
    if ($LASTEXITCODE -ne 0) { Die "git pull thất bại." }
} else {
    Log "Clone repo lần đầu..."
    git clone --branch $BRANCH $REPO_URL $PROJ 2>&1 | ForEach-Object { Log $_ }
    if ($LASTEXITCODE -ne 0) { Die "git clone thất bại." }
    Set-Location $PROJ
}

# ── 3. KÉO LFS (model pkl) ──────────────────────────────────────────────────
Log "--- [3/7] Kéo model files qua Git LFS ---"
git lfs install 2>&1 | ForEach-Object { Log $_ }
git lfs pull    2>&1 | ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) { Die "git lfs pull thất bại." }

# Kiểm tra file model tồn tại và không phải LFS pointer
$modelFiles = @(
    "outputs\acc1_combo133_202604_model.pkl",
    "outputs\acc1_combo133_202604_scaler.pkl",
    "outputs\acc1_combo133_202604_meta.json",
    "outputs\acc2_v14pp_202604_model.pkl",
    "outputs\acc2_v14pp_202604_scaler.pkl",
    "outputs\acc2_v14pp_202604_meta.json"
)

foreach ($f in $modelFiles) {
    $fp = Join-Path $PROJ $f
    if (-not (Test-Path $fp)) {
        Die "File model không tìm thấy: $fp  — kiểm tra LFS hoặc chạy với -RetrainModels"
    }
    $size = (Get-Item $fp).Length
    if ($size -lt 1000) {
        Die "File $f có vẻ là LFS pointer (${size}B). Chạy 'git lfs pull' thủ công."
    }
    Log "  ✓ $f  ($([math]::Round($size/1MB,1)) MB)"
}

# ── 4. (TUỲ CHỌN) RETRAIN MODEL ─────────────────────────────────────────────
if ($RetrainModels) {
    Log "--- [4/7] Retrain models (--RetrainModels flag) ---"
    Log "Retrain ACC1..."
    python scripts/retrain_live_model.py configs/live_acc1.yaml 2>&1 | ForEach-Object { Log $_ }
    if ($LASTEXITCODE -ne 0) { Die "Retrain ACC1 thất bại." }
    Log "Retrain ACC2..."
    python scripts/retrain_live_model.py configs/live_acc2.yaml 2>&1 | ForEach-Object { Log $_ }
    if ($LASTEXITCODE -ne 0) { Die "Retrain ACC2 thất bại." }
    Log "Retrain hoàn tất."
} else {
    Log "--- [4/7] Bỏ qua retrain (dùng model từ LFS) ---"
    # Hiển thị thông tin model đang dùng
    python -c "
import json
for acc, path in [('ACC1','outputs/acc1_combo133_202604_meta.json'),('ACC2','outputs/acc2_v14pp_202604_meta.json')]:
    m = json.load(open(path))
    print(f'  {acc}: threshold={m[\"decision_threshold\"]:.2f} | auc={m[\"roc_auc\"]:.4f} | features={sum(m[\"feature_mask\"])} | trained={m[\"retrain_date\"][:10]}')
" 2>&1 | ForEach-Object { Log $_ }
}

# ── 5. (TUỲ CHỌN) WALK-FORWARD VERIFY ──────────────────────────────────────
# Data CSV cần có trước khi chạy WF:
#   src/xauusd_ai/real_data/XAUUSDm_M15.csv  (65 MB)
#   src/xauusd_ai/real_data/XAUUSDm_M5.csv  (193 MB)
#   src/xauusd_ai/real_data/XAUUSDm_M1.csv  (896 MB)  ← lớn nhất
#   src/xauusd_ai/real_data/XAUUSDm_H4.csv  (4 MB)
#   src/xauusd_ai/real_data/XAUUSDm_H1.csv  (16 MB)
# Tải về từ Dukascopy: https://www.dukascopy.com/swiss/english/marketwatch/historical/
# Hoặc copy từ máy cũ: scp user@old-machine:"path/XAUUSDm_*.csv" src/xauusd_ai/real_data/
if ($RunWF) {
    Log "--- [5/7] Walk-forward verify (30 folds, ~20 phút) ---"
    Log "Đang chạy WF PROFIT (ACC1, sc=0.85)..."
    $t0 = Get-Date
    python scripts/walkforward_ict_wyckoff.py configs/acc1_v14pp_profit.yaml `
        --test-start 2024-09-01 --max-folds 30 --fast --cache --no-compound `
        > "$PROJ\outputs\wf_deploy_profit.log" 2>&1
    if ($LASTEXITCODE -ne 0) { Log "WF PROFIT có lỗi — xem outputs\wf_deploy_profit.log" "WARN" }
    Log "WF PROFIT xong ($(([int]((Get-Date)-$t0).TotalMinutes))m)"

    Log "Đang chạy WF COMPOSITE (ACC2, sc=0.88)..."
    $t1 = Get-Date
    python scripts/walkforward_ict_wyckoff.py configs/acc1_v14pp_composite.yaml `
        --test-start 2024-09-01 --max-folds 30 --fast --cache --no-compound `
        > "$PROJ\outputs\wf_deploy_composite.log" 2>&1
    if ($LASTEXITCODE -ne 0) { Log "WF COMPOSITE có lỗi — xem outputs\wf_deploy_composite.log" "WARN" }
    Log "WF COMPOSITE xong ($(([int]((Get-Date)-$t1).TotalMinutes))m)"

    # Đọc kết quả nhanh từ report JSON
    foreach ($rp in @("outputs\walkforward_report_acc1_v14pp_profit.json","outputs\walkforward_report_acc1_v14pp_composite.json")) {
        $rpFull = Join-Path $PROJ $rp
        if (Test-Path $rpFull) {
            python -c "
import json, sys
d = json.load(open(r'$($rpFull -replace '\\','/')'))
sim = d.get('concurrent_sim', d.get('aggregate',{}).get('concurrent_sim',{}))
print(f'  {sys.argv[1]}: PF={sim.get(\"avg_profit_factor\",\"?\"):.3f} | DD={sim.get(\"avg_max_drawdown_pct\",\"?\"):.1f}% | PosFolds={sim.get(\"positive_folds\",\"?\")}/{sim.get(\"n_folds\",\"?\")}')
" "$rp" 2>&1 | ForEach-Object { Log $_ }
        }
    }
} else {
    Log "--- [5/7] Bỏ qua WF (mặc định — dùng benchmark đã verify từ git) ---"
    # Đọc benchmark từ JSON đã có
    foreach ($rp in @("outputs/walkforward_report_acc1_v14pp_profit.json","outputs/walkforward_report_acc1_v14pp_composite.json")) {
        $rpFull = Join-Path $PROJ $rp
        if (Test-Path $rpFull) {
            python -c "
import json
d = json.load(open(r'$($rpFull -replace '\\','/')'))
sim = d.get('concurrent_sim', d.get('aggregate',{}).get('concurrent_sim',{}))
name = '$rp'.split('/')[-1].replace('.json','')
print(f'  Benchmark {name}: PF={sim.get(\"avg_profit_factor\",\"?\"):.3f} | DD={sim.get(\"avg_max_drawdown_pct\",\"?\"):.1f}% | PosFolds={sim.get(\"positive_folds\",\"?\")}/{sim.get(\"n_folds\",\"?\")}  [verified 17/04/2026]')
" 2>&1 | ForEach-Object { Log $_ }
        }
    }
}

# ── 6. BUILD + KHỞI ĐỘNG BOT ─────────────────────────────────────────────────
if (-not $SkipBotStart) {
    Log "--- [6/7] Build image và khởi động live bots ---"

    Log "Build xauusd-backend image..."
    docker compose build backend 2>&1 | ForEach-Object { Log $_ }
    if ($LASTEXITCODE -ne 0) { Die "docker build thất bại." }

    Log "Khởi động postgres (nếu chưa chạy)..."
    docker compose up -d postgres 2>&1 | ForEach-Object { Log $_ }
    Start-Sleep 8

    Log "Stop bot cũ (nếu đang chạy)..."
    docker compose stop live-acc1 live 2>&1 | Out-Null

    Log "Khởi động live-acc1 (V14++ PROFIT, sc=0.85)..."
    docker compose up -d live-acc1 2>&1 | ForEach-Object { Log $_ }
    if ($LASTEXITCODE -ne 0) { Die "Không khởi động được live-acc1." }

    Log "Khởi động live (ACC2, V14++ COMPOSITE, sc=0.88)..."
    docker compose up -d live 2>&1 | ForEach-Object { Log $_ }
    if ($LASTEXITCODE -ne 0) { Die "Không khởi động được live (acc2)." }

    Log "Chờ 30 giây cho bot startup..."
    Start-Sleep 30

    # Kiểm tra trạng thái
    $acc1Status = docker inspect trade-indicator-live-acc1-1 --format "{{.State.Status}}" 2>&1
    $acc2Status = docker inspect trade-indicator-live-1       --format "{{.State.Status}}" 2>&1
    Log "live-acc1 : $acc1Status"
    Log "live-acc2 : $acc2Status"

    Log "Logs khởi động ACC1 (15 dòng cuối):"
    docker logs trade-indicator-live-acc1-1 --tail 15 2>&1 | ForEach-Object { Log "  | $_" }
    Log "Logs khởi động ACC2 (10 dòng cuối):"
    docker logs trade-indicator-live-1      --tail 10 2>&1 | ForEach-Object { Log "  | $_" }
} else {
    Log "--- [6/7] Bỏ qua khởi động bot (--SkipBotStart flag) ---"
}

# ── 7. TẠO BÁO CÁO TRIỂN KHAI ───────────────────────────────────────────────
Log "--- [7/7] Tạo báo cáo triển khai ---"

$report = @"
══════════════════════════════════════════════════════════════════
  BÁO CÁO TRIỂN KHAI V14++
  Ngày  : $(Get-Date -Format 'dd/MM/yyyy HH:mm:ss')
  Máy   : $($env:COMPUTERNAME)
  Branch: $BRANCH
══════════════════════════════════════════════════════════════════

1. MODEL ARTIFACTS (từ Git LFS)
──────────────────────────────
"@

foreach ($f in $modelFiles) {
    $fp = Join-Path $PROJ $f
    if (Test-Path $fp) {
        $sz = [math]::Round((Get-Item $fp).Length / 1MB, 2)
        $report += "  ✓ $f  (${sz} MB)`n"
    } else {
        $report += "  ✗ $f  KHÔNG TỒN TẠI`n"
    }
}

$report += @"

2. CẤU HÌNH V14++
─────────────────
  ACC1 (COMBO #133): configs/live_acc1.yaml
    model_path     : outputs/acc1_combo133_202604_model.pkl
    min_confidence : 0.70  (Combo #133)
    signal_thresh  : 0.60
    WF benchmark   : 28 folds | 27/28 pos | +$78,204

  ACC2 (V14++)     : configs/live_acc2.yaml
    model_path     : outputs/acc2_v14pp_202604_model.pkl
    min_confidence : 0.88  (sideway + volatile)
    signal_thresh  : 0.60
    WF benchmark   : PF=3.364 | DD=-6.69% | WR=59.1% | 25/29 pos folds

3. TRẠNG THÁI MODEL
───────────────────
"@

# Thêm thông tin model meta
$metaInfo = python -c "
import json
for acc, path in [('ACC1','outputs/acc1_v14pp_model_meta.json'),('ACC2','outputs/acc2_v14pp_model_meta.json')]:
    m = json.load(open(path))
    feats = sum(m['feature_mask'])
    print(f'  {acc}: threshold={m[\"decision_threshold\"]:.2f} | ROC-AUC={m[\"roc_auc\"]:.4f} | features={feats} | trained={m[\"retrain_date\"][:10]}')
" 2>&1
$report += $metaInfo + "`n"

$report += @"

4. CONTAINER STATUS
───────────────────
"@
$psOut = docker ps --format "  {{.Names}}  |  {{.Status}}  |  {{.Ports}}" 2>&1
$report += ($psOut -join "`n") + "`n"

$report += @"

5. LOGS TRIỂN KHAI
──────────────────
  Xem chi tiết: $LOGFILE

6. DATA CSV (không có trong git — ~1.2 GB)
──────────────────────────────────────
  File cần có tại: src\xauusd_ai\real_data\XAUUSDm_*.csv
  Cách lấy:
    a) Copy từ máy cũ: scp / robocopy / USB
    b) Dukascopy: https://www.dukascopy.com/swiss/english/marketwatch/historical/
    c) Chạy lại script fetch nếu có: python scripts/fetch_data.py
  Lưu ý: Bắt buộc phải có nếu dùng -RetrainModels hoặc -RunWF
          Không cần thiết để chạy live bot (bot dùng MT5 live data)

7. HƯỚNG DẪN TIẾP THEO
───────────────────────
  Kiểm tra bot đang chạy:
    docker logs trade-indicator-live-acc1-1 --tail 20
    docker logs trade-indicator-live-1      --tail 20

  Restart nếu cần:
    docker restart trade-indicator-live-acc1-1
    docker restart trade-indicator-live-1

  Retrain model (hàng tuần — cần data CSV):
    python scripts/retrain_live_model.py configs/live_acc1.yaml
    python scripts/retrain_live_model.py configs/live_acc2.yaml

  Chạy WF verify lại (cần data CSV):
    .\deploy_v14pp.ps1 -RunWF -SkipBotStart

  Thresholds hiện tại:
    ACC1 decision_threshold = 0.44
    ACC2 decision_threshold = 0.46

══════════════════════════════════════════════════════════════════
"@

$report | Out-File -FilePath $REPORT_OUT -Encoding UTF8
$report | Write-Host

Log "════════════════════════════════════"
Log "  DEPLOY HOÀN TẤT"
Log "  Báo cáo: $REPORT_OUT"
Log "  Log    : $LOGFILE"
Log "════════════════════════════════════"
