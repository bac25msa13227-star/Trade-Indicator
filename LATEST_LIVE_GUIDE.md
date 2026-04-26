# Latest Live Guide

Last updated: `2026-04-26` · Branch: `feature/dashboard-controls-v14pp`
Config đang dùng: **Combo #133** (frozen, validated 28 folds 2024–2026, +$78K)

---

## TL;DR — Khởi động nhanh (bots + bridge + API)

```powershell
# 0. Clone repo + lấy model từ Git LFS (LẦN ĐẦU — BẮT BUỘC)
git clone https://github.com/bac25msa13227-star/Trade-Indicator.git
cd Trade-Indicator
git lfs pull   # ← tải model .pkl thật (35-37 MB/file) thay vì pointer

# 1. Copy env
Copy-Item .env.example .env; notepad .env

# 2. Khởi động core (bots + API, KHÔNG cần Grafana/MLflow/Airflow)
docker compose up postgres api live-acc1 live -d

# 3. Bridge MT5 (2 cửa sổ PS riêng)
.\scripts\windows\mt5_bridge_acc1.ps1
.\scripts\windows\mt5_bridge_acc2.ps1

# 4. Verify toàn bộ
.\scripts\windows\verify_live.ps1
```

Dashboard: **http://localhost:8000/dashboard**

---

## 0. Lấy model đúng từ Git LFS

### Model được lưu ở đâu?

File `.pkl` (35–37 MB) quá lớn để lưu trực tiếp trong git. Chúng được track qua **Git LFS** — khi clone thông thường, bạn chỉ nhận được file *pointer* 134 bytes thay vì model thật → bot sẽ lỗi ngay khi load.

**Rule LFS** (xem `.gitattributes`):
```
outputs/*.pkl filter=lfs diff=lfs merge=lfs -text
```

### Clone + lấy model lần đầu

```bash
# Bước 1: Cài Git LFS (nếu chưa có)
# macOS
brew install git-lfs && git lfs install

# Windows
winget install GitHub.GitLFS
git lfs install

# Bước 2: Clone repo
git clone https://github.com/bac25msa13227-star/Trade-Indicator.git
cd Trade-Indicator

# Bước 3: Tải model thật
git lfs pull
```

### Trên repo đã clone sẵn

```bash
# Cập nhật code + lấy model mới nhất
git pull
git lfs pull
```

### Verify model thật hay pointer?

```bash
# Nếu là pointer → hiển thị ~134 bytes và bắt đầu bằng "version https://git-lfs..."
# Nếu là model thật → hiển thị ~35-37 MB
ls -lh outputs/acc1_combo133_202604_model.pkl   # phải ≥ 30 MB
ls -lh outputs/acc2_v14pp_202604_model.pkl       # phải ≥ 30 MB

# Hoặc kiểm tra nội dung đầu file
head -c 50 outputs/acc1_combo133_202604_model.pkl
# Nếu thấy "version https://git-lfs" → chưa pull LFS → chạy lại: git lfs pull
# Nếu thấy binary (ký tự lạ) → đúng rồi
```

```powershell
# Windows PowerShell
(Get-Item outputs\acc1_combo133_202604_model.pkl).Length / 1MB   # phải > 30 MB
```

### Naming convention cho model artifacts

**Rule: `outputs/{account}_{config_id}_{YYYYMM}_{artifact}.{ext}`**

| Field | Mô tả | Ví dụ |
|---|---|---|
| `{account}` | Tài khoản | `acc1`, `acc2` |
| `{config_id}` | Tên config đóng băng | `combo133`, `v14pp`, `v15pp` |
| `{YYYYMM}` | Năm+tháng đóng băng | `202604` (April 2026) |
| `{artifact}` | Loại artifact | `model`, `scaler`, `meta` |
| `{ext}` | Extension | `.pkl` (model/scaler), `.json` (meta) |

**Khi đóng băng config mới:** tăng `{config_id}` hoặc `{YYYYMM}`, đặt tên mới, cập nhật `live_{acc}.yaml` + `auto_update_retrain.py`.

**Retrain định kỳ trong cùng version:** ghi đè file cũ, không đổi tên.

### Danh sách file được quản lý bởi LFS

| File | Kích thước | Mô tả |
|---|---|---|
| `outputs/acc1_combo133_202604_model.pkl` | ~35 MB | Model ACC1 — Combo #133 VotingClassifier (frozen 2026-04) |
| `outputs/acc1_combo133_202604_scaler.pkl` | ~3 KB | StandardScaler ACC1 |
| `outputs/acc2_v14pp_202604_model.pkl` | ~37 MB | Model ACC2 — V14++ strategy (frozen 2026-04) |
| `outputs/acc2_v14pp_202604_scaler.pkl` | ~3 KB | StandardScaler ACC2 |

### Config files (KHÔNG qua LFS — plain text, git thường)

Config files được commit thẳng vào git (không qua LFS):

| File | Mô tả |
|---|---|
| `configs/live_acc1.yaml` | Config bot ACC1 — **Combo #133** |
| `configs/live_acc2.yaml` | Config bot ACC2 — V14++ |
| `configs/xauusd_combo133_best.yaml` | Config dùng bởi `auto_update_retrain.py` |
| `outputs/acc1_combo133_202604_meta.json` | Meta: threshold, ROC-AUC, feature list |
| `outputs/acc2_v14pp_202604_meta.json` | Meta: threshold, ROC-AUC, feature list |

Meta JSON được commit trực tiếp (không qua LFS) — `git pull` bình thường là đủ.

### Sau khi retrain — push model mới lên LFS

```bash
# Retrain xong → model mới ghi vào outputs/acc1_combo133_202604_model.pkl
python scripts/auto_update_retrain.py

# Commit + push (LFS tự upload file lớn)
git add outputs/acc1_combo133_202604_model.pkl outputs/acc1_combo133_202604_scaler.pkl \
        outputs/acc1_combo133_202604_meta.json
git commit -m "chore: retrain acc1 combo133 $(date +%Y-%m-%d)"
git push   # git lfs push tự chạy kèm theo git push

# Đóng băng VERSION MỚI (ví dụ Combo #134, tháng 10/2026):
# 1. Đặt tên mới: acc1_combo134_202610_model.pkl
# 2. Cập nhật configs/live_acc1.yaml + configs/xauusd_combo133_best.yaml (rename combo id)
# 3. Cập nhật scripts/auto_update_retrain.py MODEL_PATH / SCALER_PATH / META_PATH
# 4. git mv + git add + git commit + git push
```

---

## 1. Cấu hình đang dùng — Combo #133

## 1. Cấu hình đang dùng — Combo #133

### Model artifacts (BẮT BUỘC phải có đúng file này)

| Account | Model | Scaler | Meta |
|---|---|---|---|
| ACC1 | `outputs/acc1_combo133_202604_model.pkl` | `outputs/acc1_combo133_202604_scaler.pkl` | `outputs/acc1_combo133_202604_meta.json` |
| ACC2 | `outputs/acc2_v14pp_202604_model.pkl` | `outputs/acc2_v14pp_202604_scaler.pkl` | `outputs/acc2_v14pp_202604_meta.json` |

### Live config files

- ACC1: `configs/live_acc1.yaml` — **Combo #133** (`min_conf=0.70`)
- ACC2: `configs/live_acc2.yaml` — V14++ COMPOSITE

### Combo #133 — key params đã đồng bộ WF ↔ Live

| Param | Giá trị | File |
|---|---|---|
| `min_confidence` | **`0.70`** | `live_acc1.yaml` (risk + strategy) |
| `sideway_min_confidence` | **`0.70`** | `live_acc1.yaml` |
| `volatile_min_confidence` | **`0.70`** | `live_acc1.yaml` |
| `blocked_hours_utc` | **`[3, 15, 17, 22, 23]`** | `live_acc1.yaml` |
| `d1_trend_gate` | **`false`** | `live_acc1.yaml` |
| `require_trend_alignment` | **`false`** | `live_acc1.yaml` |
| `risk_per_trade` | **`4%`** | `live_acc1.yaml` |
| `max_open_positions` | **`3`** | `live_acc1.yaml` |

> ⚠️ **KHÔNG** thay đổi `min_confidence` lên `0.85` — đó là config cũ trước Combo #133. Giá trị chuẩn là `0.70`.

### WF Benchmark (Combo #133, 28+ folds, 2024-01 → 2026-04)

| Metric | Kết quả |
|---|---|
| Total P&L | **+$78,204** |
| Trading days | 514 (357 profitable) |
| Win Rate avg | **46.4%** |
| Avg day P&L | +$152 |
| Max DD | -$838 |
| Starting balance/fold | $200 |

### Threshold hiện tại (kiểm tra sau mỗi lần retrain)

| Account | `min_confidence` | `decision_threshold` (meta) | ROC-AUC |
|---|---|---|---|
| ACC1 | `0.70` | xem `acc1_combo133_202604_meta.json` | xem meta |

---

## 2. Risk settings (key params)

| Param | ACC1 | ACC2 |
|---|---|---|
| `risk_per_trade` | `4%` | `4%` |
| `take_profit_rr` | `3.5R` | `3.5R` |
| `partial_tp_rr` | `1.2R` (50%) | `1.2R` (50%) |
| `consecutive_loss_pause_count` | `3` | `3` |
| `consecutive_loss_cooldown_bars` | `12` (1h) | `12` (1h) |
| `daily_loss_limit_pct` | `12%` | `12%` |
| `max_drawdown_kill_pct` | `20%` | `20%` |
| `kill_switch_enabled` | `true` | `true` |
| `auto_trade` | `true` | `true` |

---

## 3. Environment variables (.env)

Tạo file `.env` từ `.env.example`:

```powershell
Copy-Item .env.example .env
notepad .env
```

Các biến BẮT BUỘC:

```env
# ── MT5 credentials (map vào bridge) ─────────────────
MT5_LOGIN=270832477
MT5_PASSWORD=<password>
MT5_SERVER=Exness-MT5Trial17

# ── Telegram ACC1 ─────────────────────────────────────
TELEGRAM_BOT_TOKEN=<bot_token_acc1>
TELEGRAM_CHAT_ID=<chat_id>

# ── PostgreSQL (core, cần có dù không dùng Grafana) ───
POSTGRES_USER=trader
POSTGRES_PASSWORD=trader_secret
POSTGRES_DB=tradedb

# ── Tùy chọn nếu bật MLflow / MinIO ──────────────────
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# ── Tùy chọn nếu bật Airflow ──────────────────────────
AIRFLOW_FERNET_KEY=<generate: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())">
AIRFLOW_SECRET_KEY=changeme
AIRFLOW_USER=airflow
AIRFLOW_PASSWORD=airflow

# ── Tùy chọn nếu bật Grafana ──────────────────────────
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin
```

> **Lưu ý bảo mật:** Không commit file `.env` lên git. File này đã có trong `.gitignore`.

---

## 4. Verify cấu hình Combo #133 trước khi khởi động

Chạy lệnh sau để chắc chắn config đúng trước khi bật bot:

```powershell
cd C:\path\to\Trade-Indicator
$env:PYTHONPATH = "src"
source .venv\Scripts\Activate.ps1

python -c "
from pathlib import Path
from xauusd_ai.config import load_settings

cfg = Path('configs/live_acc1.yaml')
s = load_settings(cfg)

print('=== COMBO #133 CONFIG VERIFY ===')
print(f'model_path       : {s.app.model_path}')
print(f'min_confidence   : {s.risk.min_confidence}  (expected: 0.70)')
print(f'sideway_conf     : {s.strategy.sideway_min_confidence}  (expected: 0.70)')
print(f'volatile_conf    : {s.strategy.volatile_min_confidence}  (expected: 0.70)')
print(f'blocked_hours    : {sorted(s.strategy.blocked_hours_utc)}  (expected: [3, 15, 17, 22, 23])')
print(f'd1_trend_gate    : {s.strategy.d1_trend_gate}  (expected: False)')
print(f'require_trend    : {s.strategy.require_trend_alignment}  (expected: False)')
print(f'risk_per_trade   : {s.risk.risk_per_trade}  (expected: 0.04)')
print(f'auto_trade       : {s.execution.auto_trade}  (expected: True)')
print()

# Check model file exists
from pathlib import Path as P
model_ok = P(s.app.model_path).exists()
scaler_ok = P(s.app.scaler_path).exists()
print(f'model.pkl exists : {model_ok}')
print(f'scaler.pkl exists: {scaler_ok}')
"
```

Expected output:
```
=== COMBO #133 CONFIG VERIFY ===
model_path       : outputs/acc1_v14pp_model.pkl
min_confidence   : 0.7  (expected: 0.70)
sideway_conf     : 0.7  (expected: 0.70)
volatile_conf    : 0.7  (expected: 0.70)
blocked_hours    : [3, 15, 17, 22, 23]  (expected: [3, 15, 17, 22, 23])
d1_trend_gate    : False  (expected: False)
require_trend    : False  (expected: False)
risk_per_trade   : 0.04  (expected: 0.04)
auto_trade       : True  (expected: True)

model.pkl exists : True
scaler.pkl exists: True
```

> ❌ Nếu bất kỳ dòng nào không khớp expected — **DỪNG, không khởi động bot** cho đến khi fix xong.

---

## 5. Khởi động MT5 Bridge + Verify đặt lệnh

### Bước 1: Khởi động Bridge

Mở 2 cửa sổ PowerShell riêng (chạy trên **máy Windows có MT5**):

```powershell
# ── Cửa sổ 1 — Bridge ACC1 (port 5600) ──
cd C:\path\to\Trade-Indicator
$env:MT5_LOGIN    = "270832477"
$env:MT5_PASSWORD = "<password>"
$env:MT5_SERVER   = "Exness-MT5Trial17"
python scripts/windows/mt5_bridge.py --port 5600

# ── Cửa sổ 2 — Bridge ACC2 (port 5601) ──
$env:MT5_LOGIN    = "433326057"
$env:MT5_PASSWORD = "<password>"
$env:MT5_SERVER   = "Exness-MT5Trial7"
python scripts/windows/mt5_bridge.py --port 5601
```

### Bước 2: Verify bridge sống

```powershell
# Kiểm tra ping
Invoke-RestMethod http://localhost:5600/ping | ConvertTo-Json
# Expected: { "status": "ok", "account": 270832477, "connected": true }

Invoke-RestMethod http://localhost:5601/ping | ConvertTo-Json
# Expected: { "status": "ok", "account": 433326057, "connected": true }
```

### Bước 3: Test đặt lệnh (DRY RUN — không đặt lệnh thật)

```powershell
# Test bridge có thể nhận lệnh không (kiểm tra symbol info, không execute)
$body = @{
    symbol   = "XAUUSDm"
    action   = "check"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:5600/symbol_info `
    -Method POST -Body $body -ContentType "application/json" | ConvertTo-Json
# Expected: trả về { "symbol": "XAUUSDm", "bid": ..., "ask": ..., "spread": ... }
```

Nếu `spread` trả về giá trị hợp lý (1–5 pips với XAUUSD) → bridge kết nối được.

> ⚠️ **KHÔNG** gọi `/order` trong test — sẽ đặt lệnh thật vào tài khoản live.

---

## 6. Verify Telegram hoạt động

Telegram token được load từ `live_acc1.yaml` → `integrations.telegram.token_env: TELEGRAM_BOT_TOKEN`.

### Test gửi tin nhắn

```powershell
# Thông qua API endpoint (khi API đã chạy)
$body = @{ message = "Test Telegram ACC1 - Combo #133 live bot startup OK" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/v1/send-telegram `
    -Method POST -Body $body -ContentType "application/json"
# Expected: { "ok": true }
```

```powershell
# Thông qua Python trực tiếp (khi API chưa chạy)
$env:PYTHONPATH = "src"
python -c "
import os, requests
token   = os.environ.get('TELEGRAM_BOT_TOKEN', '')
chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
if not token or not chat_id:
    print('ERROR: TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHAT_ID chưa set trong .env')
else:
    resp = requests.post(
        f'https://api.telegram.org/bot{token}/sendMessage',
        json={'chat_id': chat_id, 'text': 'Test: ACC1 Combo #133 startup OK'},
        timeout=10
    )
    print('Status:', resp.status_code, '|', resp.json().get('ok'))
"
```

### Telegram không hoạt động — checklist debug

| Triệu chứng | Kiểm tra |
|---|---|
| `ok: false` hoặc timeout | Token sai → lấy lại từ @BotFather |
| `chat not found` | Chat ID sai → dùng `@userinfobot` để lấy chat ID |
| Không có tin nhắn | Bot chưa được `/start` trong chat → mở Telegram → gửi `/start` cho bot |
| `403 Forbidden` | Bot bị block → unblock trong Telegram |
| Token đúng nhưng từ chối | Bot token chưa set trong `.env` → `notepad .env` |

---

## 7. Khởi động Docker — Chọn theo nhu cầu

Docker Compose chia thành 4 nhóm dịch vụ. Chỉ khởi động nhóm cần thiết:

### 7a. Core (bắt buộc — bot trading + API)

```bash
docker compose up postgres api live-acc1 live -d
```

| Service | Port | Mô tả |
|---|---|---|
| `postgres` | 5432 | Database lưu trades, signals |
| `api` | 8000 | FastAPI dashboard + REST endpoints |
| `live-acc1` | — | Bot ACC1 với `configs/live_acc1.yaml` (Combo #133) |
| `live` | — | Bot ACC2 với `configs/live_acc2.yaml` |

Kiểm tra bots đang chạy:
```bash
docker compose ps
docker compose logs live-acc1 --tail 20
docker compose logs live --tail 20
```

---

### 7b. Monitoring — Grafana + Prometheus (tuỳ chọn)

**Bật:**
```bash
docker compose up prometheus grafana -d
```

| Service | Port | URL |
|---|---|---|
| `prometheus` | 9090 | http://localhost:9090 |
| `grafana` | 3000 | http://localhost:3000 (admin/admin) |

Grafana dashboard tự load từ `configs/grafana/provisioning/` nếu đã có sẵn.

**Tắt:**
```bash
docker compose stop prometheus grafana
```

**Tắt và xoá data:**
```bash
docker compose down prometheus grafana
docker volume rm trade-indicator_prometheus_data trade-indicator_grafana_data
```

---

### 7c. MLflow + MinIO (tuỳ chọn — experiment tracking)

**Bật:**
```bash
docker compose up minio mlflow -d
```

| Service | Port | URL |
|---|---|---|
| `minio` | 9000 / 9001 | http://localhost:9001 (minioadmin/minioadmin) |
| `mlflow` | 5000 | http://localhost:5000 |

Cần `postgres` đang chạy (`minio` và `mlflow` depends on `postgres`).

**Tắt:**
```bash
docker compose stop minio mlflow
```

**Tắt và xoá data (xoá toàn bộ artifact + experiments):**
```bash
docker compose down minio mlflow
docker volume rm trade-indicator_minio_data
```

> ⚠️ Xoá `minio_data` sẽ mất toàn bộ model artifacts đã push lên MinIO. Backup trước nếu cần.

---

### 7d. Airflow (tuỳ chọn — daily data ingest + model training DAG)

**Bật lần đầu (init DB + tạo user):**
```bash
docker compose up airflow-init
# Đợi cho đến khi container exit 0, rồi:
docker compose up airflow-webserver airflow-scheduler -d
```

| Service | Port | URL |
|---|---|---|
| `airflow-webserver` | 8080 | http://localhost:8080 (airflow/airflow) |
| `airflow-scheduler` | — | background |

DAGs có sẵn:
- `dags/daily_data_ingest.py` — fetch data mỗi ngày
- `dags/model_training.py` — retrain model theo lịch
- `dags/drift_monitoring.py` — monitor feature drift

**Bật lại (sau khi đã init):**
```bash
docker compose up airflow-webserver airflow-scheduler -d
```

**Tắt:**
```bash
docker compose stop airflow-webserver airflow-scheduler
```

---

### 7e. Full stack (tất cả dịch vụ)

```bash
docker compose up -d
```

Thứ tự khởi động: postgres → minio → mlflow → airflow-init → airflow-webserver + scheduler → prometheus → grafana → api → live-acc1 + live → nginx → cloudflared.

---

### Bảng tổng hợp — Enable / Disable từng service

| Service | Enable | Disable | Data volume |
|---|---|---|---|
| **Bot ACC1** | `docker compose up live-acc1 -d` | `docker compose stop live-acc1` | `./outputs/` (mount) |
| **Bot ACC2** | `docker compose up live -d` | `docker compose stop live` | `./outputs/` (mount) |
| **API / Dashboard** | `docker compose up api -d` | `docker compose stop api` | none |
| **Grafana** | `docker compose up prometheus grafana -d` | `docker compose stop prometheus grafana` | `grafana_data`, `prometheus_data` |
| **MLflow** | `docker compose up minio mlflow -d` | `docker compose stop minio mlflow` | `minio_data` |
| **Airflow** | `docker compose up airflow-webserver airflow-scheduler -d` | `docker compose stop airflow-webserver airflow-scheduler` | `airflow_logs` |

---

## 8. Dashboard features

| Feature | URL / Lệnh |
|---|---|
| Dashboard JSON | `GET http://localhost:8000/api/v1/dashboard` |
| Bật auto trade ACC1 | `POST http://localhost:8000/api/v1/auto-trade/acc1?enabled=true` |
| Tắt auto trade ACC1 | `POST http://localhost:8000/api/v1/auto-trade/acc1?enabled=false` |
| Reset kill switch ACC1 | `POST http://localhost:8000/api/v1/reset-kill-switch/acc1` |
| Gửi Telegram thủ công | `POST http://localhost:8000/api/v1/send-telegram` |
| Lấy tunnel URL | `GET http://localhost:8000/api/v1/tunnel-url` |
| Swagger UI | `GET http://localhost:8000/docs` |

---

## 9. Tunnel public dashboard (tùy chọn)

```bash
# Qua Docker (tự động)
docker compose up cloudflared -d
docker compose logs cloudflared 2>&1 | grep trycloudflare
# URL dạng: https://xxx.trycloudflare.com

# Hoặc Windows thủ công
.\cloudflared.exe tunnel --url http://localhost:8000 > outputs\tunnel_err.txt 2>&1 &
```

---

## 10. Checklist verify sau khi khởi động

```powershell
# ── 1. Config Combo #133 đúng không? ─────────────────────────
$s = Invoke-RestMethod http://localhost:8000/api/v1/dashboard
$s.accounts.acc1.config.min_confidence    # phải = 0.7
$s.accounts.acc1.config.blocked_hours     # phải có 3, 15, 17, 22, 23

# ── 2. Model file đúng không? ─────────────────────────────────
Test-Path outputs\acc1_v14pp_model.pkl     # True
Test-Path outputs\acc2_v14pp_model.pkl     # True

# ── 3. Bridge ACC1 sống không? ────────────────────────────────
(Invoke-RestMethod http://localhost:5600/ping).connected  # True

# ── 4. Bridge ACC2 sống không? ────────────────────────────────
(Invoke-RestMethod http://localhost:5601/ping).connected  # True

# ── 5. API sống không? ────────────────────────────────────────
(Invoke-RestMethod http://localhost:8000/health).status   # "ok"

# ── 6. Kill switch chưa kích hoạt? ───────────────────────────
(Get-Content outputs\risk_daily_state_acc1.json | ConvertFrom-Json).killed  # False
(Get-Content outputs\risk_daily_state_acc2.json | ConvertFrom-Json).killed  # False

# ── 7. Auto trade đang bật? ───────────────────────────────────
$s.accounts.acc1.auto_trade_enabled       # True
$s.accounts.acc2.auto_trade_enabled       # True

# ── 8. Bots đang ghi status file gần đây (không bị treo)? ────
(Get-Date) - (Get-Item outputs\live_status_acc1.json).LastWriteTime  # < 5 phút
(Get-Date) - (Get-Item outputs\live_status_acc2.json).LastWriteTime  # < 5 phút
```

---

## 11. Dừng toàn bộ

```bash
# Dừng tất cả (giữ data volumes)
docker compose stop

# Dừng + xoá containers (giữ data volumes)
docker compose down

# Dừng + xoá containers + data volumes (RESET HOÀN TOÀN)
docker compose down -v
```

Hoặc Windows native:
```powershell
Get-Process -Name python, cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
```

---

## 12. Troubleshooting

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| `min_confidence` = 0.85 trong logs | Config cũ chưa cập nhật | Kiểm tra `live_acc1.yaml` line ~120: phải là `0.70` |
| `model_binding_ok: false` | Model file không khớp config | Copy đúng `acc1_v14pp_model.pkl` vào `outputs/` |
| Bridge `/ping` trả `connected: false` | MT5 terminal chưa mở / sai credentials | Mở MT5 → đăng nhập → restart bridge |
| `killed: true` trong daily state | Kill switch bị kích hoạt do loss | `POST /api/v1/reset-kill-switch/acc1` |
| `auto_trade: false` sau restart | Config bị ghi đè | `POST /api/v1/auto-trade/acc1?enabled=true` |
| Dashboard không load | Port 8000 bị chiếm | `netstat -ano \| findstr :8000` rồi kill |
| Telegram không nhận tin | Token sai hoặc chưa `/start` | Xem Section 6 checklist |
| Airflow DAG lỗi kết nối | `postgres` chưa sẵn sàng | Đợi `postgres` healthy rồi restart scheduler |
| MLflow lỗi S3 | MinIO chưa chạy | `docker compose up minio -d` trước |
| Grafana không có data | Prometheus chưa scrape | Kiểm tra `configs/prometheus.yml` target |
| Bot không vào lệnh | Blocked hours / D1 gate / confidence gate | Xem logs `docker compose logs live-acc1 --tail 50` |

---

## 13. Retrain model (Combo #133)

Model tự retrain mỗi ~6,000 M5 bars (~20 trading days) qua `scripts/auto_update_retrain.py`:

```bash
# Cron macOS — 06:00 UTC mỗi ngày
0 6 * * * cd "/path/to/Trade-Indicator" && .venv/bin/python scripts/auto_update_retrain.py

# Chạy thủ công
python scripts/auto_update_retrain.py

# Chỉ fetch data, không retrain
python scripts/auto_update_retrain.py --data-only

# Chỉ retrain, không fetch data
python scripts/auto_update_retrain.py --train-only
```

Sau khi retrain thành công → restart bot để load model mới:
```bash
docker compose restart live-acc1
```

Kiểm tra model đã được load:
```bash
docker compose logs live-acc1 --tail 20 | grep -i "model\|threshold\|retrain"
```

---

## 14. Market open/close gate

Bot tự block lệnh khi sàn đóng. Config trong `live_acc1.yaml`:

```yaml
market:
  enforce_market_open_gate: true
  market_tick_stale_seconds: 300
  market_preopen_alert_minutes_list: [1440, 30]
  market_preclose_alert_minutes_list: [1440, 30]
```

---

## 15. Data source

Training data: `src/xauusd_ai/real_data/` (CSV, từ `2003-05-05`)

- `XAUUSDm_D1.csv`, `H4.csv`, `H1.csv`, `M30.csv`, `M15.csv`, `M5.csv`, `M1.csv`

Live data: MT5 bridge (real-time feed từ Exness).


