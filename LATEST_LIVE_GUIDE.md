# XAUUSD AI Trading System — Live Deployment Guide

> **AI PROMPT — ĐỌC TRƯỚC KHI LÀM BẤT CỨ GÌ:** Luôn nhớ phải lấy data để trade từ bridge chứ không được phép lấy từ yfinace vì sẽ làm trễ quyết định của bot, phải lấy data được chuẩn từng tick
>
> Bạn là AI assistant được giao nhiệm vụ deploy và verify hệ thống XAUUSD AI Trading trên máy mới.
> Hệ thống này là một **ML-powered scalping bot** giao dịch vàng XAU/USD trên khung M1, sử dụng
> mô hình Random Forest (scikit-learn) kết hợp ICT + Wyckoff + momentum fusion, chạy qua MT5 bridge.
>
> **Kiến trúc:** 2 tài khoản (ACC1, ACC2) chạy song song, mỗi tài khoản có config YAML + model .pkl riêng.
> Tất cả tham số đã được tối ưu bằng Walk-Forward (WF) 8-fold với friction model thực tế theo từng sàn.
>
> **QUAN TRỌNG:**
> - Project **KHÔNG cần** MLflow, Airflow, MinIO, Grafana để chạy core (train/backtest/WF/live/paper).
> - Tất cả infra services (MLflow, MinIO, Airflow, Grafana, Prometheus) là **optional** — chỉ dùng khi cần observability.
> - Core chỉ cần `requirements-core.txt` (pandas, numpy, scikit-learn, PyYAML, pydantic).
> - Docker chỉ cần `postgres` (optional, có CSV fallback) + `api` + `nginx` cho dashboard.
>
> **SAU KHI HOÀN TẤT**, bạn **PHẢI** chạy checklist ở Section 14 và xuất báo cáo theo template ở Section 15.
>
> **⚠️ CẢNH BÁO — WF-LIVE GAP:** Hệ thống từng bị live thua liên tục (1W/8L ≈ −$17/acc, tuần 14/4) dù WF $85k+ do 3 GAP kỹ thuật. Tất cả đã fix 2026-04-13. Đọc **Section 0** bên dưới trước khi deploy.

Last updated: `2026-04-13` (post-gap-fix revalidation — models retrained, WF re-run)

---

## Mục lục

0. [Cảnh báo — Tại sao live lệch WF & Cách tránh](#0-cảnh-báo--tại-sao-live-lệch-wf--cách-tránh)
1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Config & Model đang khóa](#2-config--model-đang-khóa)
3. [WF-Validated Parameters (Exness Pro)](#3-wf-validated-parameters-exness-pro)
4. [WF-Validated Parameters (VT Markets STP)](#4-wf-validated-parameters-vt-markets-stp)
5. [Friction Model theo sàn](#5-friction-model-theo-sàn)
6. [Preflight máy mới](#6-preflight-máy-mới)
7. [File .env tối thiểu](#7-file-env-tối-thiểu)
8. [Chạy LITE (không MLflow/Airflow/MinIO/Grafana)](#8-chạy-lite-không-mlflowairflowminiografana)
9. [Chạy Full Stack (optional)](#9-chạy-full-stack-optional)
10. [Train / Backtest / Walkforward (local, không Docker)](#10-train--backtest--walkforward-local-không-docker)
11. [Live trade qua MT5 Bridge](#11-live-trade-qua-mt5-bridge)
12. [Chạy trên Windows host](#12-chạy-trên-windows-host)
13. [Verify binding & health check](#13-verify-binding--health-check)
14. [Checklist cho AI / người deploy](#14-checklist-cho-ai--người-deploy)
15. [Template báo cáo sau deploy](#15-template-báo-cáo-sau-deploy)

---

## 0. Cảnh báo — Tại sao live lệch WF & Cách tránh

> **⚠️ ĐỌC PHẦN NÀY TRƯỚC KHI DEPLOY LIVE.** Hệ thống đã từng xảy ra sự cố: live −$17/acc/tuần dù WF backtest $85k+. Root cause: 4 GAP kỹ thuật giữa code path WF (train) và code path live (inference).

### 4 Nguyên nhân đã confirm & fix (2026-04-13)

| ID | Mức độ | Vấn đề | Fix đã áp dụng |
|----|--------|--------|----------------|
| **G1** | CRITICAL | `save_model.py` dùng `baseline` sample weighting. WF dùng `day_stability_strict` (giảm weight giờ xấu 02-07/17-21 UTC → 0.75×, thứ Hai → 0.80×, ATR spike → 0.85×). Model live học phân phối khác WF. | Thêm `day_stability_strict` vào `_train_dir()` trong `acc*_scalp_m1_save_model.py` |
| **G2** | HIGH | WF dataset thiếu cột `atr` → engine fallback sang ATR5 proxy (không ổn định). Live dùng `ATR(14)` thực tế. Feature drift lớn ở tất cả signal. | Thêm `m1["atr"] = _atr14_fn(m1, 14)` vào `build_scalp_dataset()` trong `scalp_dataset.py` |
| **G3** | MEDIUM | WF hardcode `volatility_regime = 1` cho mọi bar. Live tính `infer_scalp_volatility_regime()` → real 0/1/2. 100% mismatch feature này. | Thêm `m1["volatility_regime"] = infer_scalp_volatility_regime(m1).astype(int)` vào `build_scalp_dataset()` |
| **G4** | MEDIUM | `slippage_rr: 0.01` quá lạc quan — M1 execution drift thực tế ~0.4 pts ≈ 0.05 RR. WF overstate PF. | `slippage_rr: 0.01 → 0.05` trong cả 2 YAML config |

### Format CSV bắt buộc — 9 cột (đúng thứ tự)

```
time,open,high,low,close,tick_volume,spread_points,tick_volume_delta,volume_imbalance
```

| Cột | Tính như thế nào |
|-----|------------------|
| `tick_volume_delta` | `tv[i] - tv[i-1]` — order flow proxy |
| `volume_imbalance` | `(close-low)/(high-low)` — body pressure; `0.5` nếu flat bar |
| `atr` | **KHÔNG** cần trong CSV — tự tính trong `scalp_dataset.py` |
| `volatility_regime` | **KHÔNG** cần trong CSV — tự tính trong `scalp_dataset.py` |

Dùng `scripts/realtime_bar_appender.py` để append bars mới từ MT5 bridge — script tự tính `tick_volume_delta` và `volume_imbalance` đúng cách.

### 5 Nguyên tắc để live ≡ WF benchmark

1. **Đúng training objective:** Train bằng `scripts/acc*_scalp_m1_save_model.py` (đã dùng `day_stability_strict`) — KHÔNG dùng `main.py train` nếu chưa verify weighting.
2. **CSV đúng 9 cột:** Dùng `scripts/realtime_bar_appender.py` để update data realtime, không tự append thủ công nếu không tính `tick_volume_delta`/`volume_imbalance` đúng.
3. **ATR(14) trong dataset:** `scalp_dataset.py` phải có dòng `m1["atr"] = _atr14_fn(m1, 14)` ở cuối `build_scalp_dataset()` (G2 fix).
4. **Regime tính thực:** `scalp_dataset.py` phải có dòng `m1["volatility_regime"] = infer_scalp_volatility_regime(m1).astype(int)` (G3 fix).
5. **Slippage realistic:** `slippage_rr: 0.05` trong config — KHÔNG đổi về `0.01`.

### Verify nhanh sau `git clone`

```bash
# Verify G2+G3 fix có trong scalp_dataset.py
grep -n '_atr14_fn\|infer_scalp_volatility_regime' src/xauusd_ai/features/scalp_dataset.py | tail -4
# Kỳ vọng: 2 dòng assignment cuối build_scalp_dataset() — m1["atr"] và m1["volatility_regime"]

# Verify slippage config
grep 'slippage_rr' configs/live_acc1_scalp_m1.yaml configs/live_acc2_scalp_m1.yaml
# Kỳ vọng: slippage_rr: 0.05 (cả 2 file)

# Verify CSV header (nếu đã có data)
head -1 src/xauusd_ai/real_data/XAUUSDm_M1.csv 2>/dev/null || echo 'No CSV yet — cần setup bar appender'
# Kỳ vọng: time,open,high,low,close,tick_volume,spread_points,tick_volume_delta,volume_imbalance
```

---

## 1. Tổng quan hệ thống

- **Repo**: `https://github.com/bac25msa13227-star/Trade-Indicator.git`
- **Branch**: `codex/pf-optimize-from-task4-clean`
- **Python**: 3.11+
- **Execution TF**: M1 (1 phút), multi-TF analysis: D1/H4/H1/M30/M15/M5/M1
- **ML Model**: scikit-learn RandomForest trên 50+ features (momentum, order flow, microstructure, structure, session)
- **2 accounts chạy song song**:
  - `ACC1` = scalp M1 reversal (nhiều lệnh, threshold thấp hơn)
  - `ACC2` = scalp M1 freeze H10 setup-exit (bảo thủ, guarded)
- **Không dùng**: Streamlit, old configs (`live_acc1.yaml`, `live_acc2.yaml`, `live_acc2_scalp.yaml`)
- **Dashboard**: FastAPI WebSocket tại `http://localhost/dashboard`

### Dependency tối thiểu (core only)

```
pandas>=2.2.3, numpy>=1.26.4, scikit-learn>=1.5.2
PyYAML>=6.0.2, pydantic>=2.9.2, requests>=2.32.3
matplotlib>=3.9.2, python-dotenv>=1.0.1
yfinance>=0.2.54, SQLAlchemy>=2.0.35, prometheus-client>=0.20.0
```

**Không bắt buộc** (infra — `requirements-infra.txt`):
- `mlflow-skinny` — experiment tracking (graceful no-op nếu thiếu)
- `minio`, `boto3` — object storage (fallback local filesystem)
- `psycopg2-binary` — PostgreSQL (fallback CSV)
- `fastapi`, `uvicorn` — REST API/dashboard
- `evidently` — drift monitoring (graceful no-op)

---

## 2. Config & Model đang khóa

### ACC1

| Item | Path |
|------|------|
| Config | `configs/live_acc1_scalp_m1.yaml` |
| Model | `outputs/acc1_scalp_m1_reversal_model.pkl` |
| Scaler | `outputs/acc1_scalp_m1_reversal_scaler.pkl` |
| Meta | `outputs/acc1_scalp_m1_reversal_model_meta.json` |
| WF Report | `outputs/acc1_scalp_m1_reversal_walkforward_report.json` |
| Vai trò | Nhiều lệnh hơn, bắt đảo chiều tốt, threshold thấp |

### ACC2

| Item | Path |
|------|------|
| Config | `configs/live_acc2_scalp_m1.yaml` |
| Model | `outputs/acc2_scalp_m1_h10_setup_exit_model.pkl` |
| Scaler | `outputs/acc2_scalp_m1_h10_setup_exit_scaler.pkl` |
| Meta | `outputs/acc2_scalp_m1_h10_setup_exit_model_meta.json` |
| WF Report | `outputs/acc2_scalp_m1_walkforward_report.json` |
| Vai trò | Profile freeze, guarded, bảo thủ, không sửa tùy tiện |

### Docker service tương ứng

```bash
docker compose up -d live-acc1          # ACC1
docker compose up -d live-scalp-acc2    # ACC2
```

---

## 3. WF-Validated Parameters (Exness Pro)

> Cả ACC1 và ACC2 hiện tại đều đang chạy demo trên **Exness Pro**.

**Friction model Exness Pro:**
- `spread_cost_rr: 0.035` (~$0.15 spread / $4.38 1R)
- `slippage_rr: 0.05` (M1 execution drift ~0.4 pts — cập nhật 2026-04-13, trước là 0.01 quá lạc quan)
- `commission_rr: 0.0` (spread-only, no commission)

### ACC1 — Exness Pro (WF 8-fold validated)

| Parameter | Giá trị | WF Source |
|-----------|---------|-----------|
| `signal_threshold` | **0.59** | Best from 120 candidates |
| `min_confidence` | **0.59** | = threshold |
| `risk_per_trade` | **0.026** (2.6%) | WF optimized |
| `setup_exit_scale` | **0.75** | WF optimized |
| `cooldown_bars` | **4** | WF synced |
| `silver_bullet_enabled` | **false** | A/B test: OFF wins by $6,162 |
| `daily_loss_limit_pct` | **0.20** | |
| `max_drawdown_kill_pct` | **0.12** | |

**WF kết quả (8-fold, Exness Pro friction, $200 initial) — post-gap-fix 2026-04-13:**
- Net profit: **$85,327** | PF: **1.761** | Win rate: ~45%
- Fold 8 (Feb–Apr 2026): **+$18,103** | Avg fold DD: **10.8%** | Total trades: **22,786**
- Feasible candidates: **7/120**
- *(Số thấp hơn run cũ do slippage_rr 0.01→0.05 — friction thực tế hơn)*

### ACC2 — Exness Pro (WF 8-fold validated)

| Parameter | Giá trị | WF Source |
|-----------|---------|-----------|
| `signal_threshold` | **0.60** | Best from 120 candidates |
| `min_confidence` | **0.60** | = threshold |
| `risk_per_trade` | **0.020** (2.0%) | WF postgap opt 2026-04-13 (trước là 2.2%) |
| `setup_exit_scale` | **0.75** | WF locked |
| `cooldown_bars` | **4** | WF synced |
| `silver_bullet_enabled` | **true** | A/B test: ON wins by $2,399 |
| `silver_bullet_confidence_boost` | **0.03** | |
| `daily_loss_limit_pct` | **0.12** | |
| `max_drawdown_kill_pct` | **0.12** | |

**WF kết quả (8-fold, Exness Pro friction, $200 initial) — post-gap-fix 2026-04-13:**
- Net profit: **$57,356** | PF: **1.773** | Win rate: ~46%
- Fold 8 (Feb–Apr 2026): **+$14,725** | Avg fold DD: **10.1%** | Total trades: **20,937**
- Feasible candidates: **1/120**
- *(Số thấp hơn run cũ do slippage_rr 0.01→0.05 + risk giảm 2.2→2.0%)*

---

## 4. WF-Validated Parameters (VT Markets STP)

> Nếu chuyển sang **VT Markets STP**, cần thay đổi friction model trong config.

**Friction model VT Markets STP:**
- `spread_cost_rr: 0.057` (~$0.25 spread / $4.38 1R)
- `slippage_rr: 0.01`
- `commission_rr: 0.0` (spread-only)

### ACC1 — VT Markets STP

| Parameter | Giá trị |
|-----------|---------|
| `signal_threshold` | **0.62** |
| `min_confidence` | **0.62** |
| `risk_per_trade` | **0.026** |
| `setup_exit_scale` | **0.80** |
| `spread_cost_rr` | **0.057** |

**WF kết quả:** Net profit: **$83,455** | PF: **1.94** | Feasible: **4/120**

### ACC2 — VT Markets STP

| Parameter | Giá trị |
|-----------|---------|
| `signal_threshold` | **0.60** |
| `min_confidence` | **0.60** |
| `risk_per_trade` | **0.020** |
| `setup_exit_scale` | **0.80** |
| `spread_cost_rr` | **0.057** |

**WF kết quả:** Net profit: ~**$63,000** | PF: ~**1.86** | Feasible: **1/120**

### Cách chuyển sàn

Sửa trong file YAML config (`configs/live_accX_scalp_m1.yaml`):

```yaml
# Exness Pro (hiện tại)
spread_cost_rr: 0.035
commission_rr: 0.0

# VT Markets STP
# spread_cost_rr: 0.057
# commission_rr: 0.0
```

Sau đó chạy lại WF friction search:

```bash
PYTHONPATH=src python3 scripts/acc1_friction_opt_wf.py --config configs/live_acc1_scalp_m1.yaml --n-folds 8 --out-prefix acc1_wf
PYTHONPATH=src python3 scripts/acc2_friction_opt_wf.py --config configs/live_acc2_scalp_m1.yaml --n-folds 8 --out-prefix acc2_wf
```

Session spread multiplier đã cài trong script theo từng sàn. Nếu đổi sàn, sửa `_SESSION_SPREAD_MULT` trong script tương ứng.

---

## 5. Friction Model theo sàn

### Session Spread Multipliers

| Session | UTC Hours | Exness Pro | VT Markets STP |
|---------|-----------|-----------|----------------|
| Asian | 22:00–02:00 | 1.65 | 1.75 |
| Pre-London | 05:00–07:00 | 1.20 | 1.20 |
| London | 07:00–11:00 | 0.67 | 0.65 |
| Lunch | 11:00–12:30 | 0.87 | 0.90 |
| NY–London Overlap | 12:30–16:00 | 0.53 | 0.50 |
| NY Solo | 16:00–20:00 | 0.80 | 0.80 |
| NY Close | 20:00–22:00 | 1.20–1.50 | 1.20–1.50 |

### Session Slippage Multipliers (cả 2 sàn giống nhau)

| Session | Multiplier |
|---------|------------|
| Asian (22–03) | 1.80 |
| London (07–12) | 0.70 |
| NY–London Overlap (12–16) | 0.60 |
| Other hours | 1.00 |

---

## 6. Preflight máy mới

### Bước 1 — Clone repo

```bash
git clone https://github.com/bac25msa13227-star/Trade-Indicator.git
cd Trade-Indicator
git checkout codex/pf-optimize-from-task4-clean
```

### Bước 2 — Kiểm tra Docker (ưu tiên trước)

```bash
docker --version && docker compose version
# Nếu OK → dùng Docker (xem Bước 3a). Không cần cài Python packages thủ công.
# Nếu không có Docker → dùng local Python (xem Bước 3b).
```

### Bước 3a — Dùng Docker ✅ (recommended)

```bash
# Build images (lần đầu hoặc sau khi đổi code)
docker compose build

# Verify config + model bên trong container
docker compose run --rm api python3 -c "
from pathlib import Path
from xauusd_ai.config import load_settings
s1 = load_settings(Path('configs/live_acc1_scalp_m1.yaml'))
s2 = load_settings(Path('configs/live_acc2_scalp_m1.yaml'))
print(f'ACC1: thr={s1.strategy.signal_threshold}, risk={s1.risk.risk_per_trade}')
print(f'ACC2: thr={s2.strategy.signal_threshold}, risk={s2.risk.risk_per_trade}')
print('OK: Both configs loaded')
"

# Verify model files bên trong container
docker compose run --rm api ls -lh /app/outputs/*.pkl /app/outputs/*.json

# Start minimal live stack
docker compose up -d postgres api nginx
docker compose up -d live-acc1 live-scalp-acc2 healthwatch

# Kiểm tra health
docker compose ps
curl -sS http://localhost:8000/health
```

> Chuyển sang **Section 8** để biết chi tiết các service Docker. Bỏ qua Bước 3b.

### Bước 3b — Local Python (fallback khi không có Docker)

```bash
# Chỉ làm khi Docker không có hoặc không muốn dùng Docker

# Tạo venv
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install core (đủ cho train/backtest/WF/live/paper)
pip install -r requirements-core.txt
pip install -e .

# Verify config load
PYTHONPATH=src python3 -c "
from pathlib import Path
from xauusd_ai.config import load_settings
load_settings(Path('configs/live_acc1_scalp_m1.yaml'))
load_settings(Path('configs/live_acc2_scalp_m1.yaml'))
print('OK: Both configs loaded')
"
```

### Bước 4 — Verify chung (Docker hoặc local)

```bash
# Verify model files tồn tại và không phải LFS stub
ls -lh outputs/*.pkl outputs/*.json

# Verify market data
ls -lh src/xauusd_ai/real_data/XAUUSDm_M1.csv

# Verify CSV có đúng 9 cột bắt buộc
head -1 src/xauusd_ai/real_data/XAUUSDm_M1.csv
# Kỳ vọng: time,open,high,low,close,tick_volume,spread_points,tick_volume_delta,volume_imbalance

# Verify G2+G3 fix có trong scalp_dataset.py (bắt buộc để live = WF)
grep -n '_atr14_fn\|infer_scalp_volatility_regime' src/xauusd_ai/features/scalp_dataset.py | tail -4
# Kỳ vọng: thấy 2 dòng assignment — m1["atr"] và m1["volatility_regime"]
```

> **Lưu ý:** File CSV data (`src/xauusd_ai/real_data/XAUUSDm_*.csv`) **KHÔNG** được push lên git (gitignore).
> Cần copy thủ công hoặc fetch bằng `scripts/fetch_xauusd_dukascopy.py`.

---

## 7. File `.env` tối thiểu

```env
# PostgreSQL (optional — code có CSV fallback nếu không có DB)
POSTGRES_USER=trader
POSTGRES_PASSWORD=trader_secret
POSTGRES_DB=tradedb

# Telegram notifications (optional nhưng recommended)
TELEGRAM_BOT_TOKEN_ACC1=<token_acc1>
TELEGRAM_CHAT_ID_ACC1=<chat_id_acc1>
TELEGRAM_BOT_TOKEN_ACC2=<token_acc2>
TELEGRAM_CHAT_ID_ACC2=<chat_id_acc2>
```

**Không cần** (chỉ khi dùng full stack):
- `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` — MinIO object storage
- `GRAFANA_USER` / `GRAFANA_PASSWORD` — Grafana dashboards
- `AIRFLOW_*` — Airflow orchestration

---

## 8. Chạy LITE (không MLflow/Airflow/MinIO/Grafana)

> **Đây là cách chạy chính, recommended cho hầu hết trường hợp.**

### Option A: Docker (minimal services)

```bash
# Bước 1: Tạo .env
cp .env.example .env
# Sửa TELEGRAM tokens nếu cần

# Bước 2: Build & start (chỉ 3 service)
docker compose up -d postgres api nginx

# Bước 3: Chạy live bots
docker compose up -d live-acc1 live-scalp-acc2 healthwatch

# Bước 4: Verify
docker compose ps
curl -sS http://localhost:8000/health
```

**Dashboard:** `http://localhost/dashboard` hoặc `http://localhost:8000/dashboard`

### Option B: Local Python (không Docker)

```bash
source .venv/bin/activate

# Train model (nếu chưa có .pkl)
PYTHONPATH=src python3 src/xauusd_ai/main.py train --config configs/live_acc1_scalp_m1.yaml
PYTHONPATH=src python3 src/xauusd_ai/main.py train --config configs/live_acc2_scalp_m1.yaml

# Backtest
PYTHONPATH=src python3 src/xauusd_ai/main.py backtest --config configs/live_acc1_scalp_m1.yaml
PYTHONPATH=src python3 src/xauusd_ai/main.py backtest --config configs/live_acc2_scalp_m1.yaml

# Walk-Forward
PYTHONPATH=src python3 src/xauusd_ai/main.py walkforward --config configs/live_acc1_scalp_m1.yaml
PYTHONPATH=src python3 src/xauusd_ai/main.py walkforward --config configs/live_acc2_scalp_m1.yaml

# Paper trade
PYTHONPATH=src python3 src/xauusd_ai/main.py paper --config configs/live_acc1_scalp_m1.yaml
PYTHONPATH=src python3 src/xauusd_ai/main.py paper --config configs/live_acc2_scalp_m1.yaml

# Live (cần MT5 bridge)
PYTHONPATH=src python3 src/xauusd_ai/main.py live --config configs/live_acc1_scalp_m1.yaml
PYTHONPATH=src python3 src/xauusd_ai/main.py live --config configs/live_acc2_scalp_m1.yaml
```

> MLflow tự fallback sang `file:///tmp/mlruns` — không cần server.
> PostgreSQL tự fallback sang CSV — không cần DB.

---

## 9. Chạy Full Stack (optional)

Chỉ cần khi muốn đầy đủ observability: experiment tracking (MLflow), workflow scheduling (Airflow), dashboard monitoring (Grafana).

```bash
# 1. Infra
docker compose up -d postgres minio prometheus grafana api nginx healthwatch

# 2. Airflow DB (chạy 1 lần)
docker exec xauusd-postgres psql -U trader -d postgres -c "CREATE DATABASE airflow;" || true

# 3. Airflow
docker compose up -d airflow-init
sleep 15
docker compose up -d airflow-webserver airflow-scheduler

# 4. MLflow
docker compose up -d mlflow

# 5. Live bots
docker compose up -d live-acc1 live-scalp-acc2

# Health check
curl -sS http://localhost:8000/health     # API
curl -I http://localhost:9090/-/healthy    # Prometheus
curl -I http://localhost:3000/api/health   # Grafana
curl -I http://localhost:8080/health       # Airflow
curl -I http://localhost:5000              # MLflow
```

---

## 10. Train / Backtest / Walkforward (local, không Docker)

### Friction-optimized WF search (recommended)

Các script walk-forward tối ưu friction đã bao gồm session spread/slippage model:

```bash
# ACC1 friction WF (Exness Pro session mults built-in)
PYTHONPATH=src python3 scripts/acc1_friction_opt_wf.py \
  --config configs/live_acc1_scalp_m1.yaml \
  --n-folds 8 --out-prefix acc1_wf

# ACC2 friction WF (Exness Pro session mults built-in)
PYTHONPATH=src python3 scripts/acc2_friction_opt_wf.py \
  --config configs/live_acc2_scalp_m1.yaml \
  --n-folds 8 --out-prefix acc2_wf

# Silver Bullet A/B test
PYTHONPATH=src python3 scripts/silver_bullet_ab_test.py
PYTHONPATH=src python3 scripts/silver_bullet_ab_test.py --acc1-only
PYTHONPATH=src python3 scripts/silver_bullet_ab_test.py --acc2-only
```

### Standard WF (qua main.py)

```bash
PYTHONPATH=src python3 src/xauusd_ai/main.py walkforward --config configs/live_acc1_scalp_m1.yaml
PYTHONPATH=src python3 src/xauusd_ai/main.py walkforward --config configs/live_acc2_scalp_m1.yaml
```

### Docker shortcut

```bash
docker compose run --rm train           # Train ACC2
docker compose run --rm train-acc1      # Train ACC1
docker compose run --rm backtest        # Backtest ACC2
```

---

## 11. Live trade qua MT5 Bridge

macOS/Linux chạy bot trong Docker, MT5 terminal + bridge chạy trên Windows/VPS.

**Bridge ports:**
- ACC1: `http://host.docker.internal:5600`
- ACC2: `http://host.docker.internal:5601`

**Bot lấy:**
- Tick giá realtime từ bridge
- OHLCV bars đa khung từ bridge `/bars` endpoint
- Full TF stack: D1/H4/H1/M30/M15/M5/M1

```bash
# Start live bots
docker compose up -d live-acc1 live-scalp-acc2 healthwatch
docker compose logs -f live-acc1 live-scalp-acc2 healthwatch
```

> Nếu chưa có bridge, bots sẽ `unhealthy` (Connection refused) — đây là expected.
> Dashboard/API/backtest/WF vẫn hoạt động bình thường.

### Realtime Bar Appender (bắt buộc khi chạy live)

Script `scripts/realtime_bar_appender.py` fetch closed bars từ MT5 bridge và append vào CSV data, tự tính `tick_volume_delta` + `volume_imbalance` đúng cách (bảo đảm live features = WF features).

```bash
# Chạy thủ công 1 lần
PYTHONPATH=src python3 scripts/realtime_bar_appender.py

# Chạy loop liên tục (poll mỗi 60s)
RUN_LOOP=1 PYTHONPATH=src python3 scripts/realtime_bar_appender.py

# Crontab — khách hàng khuyến nghị cho production (chạy mỗi phút)
# Cài crontab: crontab -e
# * * * * * cd /path/to/Trade-Indicator && BRIDGE_HOST=localhost BRIDGE_PORT=5600 DATA_DIR=src/xauusd_ai/real_data PYTHONPATH=src python3 scripts/realtime_bar_appender.py >> /tmp/bar_appender.log 2>&1
```

**Biến ENV:**

| Var | Default | Mô tả |
|-----|---------|------|
| `BRIDGE_HOST` | `localhost` | Host của MT5 bridge |
| `BRIDGE_PORT` | `5600` | Port của MT5 bridge |
| `MT5_SYMBOL` | `XAUUSDm` | Symbol MT5 |
| `DATA_DIR` | `src/xauusd_ai/real_data` | Thư mục chứa CSV |
| `RUN_LOOP` | `0` | `1` = chạy loop vô hạn |

> CSV được ghi atomic (`.tmp_append` → rename) — không bao giờ corrupt nếu bị kill giữa chừng.
> Bridge lỗi: skip TF đó, log warning, không crash toàn bộ script.

### Public dashboard (optional)

```bash
docker compose up -d cloudflared
docker compose logs cloudflared 2>&1 | grep -i trycloudflare
```

---

## 12. Chạy trên Windows host

```bat
scripts\windows\run_acc1.bat
scripts\windows\run_acc2.bat
```

Scripts tự map `TELEGRAM_BOT_TOKEN_ACC1/ACC2`, `TELEGRAM_CHAT_ID_ACC1/ACC2`, `MT5_BRIDGE_URL`.

**MT5 bridge:**

```bat
python scripts\windows\mt5_bridge.py
```

---

## 13. Verify binding & health check

### Config load

```bash
PYTHONPATH=src python3 -c "
from pathlib import Path
from xauusd_ai.config import load_settings
s1 = load_settings(Path('configs/live_acc1_scalp_m1.yaml'))
s2 = load_settings(Path('configs/live_acc2_scalp_m1.yaml'))
print(f'ACC1: thr={s1.strategy.signal_threshold}, risk={s1.risk.risk_per_trade}, spread={s1.risk.spread_cost_rr}')
print(f'ACC2: thr={s2.strategy.signal_threshold}, risk={s2.risk.risk_per_trade}, spread={s2.risk.spread_cost_rr}')
print('OK')
"
```

**Kỳ vọng (Exness Pro, post-gap-fix 2026-04-13):**
```
ACC1: thr=0.59, risk=0.026, spread=0.035
ACC2: thr=0.60, risk=0.020, spread=0.035
```

### Model files

```bash
python3 -c "
import pickle, json, os
for f in ['outputs/acc1_scalp_m1_reversal_model.pkl', 'outputs/acc2_scalp_m1_h10_setup_exit_model.pkl']:
    sz = os.path.getsize(f)
    assert sz > 1000, f'{f} too small ({sz} bytes) — may be LFS pointer stub'
    print(f'{f}: {sz:,} bytes OK')
"
```

### WF binding reports

```bash
python3 -c "
import json
for f, label in [('outputs/acc1_scalp_m1_reversal_walkforward_report.json', 'ACC1'),
                  ('outputs/acc2_scalp_m1_walkforward_report.json', 'ACC2')]:
    r = json.load(open(f))
    a = r['aggregate']
    print(f'{label}: source={a[\"source\"]}, net=\${a[\"net_profit\"]:,.0f}, PF={a[\"global_profit_factor\"]:.3f}, folds={len(r[\"folds\"])}')
"
```

### Unit tests

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

**Kỳ vọng:** 92 tests passed.

### Dataset column integrity (verify G2 + G3 fix)

```bash
PYTHONPATH=src python3 -c "
import sys
from pathlib import Path
from xauusd_ai.config import load_settings
from xauusd_ai.features.scalp_dataset import build_scalp_dataset
cfg = load_settings(Path('configs/live_acc1_scalp_m1.yaml'))
df, _ = build_scalp_dataset(cfg)
errors = []
for col in ['atr', 'volatility_regime']:
    if col not in df.columns:
        errors.append(f'FAIL: {col} missing')
if errors:
    print('\n'.join(errors)); sys.exit(1)
regime = df.volatility_regime.value_counts().sort_index().to_dict()
print(f'atr: mean={df.atr.mean():.2f}, min={df.atr.min():.2f}')
print(f'regime distribution: {regime}  (phải có cả 0, 1, 2)')
print('OK: G2+G3 fix verified')
"
```

**Kỳ vọng:**
- `atr` mean ~10–30 (XAU/USD points), min > 0
- `regime distribution` có cả key 0, 1, 2 (không chỉ `{1: N}` — nếu chỉ có key 1 thì G3 chưa fix)

### Docker compose

```bash
docker compose config -q && echo "compose OK"
```

### Dừng hệ thống

```bash
docker compose down        # Dừng services
docker compose down -v     # Dừng + xóa volumes (CẢNH BÁO: mất DB data)
```

---

## 14. Checklist cho AI / người deploy

Sau khi clone và setup xong, chạy từng bước và đánh dấu:

- [ ] **C1.** `git checkout codex/pf-optimize-from-task4-clean` — đúng branch
- [ ] **C2.** `pip install -r requirements-core.txt && pip install -e .` — install thành công
- [ ] **C3.** Config load test — cả 2 config load không lỗi
- [ ] **C4.** Model .pkl > 1KB — không phải LFS pointer stub
- [ ] **C5.** Verify ACC1 params: `thr=0.59, risk=0.026, spread=0.035, slippage=0.05, silver_bullet=false`
- [ ] **C6.** Verify ACC2 params: `thr=0.60, risk=0.020, spread=0.035, slippage=0.05, silver_bullet=true`
- [ ] **C7.** WF re-run ACC1: net > $80k, PF > 1.7 (post-gap-fix benchmark)
- [ ] **C8.** WF re-run ACC2: net > $50k, PF > 1.7 (post-gap-fix benchmark)
- [ ] **C9.** `pytest tests/` — tất cả pass (expect 92 tests)
- [ ] **C10.** Market data CSV tồn tại: `src/xauusd_ai/real_data/XAUUSDm_M1.csv`
- [ ] **C11.** (Nếu Docker) `docker compose config -q` — không lỗi
- [ ] **C12.** (Nếu Docker) `docker compose up -d postgres api nginx` — services healthy
- [ ] **C13.** (Nếu Docker) `curl http://localhost:8000/health` — trả về OK
- [ ] **C14.** (Nếu backtest) Chạy backtest cả 2 config — không crash, có output
- [ ] **C15.** (Nếu live) MT5 bridge reachable — bots sẽ healthy
- [ ] **C16.** CSV M1 có đúng 9 cột: `time,open,high,low,close,tick_volume,spread_points,tick_volume_delta,volume_imbalance`
- [ ] **C17.** Dataset build trả về `atr` (mean > 0, không phải ATR5 proxy) và `volatility_regime` (có cả 0, 1, 2) — G2+G3 verified
- [ ] **C18.** (Nếu live) `realtime_bar_appender.py` chạy được hoặc crontab đã setup

---

## 15. Template báo cáo sau deploy

```
=== XAUUSD AI DEPLOYMENT REPORT ===
Date: YYYY-MM-DD
Machine: <OS, CPU, RAM>
Branch: codex/pf-optimize-from-task4-clean
Commit: <git rev-parse --short HEAD>

--- Checklist ---
C1  Branch:           [PASS/FAIL]
C2  Install:          [PASS/FAIL]
C3  Config load:      [PASS/FAIL]
C4  Model .pkl:       [PASS/FAIL] (ACC1: XX KB, ACC2: XX KB)
C5  ACC1 params:      [PASS/FAIL] (thr=X.XX, risk=X.XXX, spread=X.XXX)
C6  ACC2 params:      [PASS/FAIL] (thr=X.XX, risk=X.XXX, spread=X.XXX)
C7  WF report ACC1:   [PASS/FAIL] (source=XX, net=$XX)
C8  WF report ACC2:   [PASS/FAIL] (source=XX, net=$XX)
C9  Tests:            [PASS/FAIL] (XX/92 passed)
C10 Market data:      [PASS/FAIL] (M1 CSV: XX MB)
C11 Docker compose:   [PASS/FAIL/SKIP]
C12 Docker services:  [PASS/FAIL/SKIP]
C13 API health:       [PASS/FAIL/SKIP]
C14 Backtest:         [PASS/FAIL/SKIP]
C15 MT5 bridge:       [PASS/FAIL/SKIP]
C16 CSV 9 columns:    [PASS/FAIL] (header: time,open,...,tick_volume_delta,volume_imbalance)
C17 G2+G3 dataset:    [PASS/FAIL] (atr mean=XX, regime={0:XX,1:XX,2:XX})
C18 Bar appender:     [PASS/FAIL/SKIP]

--- Summary ---
Overall: [READY / NOT READY]
Issues: <nếu có>
Notes: <ghi chú thêm>
```

---

## Appendix: Cấu trúc thư mục quan trọng

```
configs/
  live_acc1_scalp_m1.yaml           # ACC1 config (Exness Pro)
  live_acc2_scalp_m1.yaml           # ACC2 config (Exness Pro)
outputs/
  acc1_scalp_m1_reversal_model.pkl  # ACC1 model
  acc1_scalp_m1_reversal_scaler.pkl
  acc2_scalp_m1_h10_setup_exit_model.pkl  # ACC2 model
  acc2_scalp_m1_h10_setup_exit_scaler.pkl
scripts/
  acc1_friction_opt_wf.py           # ACC1 WF search
  acc2_friction_opt_wf.py           # ACC2 WF search
  silver_bullet_ab_test.py          # Silver Bullet A/B test
  live_runner.py                    # Live bot launcher
  healthwatch.py                    # Telegram health monitor
  realtime_bar_appender.py          # Fetch MT5 bars → CSV (bắt buộc khi live, thiếu là data stale)
src/xauusd_ai/
  main.py                           # CLI: train/backtest/walkforward/live/paper
  config.py                         # YAML config loader
  orchestrator.py                   # Core orchestration
  backtesting/engine.py             # WF + backtest engine
  strategies/hybrid.py              # Trading strategy
  model/scalp_model.py              # ML model
  model/scalp_runtime.py            # Runtime inference
  features/scalp_dataset.py         # WF dataset builder (ATR14 + regime fixed)
dags/
  scalp_model_retrain.py            # Weekly Airflow retrain DAG (Sun 03:00 UTC)
tests/                              # 92 unit tests
```

---

## 16. Changelog

### 2026-04-13 — Post-Gap-Fix Revalidation

**Bối cảnh:** Live results tuần 14/4: ACC1 1W/8L (−$17), ACC2 1W/8L (−$15). Audit toàn bộ WF vs live code path → tìm ra 3 GAP thực sự.

#### G1 — Model training objective (CRITICAL)
- **Vấn đề:** `acc2_scalp_m1_save_model.py` dùng `baseline` weighting khi train model live, trong khi WF train bằng `day_stability_strict` (penalty giờ xấu, thứ Hai, vol spike).
- **Fix:** Thêm `day_stability_strict` vào `_train_dir()` của save_model script — giờ khớp với WF.

#### G2 — ATR computation mismatch (HIGH)
- **Vấn đề:** WF dataset không có cột `atr` → engine fallback sang `ms_atr5_norm × close / 1000` (ATR5). Live dùng `ATR(14)` thực tế.
- **Fix:** Thêm `atr = ATR(14)` vào `build_scalp_dataset()` trong `scalp_dataset.py`.

#### G3 — volatility_regime hardcoded (MEDIUM)
- **Vấn đề:** WF dataset mặc định `volatility_regime = 1` (normal) cho tất cả bars. Live tính `infer_scalp_volatility_regime()` → real 0/1/2.
- **Fix:** Thêm `volatility_regime = infer_scalp_volatility_regime(m1).astype(int)` vào `build_scalp_dataset()`.

#### Slippage correction
- **Vấn đề:** `slippage_rr: 0.01` quá lạc quan. Thực đo M1 execution: ~0.4 pts drift.
- **Fix:** `slippage_rr: 0.01 → 0.05` trong cả 2 config.

#### ACC2 risk re-optimization
- WF post-fix tối ưu: `risk_per_trade: 0.022 → 0.020` (ATR/regime mới strict hơn → giảm sizing).
- `max_risk_fraction: 0.054 → 0.048` (duy trì tỉ lệ 1.8×).

#### Models retrained
- Cả ACC1 và ACC2 retrained với `day_stability_strict` + `train_end=2026-04-13`.
- Mean proba: 0.401, std: 0.183, ≥0.55: 24%.

#### WF re-run results (post-fix)
| Account | Net (8-fold) | PF | Fold 8 | Feasible |
|---------|-------------|-----|--------|----------|
| ACC1 | $85,327 | 1.761 | +$18,103 | 7/120 |
| ACC2 | $57,356 | 1.773 | +$14,725 | 1/120 |

#### Weekly retrain DAG
- Tạo mới `dags/scalp_model_retrain.py` — chạy tự động mỗi Chủ nhật 03:00 UTC qua Airflow.
