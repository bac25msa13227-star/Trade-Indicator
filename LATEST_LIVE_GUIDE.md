# Latest Live Guide

Last updated: `2026-04-03 (Recovered exact benchmark baseline on commit 2016501)`

## 0) Baseline đã khôi phục (xác thực chính xác)

- Commit baseline: `20165010aca81a6b75a5ceb3265cff0586ad5131`
- Config replay chuẩn: `configs/snapshots/acc2_phase2_dd19_compat_20260401.yaml`
- File xác thực replay exact:
  - `outputs/wf_exact_recovery_verify_2016501.json`
- Kết quả đã khớp exact:
  - ACC1: `Net 66,590.56 | DD 39.53% | PF 1.4809 | Trades 1054 | WR 59.01%`
  - ACC2: `Net 21,057.65 | DD 23.33% | PF 2.1906 | Trades 569 | WR 64.50%`

## 1) File config nào dùng để chạy thật

Live configs (chuẩn production):

- `configs/live_acc1.yaml`
- `configs/live_acc2.yaml`

Model/scaler live tương ứng:

- ACC1:
  - Profile: `breakthrough max-net` (`Net = $66,590.56`, `DD = 39.53%`, `PF = 1.4809`)
  - Config freeze: `configs/snapshots/acc1_breakthrough_net66590_dd3953_20260401.yaml`
  - Live config đang apply: `configs/live_acc1.yaml`
  - `outputs/acc1_breakthrough_net66590_dd3953_model.pkl`
  - `outputs/acc1_breakthrough_net66590_dd3953_scaler.pkl`
  - `outputs/acc1_breakthrough_net66590_dd3953_model_meta.json`
  - WF evidence: `outputs/wf_breakthrough_dd25_acc2_run1_20260401_222454.csv` (row top-1 `net=66590.56`, `dd=39.53`, `pf=1.4809`)
- ACC2:
  - Profile: `breakthrough dd25` (`Net = $21,057.65`, `DD = 23.33%`, `PF = 2.1906`, `569 trades`, `WR = 64.5%`)
  - Config freeze: `configs/snapshots/acc2_breakthrough_net21k_dd2333_20260401.yaml`
  - Live config đang apply: `configs/live_acc2.yaml`
  - `outputs/acc2_breakthrough_net21k_dd2333_model.pkl`
  - `outputs/acc2_breakthrough_net21k_dd2333_scaler.pkl`
  - `outputs/acc2_breakthrough_net21k_dd2333_model_meta.json`
  - WF evidence: `outputs/wf_breakthrough_dd25_acc2_run1_20260401_222454.json`
  - WF full candidates: `outputs/wf_breakthrough_dd25_acc2_run1_20260401_222454.csv` (`32` candidates có `net > 21k`)

Lưu ý:

- `threshold` runtime đọc từ `*_model_meta.json` khi load artifact.
- ACC1 hiện chạy artifact `acc1_breakthrough_net66590_dd3953_*`.
- ACC2 hiện chạy artifact `acc2_breakthrough_net21k_dd2333_*`.
- Với ACC1 breakthrough, `decision_threshold` trong model meta đã set `0.62` để khớp profile WF.
- Với ACC2 breakthrough:
  - `strategy.signal_threshold = 0.76` (router gate trong config)
  - `model_meta.decision_threshold = 0.72` (ngưỡng predict của model artifact)
- `acc2_breakthrough_net21k_dd2333_*` là alias freeze để tránh nhầm lẫn giữa các profile.
- `retrain_on_startup: false` để bot vào lệnh ngay bằng model freeze.
- Self-learning vẫn chạy nền, chỉ hot-reload khi candidate đạt điều kiện acceptance.

## 2) News trade + realtime dashboard (đã áp dụng)

- Dashboard websocket:
  - URL API trực tiếp: `/dashboard` (kèm WS `/ws/live`).
  - Đã thêm cột `Entry/SL/TP`, `Outcome`, và chuẩn hóa cột `Time` robust hơn.
- News trade:
  - Alert Telegram/API hiện lọc `Medium + High` (không alert `Low`).
  - Trong live loop, khi `news_trade_override=true` thì filter tin dùng `Medium + High`.

## 3) Cách chạy FULL stack (có MLflow/Grafana/Airflow)

### 3.1 Start stack

```bash
docker compose up -d \
  postgres minio mlflow prometheus grafana \
  airflow-init airflow-webserver airflow-scheduler \
  api nginx healthwatch live live-acc1
```

Tuỳ chọn frontend Streamlit:

```bash
docker compose up -d frontend
```

### 3.2 URL kiểm tra

- API docs: `http://localhost:8000/docs`
- Realtime dashboard (WS): `http://localhost/dashboard`
- MLflow: `http://localhost:5000`
- Grafana: `http://localhost:3000`
- Airflow: `http://localhost:8080`

## 4) Cách chạy LITE (không MLflow/MinIO/Airflow/Grafana)

Mục tiêu: chỉ cần bot chạy + Telegram chạy + dashboard realtime chạy.

### 4.1 Docker lite

```bash
docker compose up -d postgres api nginx healthwatch live live-acc1
```

Trong mode này:

- Không cần bật `mlflow`, `minio`, `grafana`, `prometheus`, `airflow-*`.
- Dashboard realtime vẫn chạy qua API + websocket.
- Telegram vẫn chạy theo config/env.

### 4.2 Chạy local không docker (tuỳ chọn)

```bash
PYTHONPATH=src python3 scripts/live_runner.py live --config configs/live_acc1.yaml
PYTHONPATH=src python3 scripts/live_runner.py live --config configs/live_acc2.yaml
PYTHONPATH=src python3 -m uvicorn xauusd_ai.api.main:app --host 0.0.0.0 --port 8000
```

Mở dashboard:

- `http://localhost:8000/dashboard`

## 5) Checklist trước khi bật live

1. MT5 bridge đúng cổng account:
   - ACC1: `MT5_BRIDGE_URL=http://host.docker.internal:5600`
   - ACC2: `MT5_BRIDGE_URL=http://host.docker.internal:5601`
2. Telegram env đã set đúng token/chat_id theo account.
3. `configs/live_acc1.yaml` và `configs/live_acc2.yaml` đúng risk/threshold mong muốn.
4. Kiểm tra health:
   - `outputs/live_status_acc1.json`
   - `outputs/live_status_acc2.json`
5. Kiểm tra test:
   - `PYTHONPATH=src pytest -q tests`

## 5.1) Verify bot đang dùng đúng model nào (rất quan trọng)

Kiểm tra config live:

```bash
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
from xauusd_ai.config import load_settings
for cfg in ["configs/live_acc1.yaml", "configs/live_acc2.yaml"]:
    s = load_settings(Path(cfg))
    print(cfg, "->", s.app.model_path, "|", s.app.model_meta_path)
PY
```

Kiểm tra runtime status (sau khi bot chạy):

```bash
cat outputs/live_status_acc1.json
cat outputs/live_status_acc2.json
```

Trong JSON phải thấy các field:
- `model_path`
- `model_meta_path`
- `model_decision_threshold`

## 6) Unit test status (hiện tại)

Đã pass:

- `PYTHONPATH=src pytest -q tests`
- `56 passed`

Ghi chú:

- `scripts/test_mlflow_integration.py` là integration test, cần dịch vụ MLflow thật để chạy.

## 7) Chính sách dữ liệu Dukascopy CSV

Các file `.csv` lấy từ Dukascopy chỉ giữ local để train/backtest, không đưa lên Git remote.

- Local machine: giữ nguyên file dữ liệu để chạy.
- Git remote: không commit/push các CSV dữ liệu này.

Khi đồng bộ máy khác, dùng script fetch dữ liệu để kéo lại local data rồi train/run.
