# Latest Live Guide

Last updated: `2026-04-01`

## 1) File config nào dùng để chạy thật

Live configs (chuẩn production):

- `configs/live_acc1.yaml`
- `configs/live_acc2.yaml`

Model/scaler live tương ứng:

- ACC1:
  - `outputs/acc1_live_model.pkl`
  - `outputs/acc1_live_scaler.pkl`
  - `outputs/acc1_live_model_meta.json`
- ACC2:
  - `outputs/acc2_live_model.pkl`
  - `outputs/acc2_live_scaler.pkl`
  - `outputs/acc2_live_model_meta.json`

Lưu ý:

- `threshold` runtime đọc từ `*_model_meta.json` khi load artifact.
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

## 4) Cách chạy LITE (không MLflow/Grafana/Airflow)

Mục tiêu: chỉ cần bot chạy + Telegram chạy + dashboard realtime chạy.

### 4.1 Docker lite

```bash
docker compose up -d postgres api nginx healthwatch live live-acc1
```

Trong mode này:

- Không cần bật `mlflow`, `grafana`, `prometheus`, `airflow-*`.
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
