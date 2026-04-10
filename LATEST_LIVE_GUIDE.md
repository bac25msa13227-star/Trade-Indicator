# Latest Live Guide

Last updated: `2026-04-11`

Đây là file hướng dẫn duy nhất cần đọc để chạy đúng project trên máy khác.

Quan trọng:
- Repo này **không còn dùng Streamlit**.
- Dashboard realtime chuẩn là **dashboard WebSocket của FastAPI** tại:
  - `http://localhost/dashboard`
  - hoặc `http://localhost:8000/dashboard`
- Hai profile đang dùng:
  - `ACC1 = scalp M1 reversal`
  - `ACC2 = scalp M1 freeze H10 setup-exit`

---

## 1. Bộ config/model đang khóa

### ACC1 live
- Config: `configs/live_acc1_scalp_m1.yaml`
- Model: `outputs/acc1_scalp_m1_reversal_model.pkl`
- Scaler: `outputs/acc1_scalp_m1_reversal_scaler.pkl`
- Meta: `outputs/acc1_scalp_m1_reversal_model_meta.json`
- Live bars/tick source: `MT5 bridge`
- Vai trò: nhiều lệnh hơn, bắt đảo chiều tốt hơn ACC2

### ACC2 live
- Config: `configs/live_acc2_scalp_m1.yaml`
- Model: `outputs/acc2_scalp_m1_h10_setup_exit_model.pkl`
- Scaler: `outputs/acc2_scalp_m1_h10_setup_exit_scaler.pkl`
- Meta: `outputs/acc2_scalp_m1_h10_setup_exit_model_meta.json`
- Live bars/tick source: `MT5 bridge`
- Vai trò: profile freeze, guarded, không sửa tùy tiện

### Service live đúng
- ACC1: `docker compose up -d live-acc1`
- ACC2: `docker compose up -d live-scalp-acc2`

Không dùng:
- `configs/live_acc1.yaml`
- `configs/live_acc2.yaml`
- `configs/live_acc2_scalp.yaml`
- Streamlit / `frontend`

---

## 2. Preflight máy mới

```bash
git clone https://github.com/bac25msa13227-star/Trade-Indicator.git
cd Trade-Indicator
git checkout codex/pf-optimize-from-task4-clean
docker compose config -q
```

Kiểm tra data:

```bash
ls -lh src/xauusd_ai/real_data
```

Tối thiểu phải có bộ CSV XAUUSD/XAUUSDm từ `D1` xuống `M1` trong `src/xauusd_ai/real_data/`.

---

## 3. File `.env` tối thiểu

```env
POSTGRES_USER=trader
POSTGRES_PASSWORD=trader_secret
POSTGRES_DB=tradedb

MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

TELEGRAM_BOT_TOKEN_ACC1=<token_acc1>
TELEGRAM_CHAT_ID_ACC1=<chat_id_acc1>
TELEGRAM_BOT_TOKEN_ACC2=<token_acc2>
TELEGRAM_CHAT_ID_ACC2=<chat_id_acc2>
```

Ghi chú:
- `live-acc1` tự map `ACC1` token/chat vào `TELEGRAM_BOT_TOKEN` và `TELEGRAM_CHAT_ID`.
- `live-scalp-acc2` tự map `ACC2` token/chat vào `TELEGRAM_BOT_TOKEN` và `TELEGRAM_CHAT_ID`.
- Vì vậy cứ điền `*_ACC1` và `*_ACC2` trong `.env` là đủ.

---

## 4. Chạy LITE

Phù hợp cho Macbook hoặc máy chỉ cần dashboard/API/WF/backtest, không cần full observability stack.

```bash
docker compose up -d postgres api nginx
```

Health check:

```bash
curl -sS http://localhost:8000/health
curl -I http://localhost/dashboard
docker compose ps
```

Dashboard:
- `http://localhost/dashboard`
- `http://localhost:8000/dashboard`

---

## 5. Chạy FULL STACK

Khi cần cả `MLflow + Airflow + Grafana + MinIO + Prometheus`.

### Bước 1: bật nền

```bash
docker compose up -d postgres minio prometheus grafana api nginx healthwatch
```

### Bước 2: tạo DB Airflow một lần

```bash
docker exec xauusd-postgres psql -U trader -d postgres -c "CREATE DATABASE airflow;" || true
```

### Bước 3: bật Airflow

```bash
docker compose up -d airflow-init
docker compose up -d airflow-webserver airflow-scheduler
```

### Bước 4: bật MLflow nếu cần

```bash
docker compose up -d mlflow
```

Health check full stack:

```bash
curl -sS http://localhost:8000/health
curl -I http://localhost/dashboard
curl -I http://localhost:9090/-/healthy
curl -I http://localhost:3000/api/health
curl -I http://localhost:8080/health
docker compose ps
```

---

## 6. Live trade thật qua MT5 Bridge

Project này trên macOS không auto trade trực tiếp bằng MT5 package. Cách chạy chuẩn là:
- MT5 terminal + bridge chạy trên Windows/VPS
- Repo + bot + dashboard có thể chạy ở máy Docker hiện tại

Bridge mặc định:
- ACC1: `http://host.docker.internal:5600`
- ACC2: `http://host.docker.internal:5601`

Hiện tại bot live lấy:
- tick giá hiện tại từ bridge
- OHLCV bars đa khung từ bridge qua endpoint `/bars`
- full TF stack theo config: `D1/H4/H1/M30/M15/M5/M1`
- không dùng `yfinance` cho live nữa

Chạy live:

```bash
docker compose up -d live-acc1 live-scalp-acc2 healthwatch
docker compose logs -f live-acc1 live-scalp-acc2 healthwatch
```

Nếu chưa có bridge:
- `live-acc1` và `live-scalp-acc2` có thể `unhealthy` do `Connection refused`
- đó là expected
- khi đó vẫn dùng được `API`, `dashboard`, `backtest`, `walkforward`

Nếu vừa pull code mới sang máy Windows đang chạy bridge:
- restart lại `scripts/windows/mt5_bridge.py`
- để bridge nhận endpoint `/bars` mới nhất

---

## 7. Public dashboard

Nếu muốn public dashboard realtime:

```bash
docker compose up -d cloudflared
docker compose logs cloudflared 2>&1 | grep -i trycloudflare
```

Dashboard public sẽ proxy vào nginx/FastAPI WebSocket dashboard, không cần Streamlit.

---

## 8. Backtest / Walkforward / Train không cần MLflow server

Nếu không bật service `mlflow`, vẫn chạy được bằng local file store:

```bash
MLFLOW_TRACKING_URI=file:///tmp/mlruns PYTHONPATH=src python3 src/xauusd_ai/main.py train --config configs/live_acc1_scalp_m1.yaml
MLFLOW_TRACKING_URI=file:///tmp/mlruns PYTHONPATH=src python3 src/xauusd_ai/main.py train --config configs/live_acc2_scalp_m1.yaml

MLFLOW_TRACKING_URI=file:///tmp/mlruns PYTHONPATH=src python3 src/xauusd_ai/main.py backtest --config configs/live_acc1_scalp_m1.yaml
MLFLOW_TRACKING_URI=file:///tmp/mlruns PYTHONPATH=src python3 src/xauusd_ai/main.py backtest --config configs/live_acc2_scalp_m1.yaml

MLFLOW_TRACKING_URI=file:///tmp/mlruns PYTHONPATH=src python3 src/xauusd_ai/main.py walkforward --config configs/live_acc1_scalp_m1.yaml
MLFLOW_TRACKING_URI=file:///tmp/mlruns PYTHONPATH=src python3 src/xauusd_ai/main.py walkforward --config configs/live_acc2_scalp_m1.yaml
```

Docker shortcut:

```bash
docker compose run --rm train
docker compose run --rm train-acc1
docker compose run --rm backtest
```

Mặc định hiện tại:
- `train` = ACC2 freeze scalp M1
- `train-acc1` = ACC1 scalp M1 reversal
- `backtest` = ACC2 freeze scalp M1

---

## 9. Verify đúng binding sau khi pull

### Verify model/config runtime qua API

```bash
curl -s http://localhost:8000/api/v1/dashboard
```

Kỳ vọng:
- ACC1 model là `outputs/acc1_scalp_m1_reversal_model.pkl`
- ACC2 model là `outputs/acc2_scalp_m1_h10_setup_exit_model.pkl`

Ví dụ kiểm tra nhanh đúng artifact:

```bash
python3 -c "import json, urllib.request; obj=json.load(urllib.request.urlopen('http://localhost:8000/api/v1/dashboard')); print(obj['accounts']['acc1']['model']['binding_actual']); print(obj['accounts']['acc2']['model']['binding_actual'])"
```

### Verify config YAML load được

```bash
PYTHONPATH=src python3 -c "from pathlib import Path; from xauusd_ai.config import load_settings; load_settings(Path('configs/live_acc1_scalp_m1.yaml')); load_settings(Path('configs/live_acc2_scalp_m1.yaml')); print('settings ok')"
```

### Verify compose

```bash
docker compose config -q
```

---

## 10. Chạy trực tiếp trên Windows host

Nếu chạy bot trực tiếp trên Windows host thay vì Docker:

### ACC1
```bat
scripts\windows\run_acc1.bat
```

### ACC2
```bat
scripts\windows\run_acc2.bat
```

Hai script này tự map:
- `TELEGRAM_BOT_TOKEN_ACC1/ACC2`
- `TELEGRAM_CHAT_ID_ACC1/ACC2`
- `MT5_BRIDGE_URL`

---

## 11. Dừng hệ thống

```bash
docker compose down
```

Xóa volume local:

```bash
docker compose down -v
```

Cẩn thận vì lệnh này sẽ xóa DB/object storage local.

---

## 12. Kết luận ngắn

Nếu chỉ muốn chạy đúng, ít rủi ro nhất:

```bash
docker compose up -d postgres api nginx
docker compose up -d live-acc1 live-scalp-acc2 healthwatch
```

Rồi mở:
- `http://localhost/dashboard`

Và nhớ:
- `ACC1` dùng `live_acc1_scalp_m1.yaml`
- `ACC2` dùng `live_acc2_scalp_m1.yaml`
- Không dùng Streamlit
- Không dùng config/model cũ
