# Latest Live Guide

Last updated: `2026-04-09`  
Mục tiêu: chạy dự án trên máy khác theo 2 chế độ:
- Có đủ `MLflow + Airflow + Grafana + MinIO + Prometheus` (full stack).
- Không dùng các thành phần trên (lite stack), vẫn chạy API/Dashboard/WF/Backtest bình thường.

---

## 1) Cấu hình đang khóa

### ACC1 (M5)
- Config: `configs/live_acc1.yaml`
- Model: `outputs/acc1_live_primary_model.pkl`
- Scaler: `outputs/acc1_live_primary_scaler.pkl`
- Meta: `outputs/acc1_live_primary_model_meta.json`

### ACC2 (M5)
- Config: `configs/live_acc2.yaml`
- Model: `outputs/acc2_live_primary_model.pkl`
- Scaler: `outputs/acc2_live_primary_scaler.pkl`
- Meta: `outputs/acc2_live_primary_model_meta.json`

### ACC2 Scalp M1
- Config: `configs/live_acc2_scalp_m1.yaml`
- Model: `outputs/acc2_scalp_m1_model.pkl`
- Scaler: `outputs/acc2_scalp_m1_scaler.pkl`
- Meta: `outputs/acc2_scalp_m1_model_meta.json`

---

## 2) Preflight máy mới

```bash
git clone https://github.com/bac25msa13227-star/Trade-Indicator.git
cd Trade-Indicator
git checkout codex/pf-optimize-from-task4-clean
docker compose config -q
```

### `.env` tối thiểu

```env
POSTGRES_USER=trader
POSTGRES_PASSWORD=trader_secret
POSTGRES_DB=tradedb

MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

TELEGRAM_BOT_TOKEN_ACC1=<your_token_acc1>
TELEGRAM_CHAT_ID_ACC1=<your_chat_id_acc1>
TELEGRAM_BOT_TOKEN_ACC2=<your_token_acc2>
TELEGRAM_CHAT_ID_ACC2=<your_chat_id_acc2>
```

Nếu chạy live thật qua bridge Windows thì thêm:

```env
MT5_LOGIN=<your_login>
MT5_PASSWORD=<your_password>
MT5_SERVER=<your_server>
```

### Data CSV bắt buộc

Kiểm tra thư mục dữ liệu:

```bash
ls -lh src/xauusd_ai/real_data
```

Nếu máy mới thiếu `XAUUSDm_M1.csv`, `XAUUSDm_M5.csv`, `XAUUSDm_H1.csv`, `XAUUSDm_H4.csv`, `XAUUSDm_D1.csv` thì copy từ máy gốc hoặc tải lại.

---

## 3) Chạy LITE (không MLflow/Airflow/Grafana/MinIO/Prometheus)

Phù hợp cho Macbook local test WF/backtest/dashboard.

```bash
docker compose up -d postgres api frontend nginx
```

Health checks:

```bash
curl -sS http://localhost:8000/health
curl -I http://localhost:8501
curl -I http://localhost
docker compose ps
```

### Chạy WF/Backtest không cần MLflow service

```bash
MLFLOW_TRACKING_URI=file:///tmp/mlruns PYTHONPATH=src python3 src/xauusd_ai/main.py walkforward --config configs/live_acc1.yaml
MLFLOW_TRACKING_URI=file:///tmp/mlruns PYTHONPATH=src python3 src/xauusd_ai/main.py walkforward --config configs/live_acc2.yaml

MLFLOW_TRACKING_URI=file:///tmp/mlruns PYTHONPATH=src python3 src/xauusd_ai/main.py backtest --config configs/live_acc1.yaml
MLFLOW_TRACKING_URI=file:///tmp/mlruns PYTHONPATH=src python3 src/xauusd_ai/main.py backtest --config configs/live_acc2.yaml
```

---

## 4) Chạy FULL STACK (có MLflow/Airflow/Grafana/MinIO/Prometheus)

### Bước 1: bật nền tảng

```bash
docker compose up -d postgres minio prometheus grafana api frontend nginx healthwatch
```

### Bước 2: tạo DB Airflow (chỉ cần 1 lần)

```bash
docker exec xauusd-postgres psql -U trader -d postgres -c "CREATE DATABASE airflow;" || true
```

### Bước 3: bật Airflow

```bash
docker compose up -d airflow-init
docker compose up -d airflow-webserver airflow-scheduler
```

### Bước 4: bật MLflow (nếu port 5000 trống)

```bash
docker compose up -d mlflow
```

Nếu báo `bind: address already in use` ở port 5000:

```bash
lsof -nP -iTCP:5000 -sTCP:LISTEN
# Tắt process đang chiếm port hoặc đổi map port trong docker-compose.yml
```

### Health checks full stack

```bash
curl -sS http://localhost:8000/health
curl -I http://localhost:8501
curl -I http://localhost:9090/-/healthy
curl -I http://localhost:3000/api/health
curl -I http://localhost:8080/health
docker compose ps
```

---

## 5) Live trade thật (chỉ khi có MT5 Bridge Windows)

Bridge mặc định:
- ACC1: `http://host.docker.internal:5600`
- ACC2: `http://host.docker.internal:5601`

Start bot:

```bash
docker compose up -d live live-acc1 live-scalp-acc2
docker compose logs -f live live-acc1 live-scalp-acc2
```

Nếu không có bridge (ví dụ chạy thuần trên Mac), các service live sẽ báo `unhealthy` với `Connection refused` là expected.  
Trường hợp đó chỉ chạy mode LITE/FULL infra và dùng WF/backtest/paper.

---

## 6) Verify nhanh sau khi pull máy khác

```bash
PYTHONPATH=src python3 -m pytest -q
```

Kỳ vọng:
- Test pass toàn bộ.
- API/Dashboard lên được.
- WF/backtest chạy hết không crash.

---

## 7) Các file output quan trọng để đối chiếu

- WF ACC1: `outputs/acc1_pf3v2_net815k_dd2891_walkforward_report.json`
- WF ACC2: `outputs/acc2_pf3v2_net63k_dd1864_walkforward_report.json`
- Backtest report mới nhất: `outputs/backtest_report.json`

---

## 8) Dừng hệ thống

```bash
docker compose down
```

Dừng và xóa volume (cẩn thận vì mất DB/object storage local):

```bash
docker compose down -v
```
