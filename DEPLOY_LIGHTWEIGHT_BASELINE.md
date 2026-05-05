# Deploy Baseline Frozen Tren May Yeu (Khong MLflow/Airflow/Grafana/MinIO)

Ap dung cho Baseline Frozen Combo133.
Muc tieu: chay live bot on dinh voi tai nguyen thap, khong dung stack monitoring/tracking nang.

## 1) Co can train lai model khong?

- Khong can train lai neu ban da co dung artifact model frozen.
- Chi train lai khi:
  - Khong tai duoc model that tu Git LFS (chi co file pointer nho), hoac
  - Ban muon cap nhat model moi co chu dich.

## 2) Clone va lay model frozen

```bash
git clone https://github.com/bac25msa13227-star/Trade-Indicator.git
cd Trade-Indicator
git checkout feature/dashboard-controls-v14pp
git pull

git lfs install
git lfs pull
```

Kiem tra model that (khong phai pointer):

```bash
ls -lh outputs/acc1_combo133_202604_model.pkl
ls -lh outputs/acc1_combo133_202604_scaler.pkl
ls -lh outputs/acc1_combo133_202604_meta.json
```

Model .pkl phai co kich thuoc lon (hang chuc MB), khong phai ~100 bytes.

## 3) Cac file config live can dung

- ACC1: `configs/live_acc1.yaml`
- ACC2: `configs/live_acc2.yaml`

Hai file nay da duoc dong bo ve baseline core params:
- min_confidence: 0.70
- require_trend_alignment: false
- min_strategy_score: 0.00
- blocked_hours_utc: [3, 15, 17, 22, 23]
- d1_trend_gate: false

## 4) Chay toi gian bang Docker (khong MLflow/Airflow/Grafana/MinIO)

Chi chay cac service can thiet:

```bash
docker compose up -d postgres api live-acc1 live
```

Khong chay cac service nang:
- mlflow
- airflow-webserver
- airflow-scheduler
- grafana
- prometheus
- minio

## 5) Chay MT5 bridge (2 terminal rieng)

ACC1:

```bash
python scripts/windows/mt5_bridge.py --port 5600
```

ACC2:

```bash
python scripts/windows/mt5_bridge.py --port 5601
```

## 6) Kiem tra sau khi len may moi

```bash
docker compose ps

docker logs trade-indicator-live-acc1-1 --tail 50
docker logs trade-indicator-live-1 --tail 50
```

Dau hieu on:
- Khong bao loi load model/scaler
- Co log signal/skip/trade theo gio giao dich
- Khong bi kill switch ngay luc khoi dong

## 7) Neu may rat yeu

- Chi chay 1 account truoc (uu tien ACC1 baseline):

```bash
docker compose up -d postgres api live-acc1
```

- Sau khi on dinh moi mo them account thu 2.

## 8) Khuyen nghi dong bang runtime

- Khong sua config tren may deploy trong khi bot dang chay.
- Moi thay doi config/model phai:
  1. commit tren git
  2. pull + restart container
  3. verify log

## 9) Restart de nap config moi

```bash
docker restart trade-indicator-live-acc1-1
docker restart trade-indicator-live-1
```

Neu ten container khac, dung `docker ps` de lay dung ten.
