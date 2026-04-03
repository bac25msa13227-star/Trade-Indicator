# Latest Live Guide

Last updated: `2026-04-03` (after cleanup commit `0894ba7` on branch `model/net66kdd39_net21dd23`)

## Locked benchmark profiles (must-use)

- ACC1 benchmark: `Net 66,590.56 | DD 39.53% | PF 1.4809 | Trades 1054 | WR 59.01%`
- ACC2 benchmark: `Net 21,057.65 | DD 23.33% | PF 2.1906 | Trades 569 | WR 64.50%`

Required live configs:

- `configs/live_acc1.yaml`
- `configs/live_acc2.yaml`

Required model artifacts:

- `outputs/acc1_breakthrough_net66590_dd3953_model.pkl`
- `outputs/acc1_breakthrough_net66590_dd3953_scaler.pkl`
- `outputs/acc1_breakthrough_net66590_dd3953_model_meta.json`
- `outputs/acc2_breakthrough_net21k_dd2333_model.pkl`
- `outputs/acc2_breakthrough_net21k_dd2333_scaler.pkl`
- `outputs/acc2_breakthrough_net21k_dd2333_model_meta.json`

Quick threshold snapshot:

- ACC1: `strategy.signal_threshold = 0.62` (and model meta threshold = `0.62`)
- ACC2: `strategy.signal_threshold = 0.76` (and model meta threshold = `0.72`)

## Data source status (kept locally)

- `src/xauusd_ai/real_data`

## Locked WF verification (khớp 100%)

Để tái tạo đúng 2 benchmark:

- ACC1: `66,590.56 | DD 39.53% | PF 1.4809 | 1054 | WR 59.01%`
- ACC2: `21,057.65 | DD 23.33% | PF 2.1906 | 569 | WR 64.50%`

Dùng đúng manifest + script:

- Manifest: `configs/benchmarks/wf_locked_breakthrough_20260401.json`
- Verify script: `scripts/wf_verify_locked_benchmarks.py`

Chạy verify:

```bash
PYTHONPATH=src python3 scripts/wf_verify_locked_benchmarks.py
```

Report sẽ ghi ra:

- `outputs/wf_locked_verify_report.json`

Quan trọng:

- Bộ benchmark này chạy theo đường WF optimizer (train theo từng fold), không đọc trực tiếp live model pkl.
- Runtime benchmark lock ở feature-set hiện tại (`FEATURE_COLUMNS=59`).
- Nếu thay đổi feature engineering (ví dụ thêm 25 features mới), benchmark cũ sẽ không còn khớp 100%.

## Market open/close gate (mới)

- Bot gọi trạng thái sàn từ MT5 bridge (`/market/state`) mỗi vòng lặp.
- Khi sàn đóng: bot tự block lệnh mới (`reason=MARKET_CLOSED:*`) nên không còn spam tín hiệu vào lệnh lúc market closed.
- Có cảnh báo Telegram:
  - chuyển trạng thái mở ↔ đóng,
  - trước mở cửa `1 ngày` và `30 phút`,
  - trước đóng cửa `1 ngày` và `30 phút`.
- Thông số trong `market`:
  - `enforce_market_open_gate`
  - `market_tick_stale_seconds`
  - `market_preopen_alert_minutes_list` (ví dụ `[1440, 30]`)
  - `market_preclose_alert_minutes_list` (ví dụ `[1440, 30]`)

## Cách chạy FULL stack (có MLflow/Grafana/Airflow)

Core multi-timeframe data still available from `2003-05-05` to `2026-03-30`:

- `XAUUSDm_D1.csv`
- `XAUUSDm_H4.csv`
- `XAUUSDm_H1.csv`
- `XAUUSDm_M30.csv`
- `XAUUSDm_M15.csv`
- `XAUUSDm_M5.csv`
- `XAUUSDm_M1.csv`

## Run without MLflow / MinIO / Airflow / Grafana

Use this mode for stable live + Telegram + realtime dashboard only.

Start live bots:

```bash
PYTHONPATH=src python3 scripts/live_runner.py live --config configs/live_acc1.yaml
PYTHONPATH=src python3 scripts/live_runner.py live --config configs/live_acc2.yaml
```

Start API/dashboard websocket:

```bash
PYTHONPATH=src python3 -m uvicorn xauusd_ai.api.main:app --host 0.0.0.0 --port 8000
```

Open dashboard:

- `http://localhost:8000/dashboard`

## Verify runtime is loading the correct models

Check config binding:

```bash
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
from xauusd_ai.config import load_settings
for cfg in ["configs/live_acc1.yaml", "configs/live_acc2.yaml"]:
    s = load_settings(Path(cfg))
    print(cfg, "->", s.app.model_path, "|", s.app.model_meta_path)
PY
```

Check runtime status files:

```bash
cat outputs/live_status_acc1.json
cat outputs/live_status_acc2.json
```

These JSON files should show:

- `model_path`
- `model_meta_path`
- `model_decision_threshold`
- `benchmark_profile`

## Important cleanup note

This branch intentionally removed old experiment configs/models to avoid confusion.
If you need MLflow/Grafana/Airflow full stack again, restore those infra config files from Git history before running those services.
