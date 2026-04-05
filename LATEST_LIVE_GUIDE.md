# Latest Live Guide

Last updated: `2026-04-05` — **WF PF3 v2 locked** (target metrics verified reproducible)

---

## Mục tiêu WF PF3 v2 (source of truth)

| Metric  | ACC1              | ACC2            |
|---------|-------------------|-----------------|
| Net     | **$815,620.77**   | **$63,164.18**  |
| DD      | **28.91%**        | **18.64%**      |
| PF      | **3.0442**        | **3.9679**      |
| Trades  | **322**           | **182**         |
| WR      | **76.09%**        | **81.87%**      |

Tất cả số trên được generate từ `scripts/wf_pf3_v2_search.py` seed `20260405`,
đã verify reproducible — re-run cho kết quả bit-for-bit giống hệt.

---

## Model artifacts

Các file model artifacts **ĐÃ được commit vào git** và có sẵn trên branch — máy khác
`git clone` + `git checkout` là có ngay, **không cần copy thủ công**.

```
outputs/acc1_expand_net127313_dd3215_model.pkl      (~989 KB)  ← commit 5db0852
outputs/acc1_expand_net127313_dd3215_scaler.pkl     (~2.9 KB)  ← commit 5db0852
outputs/acc1_expand_net127313_dd3215_model_meta.json           ← commit 5db0852

outputs/acc2_expand_net35714_dd2281_model.pkl       (~983 KB)  ← commit 5db0852
outputs/acc2_expand_net35714_dd2281_scaler.pkl      (~2.9 KB)  ← commit 5db0852
outputs/acc2_expand_net35714_dd2281_model_meta.json            ← commit 5db0852
```

> **Lưu ý**: `outputs/` bị `.gitignore` nhưng các file này đã được force-add từ trước
> nên git vẫn track. Chạy `git ls-files outputs/ | grep expand` để xác nhận.

---

## Live configs (source of truth cho live trading)

| File | Dùng cho |
|------|----------|
| `configs/live_acc1.yaml` | Live bot ACC1 |
| `configs/live_acc2.yaml` | Live bot ACC2 |

Đây là **exact copy** của benchmark configs tương ứng:

- `configs/benchmarks/acc1_pf3v2_net815k_dd2891.yaml`
- `configs/benchmarks/acc2_pf3v2_net63k_dd1864.yaml`

### Key params ACC1

```yaml
app:
  model_path:       outputs/acc1_expand_net127313_dd3215_model.pkl
  scaler_path:      outputs/acc1_expand_net127313_dd3215_scaler.pkl
  model_meta_path:  outputs/acc1_expand_net127313_dd3215_model_meta.json

strategy:
  signal_threshold:         0.86
  min_strategy_score:       0.15
  require_trend_alignment:  false
  blocked_hours_utc:        [15, 22, 23]   # template: minimal
  silver_bullet_enabled:    false
  adx_gate_enabled:         true
  adx_min_trend:            12.0

risk:
  risk_per_trade:                 0.07
  stop_loss_atr_multiple:         1.2
  take_profit_rr:                 7.0
  sideway_take_profit_rr:         6.0
  volatile_take_profit_rr:        8.0
  daily_loss_limit_pct:           0.03
  consecutive_loss_pause_count:   3
  consecutive_loss_cooldown_bars: 8
  anti_martingale_factor:         0.6
  compound_cap:                   50.0
```

### Key params ACC2

```yaml
app:
  model_path:       outputs/acc2_expand_net35714_dd2281_model.pkl
  scaler_path:      outputs/acc2_expand_net35714_dd2281_scaler.pkl
  model_meta_path:  outputs/acc2_expand_net35714_dd2281_model_meta.json

strategy:
  signal_threshold:         0.80
  min_strategy_score:       0.15
  require_trend_alignment:  false
  allowed_weekday_hours_utc:               # template: focus_hours
    Monday:    [0, 1, 5, 6, 7, 13, 14, 20, 21]
    Tuesday:   [0, 1, 5, 6, 7, 13, 14, 20, 21]
    Wednesday: [0, 1, 5, 6, 7, 13, 14, 20, 21]
    Thursday:  [0, 1, 5, 6, 7, 13, 14, 20, 21]
    Friday:    [0, 1, 5, 6, 7, 13, 14, 20, 21]
  silver_bullet_enabled:    true
  adx_gate_enabled:         false

risk:
  risk_per_trade:                 0.06
  stop_loss_atr_multiple:         1.2
  take_profit_rr:                 10.0
  sideway_take_profit_rr:         9.0
  volatile_take_profit_rr:        11.0
  sideway_risk_multiplier:        0.20
  strong_volatility_risk_multiplier: 1.0
  daily_loss_limit_pct:           0.015
  consecutive_loss_pause_count:   3
  consecutive_loss_cooldown_bars: 12
  anti_martingale_factor:         0.7
  compound_cap:                   50.0
```

---

## Setup máy mới (từng bước)

### 1. Clone repo và checkout branch

```bash
git clone <repo_url>
cd "Trade Indicator"
git checkout codex/pf-optimize-from-task4-clean
```

### 2. Tạo Python environment

```bash
python3 -m venv .venv2
source .venv2/bin/activate      # macOS/Linux
# hoặc: .venv2\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Xác nhận model artifacts có sẵn

```bash
git ls-files outputs/ | grep expand
```

Expected (6 files):
```
outputs/acc1_expand_net127313_dd3215_model.pkl
outputs/acc1_expand_net127313_dd3215_model_meta.json
outputs/acc1_expand_net127313_dd3215_scaler.pkl
outputs/acc2_expand_net35714_dd2281_model.pkl
outputs/acc2_expand_net35714_dd2281_model_meta.json
outputs/acc2_expand_net35714_dd2281_scaler.pkl
```

Nếu output rỗng (git không track), chạy:

```bash
git checkout 5db0852 -- outputs/acc1_expand_net127313_dd3215_model.pkl \
  outputs/acc1_expand_net127313_dd3215_scaler.pkl \
  outputs/acc1_expand_net127313_dd3215_model_meta.json \
  outputs/acc2_expand_net35714_dd2281_model.pkl \
  outputs/acc2_expand_net35714_dd2281_scaler.pkl \
  outputs/acc2_expand_net35714_dd2281_model_meta.json
```

### 4. Verify config load đúng

```bash
source .venv2/bin/activate
PYTHONPATH=src python3 -c "
from pathlib import Path
from xauusd_ai.config import load_settings
for label, cfg in [('ACC1','configs/live_acc1.yaml'),('ACC2','configs/live_acc2.yaml')]:
    s = load_settings(Path(cfg))
    print(label, '-> model:', s.app.model_path, '| threshold:', s.strategy.signal_threshold, '| risk:', s.risk.risk_per_trade, '| TP_RR:', s.risk.take_profit_rr)
"
```

Expected output:

```
ACC1 -> model: outputs/acc1_expand_net127313_dd3215_model.pkl | threshold: 0.86 | risk: 0.07 | TP_RR: 7.0
ACC2 -> model: outputs/acc2_expand_net35714_dd2281_model.pkl | threshold: 0.8 | risk: 0.06 | TP_RR: 10.0
```

### 5. Chạy unit tests

```bash
PYTHONPATH=src python -m pytest tests/ -q --tb=short \
  --ignore=tests/test_api_dashboard_journal.py \
  --ignore=tests/test_api_news_alerts.py \
  --ignore=tests/test_db_helpers.py
```

Expected: **324/324 passed**

---

## Verify WF artifacts (fast — không cần re-run search)

Kiểm tra benchmark YAML params và JSON artifact metrics khớp với target:

```bash
source .venv2/bin/activate
PYTHONPATH=src python -m pytest tests/test_pf3v2_wf_results.py -v
```

Expected: **176/176 passed** — bao gồm kiểm tra exact metrics cho cả 2 accounts.

---

## Verify reproducibility WF (tùy chọn — mất ~15 phút mỗi account)

Re-run WF search với cùng seed để xác nhận kết quả giống hệt:

```bash
source .venv2/bin/activate

# ACC1 — expected most_trades: Net=815,620.77  DD=28.91%  PF=3.0442  Trades=322  WR=76.09%
PYTHONPATH=src python scripts/wf_pf3_v2_search.py \
  --base-config configs/benchmarks/acc1_expand_net127313_dd3215.yaml \
  --old-net 127313.86 --old-dd 32.15 --pf-min 3.0 \
  --out-prefix wf_pf3v2_acc1_verify \
  --explore 2000 --refine 500 --max-rounds 1 --seed 20260405

# ACC2 — expected most_trades: Net=63,164.18  DD=18.64%  PF=3.9679  Trades=182  WR=81.87%
PYTHONPATH=src python scripts/wf_pf3_v2_search.py \
  --base-config configs/benchmarks/acc2_expand_net35714_dd2281.yaml \
  --old-net 35714.67 --old-dd 22.81 --pf-min 3.0 \
  --out-prefix wf_pf3v2_acc2_verify \
  --explore 2000 --refine 500 --max-rounds 1 --seed 20260405
```

> **Lưu ý**: Chỉ cần `--max-rounds 1` để verify reproducibility.
> `most_trades` trong output JSON phải khớp chính xác với target metrics trên.

---

## Chạy live bots (không cần MLflow/Airflow/Grafana)

```bash
source .venv2/bin/activate

# ACC1
PYTHONPATH=src python3 scripts/live_runner.py live --config configs/live_acc1.yaml &

# ACC2
PYTHONPATH=src python3 scripts/live_runner.py live --config configs/live_acc2.yaml &

# API + Dashboard
PYTHONPATH=src python3 -m uvicorn xauusd_ai.api.main:app --host 0.0.0.0 --port 8000
```

Dashboard: `http://localhost:8000/dashboard`

Check runtime status sau khi bots khởi động:

```bash
cat outputs/live_status_acc1.json
cat outputs/live_status_acc2.json
```

---

## Chạy full stack với Docker

```bash
docker compose up --build
```

Services:
- `live-acc1` — bot ACC1, dùng `configs/live_acc1.yaml`
- `live` (acc2) — bot ACC2, dùng `configs/live_acc2.yaml`
- `api` — REST + WebSocket API tại port 8000
- `mlflow` — MLflow tracking tại port 5000
- `prometheus` + `grafana` — monitoring
- `nginx` — reverse proxy (WebSocket support)

Infra configs đã có sẵn trong repo:

```
configs/nginx.conf
configs/prometheus.yml
configs/grafana/provisioning/dashboards/dashboards.yml
configs/grafana/dashboards/trading.json
```

---

## Kiến trúc model

- **Feature set**: 59 features — ICT structure (BOS, CHoCH, FVG, OB), VSA, Wyckoff, multi-TF, news, momentum
- **2-stage model**: Entry classifier (LightGBM) + Meta calibrator (isotonic, 16 features)
- **Decision threshold**: ACC1 = 0.68 (model meta), override bởi `signal_threshold = 0.86` trong config
- **WF search**: 8 folds sequential, `compound_cap=50`, `initial_balance=200`
- **Fold period**: ~28,000 bars M5 mỗi fold (≈ ~3 tuần)

---

## Data source

Multi-timeframe XAUUSD data từ `2003-05-05` đến `2026-03-30`:

```
src/xauusd_ai/real_data/XAUUSDm_M5.csv    ← primary (WF folds dùng M5)
src/xauusd_ai/real_data/XAUUSDm_M15.csv
src/xauusd_ai/real_data/XAUUSDm_H1.csv
src/xauusd_ai/real_data/XAUUSDm_H4.csv
src/xauusd_ai/real_data/XAUUSDm_D1.csv
```

---

## Git branch

```
codex/pf-optimize-from-task4-clean
```

Commits chính:

| Commit  | Nội dung |
|---------|----------|
| `21f20f6` | feat(wf-v2): lock pf3v2 configs + restore infra configs + 324 passing tests |
| `558e185` | chore(scripts): add WF analysis & debug scripts + reset peak balance |
