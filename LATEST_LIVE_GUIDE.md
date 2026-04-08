# Latest Live Guide

Last updated: `2026-04-08` — **M1 DualScalpM1** live bot (18/18 WF folds positive, ATH-proof)

---

## Tóm tắt kiến trúc mới

| | M1 Scalp (ACC2 — **mới, ưu tiên**) | M5 PF3v2 (ACC1 — cũ) |
|---|---|---|
| Service | `live-scalp-acc2` | `live-acc1` |
| Config | `configs/live_acc2_scalp_m1.yaml` | `configs/live_acc1.yaml` |
| Model | `DualScalpM1` (BUY+SELL độc lập) | HistGBDT + isotonic calibrator |
| Timeframe | M1 | M5 |
| Risk/trade | **0.3%** | 0.07% |
| WR | 52–56% | 76% |
| PF | 1.71–2.12 (18/18 folds) | 3.0+ |
| Trades/day | ~120 | ~1–2 |
| ATH gold (Aug23–Jul24) | ✅ 7/7 folds positive | ❌ WR = 0% |

---

## WF Results — M1 Scalp ACC2

Kết quả 18 folds walk-forward (Sep 2023 → Mar 2026), script `scripts/acc2_scalp_m1_wf.py`:

| Metric | Giá trị |
|--------|---------|
| Folds positive | **18/18** |
| ATH gold folds 1–7 | **7/7 positive** (M5 WR = 0% cùng kỳ) |
| Win Rate | **52–56%** |
| Profit Factor | **1.71–2.12** |
| Max DD @ 0.3% risk | **9.5%** |
| Non-compound return/yr | **+12,015%** |
| Paper trade holdout | WR=52.2% · PF=1.73 · DD=7.6% |

> Paper trade = fold 18 hold-out (Feb–Mar 2026, ~4,798 trades).  
> Non-compound @ 0.3% risk: +439%/54 ngày = +3,290%/yr.

---

## Model artifacts

Model **KHÔNG có trong git** (`outputs/` bị `.gitignore`).
Service `live-scalp-acc2` tự train khi lần đầu start (~4 phút).

```
outputs/acc2_scalp_m1_model.pkl        <- DualScalpModel (BUY+SELL CalibratedDirModel)
outputs/acc2_scalp_m1_scaler.pkl       <- identity transformer
outputs/acc2_scalp_m1_model_meta.json  <- feature list, thresholds, train_end
```

Train thủ công:

```bash
PYTHONPATH=src python scripts/acc2_scalp_m1_save_model.py
# Cần: src/xauusd_ai/real_data/XAUUSDm_M1.csv
```

---

## Live config — `configs/live_acc2_scalp_m1.yaml`

Key parameters:

```yaml
app:
  model_path:      outputs/acc2_scalp_m1_model.pkl
  scaler_path:     outputs/acc2_scalp_m1_scaler.pkl
  model_meta_path: outputs/acc2_scalp_m1_model_meta.json
  poll_seconds:    60         # 1 bar M1 = 1 phut

market:
  execution_timeframe: M1
  csv_data_path: src/xauusd_ai/real_data/XAUUSDm_M1.csv

strategy:
  signal_threshold: 0.55      # SELL gate (BUY gate = 0.58 trong DualScalpModel)
  adx_gate_enabled: false
  require_trend_alignment: false
  min_strategy_score: 0.0

risk:
  risk_per_trade:         0.003   # 0.3% -- MaxDD WF = 9.5%
  stop_loss_atr_multiple: 0.8
  take_profit_rr:         1.5     # aligned label_tp_rr
  max_open_positions:     2
  daily_loss_limit_pct:   0.008   # kill sau ~3 lenh thua lien tiep
  max_drawdown_kill_pct:  0.12
  partial_tp_enabled:     true
  partial_tp_rr:          1.0     # chot 50% o 1R, de 50% chay 1.5R
  trailing_sl:
    enabled: true
    breakeven_at_rr: 0.5
    activation_rr:   0.8
    trail_atr_multiple: 0.5

execution:
  magic_number: 20260408
  comment: xauusd-ai-acc2-scalp-m1-v1
  close_opposite_on_signal: true  # flip khi model doi chieu

training:
  retrain_on_startup: false   # KHONG retrain qua ModelTrainer
  sltp_label_max_horizon: 8   # 8 M1 bars = 8 phut
  label_tp_rr: 1.5
  walkforward_train_size: 250000  # ~6 thang M1 bars
  walkforward_test_size:   50000  # ~8 tuan
```

---

## Setup máy mới — từng bước

### Bước 1 — Clone repo

```bash
git clone https://github.com/bac25msa13227-star/Trade-Indicator.git
cd Trade-Indicator
git checkout codex/pf-optimize-from-task4-clean
```

### Bước 2 — Copy data CSV (bắt buộc — không có trong git)

Chạy **từ máy cũ (Mac)**, thay `user@TARGET_IP` và `/path/to/`:

```bash
cd "/Users/dodoannang/Documents/Thac si MSE/Trade Indicator"

for f in XAUUSDm_M1.csv XAUUSDm_M5.csv XAUUSDm_H1.csv XAUUSDm_H4.csv XAUUSDm_D1.csv; do
  scp "src/xauusd_ai/real_data/$f" \
      user@TARGET_IP:/path/to/Trade-Indicator/src/xauusd_ai/real_data/
done
```

### Bước 3 — Tạo file `.env`

```env
TELEGRAM_BOT_TOKEN=8591115022:AAEhZNq96Cw3mZmEsAADd1Mc7QfKOqXSMtw
TELEGRAM_CHAT_ID=1638555472
TELEGRAM_BOT_TOKEN_ACC2=8591115022:AAEhZNq96Cw3mZmEsAADd1Mc7QfKOqXSMtw
TELEGRAM_CHAT_ID_ACC2=1638555472
MT5_LOGIN=433326057
MT5_PASSWORD=07032001bB@
MT5_SERVER=Exness-MT5Trial7
```

> MT5 Bridge cần chạy trên Windows host tại **port 5601** (`http://host.docker.internal:5601`).

### Bước 4 — Build Docker image

```bash
docker compose build live-scalp-acc2
```

### Bước 5 — Start bot

```bash
docker compose up live-scalp-acc2 -d
```

- **Lần đầu**: tự train DualScalpM1 (~4 phút) rồi start live loop
- **Lần sau**: pkl có sẵn trong `outputs/` → skip train, start ngay

### Bước 6 — Kiểm tra

```bash
# Logs realtime
docker compose logs live-scalp-acc2 -f

# Trạng thái container
docker compose ps

# Trạng thái bot (cập nhật mỗi poll = 60 giây)
cat outputs/live_status_acc2.json
```

---

## Train lại model với data mới

Khi có thêm M1 data mới nhất:

```bash
docker compose run --rm live-scalp-acc2 bash -c \
  "rm -f outputs/acc2_scalp_m1_model.pkl && \
   PYTHONPATH=src python scripts/acc2_scalp_m1_save_model.py"

docker compose up live-scalp-acc2 -d
```

---

## Kiến trúc kỹ thuật — DualScalpM1

### Luồng inference (production)

```
M1 OHLCV (last 1000 bars)
   |
   v
build_all_scalp_features()      50 SCALP_FEATURE_COLUMNS (M1-native)
   + M5 context merge            m5_bias, m5_rsi_14, m5_atr_norm
   |
   v
DualScalpModel.score_live_row()
   +-- BUY model  (HistGBDT, trained on expected_direction=+1)
   |     P(TP_hit) >= 0.58  -->  trade_side = "buy"
   +-- SELL model (HistGBDT, trained on expected_direction=-1)
         P(TP_hit) >= 0.55  -->  trade_side = "sell"
   |
   v
HybridStrategy.build_trade_decision()
   trade_side = model_signal["trade_side"]   (override strategy_score sign)
   SL = 0.8 x ATR5,  TP = 1.5R
   |
   v
RiskManager -> 0.3% sizing -> MT5 execute
```

### Label (training)

- **Positive (1)** = TP (1.5R) chạm trước SL (0.8×ATR) trong vòng **8 bars (8 phút)**
- Tỷ lệ positive tự nhiên ~40%, balanced bằng `class_weight` + time decay
- Isotonic calibration trên 15% val cuối mỗi direction model

### Feature groups (50 tổng)

| Group | Count | Ví dụ |
|-------|-------|-------|
| Fast momentum | 8 | ema3_ema8_cross, rsi5, macd_hist |
| Order flow | 8 | tick_vol_delta_5, obv_slope_8, pressure_ratio |
| Microstructure | 10 | wick_rej_bull, body_ratio, inside_bar |
| Structure | 10 | fvg3_bull, bos_m1_bull, liq_sweep_low |
| Session | 7 | vwap_dev, hour_sin, session_id |
| BB + Stoch | 4 | bb_pct, stoch_k_5, stoch_d_5 |
| M5 context | 3 | m5_bias, m5_rsi_14, m5_atr_norm |

---

## Files source code quan trọng

```
src/xauusd_ai/model/scalp_model.py          DualScalpModel + CalibratedDirModel
src/xauusd_ai/features/scalp_features.py    50 SCALP_FEATURE_COLUMNS
src/xauusd_ai/features/scalp_dataset.py     build_scalp_dataset()
src/xauusd_ai/model/trainer.py              score_live_row: DualScalpModel fast path
src/xauusd_ai/features/dataset.py           build_live_feature_frame: scalp branch
src/xauusd_ai/strategies/hybrid.py          trade_side override from DualScalpModel
configs/live_acc2_scalp_m1.yaml             live config chinh
scripts/acc2_scalp_m1_save_model.py         train + save pkl (~4 phut)
scripts/acc2_scalp_m1_paper.py              paper trade validator
scripts/acc2_scalp_m1_wf.py                 18-fold WF script
docker-compose.yml                          live-scalp-acc2 service
```

---

## Model M5 PF3v2 — ACC1 (vẫn hoạt động)

```bash
docker compose up live-acc1 -d
```

| Metric | ACC1 |
|--------|------|
| Net | $815,620.77 |
| DD | 28.91% |
| PF | 3.04 |
| WR | 76.09% |
| Model | `outputs/acc1_expand_net127313_dd3215_model.pkl` |
| Config | `configs/live_acc1.yaml` |

Model M5 ACC1 đã force-commit vào git — có sẵn sau khi clone.

---

## Git

```
Branch: codex/pf-optimize-from-task4-clean
Repo:   https://github.com/bac25msa13227-star/Trade-Indicator.git
```

| Commit | Nội dung |
|--------|----------|
| `936bb99` | feat(scalp): M1 DualScalpM1 live — 18/18 WF positive, ATH-proof |
| `02d0b87` | docs(guide): rewrite guide for WF PF3 v2 |
| `21f20f6` | feat(wf-v2): lock pf3v2 configs + 324 passing tests |
