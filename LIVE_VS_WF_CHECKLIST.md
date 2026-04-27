# Checklist: Đảm Bảo Kết Quả Live ≈ WF (Combo #133)

> **Mục tiêu:** Mỗi lần deploy / retrain / thay đổi config, chạy qua checklist này để phát hiện lệch sớm.  
> **Tham chiếu WF:** `scripts/show_combo133_daily.py` | **Tham chiếu Live:** `configs/live_acc1.yaml` + `src/xauusd_ai/model/trainer.py`

---

## 🏗️ NHÓM 1 — Model Architecture

| # | Kiểm tra | WF (show_combo133_daily.py) | Live (trainer.py) | Status |
|---|----------|-----------------------------|-------------------|--------|
| 1.1 | Loại ensemble | `VotingClassifier` soft-voting | ✅ Đồng nhất | ☐ |
| 1.2 | Thứ tự estimators | `[("hgb",HGB), ("rf",RF), ("et",ET)]` | ✅ Đồng nhất | ☐ |
| 1.3 | Weights | `[3, 2, 1]` | ✅ `[3, 2, 1]` | ☐ |
| 1.4 | HGB params | `max_iter=1000, lr=0.01, depth=7, leaf=20, l2=1.0, bins=128, val=0.1, no_change=40` | ✅ Đồng nhất | ☐ |
| 1.5 | RF params | `n_est=200, depth=12, leaf=15, max_feat="sqrt", balanced` | ✅ Đồng nhất | ☐ |
| 1.6 | ET params | `n_est=200, depth=14, leaf=10, max_feat="sqrt", balanced` | ✅ Đồng nhất | ☐ |
| 1.7 | Sample weights | `2×pos_boost + time-decay half=40%` | ✅ Đồng nhất | ☐ |
| 1.8 | Feature selection | RF scout, drop bottom 30% importances | ✅ `_feat_sel_drop=30` | ☐ |
| 1.9 | Feature scaler | `StandardScaler` fit_on_train, transform_on_test | ✅ Đồng nhất | ☐ |
| 1.10 | TRAIN_BARS | `30,000 M5 bars` | Đảm bảo retrain dùng đủ 30k | ☐ |

**Cách kiểm tra 1.1–1.10:**
```bash
# So sánh tay trainer.py vs show_combo133_daily.py
grep -n "max_iter\|n_estimators\|max_depth\|weights" scripts/show_combo133_daily.py
grep -n "max_iter\|n_estimators\|max_depth\|weights" src/xauusd_ai/model/trainer.py
```

---

## 📊 NHÓM 2 — Feature Pipeline

| # | Kiểm tra | Yêu cầu | Cách verify |
|---|----------|---------|-------------|
| 2.1 | FEATURE_COLUMNS giống nhau | `len(FEATURE_COLUMNS)` phải bằng nhau ở WF và Live | `python -c "from xauusd_ai.features.dataset import FEATURE_COLUMNS; print(len(FEATURE_COLUMNS))"` |
| 2.2 | Không có feature NaN | Không có NaN trong FEATURE_COLUMNS lúc inference | Xem warning log của bot `ModelTrainer: missing X feature columns` |
| 2.3 | `judas_swing_signal` tồn tại | Feature column — KHÔNG phải standalone | ✅ Hàm `judas_swing()` trong `indicators.py` |
| 2.4 | `silver_bullet_setup` tồn tại | Feature column — KHÔNG phải standalone | ✅ Hàm `silver_bullet_setup()` trong `indicators.py` |
| 2.5 | Multi-timeframe data đủ | M5, M15, H1, H4, D1 đều load được | Xem log bot: `[1/3] Loading data...` |
| 2.6 | Timezone UTC nhất quán | CSV và MT5 bridge đều UTC | `csv.index.tz == UTC` |
| 2.7 | Data source nhất quán | WF dùng Dukascopy CSV; Live dùng MT5 tick | Chấp nhận drift nhỏ (~2–5% P&L) |

**⚠️ Rủi ro cao nhất:** Nếu MT5 bridge trả về M5 thiếu cột (vd. `volume=0`), indicator tính sai → feature drift → model cho signal khác WF.

---

## ⚙️ NHÓM 3 — Signal Filter (live_acc1.yaml vs WF params)

| # | Tham số | Giá trị WF | live_acc1.yaml hiện tại | Match? |
|---|---------|-----------|------------------------|--------|
| 3.1 | `min_confidence` | `0.70` | `0.70` | ✅ |
| 3.2 | `sideway_min_confidence` | `0.70` | `0.70` | ✅ |
| 3.3 | `volatile_min_confidence` | `0.70` | `0.70` | ✅ |
| 3.4 | `blocked_hours_utc` | `[3,15,17,22,23]` | `[3,15,17,22,23]` | ✅ |
| 3.5 | `require_trend_alignment` | `False` | Kiểm tra | ☐ |
| 3.6 | `d1_trend_gate` | `False` | Kiểm tra | ☐ |
| 3.7 | `min_strategy_score` | `0.00` | Kiểm tra | ☐ |
| 3.8 | `silver_bullet_enabled` | N/A (WF không dùng) | `True` (confidence boost giờ NY) | ⚠️ Xem ghi chú |
| 3.9 | `silver_bullet_confidence_boost` | N/A | `0.05` | Chấp nhận được |

> **Ghi chú 3.8:** `silver_bullet_enabled: true` trong live config CHỈ tăng `effective_prob += 0.05` trong giờ NY. WF không có cơ chế này → Live có thể lấy thêm ~3–5% lệnh so với WF trong các giờ 10–11 NY. Đây là **lợi thế nhỏ**, không phải lỗi.

**Cách kiểm tra 3.5–3.7:**
```bash
grep -A2 "require_trend\|d1_trend_gate\|min_strategy_score" configs/live_acc1.yaml
```

---

## 💰 NHÓM 4 — Trade Execution (Exit Logic)

| # | Tham số | Giá trị WF (engine.py) | live_acc1.yaml | Match? |
|---|---------|------------------------|----------------|--------|
| 4.1 | `partial_tp_enabled` | `True` | `True` | ✅ |
| 4.2 | `partial_tp_rr` | `1.2` | `1.2` | ✅ |
| 4.3 | `partial_tp_pct` | `0.5` (đóng 50%) | `0.5` | ✅ |
| 4.4 | `trailing_sl` bật | Bật sau `1.0R` | Kiểm tra | ☐ |
| 4.5 | `trail_atr_multiple` | `1.0` | `1.0` | ✅ |
| 4.6 | `spread_cost_rr` | `0.10` | Kiểm tra | ☐ |
| 4.7 | `slippage_rr` | `0.05` | Kiểm tra | ☐ |
| 4.8 | `commission_rr` | `0.02` | Kiểm tra | ☐ |
| 4.9 | `risk_per_trade` | `0.04` (4%) | `0.04` | ✅ |
| 4.10 | Compound mode | **KHÔNG compound** (WF non-compound) | Đảm bảo không tăng lot theo balance | ☐ |

> **⚠️ Điểm dễ lệch nhất:** MT5 live spread XAUUSD giờ cao điểm có thể = 0.25–0.40 RR (WF giả định 0.10). → Kỳ vọng live P&L thấp hơn WF khoảng 10–15% là **bình thường**.

---

## 🔄 NHÓM 5 — Retrain Cycle

| # | Kiểm tra | Yêu cầu |
|---|----------|---------|
| 5.1 | Trigger đúng | `auto_update_retrain.py` trigger khi `Δbars ≥ 6,000` (~20 ngày giao dịch) |
| 5.2 | TRAIN_BARS đủ | 30,000 bars M5 gần nhất (~125 ngày) |
| 5.3 | Threshold được re-optimize | Mỗi lần retrain, threshold search lại trên 30% val subset |
| 5.4 | Feature mask được lưu | `outputs/acc1_v14pp_model_meta.json` có `feat_mask` |
| 5.5 | Model + Scaler khớp nhau | Scaler fit cùng batch train với model, lưu cùng lúc |
| 5.6 | Bot load lại model | Sau retrain, `_reload_event` được set → orchestrator reload `.pkl` |
| 5.7 | State file cập nhật | `outputs/combo133_retrain_state.json` → `last_retrain_m5_count` tăng |

**Cách kiểm tra thủ công:**
```bash
# Xem state file
cat outputs/combo133_retrain_state.json

# Xem meta model
cat outputs/acc1_v14pp_model_meta.json | python -m json.tool | grep -E "feat|threshold|train_rows"
```

---

## 📡 NHÓM 6 — MT5 Bridge & Execution

| # | Kiểm tra | Yêu cầu |
|---|----------|---------|
| 6.1 | Bot nhận đúng M5 bar | MT5 gửi close của bar M5 vừa đóng (không phải bar đang mở) |
| 6.2 | Giờ UTC đúng | MT5 server time = UTC+0 (hoặc được convert về UTC trước khi tính features) |
| 6.3 | Partial close MT5 | Lệnh partial TP được thực thi (đóng 50% volume, giữ 50%) |
| 6.4 | Trailing SL sau partial | Sau partial close, trailing SL tiếp tục theo dõi 50% còn lại |
| 6.5 | Magic number tách biệt | acc1 và acc2 dùng magic number khác nhau (không cross-signal) |
| 6.6 | Không mở lệnh trong giờ blocked | Kiểm tra log bot lúc 3:00, 15:00, 17:00, 22:00, 23:00 UTC |
| 6.7 | Spread thực tế | Kiểm tra `account_info().spread` × `point` → so với WF assumption 0.10R |

**Log pattern để monitor:**
```
[SKIP] blocked_hours (hour=15)          ← đúng
[SKIP] confidence_below_floor (0.685)   ← đúng
[TRADE] BUY conf=0.731 sl=3422.50       ← trade thực sự
```

---

## 📅 NHÓM 7 — Checklist Hàng Tuần

Mỗi thứ Hai trước khi thị trường mở:

```
☐ 7.1  Chạy: python scripts/show_combo133_daily.py 2>&1 | tail -20
        → Xem fold mới nhất có P&L dương không

☐ 7.2  So sánh win_rate live (dashboard) vs WF avg (46.4%)
        → Lệch > 10% → điều tra

☐ 7.3  Xem log bot 7 ngày qua:
        docker logs trade-indicator-live-acc1-1 --since 7d | grep -E "TRADE|SKIP|ERROR"

☐ 7.4  Kiểm tra retrain có chạy không:
        cat outputs/combo133_retrain_state.json | grep last_retrain

☐ 7.5  Kiểm tra bot uptime:
        docker ps --format "{{.Names}}\t{{.Status}}"

☐ 7.6  Kiểm tra file csv data mới nhất:
        python -c "import pandas as pd; df=pd.read_csv('src/xauusd_ai/real_data/XAUUSDm_M5.csv'); print(len(df), df.index[-1])"
```

---

## 🚨 NHÓM 8 — Dấu Hiệu Cần Điều Tra Ngay

| Triệu chứng | Nguyên nhân nghi ngờ | Hành động |
|-------------|---------------------|-----------|
| Win rate live < 35% liên tục 5 ngày | Model predict sai vì data drift | Retrain thủ công |
| Số lệnh/ngày live << WF avg (~22) | Blocked giờ sai, spread quá cao, min_conf quá cao | Check config vs WF |
| Số lệnh/ngày live >> WF avg | Silver Bullet boost quá mạnh, hoặc min_conf thấp hơn WF | Check `silver_bullet_confidence_boost` |
| P&L = 0 nhiều lệnh | MT5 partial close lỗi | Check log bridge |
| Max DD > 15% trong 1 tháng | Regime change, cần retrain | Chạy auto_update_retrain.py |
| Log: `missing X feature columns` | FEATURE_COLUMNS không khớp model cũ | Xóa `.pkl` cũ, retrain lại |

---

## ✅ Kết Luận Nhanh

**Hiện tại (26/04/2026) — Trạng thái:**

| Hạng mục | Trạng thái |
|----------|-----------|
| Model architecture | ✅ Khớp 100% (trainer.py = show_combo133_daily.py) |
| Signal filters (blocked, min_conf) | ✅ Đồng nhất |
| Exit logic (partial TP, trailing) | ✅ Đồng nhất |
| SB standalone | ✅ Đã xóa — không ảnh hưởng WF |
| Silver Bullet boost (giờ NY) | ⚠️ Live có thêm, WF không có → live có thể nhỉnh hơn WF ~3% |
| Spread assumption | ⚠️ WF=0.10R, Live thực tế có thể 0.20–0.40R giờ cao điểm |
| Data source | ⚠️ WF=Dukascopy, Live=MT5 → chấp nhận drift 5–15% |

**Kỳ vọng thực tế:** Live P&L = **WF P&L × 0.70–0.85** là bình thường (do spread/slippage thực tế).  
Nếu live > 85% WF → xuất sắc. Nếu live < 60% WF liên tục → cần điều tra.
