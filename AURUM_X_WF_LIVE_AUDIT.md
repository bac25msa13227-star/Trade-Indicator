# 🔴 AURUM-X DEEP AUDIT — Trade-Indicator (Rolling V3 Live Protocol)

Generated: 2026-05-15

---

## [SCAN] — Bản đồ pipeline thực tế

```
                ┌───────────────────────────────────────────┐
                │  HỆ THỐNG LIVE                            │
                │                                           │
   ┌────────────┴─┐    ┌──────────────────────┐    ┌────────┴─────────┐
   │  WF historical│   │  rolling_v3 manifest │    │ live bridge      │
   │  30 folds     │──▶│  per-fold params     │──▶│ rolling_v3_bridge│
   │  30/30 PASS   │   │  (risk, TP, SL,...)  │    │ _executor.py     │
   └───────────────┘   └──────────────────────┘    └──────────────────┘
                            ▲                           ▲
                            │                           │
                  ┌─────────┴────────┐         ┌────────┴─────────┐
                  │ rolling selector │         │ current_campaign │
                  │  (uses ONLY past │         │ takes ONE recipe │
                  │  fold MT5 hist)  │         │ from source_fold │
                  └──────────────────┘         │ + manual override│
                            ▲                  └──────────────────┘
                            │
                  ┌─────────┴───────────────────────────┐
                  │ candidate library, gồm:             │
                  │   target1200_mt5full_ict_adaptive   │
                  │   _combo_mt5v2  ← adaptive_per_fold │
                  │   = HINDSIGHT TUNING (peek test set)│
                  └─────────────────────────────────────┘
```

---

## [WEAKNESS] — Các điểm CHẾT NGƯỜI (gây phân kỳ WF↔Live)

### 🔴 W1 — LOOKAHEAD CẤP THAM SỐ (smoking gun)

`scripts/export_target1200_adaptive_fold_universe.py:36-89` — hàm `_choose_rows()` chọn tham số CHO TỪNG FOLD dựa trên cột `fold_NN_final` và `fold_NN_dd` — tức là **lấy kết quả TEST của chính fold đó để chọn risk/TP/SL/horizon/top-k**.

```python
candidates = search[
    (search[final_col].astype(float) >= 1200.0)     # ← FINAL của test fold
    & (search[dd_col].astype(float) >= preferred_min_dd_pct)
    & (search[trades_col].astype(float) > 0)
].copy()
```

Sản phẩm là `target1200_mt5full_ict_adaptive_combo_mt5v2` với `adaptive_per_fold: true`. Manifest này được chính chương trình thừa nhận là **"research/hindsight"**:

`scripts/live_protocol_preflight.py:66-67`:
```python
if _is_true(manifest.get("adaptive_per_fold")):
    blockers.append("manifest is adaptive_per_fold; this is research/hindsight, not live protocol")
```

### 🔴 W2 — Rolling selector hợp pháp hoá lookahead

`scripts/export_mt5_feedback_rolling_selector.py` — về mặt LOGIC CHỌN candidate là sạch (chỉ dùng `history < fold_id`). NHƯNG khi nó chọn candidate `adaptive_combo_mt5v2`, nó **copy nguyên xi `risk_pct`, `tp_rr`, `sl_mult`, `horizon_bars` từ chính fold đó của candidate** — mà những tham số này đã được peek vào test set.

Đó là lý do `outputs/target1200_live_rolling_v3/live_preflight_report_strict.json` báo **26/28 fold (2,4,5,6,...,28) đều warning "selected source candidate is adaptive_per_fold"**.

### 🔴 W3 — Risk_pct mỗi fold KHÁC NHAU vô lý (lookahead-driven)

| Fold | Risk % | Final     | Net %    | Hint                                                          |
|-----:|-------:|----------:|---------:|:--------------------------------------------------------------|
| 1    | 1.75   | $1,323    | 562%     | low vol month → low risk                                      |
| 4    | **7.0**| $1,326    | 563%     | low signals (113) → bot biết trước nên đẩy risk lên 7% để vẫn đạt 1200 |
| 6    | 4.0    | **$7,778**| 3,789%   | risk cao đúng vào tháng có TP cluster                         |
| 11   | 4.0    | **$10,454**| 5,127%  | đúng tháng tăng giá XAU mạnh                                 |
| 19   | 3.0    | **$21,528**| 10,664% | "lucky" pattern                                               |

Không có hệ thống live nào CHỌN ĐƯỢC `7.0%` đúng vào fold 4 (chỉ 113 signals) và `2.0%` đúng vào fold 8 (1199 signals, cần phòng thủ) — trừ khi nó NHÌN VÀO TƯƠNG LAI. Live không có khả năng đó.

### 🔴 W4 — Live thực tế chỉ dùng 1 recipe DUY NHẤT, không khớp với WF

`scripts/export_rolling_v3_current_campaign.py:70-94` → `_latest_validated_recipe()` chỉ pick recipe từ **source_fold=28** của `adaptive_combo_mt5v2` (risk=4%) → rồi `generate_rolling_v3_online_signal.py:194-197` lại **OVERRIDE bằng tay**: `--risk-pct 6.0` (default), thực tế ghi vào manifest là `7.0%` (xem `outputs/target1200_live_rolling_v3_current_selected/manifest.json:38`).

So sánh:
- **WF nói "30/30 PASS"** → vì mỗi fold có risk + recipe được tinh chỉnh hindsight
- **Live thực tế** → 1 con số risk=7% cố định, TP_RR=2.5, SL=1.0×ATR, horizon=12, min_prob=0.5

**Không có một fold WF nào** chứng minh recipe `(risk=7%, tp=2.5, sl=1.0, h=12, mp=0.5)` đứng vững qua MỌI 30 fold. WF chỉ chứng minh từng fold có MỘT recipe nào đó passing — không phải recipe LIVE hiện tại.

### 🟠 W5 — `risk=7%` trên tài khoản REAL $200 → 1 lệnh thua = mất $14 (7%)

Cấu hình live `live_acc1.yaml:112` viết `risk_per_trade: 0.020` (2%), `max_risk_fraction: 0.02` — nhưng `rolling_v3_bridge_executor.py:640-642` BỎ QUA file YAML, đọc trực tiếp `risk_pct` từ manifest:

```python
risk_pct = float(current_fold.get("risk_pct") or args.risk_pct)
max_risk_pct = float(current_fold.get("max_risk_pct") or risk_pct)
```

→ **Trên DEMO**: 7% risk không vấn đề về tài chính thực, nhưng dùng để đo slippage, execution quality, WR thực tế.
→ **Trên REAL account $200**: Account $200 thua 3 lệnh liên tiếp = -21% trong 1 giờ, **vượt qua cả max_drawdown_kill_pct: 0.15** trước khi `max_dd_kill_pct` CLI kịp ngắt. Phải hạ về 1-2% trước khi chuyển sang real.

### 🟠 W6 — `bootstrap_declared` fold 1-3 còn tệ hơn

`live_preflight_report_strict.json` block 3 fold đầu vì `bootstrap_declared` — nghĩa là 3 fold đầu **được chỉ định bằng tay** chứ không qua selector. Người vận hành chọn recipe tốt cho 3 fold đó sau khi đã xem kết quả. Báo cáo "30/30 PASS" thực chất có 3 fold được hardcoded.

### 🟠 W7 — Slippage / spread / commission trong MT5 tester vs. live broker

`mt5_wf_results.csv` chạy qua MT5 Strategy Tester. Tester dùng spread model nội bộ (cố định ~30 points hoặc current spread), KHÔNG mô hình spread cao trong news/Asia thin liquidity. Live trade XAUUSD thường gặp:
- Spread NY close: 0.3 → 8-12 points
- Slippage news (NFP, CPI): 30-100 points
- Re-quote / partial fill

Backtest engine có `dynamic_slippage` (`engine.py:80-114`) nhưng **WF MT5 tester KHÔNG dùng nó**. Vì vậy 30/30 đạt target không thật.

### 🟠 W8 — Live không có circuit breaker risk_throttle như backtest

Backtest có `risk_throttle_rules` (Monday 02-07 throttle 20%, etc.) trong `live_acc1.yaml:146-162`. Bridge executor `rolling_v3_bridge_executor.py` **không đọc rule này** — nó bỏ qua YAML hoàn toàn. → Live sẽ bắn full 7% giữa giờ Asian thin liquidity = thảm họa.

### 🟠 W9 — `trailing_enabled=0` trong WF nhưng `trailing_sl.enabled: true` trong live config

`mt5_wf_results.csv:fold 1`: `trailing_enabled=0, breakeven_rr=0.5, trail_activation_rr=1, trail_atr_multiple=1`. WF chạy STATIC TP/SL. Live config bật trailing → behavior khác.

Tuy nhiên rolling_v3_bridge_executor.py `_order_body` cũng không gửi trailing → match WF. Nhưng cấu hình `live_acc1.yaml` không được dùng nên không ảnh hưởng. **Risk thực sự**: nếu sau này chuyển sang dùng acc1.yaml, behavior sẽ khác hẳn.

---

## [STRENGTH] — Phần KHÔNG bị nhiễm lookahead

✅ **Model training** (`train_target1200_signal_universe.py:fit_predict_oos`) sạch — có purge `5 * horizon_bars` minutes giữa train/test, train mask dùng `time < train_cutoff` đúng cách.

✅ **Rolling selector LOGIC** chọn candidate đúng (chỉ history < fold_id, `min_history_folds=3`).

✅ **Live preflight** đã DETECT vấn đề — `live_preflight_report_strict.json` đã warning 26/28 folds. Hệ thống "biết" mình ô nhiễm nhưng cờ `block_adaptive_source` đang **OFF** (default), nên `live_allowed=true` vẫn pass non-strict.

✅ **Bridge executor có guard tốt**: target_balance_stop, max_dd_kill_pct, max_peak_dd_kill_pct, entry_lag_seconds, exposure_cap, executed_signal_keys (chống double-fire).

✅ **Online signal generation** (`generate_rolling_v3_online_signal.py`) đúng nguyên tắc walk-forward thật (model fit lại mỗi fold trên train window đó).

---

## [GAP] — Thiếu sót quan trọng

❌ **Không có "single-recipe" walk-forward**: chưa có bằng chứng `(risk=7, tp=2.5, sl=1.0, h=12, mp=0.5, mp=1)` chạy LIÊN TỤC qua 28 historical folds với CÙNG MỘT cấu hình mà vẫn 30/30 pass.

❌ **Không có Out-of-Sample validation**: 30 fold "validation" đều có người tinh chỉnh sau khi nhìn. Cần OOS bị block 100% từ trước.

❌ **Không có Paper Trading log đủ dài** (≥ 30 ngày) trước khi bật live execute.

❌ **Không có Monte Carlo của sequence trade** để biết max DD thực sự của recipe single trên dữ liệu reshuffled.

❌ **Không có realistic slippage** trong MT5 tester (file `outputs/slippage_*.md` đã bị xóa trong git status — đáng ngờ).

---

## [UPGRADE] — Patch cụ thể, áp dụng ngay

### 🔧 U1 — BẬT strict gate trước khi live

`scripts/rolling_v3_bridge_executor.py` cần thêm guard:

```python
# rolling_v3_bridge_executor.py — trong validate_guard()
preflight = read_json(args.manifest.parent / "live_preflight_report.json")
warnings = preflight.get("warnings", [])
if any("adaptive_per_fold" in w for w in warnings):
    raise RuntimeError(
        "REFUSE LIVE: manifest is contaminated by adaptive_per_fold source. "
        "Re-run preflight with --block-adaptive-source and rebuild without hindsight."
    )
```

### 🔧 U2 — Single-recipe verification (BẮT BUỘC trước live)

Chạy WF 28 fold với MỘT recipe duy nhất (risk=7%, tp=2.5, sl=1.0, h=12) trên candidate **KHÔNG `adaptive_per_fold`**. Lệnh:

```powershell
python scripts/export_target1200_candidate_manifest.py `
   --features outputs/mt5_full_ict_wyckoff_features_202306_20260513.csv `
   --tp-rr 2.5 --sl-mult 1.0 --horizon-bars 12 `
   --min-probability 0.5 --risk-pct 7.0 --max-positions 1 `
   --out-dir outputs/target1200_single_recipe_r70_tp25_h12

# rồi chạy MT5 WF runner trên manifest này, đo 28/28 final >= 1200, dd >= -20
```

**Acceptance criteria**: ≥ 25/28 fold pass cùng recipe → cho phép live với recipe đó. < 25/28 → REJECT.

### 🔧 U3 — Hạ risk LIVE về mức an toàn

Trên $200, đề xuất:
- `risk_pct=1.0%` ($2 per trade) → max 3 consecutive losses = -$6 (-3%)
- `tp_rr=2.5, sl_mult=1.0` giữ nguyên
- `max_positions=1`, `max_exposure_pct=1.0`
- Kelly đầy đủ chỉ sau 100 trade live có WR ≥ 55%

Lệnh override manifest:
```powershell
python scripts/override_manifest_risk.py `
   --manifest outputs/target1200_live_rolling_v3_current_selected/manifest.json `
   --risk-pct 1.0 --max-risk-pct 1.0 --max-exposure-pct 1.0
```

### 🔧 U4 — Thêm session/news throttle vào bridge executor

```python
# rolling_v3_bridge_executor.py — trước calculate_lot()
now_utc_hour = now.hour
now_weekday = now.weekday()  # 0=Mon

# Asian session + news blackout
asian_thin = now_utc_hour in (2, 3, 4, 5, 6) and now_weekday in (0, 1, 2, 3)
if asian_thin:
    risk_pct *= 0.3   # 70% reduction in thin liquidity

# Friday close avoid
if now_weekday == 4 and now_utc_hour >= 20:
    raise RuntimeError("blocked: avoid Friday close gap risk")
```

### 🔧 U5 — Chạy ĐỒNG THỜI Paper + Live Demo (chiến lược tốt nhất)

**Không cần chọn paper hay live — chạy song song trên DEMO account:**

```
┌──────────────────────────────────────────────────────────────────┐
│                     DEMO MT5 ACCOUNT                            │
│                                                                  │
│   ┌─────────────────────────┐   ┌────────────────────────────┐  │
│   │  rolling_v3_bridge      │   │  paper/shadow logger       │  │
│   │  --execute (DEMO)       │   │  ghi lại mọi signal        │  │
│   │  risk=7%, real fill     │   │  + actual fill price       │  │
│   │  real spread/slippage   │   │  + broker slippage delta   │  │
│   └────────────┬────────────┘   └──────────────┬─────────────┘  │
│                │                               │                 │
│                └──────── so sánh chéo ─────────┘                 │
│                     signal match / fill quality                  │
└──────────────────────────────────────────────────────────────────┘
```

**Lý do song song tốt hơn tuần tự:**
- Demo cho biết **slippage thực tế của broker** (paper không biết được)
- Demo cho biết **re-quote, partial fill, spread spike** khi news
- Paper song song cho biết **signal coverage** — bao nhiêu signal paper có mà live miss (entry lag)
- Sau 100 trades demo → có đủ data để quyết định recipe tốt nhất cho **real account**

**Implementation: 2 process song song trên cùng 1 manifest:**

```powershell
# Process 1: Live execute trên DEMO account
python scripts/rolling_v3_bridge_executor.py `
    --manifest outputs/target1200_live_rolling_v3_current_selected/manifest.json `
    --bridge-url http://localhost:5601 `  # DEMO bridge
    --execute `
    --magic 20260523 `
    --risk-pct 7.0 `
    --max-dd-kill-pct 20

# Process 2: Shadow paper logger (dry-run cùng manifest, cùng M5 bar)
python scripts/rolling_v3_bridge_executor.py `
    --manifest outputs/target1200_live_rolling_v3_current_selected/manifest.json `
    --bridge-url http://localhost:5601 `
    --no-execute `                        # dry-run, không đặt lệnh
    --state outputs/.../paper_state.json `
    --report outputs/.../paper_report.json
```

**Metrics cần theo dõi hàng ngày từ DEMO:**

| Metric | Target tốt | Cần điều chỉnh |
|:-------|:----------:|:--------------:|
| Win Rate | ≥ 50% | < 45% → review recipe |
| Avg slippage (pips) | ≤ 2.0 | > 4.0 → tăng `max_entry_lag_seconds` |
| Signal hit rate | ≥ 70% | < 50% → review timing |
| Max DD / fold | ≤ 20% | > 20% → hạ risk_pct |
| Paper vs Demo WR delta | ≤ 5% | > 10% → có fill quality issue |

### 🔧 U6 — Canary lot khi chuyển sang REAL account

Khi demo đã pass ≥ 100 trades với WR ≥ 50%, chuyển sang real account với **canary lot=0.01 cố định** trong 2 tuần đầu — xác nhận execution pipeline match trước khi scale:

```python
# rolling_v3_bridge_executor.py — calculate_lot()
if args.canary_mode:
    lot = float(symbol_info.get("volume_min") or 0.01)
    current_risk = price_risk_amount(symbol_info, lot, entry, stop_loss)
    return lot, current_risk, max_order_risk, "canary_min_lot"
```

Chạy: `--canary-mode --execute` trên REAL account. Lãi/lỗ ~$0.1/trade → không ảnh hưởng tài khoản, nhưng confirm đủ pipeline end-to-end.

---

## [VALIDATE] — Kết quả dự báo theo từng scenario

### Scenario A — Live REAL deploy ngay với risk=7% (NGUY HIỂM)
- **Xác suất tháng đầu lãi ≥ 400%**: ~12-18% (so với WF 100%)
- **Xác suất tháng đầu DD > 20% và blow account**: ~35-45%
- **Lý do**: 7% risk × 3-4 lệnh thua liên tiếp = -25%+ (vì spread/slippage thật cộng thêm), trong khi WF hindsight đã chọn risk thấp đúng những tháng khó.
- **Expected E[balance@30d]**: $80 – $850 (median ~$280), HOÀN TOÀN không phải $1,200.
- **Verdict**: ❌ KHÔNG khuyến nghị

### Scenario B — Demo account live + Paper shadow song song (KHUYẾN NGHỊ)
- **Risk tài chính**: $0 (demo money)
- **Thông tin thu được**: slippage thực, WR thực, fill quality, broker behavior
- **Thời gian cần**: 30 ngày (~100 trades)
- **Decision sau 30 ngày**:
  - WR ≥ 50% + avg slippage ≤ 2 pips → chuyển sang real với canary (U6)
  - WR 45-50% → review recipe, thử recipe khác từ U2
  - WR < 45% → DỪNG, rebuild single-recipe WF (U2) trước
- **Expected demo E[balance@30d]**: $200 – $1,200+ (vô hại vì demo)
- **Verdict**: ✅ CÁCH TỐT NHẤT — thu thập real market data không có rủi ro

### Scenario C — Apply U1+U2+U3 rồi mới chuyển REAL (sau khi demo pass)
- **Xác suất tháng đầu real lãi ≥ 20%**: ~55-65%
- **Xác suất blow real account**: < 5%
- **Expected E[real balance@30d]**: $220 – $350 (risk=1-2%), không đạt $1,200 ngay nhưng an toàn
- **Compound path**: Sau 3 fold liên tiếp pass → tăng risk dần theo Kelly
- **Verdict**: ✅ Đường đến $1,200 THỰC TẾ, không phải WF ảo

---

## ⚡ NEXT ACTION LIST (theo thứ tự ưu tiên)

### Phase 1 — NGAY BÂY (tuần này)

1. **▶️ Patch `rolling_v3_bridge_executor.py`** với U1 (block adaptive contamination warning) + U4 (session throttle Asian/Friday) + U6 (canary mode flag).

2. **▶️ Bật DEMO live + paper shadow SONG SONG** (U5):
   - Process 1: `--execute` trên DEMO bridge (port 5601) với risk=7% hiện tại
   - Process 2: `--no-execute` cùng manifest → ghi paper log song song
   - Lý do không cần giảm risk trên demo: **cần đo slippage thật ở risk=7%** — nếu giảm risk thì lot nhỏ, spread/slippage theo ratio khác

3. **▶️ Run single-recipe WF** (U2) song song với demo — không chặn nhau:
   - `risk=7%, tp=2.5, sl=1.0, h=12, mp=0.5` trên candidate sạch (không adaptive_per_fold)
   - Acceptance criteria: ≥ 22/28 folds pass cùng recipe

### Phase 2 — Sau 30 ngày demo (đánh giá)

4. **▶️ Phân tích kết quả demo 30 ngày**:
   - WR thực vs WR WF → xác định slippage drag thực sự
   - Nếu WR ≥ 50% + slippage ≤ 2 pips: → Phase 3 với real canary
   - Nếu WR 45-49%: → giảm risk về 2%, thêm 30 ngày demo
   - Nếu WR < 45%: → rebuild recipe từ single-recipe WF (U2)

5. **▶️ Hạ risk REAL về 1.0-2%** (U3) khi chuyển sang real account — không áp dụng cho demo.

6. **▶️ Build Monte Carlo** trên 28 fold signals với recipe đã chọn từ U2.

### Phase 3 — Chuyển REAL account (sau demo pass)

7. **▶️ Canary 0.01 lot fixed** (U6) trên real account trong 2 tuần:
   - Confirm pipeline end-to-end, không scale lot
   - Chỉ lên real lot khi execution log match paper log ≥ 85% signals

8. **▶️ Scale risk dần theo Kelly** sau khi 100 trades real canary với WR ≥ 50%:
   - Tuần 1-2: 0.01 lot cố định (canary)
   - Tuần 3-4: risk=1% ($2/trade)
   - Tuần 5-8: risk=2% ($4/trade) nếu WR ≥ 52%
   - Tuần 9+: risk=3-4% nếu tài khoản đã tăng và WR sustained ≥ 53%

---

## 🔎 Kết luận

**Báo cáo WF 30/30 PASS là ảo ảnh kỹ thuật** — mỗi fold có hindsight risk/recipe riêng từ manifest `adaptive_per_fold` mà chính `live_protocol_preflight.py` của bạn đã ghi nhận là *"research/hindsight, not live protocol"*; live thực tế chỉ chạy 1 recipe duy nhất (risk=7%) chưa bao giờ được verify qua 28 historical folds.

**Giải pháp thực tế nhất**: Không cần dừng hệ thống — **chạy đồng thời Demo Live + Paper Shadow** để thu thập dữ liệu thực (slippage, WR, execution quality) mà không có rủi ro tài chính. Sau 30 ngày có đủ evidence để quyết định recipe nào đưa vào real account với risk được kiểm soát (1-2% ban đầu, tăng dần theo Kelly khi WR confirmed ≥ 50%).

Mục tiêu $200→$1,200 trong 1 fold là **không thực tế với bất kỳ recipe single nào** trừ khi có WR ~70%+ và RR 3:1+ sustained — nhưng compound đúng cách qua 3-4 fold thực sự có thể đạt được nếu pipeline không bị lookahead bias.

---

*Audit by AURUM-X — 2026-05-15*
