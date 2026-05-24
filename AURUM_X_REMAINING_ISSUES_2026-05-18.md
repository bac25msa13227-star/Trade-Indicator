# 🔍 AURUM-X RE-AUDIT — Code mới nhất 2026-05-18

**Mục tiêu**: $200 → $1,200 per fold (30 ngày). **Hiện trạng**: ACC2 DEMO live + paper shadow song song. Real-money vẫn bị block.

---

## ✅ ĐÃ FIX (so với audit cũ)

| Issue cũ | Status | Bằng chứng |
|:---|:---:|:---|
| W1: adaptive_per_fold lookahead | ✅ FIX | Manifest mới `adaptive_per_fold: false`, dùng `mt5_profitr_model_selector` train từ `feedback[feedback["fold"] < fold_id]` |
| W2: rolling selector copy hindsight params | ✅ FIX | `selection_mode: prior_mt5_profitr_model`, `selection_uses_current_fold_metrics: false` |
| W3: risk_pct vô lý mỗi fold | ✅ FIX | `signal_density_risk_policy` rule-based: 3.5 / 6.0 / 7.0% theo signal count thuần (không peek outcome) |
| W4: live recipe ≠ WF recipe | ✅ FIX | Live online_signal dùng cùng model trained on past feedback, output qua manifest tự sinh |
| W5: risk=7% trên $200 | ✅ FIX | `--live-risk-cap-pct` default 1.0%, demo override 7% (chỉ trên demo) |
| W6: bootstrap_declared fold 1-3 | ✅ FIX | Fold 1-3 dùng `cold_start_source_signals` (đầu vào base manifest, không hardcode) |
| W7: spread tester ≠ live | ⚠️ PARTIAL | Có stress_full.json với 30/50/100 extra points; nhưng **stress fail: realistic_gate_pass=false** |
| W8: không có session throttle | ✅ FIX | `session_guard()` Asian thin (hours 2-6, weekdays Mon-Thu, ×0.3) + Friday close block |
| W9: trailing config mismatch | ⚠️ PARTIAL | WF tester `trailing_enabled=0`, executor cũng không gửi trailing; vẫn match |

**Khung gates mới đã có**:
- `validate_live_protocol()` — block adaptive_per_fold, hindsight, bootstrap warnings
- `validate_execute_gate()` — paper gate (≥100 trades, ≥30 days, WR≥50%) + canary gate + realistic gate
- `session_guard()` — Asian thin + Friday close
- `--live-risk-cap-pct` (default 1.0%) — hard cap
- `--canary-mode` — force min lot
- `target_balance_stop` — auto-stop tại $1200
- `max_dd_kill_pct` — close all + halt khi DD từ initial ≥ X%
- `max_spread_points` — skip bar khi spread quá lớn

---

## 🔴 VẤN ĐỀ NGHIÊM TRỌNG CÒN LẠI

### 🔴 R1 — **REALISTIC GATE FAIL** (showstopper)

`outputs/rolling_v3_acc2_demo_frozen_density_score_20260518/stress_full.json:419`:
```json
"realistic_gate_pass": false,
"target_pass_folds": 26,        // chỉ 26/30 nếu có friction thực
"dd_pass_folds": 18,             // chỉ 18/30 DD pass
"all_pass_folds": 17,            // 17/30 cả hai pass
"worst_dd_pct": -27.35,          // fold 18 chạm -27%
"min_final_balance": 1148.21     // dưới $1200 target
```

Khi áp dụng friction realistic (30 extra points roundtrip, 50 thin-hour points, 100 Friday points):
- 13/30 fold KHÔNG đạt mục tiêu
- Worst DD vượt -25% (cả hai bộ lọc khả thi `dd20_ls0` và `dd18_ls0` đều có fold thua → $167 balance)
- `dd_throttle_sweep.json:9-26` xác nhận: best config cũng có **1 loss fold ($167)** và 7 target_fail folds

**Lý do**: framework có gates đầy đủ — nhưng kết quả OOS thực tế chỉ 17-23/30 pass thay vì 30/30. Không có recipe nào hiện tại đứng vững khi friction thực tế áp vào.

### 🔴 R2 — **BROKER SPREAD GẤP 2.5x CAP** (showstopper thực địa)

`outputs/rolling_v3_acc2_demo_frozen_density_score_20260518/bridge_live_state.json:8-156`:
```
21 M5 bars liên tiếp (1.75h) đều có spread=308 points,
cap=120 points → tất cả signals bị SKIP
```

Broker = `Exness-MT5Trial7`. XAUUSDm spread thực tế **308 points (30.8 pips)** trong giờ giao dịch, vượt xa:
- `max_spread_points=120` (12 pips) của bridge executor
- `extra_roundtrip_points=30` (3 pips) của stress test

**Hậu quả**:
1. Bot live signal `decision="trade"` được sinh ra (vẫn $1.21 lãi sau 1 ngày), nhưng 95%+ thời gian **không trade được** vì spread.
2. Stress test với 30 points extra đã fail → broker thực 300 points sẽ thua thảm.
3. **Recipe `tp_rr=2.5, sl_mult=1.0×ATR`** với ATR=6.7 → SL khoảng 67 points → spread 308 points = 4.6× SL distance → entry đã -4.6R trước khi giá di chuyển. **Hệ thống không thể chạy lãi với broker này.**

### 🟠 R3 — **Target $1200/fold đã đạt bằng `target_balance_stop`, không phải bằng strategy**

WF `mt5_wf_results.csv` có 30/30 target pass với `target_balance_stop=1300`. Khi balance chạm $1300, bot ngừng trade. Đây là **selection bias**: nếu fold sau khi đạt target có trades tiếp tục, có thể giảm balance về dưới $1200.

Bằng chứng: trong `dd_throttle_sweep.json` (chạy không có target_balance_stop), median final = $1221, MIN final = **$167** (loss fold).

→ Target $1200 chỉ được "guarantee" bằng cách dừng sớm. Live cũng có target_balance_stop=$1200 — OK với edge case này nhưng cần hiểu rõ.

### 🟠 R4 — **DD spike đến -27% trong 1 fold (Apr 2025)**

Fold 18 (2025-04-16 → 2025-05-19) có DD = **-26.76%** trong WF, **-27.35%** trong stress test, mặc dù risk_pct=3.5%. Đây là 1.3σ tail event với:
- 696 signals (cao)
- 436 trades, WR=56.7%
- Final balance $1300 (đạt target nhưng có lúc xuống -27%)

**Vấn đề**: với account thực $200, DD -27% = balance $146. `max_dd_kill_pct=20%` sẽ KILL ở $160 — nhưng MT5 tester không kill (tester ignore kill switch). Live bridge có kill nhưng chỉ kill khi DD realized; intra-bar DD có thể vượt qua trước khi kill kích hoạt.

### 🟠 R5 — **Hyperparameter tuning trên ALL FOLDS = in-sample leak gián tiếp**

Các ngưỡng trong `signal_density_risk_policy`:
- `low_signal_threshold: 250`
- `very_low_signal_threshold: 120`
- `score_boost_min_mean_score: 0.54`
- `score_boost_max_signals: 450`
- `min_train_rows_for_boost: 400`

Những con số này KHÔNG có lý thuyết đằng sau — chúng là kết quả tinh chỉnh thủ công trên 28-30 fold lịch sử. Đây là **hyperparameter overfitting** gián tiếp dù mỗi fold không "peek" outcome riêng nó.

Bằng chứng: thư mục `outputs/density_score_risk_policy_*_cfg026_*` và `density_risk_policy_base_cfg026_thr250_r6_*` cho thấy đã chạy MULTIPLE configs (thr250_r6, thr250_r60, ...) — chọn config tốt nhất qua tất cả folds = in-sample selection.

### 🟠 R6 — **WR fold 1 thấp (46%) — recipe không robust ở fold đầu**

Fold 1 (2023-11): risk=3.5%, 328 trades, WR=**46%**, DD=**-21.86%**. Đây là fold không có model history (cold_start_source_signals). Real-money startup sẽ giống fold 1 — chưa có feedback model để hỗ trợ → WR thấp.

**Hậu quả**: 30-day đầu live, expected WR ~46% với risk=3.5% → -21% DD, có thể blow account.

### 🟠 R7 — **WF MT5 chưa bao gồm execution latency**

Bridge executor có `--max-entry-lag-seconds 90` để skip signal vào bar muộn. Nhưng WF MT5 backtest chạy trên close-bar prices, không mô hình:
- 5-15s latency từ M5 close → bridge nhận → executor process → broker fill
- Re-quote ở broker
- Partial fill

Đối với chiến lược M5 với SL ~67 points, 15 seconds delay ở Gold = giá có thể di chuyển 20-30 points → entry skew làm SL hit sớm.

### 🟠 R8 — **`min_paper_win_rate_pct=50` quá thấp, không phòng vệ overfit**

`rolling_v3_bridge_executor.py:1173`: paper gate yêu cầu WR ≥ 50%. Nhưng:
- Recipe có expected WR ~58% trên backtest
- Live nếu thực sự hit chỉ 50% WR → âm EV với RR 2.5:1 và spread cao
- Cần WR ≥ 55% để có safety margin

### 🟡 R9 — **`positive_r_threshold=0.0` quá lỏng cho classifier**

`scripts/export_mt5_profitr_model_manifest.py:165` — label "positive" khi `realized_r > 0`. Trade có R=0.01 (lãi $0.01 sau spread/commission) vẫn được coi là positive. Class imbalance không đo lường được lợi nhuận thực.

Nên dùng `positive_r_threshold ≥ 0.5` để chỉ count trades có lãi đáng kể.

### 🟡 R10 — **Chưa có Monte Carlo / bootstrap reshuffle**

Không tìm thấy script Monte Carlo nào shuffle thứ tự trades trong fold để đo P5/P95 của final balance & max DD. Sequential luck rất quan trọng cho account $200 — 3-5 thua liên tiếp đầu fold có thể giết account.

---

## 📊 BẢNG TỔNG HỢP (raw vs realistic)

| Metric | WF raw (target_stop=$1300) | Stress full | Throttle-sweep best | Live target |
|:---|---:|---:|---:|---:|
| Folds | 30 | 30 | 30 | 30 |
| Target pass | 30/30 (100%) | **26/30 (87%)** | 23/30 (77%) | 30/30 |
| DD pass (≥-20%) | 23/30 (77%) | **18/30 (60%)** | 23/30 (77%) | 30/30 |
| All-pass | 23/30 | **17/30 (57%)** | 23/30 (77%) | 30/30 |
| Loss folds | 0 | 0 | **1 ($167)** | 0 |
| Min final | $1200 | **$1148** | **$167** | ≥$200 |
| Worst DD | -26.76% | **-27.35%** | -21.49% | -20% |

**Gap thực sự**: từ 30/30 trên giấy tờ → **17-23/30 trong thực tế** ⇒ tỷ lệ thành công tháng đầu **57-77%**, không phải 100%.

---

## 🎯 ROADMAP ĐỂ THỰC SỰ ĐẠT TARGET

### Phase 0 — Khắc phục R2 ngay (1-3 ngày)
1. **Đổi broker hoặc symbol**: Exness-MT5Trial7 spread 30 pips không khả thi. Thử:
   - Exness Raw Spread / Pro account (typically 1-3 pips trên XAUUSD)
   - IC Markets Raw (0.1-0.5 pips)
   - Pepperstone Razor
2. **Validate spread khi market open**: chạy `python -c "import requests; print(requests.get('http://localhost:5601/tick?symbol=XAUUSD').json())"` nhiều lần, đo spread distribution. Nếu p99 > 50 points → broker không phù hợp.
3. **Update `max_spread_points`** dựa trên broker thực: cap = max(50, p95_spread_observed).

### Phase 1 — Hardening (1 tuần)
4. **Tăng `min_paper_win_rate_pct` lên 55%** (R8 fix)
5. **Tăng `positive_r_threshold` lên 0.5** trong profitr model (R9 fix)
6. **Thêm Monte Carlo** module: `scripts/monte_carlo_fold_simulation.py` shuffle 1000× per fold, đo P5(final) và P95(max_dd). Block real-money nếu P5(final) < $1000.
7. **Rebuild stress test** với REAL broker spread distribution (đo từ demo loop logs).

### Phase 2 — Validation thực tế (2-4 tuần)
8. **Demo live 30 ngày**: hiện tại đã chạy, để chạy đủ 30 ngày + 100 trades minimum.
9. **Phân tích slippage_pips từ demo**: nếu avg > 5 pips → cần đổi broker hoặc tăng SL distance.
10. **Phân tích entry_drift_r distribution**: nếu p95 drift > 0.5R → tăng `max_entry_lag_seconds` xuống 30s.
11. **Cross-validate**: chạy WF với 7 fold rolling test (1 train year, 1 test month, hold out 6 tháng cuối) — kiểm tra recipe tự nó không thay đổi nhiều theo tuning.

### Phase 3 — Real account canary (sau Phase 2 pass)
12. **Bật full gates**: bỏ tất cả `--no-paper-gate --no-canary-gate --no-realistic-gate --allow-execute-without-paper`.
13. **Force `--canary-mode`** lot=0.01 trong 2 tuần đầu.
14. **Risk cap 1%** (default, không override).
15. **Daily review**: nếu 1 ngày DD > 5% → pause 24h, review.

### Phase 4 — Scale-up theo Kelly (3-6 tháng)
16. Risk = min(2%, 0.5 × Kelly) sau 100 real trades có WR ≥ 55%.
17. Nâng lot từ 0.01 → 0.02 → 0.05 mỗi 50 trades thành công.
18. Mục tiêu hợp lý: **$200 → $400 trong 60-90 ngày** (không phải 30 ngày).
19. $200 → $1200 trong 30 ngày = +6× = vi phạm Kelly cho mọi recipe có WR < 70%; chỉ khả thi nếu chấp nhận P(blow) > 30%.

---

## 🚦 VERDICT (hiện tại 2026-05-18)

| Tiêu chí | Status |
|:---|:---:|
| Lookahead bias trong WF | ✅ FIX rồi |
| Live preflight gates | ✅ FIX rồi |
| Risk cap & canary mode | ✅ FIX rồi |
| Session throttle | ✅ FIX rồi |
| Realistic friction gate | ❌ **FAIL** (17/30 only) |
| Broker spread compatibility | ❌ **FAIL** (308 vs 120 cap) |
| Hyperparameter overfit risk | ⚠️ MEDIUM |
| Monte Carlo stress | ⚠️ Chưa có |
| **TỔNG: Sẵn sàng REAL?** | ❌ **CHƯA** |
| **TỔNG: Sẵn sàng DEMO 30-day?** | ✅ Đang chạy đúng cách |

### 🎯 Mục tiêu $200→$1200 trong 30 ngày: KHẢ THI? 

**Theoretically**: Có, theo WF với target_balance_stop = 30/30. 

**Realistically**: 
- Median expected: $400-$600 (50-90% lợi nhuận, không phải 500%)
- P50 đạt $1200: ~17-23/30 fold = **57-77%** trên data cleaned, < 50% trên real broker với spread 30 pips
- P(blow $200 → < $100): 10-15% nếu xảy ra fold 1-style cold start với WR 46%

### 🎯 Đề xuất mục tiêu hiện thực:
- **30-day Phase 1 (demo)**: chứng minh recipe đạt $200→$400 (+100%) với DD < 15%
- **30-day Phase 2 (real canary 0.01)**: $200→$220 (+10%, lot nhỏ chỉ confirm pipeline)
- **30-day Phase 3 (real 1% risk)**: $220→$280 (+27%, scale Kelly)
- **30-day Phase 4 (real 2% risk)**: $280→$420 (+50%) nếu WR ≥ 55%

→ **$200 → $1200 trong 4-5 tháng real** thay vì 30 ngày, là mục tiêu thực tế và bảo vệ vốn.

---

*Re-audit by AURUM-X — 2026-05-18 19:30 +07:00*
*Files reviewed: rolling_v3_bridge_executor.py, generate_density_score_online_signal.py, apply_signal_density_risk_policy.py, export_mt5_profitr_model_manifest.py, manifest.json (frozen_density_score), stress_full.json, dd_throttle_sweep.json, bridge_live_state.json, bridge_live_last_report.json*
