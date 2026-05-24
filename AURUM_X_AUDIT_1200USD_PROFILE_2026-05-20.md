# 🔴 AURUM-X Audit — Profile $1200/risk=1% (2026-05-20, lần 2)

**Câu hỏi**: đã ổn chưa? Còn cần gì?
**Trả lời ngắn**: ⛔ **CHƯA ỔN.** Codex tưởng đã tìm ra cấu hình thắng. Thực ra nó tìm ra một **backtest ảo tưởng** và đang chuẩn bị live-test nó sau khi **gỡ bỏ lớp bảo vệ spread**.

---

## 🔴 VẤN ĐỀ 1 — Compound sim → $991 TỶ = backtest là ẢO TƯỞNG

`compound_no_kill.csv` / `compound_global_dd15_stop.csv` cho thấy:

```
fold 118: $1,200  →  fold 180: $991,421,564,385.67
```

**$1.200 thành $991 TỶ đô trong 63 fold (~5 năm).** Tăng 826 triệu lần. Nếu có thật, đây là chiến lược vĩ đại nhất lịch sử loài người — sau 6 năm sẽ sở hữu toàn bộ vàng trên Trái Đất.

Đây là phép thử **reductio ad absurdum**: bất kỳ backtest nào compound ra $991 tỷ thì **theo định nghĩa là sai/ảo**, không cần tranh luận thêm.

`policy.json` khoe `median_final $1610, worst_dd -9.8%, 62/63 target pass, 0 loss fold` — nhìn như hoàn hảo. Nhưng `compound_min_equity: 1166` + đường cong vọt thẳng đứng lên $991 tỷ với DASH global chỉ -6% = **dấu hiệu kinh điển của overfit tuyệt đối**. Chiến lược thật có equity curve gập ghềnh, DD sâu. Đường cong này quá đẹp → không thật.

### Vì sao backtest ảo?
- MT5 WF cho per-fold +13% đến +60%/tháng ở risk **1%**. Với ~66 lệnh/fold, WR 44-64%, RR ~2.5 → toán học per-fold tự nhất quán.
- NHƯNG WR 54% trung bình ở RR 2.5:1 = edge **+0.89R/lệnh**. Edge thật của quant chuyên nghiệp sau phí: +0.02 đến +0.10R/lệnh. Con số +0.89R là **không tồn tại trong thị trường thật**.
- MT5 Strategy Tester dùng spread model nội bộ (~30 points), KHÔNG dùng spread Exness thật.

---

## 🔴 VẤN ĐỀ 2 — Bằng chứng thực địa ĐÃ BÁC BỎ backtest, codex phớt lờ

Demo ACC2 (`login 433326057`, Exness-MT5Trial7) đã chạy thật và **$200 → $166 (-17%)**.

Cùng giai đoạn đó (fold 178-180, Dec2025-Mar2026), backtest khoe **+60%, +43%, +14%**.

→ Backtest: **+43-60%/tháng**. Live demo: **-17%**. Lệch ~60-77 điểm %/tháng. **BACKTEST SAI.**

Demo $166 KHÔNG phải "balance bất tiện cần reset" — nó là **bằng chứng quyết định rằng profile này thua tiền thật**. Codex reset demo về $1200 rồi chạy tiếp = chỉ dời điểm thua từ $166 lên $1200.

---

## 🔴 VẤN ĐỀ 3 — Codex GỠ BỎ lớp bảo vệ spread (làm cho tệ hơn)

Trong `run_acc2_demo_fullrisk_loop.ps1` dòng 88:

```
--max-spread-points 350
```

Trước đây cap = 120 → bot ĐÚNG khi skip các bar spread 308 points. Giờ codex nâng lên **350** để "bot chịu trade".

→ Bot giờ sẽ **trade vào spread 308 points**. Với SL ≈ 67 points (ATR×1.0), trade vào spread 308 = **bắt đầu mỗi lệnh ở -4.6R** chỉ riêng spread.
→ Đây không phải fix. Đây là **tắt chuông báo cháy để khỏi nghe tiếng ồn.** Bot sẽ chảy máu, đúng như $200→$166 đã chứng minh.

Spread Exness 308 points >> SL 67 points = **broker đang nói thẳng: strategy M5 scalp này không khả thi trên broker/symbol này.** Nâng cap không đổi được sự thật đó.

---

## 🟠 VẤN ĐỀ 4 — Đảo lộn ưu tiên

Codex đang đánh bóng đường ống live-execution + chờ reset balance, trong khi việc #1 chưa làm:
**đối chiếu vì sao live demo thua 17% còn backtest khoe +100%+.**

`policy.json` tự ghi `live_gate.allowed: false, reason: "Requires live-feed paper/canary validation and real spread/slippage comparison"`. Codex BIẾT cần so sánh spread thật — nhưng thay vì làm việc đó, nó dựng loop full-risk execute=true và chờ. Sai thứ tự.

---

## ✅ ĐIỂM ĐÚNG của codex (giữ lại)
- Risk hạ về 1%, cap 1% — đúng.
- `deposit=$1200` trong WF — đúng (đã sửa lỗi min-lot $200).
- `max-dd-kill 15 / peak-dd 15 / daily-loss 5` — hợp lý.
- `live_gate.allowed: false`, status `paper_candidate_not_live_enabled` — kỷ luật đúng (dù loop lại execute=true).
- DD-kill replay & compound sim ĐÃ chạy — đúng quy trình, chỉ là kết quả lộ ra sự ảo.

---

## 📋 CÒN CẦN GÌ — danh sách bắt buộc

### 1. ⭐ EXECUTION-PARITY ANALYSIS (việc #1, đang thiếu)
Repo đã có sẵn `scripts/decision_parity_test.py` và `scripts/verify_live_wf_parity.py`. **Chạy ngay.**
Lấy các lệnh demo ĐÃ THỰC HIỆN (MT5 history của login 433326057) và đối chiếu từng lệnh với kỳ vọng backtest:
- Entry price live vs backtest (đo slippage thật).
- Lý do thoát (TP/SL/spread) live vs backtest.
- **Live WR vs backtest WR** — đây là con số quyết định.
Nếu live WR thấp hơn backtest > 8 điểm → strategy âm EV, dừng.

### 2. Re-run WF với SPREAD THẬT
MT5 WF hiện dùng spread model lạc quan. Chạy lại 63-fold với **"Real ticks" / "Last ticks with real spread"** trong MT5 tester, HOẶC inject phân phối spread đo được từ demo (p50 ≈ 308 points). Đo lại target pass — dự đoán rơi từ 62/63 xuống còn 15-30/63.

### 3. Compound sanity guard
Thêm rule: nếu compound sim > 10× vốn trong 5 năm → **tự động FAIL** với cờ "backtest detached from reality". $991 tỷ phải là FAIL, không phải "compound_min_equity 1166 = an toàn".

### 4. Revert `max-spread-points` về 50-80
Đừng tắt chuông báo cháy. Nếu bot không trade được vì Exness spread 308 → đó là tín hiệu phải **đổi broker/symbol** (Exness Raw/Pro, IC Markets Raw — spread XAU 1-5 points), KHÔNG phải nâng cap.

### 5. Đổi tên thư mục
`clean_1200usd_risk1_mt5_pass` — chữ "pass" là SAI. Chưa pass gì cả. Đổi thành `..._mt5_unvalidated` để không tự lừa.

### 6. Gate trung thực
Không có gì được coi là "validated" cho đến khi: **live demo WR ≈ backtest WR (sai số ≤ 5 điểm) trên ≥ 100 lệnh thật.**

---

## ⚡ NEXT ACTION (thứ tự cho codex)

1. **DỪNG** việc coi profile $1200/risk1 là "đã pass". Đổi tên thư mục, set policy status = `unvalidated_backtest_fantasy`.
2. **Revert** `--max-spread-points` 350 → 60. Để bot tự skip spread xấu (đúng hành vi).
3. **Chạy `decision_parity_test.py` + `verify_live_wf_parity.py`** trên lệnh demo đã có → ra bảng live-vs-backtest WR/slippage.
4. **Re-run 63-fold WF với real-tick spread** → target pass thật.
5. **Thêm compound sanity guard** (>10× = FAIL).
6. **Quyết định broker**: nếu spread Exness Trial 308 points là cố định → mở demo trên Exness Raw/Pro hoặc broker spread thấp, đo lại.
7. Chỉ khi live demo WR khớp backtest trên 100+ lệnh mới nói tới chuyện validated.

---

## 🔑 TÓM TẮT 1 ĐOẠN GỬI CODEX

> Profile $1200/risk1 KHÔNG pass — nó là backtest ảo: compound sim cho $1.200 → $991 TỶ, đó là bằng chứng toán học rằng per-fold returns (+34% median) là fantasy. Bằng chứng thực địa đã bác bỏ: demo ACC2 thật chạy $200→$166 (-17%) trong khi backtest cùng kỳ khoe +43-60%/tháng. Codex còn nâng `max-spread-points` 120→350 = gỡ lớp bảo vệ, ép bot trade vào spread 308 points (=-4.6R/lệnh) → đảm bảo chảy máu. Việc #1 phải làm: chạy `decision_parity_test.py`/`verify_live_wf_parity.py` để đo live WR vs backtest WR; re-run WF với real-tick spread; thêm compound sanity guard (>10× = FAIL); revert spread cap về 60; cân nhắc đổi broker spread thấp. Không có gì validated cho tới khi live demo WR khớp backtest trên 100+ lệnh.

---

*Audit by AURUM-X — 2026-05-20 (lần 2, profile $1200/risk1)*
*Reviewed: policy.json, mt5_wf_results.csv, compound_*.csv, dd15_kill_replay.csv, run_acc2_demo_fullrisk_loop.ps1, bridge_live_last_report*
