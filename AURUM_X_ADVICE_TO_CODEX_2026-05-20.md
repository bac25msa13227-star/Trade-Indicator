# 🧭 AURUM-X — Lời khuyên định hướng cho Codex (2026-05-20)

**Bối cảnh**: Codex vừa sweep/prune 180 folds, kết luận "cần build pairwise/ranking selector để đạt ≥170/180 strict rồi mới export MT5 WF".

**Phán quyết AURUM-X**: ⛔ **Hướng đi này SAI về mặt thống kê. Dừng lại trước khi đốt thêm thời gian.**

---

## 1. ⚠️ ORACLE 177/180 LÀ ẢO ẢNH THỐNG KÊ — KHÔNG PHẢI MỤC TIÊU

Codex đang hiểu: *"Universe có nhiều recipe tốt theo regime, selector chưa chọn đúng."*

**Đây là sai lầm kinh điển: nhầm nhiễu với cấu trúc.**

### Bằng chứng toán học

Universe hiện có **112 candidate configs** (16+24+36+36). Oracle = chọn candidate tốt nhất cho MỖI fold *sau khi biết kết quả*.

Với 112 candidate, xác suất oracle pass 1 fold bất kỳ:

| Giả định avg candidate pass rate | P(≥1/112 candidate pass fold đó) | Oracle kỳ vọng |
|:---:|:---:|:---:|
| 10% (thấp như family D) | 0.99999 | **180/180** |
| 15% | 1.00000 | **180/180** |
| 25% | 1.00000 | **180/180** |

→ **Oracle 177/180 là KẾT QUẢ BẮT BUỘC khi có 112 candidate.** Codex thực ra còn nhận được THẤP HƠN kỳ vọng thuần ngẫu nhiên (177 < 180) — nghĩa là có 3 fold mà *không candidate nào* sống nổi.

Oracle KHÔNG đo "universe có recipe tốt". Nó chỉ đo "universe đủ lớn". Đây là **maximum-of-N effect / multiple-comparisons trap**: cứ thêm candidate thì oracle tự lên gần 180, dù mỗi candidate chỉ là tung đồng xu.

### Hệ quả

Mục tiêu *"selector sạch phải đạt ≥170/180 strict"* = yêu cầu selector tái tạo lại oracle từ dữ liệu quá khứ. **Bất khả thi**, vì oracle được sinh từ NHIỄU (đường giá cụ thể của từng tháng), không phải từ quan hệ regime→winner ổn định.

Codex sẽ tune pairwise selector mãi mãi và không bao giờ chạm 170/180. Mỗi lần thêm feature/độ phức tạp, nó chỉ overfit thêm in-sample.

---

## 2. 📏 TRẦN THẬT CỦA SELECTOR LÀ ~73/180 — VÀ ĐÓ MỚI LÀ TÍN HIỆU

Ba selector thiết kế độc lập đều rơi vào cùng một dải:

| Selector | Strict pass |
|:---|:---:|
| Rolling prior | 73/180 |
| Regime-distance | 69/180 |
| ML prior | 66/180 |

**Ba cách tiếp cận khác nhau hội tụ về 66-73 — ĐÓ là thành phần thật sự dự đoán được.** Phần còn lại (đến 177) là nhiễu. Một pairwise/ranking selector chỉ là "ML selector phức tạp hơn" → in-sample đẹp hơn (~75-80), out-of-sample vẫn ~70. Thêm "candidate metadata features" = thêm sức chứa overfit, không thêm tín hiệu.

**Quy luật**: khi 3 mô hình khác họ cho cùng đáp số, đáp số đó là trần. Đừng đập đầu vào trần.

---

## 3. 🔬 CODEX ĐANG TỐI ƯU SAI BIẾN SỐ — BỆNH NẰM Ở RISK 7%, KHÔNG PHẢI SELECTOR

Đọc `prune_a_extratrees_stride3/best_summary.json` (best single recipe, risk **7%**):

```
target_folds:        85/180   (47%)
loss_folds:          24/180   (13%)
dd_fail_folds:       84/180   (47%)
median_final:        $1,019.59   ← THẤP HƠN target $1200
mean_final:          $2,647.63
best_final:          $76,811.65  ← 1 fold nhân 384 lần
worst_dd:            -25.19%
win_rate:            56.08%
fold_01: final $163, dd -25.19%, 6 trades
fold_07: final $163, dd -25.19%, 6 trades
fold_08: final $278, dd -25.19%, 18 trades
```

### Đây là phân phối VÉ SỐ, không phải chiến lược

- **Median $1,019 < target $1,200** → fold ĐIỂN HÌNH thất bại. 85/180 pass hoàn toàn nhờ đuôi phải.
- **Mean $2,647 >> median $1,019** → lệch phải cực mạnh: vài fold nổ $20k-76k kéo trung bình lên.
- **worst_dd = -25.19% lặp lại y hệt** ở fold 1, 7, 8... và cả 4 family (-25.19/-25.15/-25.17). Đó là chữ ký của **7% risk × chuỗi thua trên fold ít lệnh**: 6 lệnh, thua 3-4 lệnh liên tiếp ở 7% compounding ≈ -25%.

### Kết luận sắt đá

`target=85/180` của "best recipe" KHÔNG phải edge — nó là **biến động của đòn bẩy 7%**. Strip vài fold trúng số ra thì chiến lược là đồng xu thua phí giao dịch.

Family B (min-lot-aware, sát thực tế hơn) chỉ đạt 67/180 — vì min-lot-aware **giết bớt vé số**. Đó là con số thật hơn.

→ Codex sweep recipe + selector trong khi gốc rễ là **risk sizing**. Không selector nào cứu được phân phối lottery.

---

## 4. ✅ ĐIỀU SELECTOR THỰC SỰ LÀM ĐƯỢC — VÀ NÊN ĐỔI MỤC TIÊU THEO ĐÓ

So sánh:

| | Target | Loss folds |
|:---|:---:|:---:|
| Best single recipe | 85 | **24** |
| Rolling selector sạch | 80 | **17** |

Selector **không** tăng target (80 < 85) nhưng **giảm loss fold 24→17 (-29%)**.

→ Giá trị thật của selector là **TRÁNH FOLD NỔ**, không phải "chọn người thắng". Và "candidate nào KHÔNG nổ trong regime này" là mục tiêu **ổn định hơn nhiều** giữa các regime so với "candidate nào thắng to nhất".

**Đổi objective của selector**:
- ❌ Bỏ: "rank candidate theo final balance / chọn winner"
- ✅ Thay: "phân loại candidate AN TOÀN vs candidate NỔ trong pre-regime hiện tại" — bài toán nhị phân, ít nhiễu hơn, learnable hơn.

---

## 5. 🧪 3 TEST QUYẾT ĐỊNH — CHẠY TRƯỚC KHI VIẾT BẤT KỲ SELECTOR MỚI NÀO

### Test A — Random selector baseline (30 phút)
Gán NGẪU NHIÊN 1 trong 112 candidate cho mỗi fold, lặp 1000 lần, đo phân phối strict pass.
- Nếu random ≈ 60-73/180 → selector "thông minh" hiện tại **cộng giá trị ≈ 0**. Bỏ cả hướng selector.
- Nếu random ≈ 35-45/180 → selector thật sự thêm ~30 fold giá trị → đáng tiếp tục, nhưng trần vẫn ~73.

### Test B — Walk-forward chính selector (1-2 giờ)
Tune selector CHỈ trên fold 1-120. Đóng băng. Đo trên fold 121-180 chưa từng chạm.
- Nếu train 75% nhưng test 40% → selector overfit, vô giá trị.
- Bất kỳ selector nào không qua được Test B thì **cấm export MT5 WF**.

### Test C — Shuffled-oracle (15 phút)
Lấy đúng các lựa chọn oracle nhưng hoán vị ngẫu nhiên giữa các fold. Nếu vẫn ~150/180 → xác nhận oracle thuần là hiệu ứng kích thước universe.

**Nếu Test A cho random ≥ selector, hoặc Test B cho test-set sụp → DỪNG hướng selector hoàn toàn.**

---

## 6. 🩹 BLOCKER THẬT: DRAWDOWN, KHÔNG PHẢI SELECTION

`dd_fail_folds = 84/180` cho best recipe; selector tốt nhất vẫn 57/180 DD fail. **~47% fold vỡ gate -20% DD.** Ngay cả oracle worst DD -19.56% (sát mép).

### Việc phải làm trước selector:
1. **Xác định fold nào LUÔN vỡ DD** ở mọi family. `-25.19%` lặp lại → khả năng là cùng vài tháng lịch sử (2008 crisis, 2013-04 gold crash -13%/2 ngày, 2020-03 COVID, 2025-04). Liệt kê 3 fold oracle fail (43, 60, 86) + các fold -25% chung.
2. Nếu 3-5 tháng cố định luôn gây -25% → đó là **tail event không recipe nào sống nổi**. Cần **volatility OFF-switch** (ATR percentile cap, news blackout, gap filter), KHÔNG phải selector.
3. **Hạ risk** từ 7% → quét lại 1.5% / 2% / 3%. Đo lại median final + DD. Dự đoán: ở 2%, đuôi phải sụp (median ~$280-350, không phải $1200) NHƯNG loss fold rơi về <5% và worst DD <15%.

---

## 7. 🎯 MÂU THUẪN GỐC PHẢI ĐỐI MẶT — TARGET vs SỐNG SÓT

$200 → $1,200 trong 30 ngày = **+500%/tháng**. Toán học bắt buộc:
- +500%/tháng với WR 56% và RR ~2.5:1 → đòi hỏi risk ≥ 6-7% per trade.
- Risk 7% → phân phối lottery → 13% fold mất tiền, 47% fold vỡ DD, median fold THUA target.

**Không tồn tại recipe/selector nào hóa giải mâu thuẫn này.** Target đòi risk 7%; sống sót đòi risk ≤2%. Codex đang sweep universe để né sự thật này — nhưng dữ liệu 180-fold đã nói rõ.

### Hai lựa chọn trung thực:

**A. Giữ target $1200/30 ngày** → chấp nhận đây là "vé số có kỷ luật": P(thành công 1 fold) ≈ 40-47%, P(cháy tài khoản) ≈ 13%. Phải nói rõ điều này với chủ tài khoản.

**B. Đổi sang target sống sót + compound** (AURUM-X khuyến nghị):
- Risk 1.5-2%, 1 recipe robust nhất (không cần selector).
- Mục tiêu mỗi fold: +15-30% với loss fold <5%, DD <12%.
- $200 → $1,200 trong **5-7 fold** (5-7 tháng) bằng compound, không phải 1 fold.
- Đây là con đường DUY NHẤT có P(cháy) < 5%.

---

## 8. ⚡ NEXT ACTION LIST CHO CODEX (đúng thứ tự)

1. **🚫 ĐÓNG BĂNG mọi việc build pairwise/ranking selector.** Không viết thêm dòng selector nào cho đến khi qua Test A + B.

2. **▶️ Chạy Test A (random selector baseline)** — nếu random ≥ 65/180 thì cả ý tưởng "selector sạch" là ngõ cụt; báo cáo ngay.

3. **▶️ Chạy Test B (walk-forward selector trên 120/60 split)** — bất kỳ selector tương lai nào cũng phải pass test này mới được export.

4. **▶️ Risk sweep**: chạy lại best recipe family A ở risk 1.5% / 2.0% / 3.0% / 5.0% trên 180 fold. Lập bảng: median final, loss folds, DD fail folds, worst DD theo từng mức risk. **Đây là biến số quan trọng nhất, chưa được quét.**

5. **▶️ Phân tích tail folds**: liệt kê mọi fold có DD ≤ -22% ở cả 4 family. Map ra tháng lịch sử. Kiểm tra chúng có chung đặc điểm (ATR spike, news, gap) → thiết kế OFF-switch.

6. **▶️ Đổi objective selector** (nếu Test A/B cho phép tiếp tục): từ "rank winner" sang "phân loại blow-up vs safe" — binary classifier dự đoán `loss_fold OR dd_fail` cho từng candidate, chọn candidate có P(blow-up) thấp nhất.

7. **▶️ Thu nhỏ universe**: từ 112 → giữ lại 3-5 recipe mà MỖI recipe tự đứng pass ≥55% fold standalone với <8% loss fold. Nếu KHÔNG có recipe nào đạt tiêu chí đó → kết luận trung thực: **edge chưa đủ, không deploy real money**.

8. **▶️ Chỉ export MT5 WF khi**: (a) qua Test B, (b) risk đã hạ về mức loss_folds ≤ 5/180 và DD fail ≤ 20/180, (c) recipe đứng vững standalone — KHÔNG dùng tiêu chí "≥170/180 strict" (bất khả thi).

---

## 9. 📐 ĐIỀU CHỈNH TIÊU CHÍ GATE CỦA CODEX

| Tiêu chí Codex đề ra | Vấn đề | Tiêu chí AURUM-X đề xuất |
|:---|:---|:---|
| Selector sạch ≥170/180 strict | Bất khả thi (oracle là nhiễu) | Selector test-set (Test B) ≥ best single recipe test-set |
| ≤2 loss folds | Chỉ đạt được bằng risk thấp | loss_folds ≤ 5/180 ở risk đã hạ |
| Đuổi theo oracle | Oracle = artifact 112-candidate | Bỏ oracle khỏi mọi mục tiêu |
| (chưa có) | Risk 7% chưa bị chất vấn | DD fail ≤ 20/180, worst DD ≥ -18% |
| (chưa có) | — | Median final ≥ $1200 (fold điển hình phải thắng, không chỉ đuôi phải) |

---

## 🔑 TÓM TẮT 1 ĐOẠN GỬI CODEX

> Oracle 177/180 không phải mục tiêu — nó là hệ quả toán học bắt buộc của việc có 112 candidate (P(oracle pass) ≈ 1.0 ngay cả khi mỗi candidate chỉ là đồng xu). Selector sạch hội tụ ~70/180 qua 3 thiết kế độc lập — đó là trần thật. Pairwise/ranking selector sẽ không phá trần, chỉ overfit thêm. Bệnh thật nằm ở **risk 7%** tạo phân phối vé số (median fold $1019 < target $1200, mean $2647, đuôi $76k, DD wall -25%) và ở **drawdown** (47% fold vỡ gate). Trước khi viết thêm selector: chạy Test A (random baseline) + Test B (walk-forward selector) + risk sweep 1.5-5%. Đổi objective selector từ "chọn winner" sang "tránh blow-up". Và đối mặt sự thật: $200→$1200/30 ngày đòi risk 7% = 13% xác suất cháy; con đường sống sót là risk 2% + compound qua 5-7 tháng.

---

*Advice by AURUM-X — 2026-05-20*
*Reviewed: PROXY_PRUNE_REPORT_20260520.md, prune_a..d best_summary.json (112 configs), monthly180 sweep structure*
