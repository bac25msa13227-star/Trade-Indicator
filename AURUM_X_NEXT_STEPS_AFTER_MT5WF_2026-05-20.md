# 🧭 AURUM-X — Bước tiếp theo cho Codex sau MT5 WF (2026-05-20)

**Bối cảnh**: Codex đã đưa candidate aggressive (risk 7%) qua MT5 WF thật, chạy được `63 fold` (118-180, 2021-2026) trên history broker Exness thật.

**Kết quả MT5 WF**: target 30/63 (47.6%), DD fail -20%: 32/63, DD fail -30%: 14/63, median $1157.79, mean $910.65, min $265.14, **max $1202.12**, worst DD **-41.81%**.

---

## 1. ✅ ĐIỀU CODEX LÀM ĐÚNG

- Đã đưa friction THẬT vào vòng lặp (MT5 broker history, không còn proxy).
- Loaded signal mismatch = 0 → pipeline tín hiệu khớp.
- Báo cáo trung thực 47.6%, không tô hồng thành 100%.

→ Đây là tiến bộ thật. Lần đầu tiên có con số đáng tin.

---

## 2. 🔴 3 SỰ THẬT MT5 WF VỪA XÁC NHẬN

### 2.1 — Median fold VẪN THUA target
`median final = $1157.79 < $1,200`. **Fold điển hình thất bại** ngay cả ở risk 7% aggressive. Đúng như audit trước đã dự báo: friction thật cắt đuôi phải.

### 2.2 — `max final = $1,202.12` → upside đã bị KHÓA, downside mở toang
Vì `target stop = $1200`, mọi fold thắng bị chặn ở ~$1200. Nhưng fold thua rơi tự do đến $265.
→ Đây là shape **lợi nhuận bất đối xứng xấu**: thắng tối đa +$1000, thua tối đa -$935, chỉ 47.6% chạm cap.
→ `mean $910 < median $1157` = lệch TRÁI: vài fold thua nặng kéo trung bình xuống. Strategy đang "ăn xôi, trả thịt".

### 2.3 — Worst DD -41.81%, median DD -20.22%
**Fold ĐIỂN HÌNH (median) đã breach gate -20%.** Một nửa số tháng dính DD ≥ 20%.

---

## 3. ⚠️ ẢO ẢNH SỐ 1: "0/63 LOSS FOLDS" LÀ GIẢ

Codex kết luận *"no fold ended below $200 → không cháy vốn"*. **Sai nguy hiểm.**

MT5 Strategy Tester **KHÔNG thực thi kill switch** `max_dd_kill_pct`. Nó để fold chạy xuyên qua -41% DD rồi hồi phục về > $200.

Nhưng tài khoản THẬT có `max_dd_kill_pct = 20` trong `rolling_v3_bridge_executor.py` → khi DD chạm -20%, bot **force-close toàn bộ + halt**. Fold đó **chốt lỗ tại -20%**, không có cơ hội hồi.

→ 32 fold breach -20% DD: trên tài khoản thật, phần lớn bị cắt tại -20% → biến thành **loss fold thật**.
→ "0 loss folds" chỉ tồn tại vì tester cho lệnh chạy qua -41% DD. **Con số loss fold thật chưa được đo.**

---

## 4. ⚡ NEXT ACTION LIST CHO CODEX (đúng thứ tự ưu tiên)

### 🥇 BƯỚC 1 — Replay 63 fold VỚI DD-kill ENFORCED (quan trọng nhất)
Chạy lại đúng 63 fold đó trong sim Python (`rolling_v3_bridge_executor` dry-run hoặc replay engine) với `max_dd_kill_pct = 20` **được thực thi**: chạm -20% → close all + stop fold.

Đo: số loss fold THẬT, median final thật, % fold sống.
**Đây là con số quyết định** — biến "0 loss giả" thành sự thật. Dự đoán AURUM-X: 10-20/63 fold thành loss fold thật.

### 🥈 BƯỚC 2 — Risk sweep trên CHÍNH 63-fold MT5 harness
Codex giờ ĐÃ CÓ harness MT5 chạy được. Chạy lại đúng candidate này ở **risk = 1.5% / 2% / 3% / 5%** trên 63 fold.

Lập bảng:

| Risk | Median final | Target pass | DD fail >20% | Worst DD | Loss folds (DD-kill on) |
|:----:|:---:|:---:|:---:|:---:|:---:|
| 1.5% | ? | ? | ? | ? | ? |
| 2.0% | ? | ? | ? | ? | ? |
| 3.0% | ? | ? | ? | ? | ? |
| 5.0% | ? | ? | ? | ? | ? |
| 7.0% | $1157 | 30/63 | 32/63 | -41.81% | (cần đo) |

→ Tìm mức risk mà **worst DD ≥ -18%** và **loss folds ≤ 3/63**. Đó là mức risk deploy được. Đây là biến số quan trọng nhất và VẪN chưa được quét trên harness thật.

### 🥉 BƯỚC 3 — Compound continuity sim (trải nghiệm tài khoản thật)
Codex đang reset $200 mỗi fold. Tài khoản thật COMPOUND. Chạy liên tục:
- Bắt đầu $200, mang balance fold N → fold N+1.
- Risk scale theo balance (vd risk USD = balance × risk%).
- Khi balance < $200 hoặc DD-kill → ghi nhận "blow-up", restart $200 (mô phỏng nạp lại).

Đo: sau 63 fold (5 năm) terminal balance thật, số lần phải nạp lại, max DD đường cong equity.
→ Đây mới là câu trả lời "$200 thành bao nhiêu", không phải 63 con số rời rạc.

### 4️⃣ BƯỚC 4 — Phân rã tail folds
Các fold DD > -30%: `119,120,121,122,124,128,138,139,140,147,149,152,160,164`.
- 119-122 cụm liền (≈ Q1 2021), 138-140 cụm, 147-152 cụm.
- Map ra tháng lịch sử + đo ATR percentile, news density, gap mở cửa của các fold này.
- Nếu chúng chung đặc điểm vol cao → thiết kế **OFF-switch** (skip trade khi ATR percentile > X, hoặc giảm risk ×0.3 trong regime đó). Đây là cách giảm DD tail HIỆU QUẢ hơn mọi selector.

### 5️⃣ BƯỚC 5 — Quyết định shape: bỏ target cap hay không
`max final $1202` = target_stop khóa upside. Hai lựa chọn:
- **Bỏ target_stop**: cho winner chạy → đuôi phải sống lại, nhưng cần đo DD có tệ hơn không.
- **Giữ target_stop**: chấp nhận shape bất đối xứng xấu → strategy này chỉ là "cố đạt $1200 rồi nghỉ", EV thấp.
Chạy A/B 63 fold với/không target_stop, so sánh EV và DD.

---

## 5. 🚦 DECISION GATE — Khi nào codex được phép sang live

Candidate hiện tại (risk 7%) **KHÔNG đạt** bất kỳ điều kiện nào dưới đây. Chỉ deploy real-money khi:

| Tiêu chí | Ngưỡng | Hiện tại (risk 7%) |
|:---|:---:|:---:|
| Loss folds (DD-kill enforced) | ≤ 3/63 | ❓ chưa đo (dự đoán 10-20) |
| Worst DD | ≥ -18% | ❌ -41.81% |
| Median final | ≥ $1,200 | ❌ $1,157 |
| DD fail > -20% | ≤ 10/63 | ❌ 32/63 |
| Compound terminal (63 fold) | > $200 không phải nạp lại | ❓ chưa đo |

→ Hiện tại candidate này chỉ là **demo aggressive**, đúng như codex tự nhận. Không được sang real.

---

## 6. 🎯 SỰ THẬT VỀ TARGET $200→$1200/THÁNG

MT5 WF thật đã chốt: ở risk 7% (mức cao nhất hợp lý), **median fold = $1,157 < $1,200**. Nghĩa là:
- Target $1200/fold **không đạt được ổn định** kể cả với đòn bẩy aggressive.
- Đẩy risk cao hơn 7% → DD vượt -50% → cháy chắc chắn khi DD-kill bật.

**Kết luận bất khả xâm phạm**: $200→$1200 trong 1 fold (30 ngày) là **không khả thi an toàn**. Bằng chứng là dữ liệu thật 63 fold, không phải ý kiến.

### Con đường THẬT đến $1200:
Sau khi BƯỚC 1-3 xong, codex sẽ có mức risk an toàn (dự đoán ~2%). Ở mức đó:
- Median fold ≈ +15-30% (vd $200 → $240-260), loss fold < 5%.
- Compound: $200 → $1,200 cần ~7-10 fold thắng (≈ 8-12 tháng) — với P(sống sót) > 90%.
- Đó là target codex NÊN khóa vào: **"đạt $1200 bằng compound 8-12 tháng, không loss fold, DD < 15%"**.

---

## 🔑 TÓM TẮT 1 ĐOẠN GỬI CODEX

> MT5 WF thật là tiến bộ — nhưng nó vừa xác nhận strategy chưa deploy được: median fold $1157 < target, worst DD -41.81%, và "0 loss folds" là GIẢ vì MT5 tester không thực thi kill switch -20%. Bước 1 bắt buộc: replay 63 fold với DD-kill enforced để biết số loss fold thật. Bước 2: risk sweep 1.5-5% trên chính harness MT5 này (vẫn chưa làm). Bước 3: compound continuity sim thay vì reset $200 mỗi fold. Bước 4: phân rã 14 tail fold DD>-30% để làm OFF-switch. Đừng đẩy risk cao hơn 7% — DD sẽ vượt -50%. Target $200→$1200/30 ngày đã được dữ liệu thật chứng minh là bất khả thi an toàn; khóa lại target mới: $1200 bằng compound 8-12 tháng với loss fold = 0.

---

*Advice by AURUM-X — 2026-05-20 (sau MT5 WF 63-fold)*
*Reviewed: MT5_WF_REPORT_118_180.md, monthly180_aggressive_highdd_sweep_20260520*
