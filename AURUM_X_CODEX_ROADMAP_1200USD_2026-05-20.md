# 🧭 AURUM-X — Roadmap cho Codex: tài khoản $1.200 trade trực tiếp (2026-05-20)

**Quyết định của chủ tài khoản**: vốn khởi điểm `$1.200`, giao dịch trực tiếp (không dùng quỹ cấp vốn). Mục tiêu cuối: thu nhập $1.200/tháng.

**Nhiệm vụ codex**: tìm cấu hình AN TOÀN nhất cho tài khoản $1.200, để nó compound bền vững.

---

## 0. ⚠️ LỖI GỐC PHẢI SỬA TRƯỚC: mọi sweep cũ dùng sai `deposit`

Tất cả sweep (`monthly180_*`, `aggressive_highdd_sweep`, 63-fold MT5 WF) đều chạy `deposit = $200`. Điều này làm méo toàn bộ kết quả:

### Toán min-lot XAUUSD (XAUUSDm, contract 100 oz)
- MT5 min lot `0.01`. SL ≈ ATR × 1.0 ≈ `6.7` giá → rủi ro ≈ `$6.7` cho 0.01 lot.
- Deposit `$200`: 0.01 lot = $6.7 = **3.35% risk** → KHÔNG thể trade dưới 3.35%. Bị ép vào vùng aggressive → đó là lý do "best recipe" toàn risk 7% và DD -41%.
- Deposit `$1.200`: 0.01 lot = $6.7 = **0.56% risk** → trade được ở 1-2% risk với 0.02-0.04 lot, độ phân giải tốt.

### Hệ quả
→ "Aggressive profile -41% DD" một phần là **artifact của tài khoản $200 quá nhỏ**, không phải bản chất strategy.
→ **Bắt buộc**: chạy lại mọi thứ với `deposit = $1.200`. Kết quả risk thấp sẽ tốt hơn nhiều so với khi ép trên $200.

---

## 1. BƯỚC 1 — Re-run 63-fold MT5 WF với `deposit = $1.200`

Chạy lại đúng candidate, đúng 63 fold (118-180), chỉ đổi `deposit` $200 → $1.200.

Đo lại: target pass, DD distribution, worst DD, lot size trung bình.
**Mục đích**: thấy DD/lot thay đổi thế nào khi không còn bị min-lot ép. Dự đoán: DD giảm rõ vì lot giờ chia mịn được.

---

## 2. BƯỚC 2 — Risk sweep trên harness $1.200 (biến số QUAN TRỌNG NHẤT)

Chạy candidate trên 63-fold MT5, `deposit=$1200`, quét:

| Risk/lệnh | Lot tương ứng (~) | Cần đo |
|:---:|:---:|:---|
| 1.0% | 0.01-0.02 | median final, target pass, worst DD, loss folds |
| 1.5% | 0.02-0.03 | " |
| 2.0% | 0.03-0.04 | " |
| 2.5% | 0.04-0.05 | " |
| 3.0% | 0.05-0.06 | " |

Lập bảng kết quả đầy đủ. **Chọn mức risk cao nhất mà vẫn đạt profile an toàn ở Bước 6.**
Không quét trên 3% — trên đó là vùng aggressive đã loại.

---

## 3. BƯỚC 3 — DD-kill replay ENFORCED (xóa ảo ảnh "0 loss fold")

MT5 tester KHÔNG thực thi kill switch — nó cho fold chạy xuyên -41% DD rồi hồi. Tài khoản thật KHÔNG vậy.

Chạy replay 63 fold trong sim Python, **thực thi** `max_dd_kill_pct`:
- Chạm -15% DD → đóng toàn bộ + dừng fold đó.
- Ghi nhận final balance tại điểm kill.

Đo: số loss fold THẬT (final < deposit), median final thật.
→ Đây là con số trung thực. "0/63 loss fold" hiện tại là giả.

---

## 4. BƯỚC 4 — Compound continuity sim (trải nghiệm tài khoản thật)

Hiện codex reset deposit mỗi fold. Tài khoản thật COMPOUND. Chạy liên tục:
- Bắt đầu `$1.200`, mang balance fold N → fold N+1.
- Risk USD/lệnh = `balance hiện tại × risk%` (lot tăng dần theo balance).
- DD-kill enforced (-15%). Nếu balance giảm > 15% từ đỉnh → dừng, ghi nhận.
- KHÔNG reset giữa chừng.

Đo qua 63 fold (5 năm 2021-2026):
- Đường cong equity ($1.200 → ?).
- Max DD trên toàn đường cong.
- Tháng tệ nhất (worst single-fold drawdown trong chuỗi compound).
- Số tháng âm liên tiếp dài nhất.

→ Đây là câu trả lời thật: "$1.200 sau 5 năm thành bao nhiêu, có lúc nào về dưới $1.000 không".

---

## 5. BƯỚC 5 — Tail OFF-switch cho fold DD sâu

Các fold DD > -30% (từ report cũ): `119,120,121,122,124,128,138,139,140,147,149,152,160,164` — có cụm liền (Q1 2021, giữa 2022...).

Việc cần làm:
1. Với mỗi tail fold, đo: ATR percentile, mật độ news, biên độ gap mở cửa tuần, spread trung bình.
2. Tìm đặc điểm chung → xây bộ lọc OFF-switch trong `rolling_v3_bridge_executor.py` / generator:
   - Skip trade khi `atr_percentile > X` (vol quá cao).
   - Giảm risk × 0.3 trong tuần có news mật độ cao.
   - Skip Friday sau giờ cutoff (đã có `block_friday_close`).
3. Chạy lại 63-fold với OFF-switch → so sánh worst DD trước/sau.

→ Giảm DD tail bằng filter hiệu quả hơn mọi selector.

---

## 6. ✅ TIÊU CHÍ PASS — "đúng hướng" nghĩa là gì (đo được)

Codex CHỈ được coi là đi đúng hướng khi cấu hình $1.200 đạt TẤT CẢ trên 63-fold MT5 (DD-kill enforced, compound sim):

| Tiêu chí | Ngưỡng pass |
|:--|:--:|
| Loss fold (DD-kill -15% enforced) | **0/63** |
| Worst DD (đường cong compound) | **≥ -15%** |
| Median fold return | **dương, ≥ +5%** |
| Số fold target ($final ≥ deposit × 1.05) | **≥ 40/63** |
| Compound terminal sau 63 fold | **> $1.200 nhiều lần, không lần nào về < $900** |

Nếu KHÔNG có mức risk nào đạt cả 5 → strategy chưa đủ tốt, cần cải thiện model/filter trước khi nói tới live.

---

## 7. 📈 LỘ TRÌNH THỰC TẾ cho tài khoản $1.200

Sau khi codex tìm được risk an toàn (giả định strategy nét `+8%/tháng` — codex phải xác nhận bằng Bước 4):

| Mốc | Balance (compound +8%/th) | Thu nhập tháng đó |
|:---:|:---:|:---:|
| Tháng 0 | $1.200 | — |
| Tháng 6 | ~$1.900 | ~$140 |
| Tháng 12 | ~$3.020 | ~$224 |
| Tháng 18 | ~$4.800 | ~$355 |
| Tháng 24 | ~$7.600 | ~$565 |
| Tháng 30 | ~$12.000 | **~$890** |
| Tháng 33 | ~$15.100 | **~$1.120** ✅ |

→ Với edge +8%/tháng, tài khoản $1.200 đạt mốc tạo ra **$1.200/tháng** sau khoảng **30-34 tháng** compound (không rút giữa chừng).
→ Nếu edge thật cao hơn (12-15%/tháng) thì rút ngắn còn ~18-22 tháng. Edge thật bao nhiêu — Bước 2+4 sẽ trả lời.

**Quan trọng**: các tháng đầu thu nhập rất thấp ($100-300). $1.200/tháng là kết quả của compound, không phải tháng đầu. Đây là vật lý của tài khoản nhỏ — không có đường tắt an toàn.

---

## 8. ⚡ NEXT ACTION (thứ tự cho codex)

1. **Re-run 63-fold MT5 WF, `deposit=$1200`** — sửa lỗi gốc.
2. **Risk sweep 1.0-3.0%** trên harness $1.200 — lập bảng.
3. **DD-kill replay enforced (-15%)** — đo loss fold thật.
4. **Compound continuity sim** bắt đầu $1.200 — đường cong equity 63 fold.
5. **Phân rã tail fold + xây OFF-switch** — chạy lại, so DD.
6. **Báo cáo**: mức risk nào đạt cả 5 tiêu chí Bước 6. Nếu không có → đề xuất cải thiện model.
7. **Không export sang live** cho đến khi 1 cấu hình pass đủ 5 tiêu chí.

---

## 🔑 TÓM TẮT 1 ĐOẠN GỬI CODEX

> Mọi sweep cũ chạy `deposit=$200` — sai. Trên $200, min-lot 0.01 ép risk tối thiểu 3.35% nên strategy "trông như cần 7%". Trên `$1.200`, 0.01 lot chỉ là 0.56% risk → trade được an toàn ở 1-2%. Bước 1: re-run 63-fold MT5 WF với `deposit=$1200`. Bước 2: risk sweep 1-3%. Bước 3: DD-kill replay enforced -15% (xóa ảo ảnh "0 loss fold" của MT5 tester). Bước 4: compound sim liên tục từ $1.200. Bước 5: OFF-switch cho tail fold DD sâu. Tiêu chí đúng hướng: 0 loss fold, worst DD ≥ -15%, median dương, compound không về dưới $900. Mục tiêu $1.200/tháng đạt được sau ~30 tháng compound nếu edge +8%/tháng — các tháng đầu thu nhập thấp, đó là vật lý tài khoản nhỏ.

---

*Roadmap by AURUM-X — 2026-05-20*
