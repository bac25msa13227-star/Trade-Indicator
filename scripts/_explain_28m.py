"""Explain why $28M best-net result is a math artifact, not reality."""

risk = 0.25   # 25% per trade
tp_rr = 4.0   # TP = 4:1
n_wins = 91
n_losses = 15

print("=== CONFIG GÂY RA $28M ===")
print(f"  compound_cap = 0 (UNLIMITED - không giới hạn)")
print(f"  risk_per_trade = {risk*100}% (25% tài khoản / lệnh)")
print(f"  take_profit_rr = {tp_rr}:1")
print(f"  trades = {n_wins + n_losses}, wins = {n_wins}, losses = {n_losses}")
print(f"  win_rate = {n_wins/(n_wins+n_losses)*100:.1f}%")
print()

# Simulate win multiplier per trade
win_mult = 1 + risk * tp_rr  # 1 + 25% * 4 = 2.0 (double each win)
loss_mult = 1 - risk          # 1 - 25% = 0.75 (lose 25%)
print(f"  Mỗi WIN:  balance x{win_mult:.2f} (tăng gấp đôi vì 25% * 4RR = +100%)")
print(f"  Mỗi LOSS: balance x{loss_mult:.2f} (mất 25%)")
print()

# Simple simulation: all wins first then losses
bal = 200.0
for _ in range(n_wins):
    bal *= win_mult
for _ in range(n_losses):
    bal *= loss_mult
print(f"91 wins rồi 15 losses: ${bal:,.0f}")

# More interspersed
bal2 = 200.0
import random
random.seed(42)
seq = [True]*n_wins + [False]*n_losses
random.shuffle(seq)
for w in seq:
    if w:
        bal2 *= win_mult
    else:
        bal2 *= loss_mult
print(f"Ngẫu nhiên xen kẽ:    ${bal2:,.0f}")
print()

print("=== VẤN ĐỀ THỰC TẾ ===")
print()
print("1. MARGIN CALL NGAY LẬP TỨC:")
bal3 = 200.0
for i in range(4):
    bal3 *= loss_mult
    print(f"   Sau {i+1} thua liền: ${bal3:.2f} (còn {bal3/200*100:.1f}% tài khoản)")
print()

print("2. WR=85.85% TRONG 106 LỆNH LÀ OVERFITTING:")
print("   Model trained trên CÙNG dữ liệu với WF folds")
print("   WF 8 folds x 4000 bars = backtest, không phải out-of-sample thực sự")
print("   Threshold=0.88 rất cao -> chỉ lấy những tín hiệu CHẮC NHẤT trong history")
print("   Trong live trading, win rate thực sẽ thấp hơn đáng kể")
print()

print("3. TẠI SAO BACKTEST CÓ THỂ CHO $28M:")
print("   200 -> x2^91 x(0.75)^15 = TOÁN HỌC THUẦN TÚY")
print(f"   2^91 = {2**91:.2e}")
print(f"   0.75^15 = {0.75**15:.4f}")
print(f"   Tổng: $200 x {2**91 * 0.75**15:.2e} = ${200 * 2**91 * 0.75**15:,.0f}")
print()

print("=== SO SÁNH CONFIG ĐƯỢC KHUYẾN NGHỊ (compound_cap=50) ===")
print("   risk=0.07 (7%), TP=7.0, cap=50 -> THỰC TẾ HƠN NHIỀU")
print("   Mỗi win: x1.49 (+49%), mỗi loss: x0.93 (-7%)")
print("   cap=50 -> balance tối đa = fold_start * 50 -> CÓ GIỚI HẠN")
print("   Net $815k với 322 trades -> hợp lý hơn nhiều")
