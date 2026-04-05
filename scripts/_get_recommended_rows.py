"""Extract full parameter rows for ACC1 and ACC2 recommended configs."""
import pandas as pd

# ACC1 recommended: threshold=0.88, risk=0.12, tp=4.0, template=minimal, pause=2, cool=16
df1 = pd.read_csv("outputs/wf_pf3_acc1_r1_round4.csv")
mask1 = (
    (df1["threshold"] == 0.88)
    & (df1["risk_per_trade"] == 0.12)
    & (df1["take_profit_rr"] == 4.0)
    & (df1["template"] == "minimal")
    & (df1["consecutive_loss_pause_count"] == 2)
    & (df1["consecutive_loss_cooldown_bars"] == 16)
)
row1 = df1[mask1].sort_values("total_trades", ascending=False).head(1)
print("=== ACC1 RECOMMENDED ROW ===")
for col in row1.columns:
    print(f"  {col}: {row1[col].values[0]}")

# ACC2 recommended: threshold=0.90, risk=0.10, tp=8.0, template=focus_hours, pause=2, cool=32
df2 = pd.read_csv("outputs/wf_pf3_acc2_r1_round3.csv")
mask2 = (
    (df2["threshold"] == 0.90)
    & (df2["risk_per_trade"] == 0.10)
    & (df2["take_profit_rr"] == 8.0)
    & (df2["template"] == "focus_hours")
    & (df2["consecutive_loss_pause_count"] == 2)
    & (df2["consecutive_loss_cooldown_bars"] == 32)
)
row2 = df2[mask2].sort_values("total_trades", ascending=False).head(1)
print()
print("=== ACC2 RECOMMENDED ROW ===")
for col in row2.columns:
    print(f"  {col}: {row2[col].values[0]}")
