"""Debug script to understand filtering behavior for acc2 v2 config."""
from pathlib import Path
from xauusd_ai.config import load_settings
from xauusd_ai.strategies.hybrid import HybridStrategy
from xauusd_ai.backtesting.engine import simulate_prediction_backtest
from xauusd_ai.execution.risk import RiskManager
import pandas as pd

s = load_settings(Path("configs/benchmarks/acc2_pf3v2_net63k_dd1864.yaml"))
s.risk.kill_switch_enabled = False

print("=== ACC2 Config ===")
print("risk.min_confidence:", s.risk.min_confidence)
print("strategy.signal_threshold:", s.strategy.signal_threshold)
print("sideway_min_confidence:", s.strategy.sideway_min_confidence)
print("volatile_min_confidence:", s.strategy.volatile_min_confidence)
print("adx_gate_enabled:", s.strategy.adx_gate_enabled)
print("min_strategy_score:", s.strategy.min_strategy_score)
print("silver_bullet_enabled:", s.strategy.silver_bullet_enabled)
awh = s.strategy.allowed_weekday_hours_utc
print("monday allowed hours:", sorted(awh.get("Monday", [])))
print()

strategy = HybridStrategy(s)

# Test time filter
for h in [0, 5, 7, 8, 9, 13, 20, 21, 22]:
    ts = pd.Timestamp(f"2026-01-05 {h:02d}:00:00+00:00")
    blocked, reason = strategy._blocked_by_time(ts)
    print(f"  hour={h}: blocked={blocked} reason={reason}")

print()
# Test full should_allow_row
def make_row(hour=7, prob=0.85, strategy_score=0.9, adx=25.0, regime=1):
    return pd.DataFrame([{
        "split": "test",
        "time": pd.Timestamp(f"2026-01-05 {hour:02d}:00:00+00:00"),
        "prediction": 1, "probability": prob, "trade_side": "buy",
        "close": 2000.0, "future_return": 0.001, "directional_return": 0.001,
        "realized_rr": 10.0, "bars_held": 1, "strategy_score": strategy_score,
        "volatility_regime": regime, "trend_alignment": 1, "session_spread_mult": 1.0,
        "adx": adx,
    }]).itertuples(index=False).__next__()

for prob in [0.75, 0.80, 0.81, 0.85, 0.90]:
    r = make_row(hour=7, prob=prob)
    allowed, reason = strategy.should_allow_row(r, prob)
    print(f"  prob={prob}: allowed={allowed} reason={reason}")

print()
# What does probability 0.85 do regarding signal_threshold 0.80?
# The threshold is likely applied at MODEL level; should_allow_row uses risk.min_confidence
print(f"signal_threshold check: prob=0.81 >= threshold=0.80? {0.81 >= s.strategy.signal_threshold}")
print(f"min_confidence check: prob=0.81 >= min_conf=0.75? {0.81 >= s.risk.min_confidence}")

print()
# Run actual backtest with known allowed hours
rows = []
t = pd.Timestamp("2026-01-05 07:00:00+00:00")
for i in range(17):
    is_win = i < 15
    rows.append({
        "split": "test", "time": t, "prediction": 1, "probability": 0.85,
        "trade_side": "buy", "close": 2000.0,
        "future_return": 0.001 if is_win else -0.001,
        "directional_return": 0.001, "realized_rr": 10.0 if is_win else -1.0,
        "bars_held": 1, "strategy_score": 0.9, "volatility_regime": 1,
        "trend_alignment": 1, "session_spread_mult": 1.0, "adx": 25.0,
    })
    t += pd.Timedelta(minutes=5)

preds = pd.DataFrame(rows)
rm = RiskManager(s)
result = simulate_prediction_backtest(preds, s, rm, compound=False)
print("Backtest result:")
print("  trades:", result.report["trades"])
print("  profit_factor:", result.report["profit_factor"])
print("  signals_filtered_out:", result.report["signals_filtered_out"])
print("  net_profit:", result.report["net_profit"])
