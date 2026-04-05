"""Debug exactly which rows get filtered and why."""
from pathlib import Path
from xauusd_ai.config import load_settings
from xauusd_ai.strategies.hybrid import HybridStrategy
import pandas as pd

s = load_settings(Path("configs/benchmarks/acc2_pf3v2_net63k_dd1864.yaml"))
s.risk.kill_switch_enabled = False
strategy = HybridStrategy(s)

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
for row in preds.itertuples(index=False):
    allowed, reason = strategy.should_allow_row(row, float(row.probability))
    print(f"  t={row.time} rr={row.realized_rr} allowed={allowed} reason={reason}")
