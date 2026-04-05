"""Debug compound cap and anti-martingale behavior."""
from xauusd_ai.config import Settings
from xauusd_ai.backtesting.engine import simulate_prediction_backtest
from xauusd_ai.execution.risk import RiskManager
import pandas as pd

def make_settings(cap: float) -> Settings:
    s = Settings()
    s.training.backtest_initial_balance = 200.0
    s.risk.risk_per_trade = 0.25
    s.risk.compound_cap = cap
    s.risk.min_confidence = 0.5
    s.risk.kill_switch_enabled = False
    s.risk.spread_cost_rr = 0.0
    s.risk.slippage_rr = 0.0
    s.risk.commission_rr = 0.0
    s.strategy.require_trend_alignment = False
    return s

def make_preds(n=10):
    rows = []
    t = pd.Timestamp("2026-01-05 09:00:00+00:00")
    for _ in range(n):
        rows.append({"split": "test", "time": t, "prediction": 1,
            "probability": 0.95, "trade_side": "buy", "close": 2000.0,
            "future_return": 0.001, "directional_return": 0.001,
            "realized_rr": 4.0, "bars_held": 1, "strategy_score": 0.9,
            "volatility_regime": 1, "trend_alignment": 1,
            "session_spread_mult": 1.0, "adx": 30.0})
        t += pd.Timedelta(minutes=5)
    return pd.DataFrame(rows)

preds = make_preds(10)
s50 = make_settings(50.0)
s0 = make_settings(0.0)

rm50 = RiskManager(s50)
rm0 = RiskManager(s0)

print("=== Settings check ===")
print("risk_per_trade:", s50.risk.risk_per_trade)
print("compound_cap:", s50.risk.compound_cap)
print("max_risk_fraction:", s50.risk.max_risk_fraction)
print("min_confidence:", s50.risk.min_confidence)
print("normal_risk_multiplier:", s50.risk.normal_risk_multiplier)
print()

r50 = simulate_prediction_backtest(preds, s50, rm50, compound=True)
r0 = simulate_prediction_backtest(preds, s0, rm0, compound=True)

print("cap=50 ending:", r50.report["ending_balance"])
print("cap= 0 ending:", r0.report["ending_balance"])
print("cap=50 trades:", r50.report["trades"])
if not r50.trades.empty:
    print("First few trades:")
    for _, row in r50.trades.head(4).iterrows():
        print(f"  bal_before={row['balance_before']:.2f} rr={row['realized_rr']:.2f} risk={row['risk_fraction']:.4f} pnl={row['pnl']:.2f} bal_after={row['balance_after']:.2f}")

# Also debug anti-martingale for ACC1
from pathlib import Path
from xauusd_ai.config import load_settings

print()
print("=== Anti-Martingale ACC1 ===")
s1 = load_settings(Path("configs/benchmarks/acc1_pf3v2_net815k_dd2891.yaml"))
s1.risk.kill_switch_enabled = False
s1.risk.anti_martingale_max_reductions = 3
rm1 = RiskManager(s1)
rm1._consecutive_losses = 2

rf = rm1.risk_fraction(confidence=0.95, volatility_regime=1, strategy_score=1.0,
                       current_balance=1000.0, market_row=None, side="buy")
print(f"risk_per_trade={s1.risk.risk_per_trade}, factor={s1.risk.anti_martingale_factor}, 2 losses")
print(f"min_confidence={s1.risk.min_confidence}, max_risk_fraction={s1.risk.max_risk_fraction}")
print(f"base_fraction = {s1.risk.risk_per_trade} * (0.95/{s1.risk.min_confidence}) = {s1.risk.risk_per_trade * (0.95/s1.risk.min_confidence):.6f}")
print(f"anti_mart = {s1.risk.anti_martingale_factor}^2 = {s1.risk.anti_martingale_factor**2:.4f}")
# raw = 0.07 * (0.95/0.85) * 1.0 * 1.0 * 0.36
raw = s1.risk.risk_per_trade * (0.95/s1.risk.min_confidence) * (s1.risk.anti_martingale_factor**2)
print(f"expected raw = {raw:.6f}")
print(f"actual rf = {rf:.6f}")

print()
print("=== Anti-Martingale ACC2 ===")
s2 = load_settings(Path("configs/benchmarks/acc2_pf3v2_net63k_dd1864.yaml"))
s2.risk.kill_switch_enabled = False
s2.risk.anti_martingale_max_reductions = 3
rm2 = RiskManager(s2)
rm2._consecutive_losses = 1

rf2 = rm2.risk_fraction(confidence=0.95, volatility_regime=1, strategy_score=1.0,
                        current_balance=1000.0, market_row=None, side="buy")
print(f"risk_per_trade={s2.risk.risk_per_trade}, factor={s2.risk.anti_martingale_factor}, 1 loss")
print(f"min_confidence={s2.risk.min_confidence}, max_risk_fraction={s2.risk.max_risk_fraction}")
raw2 = s2.risk.risk_per_trade * (0.95/s2.risk.min_confidence) * s2.risk.anti_martingale_factor
print(f"expected raw = {raw2:.6f}")
print(f"actual rf = {rf2:.6f}")
