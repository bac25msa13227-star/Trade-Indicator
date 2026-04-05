import sys
from pathlib import Path
sys.path.insert(0, 'src')
from xauusd_ai.config import load_settings

_REPO = Path(__file__).resolve().parents[1]
s1 = load_settings(_REPO / 'configs/benchmarks/acc1_pf3v2_net815k_dd2891.yaml')
s2 = load_settings(_REPO / 'configs/benchmarks/acc2_pf3v2_net63k_dd1864.yaml')

risk_attrs = [
    'risk_per_trade', 'compound_cap', 'take_profit_rr', 'stop_loss_atr_multiple',
    'max_open_positions', 'min_confidence', 'consecutive_loss_pause_count',
    'consecutive_loss_cooldown_bars', 'anti_martingale_factor', 'sideway_risk_multiplier',
    'strong_volatility_risk_multiplier', 'partial_tp_enabled', 'daily_loss_limit_pct',
    'spread_cost_rr', 'slippage_rr', 'commission_rr',
]
strategy_attrs = [
    'signal_threshold', 'silver_bullet_enabled', 'adx_gate_enabled', 'adx_min_trend',
    'min_strategy_score', 'require_trend_alignment', 'sideway_min_confidence',
    'volatile_min_confidence',
]

for label, s in [('ACC1', s1), ('ACC2', s2)]:
    print(f'=== {label} RISK ===')
    for a in risk_attrs:
        print(f'  {a}: {getattr(s.risk, a, "MISSING")}')
    print(f'=== {label} STRATEGY ===')
    for a in strategy_attrs:
        print(f'  {a}: {getattr(s.strategy, a, "MISSING")}')
    print()
