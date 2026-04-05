from pathlib import Path
from src.xauusd_ai.config import load_settings

# Validate ACC1
s1 = load_settings(Path("configs/benchmarks/acc1_pf3v2_net815k_dd2891.yaml"))
assert s1.strategy.signal_threshold == 0.86
assert s1.risk.risk_per_trade == 0.07
assert s1.risk.take_profit_rr == 7.0
assert s1.risk.compound_cap == 50.0
assert s1.risk.consecutive_loss_pause_count == 3
assert s1.risk.partial_tp_enabled == False
assert s1.strategy.silver_bullet_enabled == False
assert s1.strategy.adx_gate_enabled == True
assert s1.strategy.adx_min_trend == 12.0
assert s1.strategy.blocked_hours_utc == [15, 22, 23]
assert s1.strategy.allowed_weekday_hours_utc == {}
print("ACC1 OK:", s1.execution.comment)

# Validate ACC2
s2 = load_settings(Path("configs/benchmarks/acc2_pf3v2_net63k_dd1864.yaml"))
assert s2.strategy.signal_threshold == 0.80
assert s2.risk.risk_per_trade == 0.06
assert s2.risk.take_profit_rr == 10.0
assert s2.risk.compound_cap == 50.0
assert s2.risk.consecutive_loss_pause_count == 3
assert s2.risk.consecutive_loss_cooldown_bars == 12
assert s2.risk.partial_tp_enabled == False
assert s2.strategy.silver_bullet_enabled == True
assert s2.strategy.adx_gate_enabled == False
assert s2.strategy.adx_min_trend == 22.0
assert s2.strategy.blocked_hours_utc == []
assert "Monday" in s2.strategy.allowed_weekday_hours_utc
assert s2.strategy.allowed_weekday_hours_utc["Monday"] == [0, 1, 5, 6, 7, 13, 14, 20, 21]
print("ACC2 OK:", s2.execution.comment)
print("Both configs validated successfully.")
