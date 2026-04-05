"""Show final v2 search results."""
import json

for acc, f in [
    ("ACC1", "outputs/wf_pf3v2_acc1_best_feasible.json"),
    ("ACC2", "outputs/wf_pf3v2_acc2_best_feasible.json"),
]:
    d = json.load(open(f))
    mt = d["most_trades"]
    bn = d.get("best_net", mt)
    bl = d.get("baseline", {})
    print(f"=== {acc} ===")
    print(f"  Total feasible: {d['total_feasible']}")
    print(f"  BASELINE (current code): net=${bl.get('sum_net_profit','?'):,}  PF={bl.get('profit_factor_global','?')}  DD={bl.get('global_max_drawdown_pct','?')}%  trades={bl.get('total_trades','?')}")
    print(f"  MOST-TRADES RECOMMENDED:")
    for k in ["profit_factor_global","sum_net_profit","global_max_drawdown_pct","total_trades","win_rate",
              "threshold","risk_per_trade","take_profit_rr","compound_cap",
              "consecutive_loss_pause_count","consecutive_loss_cooldown_bars","template",
              "min_strategy_score","adx_min_trend","silver_bullet_enabled","adx_gate_enabled",
              "sideway_risk_multiplier","strong_volatility_risk_multiplier","stop_loss_atr_multiple"]:
        if k in mt:
            print(f"    {k}: {mt[k]}")
    print()
    print(f"  BEST-NET:")
    for k in ["profit_factor_global","sum_net_profit","global_max_drawdown_pct","total_trades","win_rate",
              "threshold","risk_per_trade","take_profit_rr","compound_cap"]:
        if k in bn:
            print(f"    {k}: {bn[k]}")
    print()
