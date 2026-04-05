"""Find feasible ACC1 configs with more trades from existing search results."""
import pandas as pd, glob

dfs = [pd.read_csv(f) for f in sorted(glob.glob("outputs/wf_pf3_acc1_r1_round*.csv"))]
df = pd.concat(dfs, ignore_index=True)

feasible = df[df["feasible"]==True].copy()
print(f"Total feasible: {len(feasible)}")
print(f"Trade count stats: min={feasible['total_trades'].min()}  mean={feasible['total_trades'].mean():.0f}  max={feasible['total_trades'].max()}")
print()

# Find configs with more trades
more_trades = feasible[feasible["total_trades"] >= 150].sort_values("total_trades", ascending=False)
print(f"Feasible with >= 150 trades: {len(more_trades)}")
print()
print("Top 10 by most trades (all constraints: PF>=3, net>=127k, DD<32.15%):")
top = more_trades.head(10)
for _, r in top.iterrows():
    print(f"  thr={r['threshold']:.2f} risk={r['risk_per_trade']:.2f} TP={r['take_profit_rr']:.1f} cap={r['compound_cap']:.0f} pause={r['consecutive_loss_pause_count']} trades={r['total_trades']:>4} PF={r['profit_factor_global']:.3f} net=${r['sum_net_profit']:>12,.0f} DD={r['global_max_drawdown_pct']:.2f}% WR={r['win_rate']:.2%} tpl={r['template']}")
