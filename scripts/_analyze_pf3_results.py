#!/usr/bin/env python3
"""Analyze PF3 joint search results."""
import pandas as pd
import glob
import os

ACCOUNTS = [
    ("ACC1", "outputs/wf_pf3_acc1_r1", 127313.86, 32.15, 1.357, "net=$127k"),
    ("ACC2", "outputs/wf_pf3_acc2_r1",  35714.67, 22.81, 1.655, "net=$35k"),
]

for acct, prefix, old_net, old_dd, old_pf, note in ACCOUNTS:
    csvs = sorted(glob.glob(f"{prefix}_round*.csv"))
    print(f"\n{'='*70}")
    print(f"  {acct}  ({len(csvs)} rounds found)")
    print(f"  OLD: net=${old_net:,.2f}  DD={old_dd}%  PF={old_pf}  ({note})")
    print(f"  TARGET: PF>=3.0  net>=${old_net:,.0f}  DD<{old_dd}%")

    dfs = []
    for f in csvs:
        try:
            df = pd.read_csv(f)
            df["source"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"  ERROR: {f}: {e}")
    if not dfs:
        print("  No data found yet")
        continue

    df = pd.concat(dfs, ignore_index=True)
    feasible = df[
        (df["profit_factor_global"] >= 3.0) &
        (df["sum_net_profit"] >= old_net) &
        (df["global_max_drawdown_pct"] < old_dd)
    ].copy()

    print(f"  Total candidates evaluated: {len(df):,}")
    print(f"  Total FEASIBLE (all 3 constraints met): {len(feasible):,}")

    if len(feasible) == 0:
        # Pareto best
        best_pf = df.nlargest(1, "profit_factor_global").iloc[0]
        best_net = df.nlargest(1, "sum_net_profit").iloc[0]
        pf_feas = df[df["profit_factor_global"] >= 3.0]
        best_dd = pf_feas.nsmallest(1, "global_max_drawdown_pct").iloc[0] if len(pf_feas) > 0 else None
        print(f"\n  ⚠  No joint-feasible found!")
        print(f"  Best PF: {best_pf['profit_factor_global']:.3f}  net=${best_pf['sum_net_profit']:,.0f}  DD={best_pf['global_max_drawdown_pct']:.2f}%")
        print(f"  Best net: PF={best_net['profit_factor_global']:.3f}  net=${best_net['sum_net_profit']:,.0f}  DD={best_net['global_max_drawdown_pct']:.2f}%")
        if best_dd is not None:
            print(f"  Best DD (when PF>=3): PF={best_dd['profit_factor_global']:.3f}  net=${best_dd['sum_net_profit']:,.0f}  DD={best_dd['global_max_drawdown_pct']:.2f}%")
        continue

    print(f"\n  ✅ FEASIBLE CONFIGS FOUND!")

    # Balanced: PF 3-15, trades >= 50
    balanced = feasible[
        (feasible["profit_factor_global"] < 15.0) &
        (feasible["total_trades"] >= 50)
    ].sort_values("total_trades", ascending=False)
    print(f"  Balanced group (PF 3-15, trades>=50): {len(balanced)}")
    if len(balanced) > 0:
        print("  Top 5 balanced (most trades):")
        for _, r in balanced.head(5).iterrows():
            print(f"    PF={r['profit_factor_global']:.3f}  net=${r['sum_net_profit']:,.0f}"
                  f"  DD={r['global_max_drawdown_pct']:.2f}%"
                  f"  trades={r['total_trades']}  WR={r['win_rate']:.2%}"
                  f"  risk={r['risk_per_trade']}  tp={r['take_profit_rr']}:1"
                  f"  cap={r['compound_cap']}")

    # Minimum DD
    print("\n  Top 3 by LOWEST DD:")
    for _, r in feasible.nsmallest(3, "global_max_drawdown_pct").iterrows():
        print(f"    PF={r['profit_factor_global']:.3f}  net=${r['sum_net_profit']:,.0f}"
              f"  DD={r['global_max_drawdown_pct']:.2f}%"
              f"  trades={r['total_trades']}  WR={r['win_rate']:.2%}"
              f"  risk={r['risk_per_trade']}  tp={r['take_profit_rr']}:1")

    # Best overall  
    print("\n  Best by net:")
    r = feasible.nlargest(1, "sum_net_profit").iloc[0]
    print(f"    PF={r['profit_factor_global']:.3f}  net=${r['sum_net_profit']:,.0f}"
          f"  DD={r['global_max_drawdown_pct']:.2f}%"
          f"  trades={r['total_trades']}  WR={r['win_rate']:.2%}"
          f"  risk={r['risk_per_trade']}  tp={r['take_profit_rr']}:1  cap={r['compound_cap']}")
    r = feasible.nlargest(1, "profit_factor_global").iloc[0]
    print(f"  Best by PF:")
    print(f"    PF={r['profit_factor_global']:.3f}  net=${r['sum_net_profit']:,.0f}"
          f"  DD={r['global_max_drawdown_pct']:.2f}%"
          f"  trades={r['total_trades']}  WR={r['win_rate']:.2%}"
          f"  risk={r['risk_per_trade']}  tp={r['take_profit_rr']}:1")

    # Recommended: highest trades with PF 3-10 and net above old_net * 2
    moderate = feasible[(feasible['profit_factor_global'] < 12) & (feasible['total_trades'] >= 40)]
    if len(moderate) > 0:
        rec = moderate.sort_values("total_trades", ascending=False).iloc[0]
        print(f"\n  ★ RECOMMENDED (most trades, PF<12):")
        print(f"    PF={rec['profit_factor_global']:.4f}")
        print(f"    net       = ${rec['sum_net_profit']:,.2f}  (old: ${old_net:,.2f}  → +{rec['sum_net_profit']/old_net:.1f}x)")
        print(f"    DD        = {rec['global_max_drawdown_pct']:.2f}%  (old: {old_dd}%  → better)")
        print(f"    trades    = {rec['total_trades']}  WR={rec['win_rate']:.2%}")
        print(f"    threshold = {rec['threshold']}")
        print(f"    risk/tr   = {rec['risk_per_trade']}  ({rec['risk_per_trade']*100:.1f}%)")
        print(f"    TP ratio  = {rec['take_profit_rr']}:1  floor={rec['risk_tier_floor']}")
        print(f"    template  = {rec['template']}")
        print(f"    comp_cap  = {rec['compound_cap']}")
        print(f"    consec    = pause={rec['consecutive_loss_pause_count']}  cool={rec['consecutive_loss_cooldown_bars']}bars")
        print(f"    source    = {rec['source']}")
