#!/usr/bin/env python3
"""
Fast parameter sweep using existing WF predictions + parquet cache realized_rr.
No model retraining — sweeps risk_per_trade, max_positions, friction_rr combinations.
Takes <10 seconds vs hours for full WF rerun.

Usage:
  python scripts/fast_resim_sweep.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
SIGNALS_CSV  = "outputs/walkforward_trades_acc1_v14pp_profit.csv"
CACHE_PARQUET = "outputs/.wf_cache/xauusd_combo133_best_e9bbf0b6a41b.parquet"

STARTING_BAL   = 200.0    # USD per fold, reset each fold (no cross-fold compound)
SL_MULT        = 1.5      # stop_loss_atr_multiple
MAX_LOT        = 5.0      # max lot cap
COMPOUND_CAP   = 500.0    # max balance = start_bal × cap (prevent explosions)
MAX_DD_KILL    = 0.20     # stop fold if drawdown from peak exceeds 20%
DAILY_LOSS_LIM = 0.12     # stop day if daily loss exceeds 12% of balance

TARGET_TOTAL_PNL = 70_000.0  # $70k target

# ── Parameter grid ────────────────────────────────────────────────────────────
RISK_VALUES    = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
MAX_POS_VALUES = [3, 5, 8, 10, 15, 20]
FRICTION_VALUES = [0.17, 0.20, 0.23, 0.26]  # 0.17=original, 0.26=current realistic

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading signals CSV...")
sig = pd.read_csv(SIGNALS_CSV, parse_dates=["time"])
sig["time_utc"] = pd.to_datetime(sig["time"], utc=True)

print("Loading parquet cache...")
cache = pd.read_parquet(CACHE_PARQUET, columns=["time", "realized_rr", "bars_held", "atr", "trade_side"])
cache["time_utc"] = pd.to_datetime(cache["time"], utc=True)

print("Joining...")
merged = sig.merge(
    cache[["time_utc", "realized_rr", "bars_held", "atr", "trade_side"]],
    on="time_utc", how="left"
)
merged = merged.dropna(subset=["realized_rr", "atr"])
merged["bars_held"] = merged["bars_held"].fillna(20).astype(int).clip(1, 200)
merged["atr"] = merged["atr"].clip(lower=0.1)  # safety: no zero ATR

# Sort within each fold by time
merged = merged.sort_values(["fold", "time_utc"]).reset_index(drop=True)
folds = sorted(merged["fold"].unique())
print(f"  {len(merged):,} signals across {len(folds)} folds")


# ── Simulation core ───────────────────────────────────────────────────────────
def simulate_fold(
    fold_df: pd.DataFrame,
    start_bal: float,
    risk_per_trade: float,
    max_positions: int,
    friction_rr: float,
    m15_per_bar: int = 15,  # minutes per bar (M15)
) -> dict:
    """
    Simulate one fold with concurrent positions, lot-snap, max drawdown kill.
    Uses TIME-based position closing (not signal-index based) for accuracy.
    Returns fold report dict.
    """
    rows = fold_df.reset_index(drop=True)
    n = len(rows)
    balance = start_bal
    peak_bal = start_bal
    max_balance = start_bal * COMPOUND_CAP

    open_positions: list[dict] = []  # {close_time, pnl}
    wins = losses = 0
    daily_loss: dict[str, float] = {}  # date_str → cumulative day loss
    daily_blocked: set[str] = set()    # date strings blocked for the day

    for i in range(n):
        row = rows.iloc[i]
        current_time = row["time_utc"]

        # ── close matured positions (time-based) ──────────────────────────────
        still_open = []
        for pos in open_positions:
            if current_time >= pos["close_time"]:
                pnl = pos["pnl"]
                balance += pnl
                peak_bal = max(peak_bal, balance)
                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1
                    d = str(current_time.date())
                    daily_loss[d] = daily_loss.get(d, 0.0) + abs(pnl)
                    if DAILY_LOSS_LIM > 0 and daily_loss[d] >= balance * DAILY_LOSS_LIM:
                        daily_blocked.add(d)
            else:
                still_open.append(pos)
        open_positions = still_open

        # ── max drawdown kill ─────────────────────────────────────────────────
        dd_from_peak = (peak_bal - balance) / peak_bal if peak_bal > 0 else 0.0
        if dd_from_peak >= MAX_DD_KILL:
            break

        # ── daily block check ─────────────────────────────────────────────────
        d = str(current_time.date())
        if d in daily_blocked:
            continue

        # ── max concurrent positions ──────────────────────────────────────────
        if len(open_positions) >= max_positions:
            continue

        # ── size position ─────────────────────────────────────────────────────
        atr = float(row["atr"])
        sl_dist = atr * SL_MULT
        if sl_dist <= 0:
            continue

        eff_bal = min(balance, max_balance)
        if eff_bal <= 0:
            break

        ideal_lot = (eff_bal * risk_per_trade) / (100.0 * sl_dist)
        actual_lot = max(0.01, round(ideal_lot / 0.01) * 0.01)
        actual_lot = min(actual_lot, MAX_LOT)
        # Recompute actual RF after lot-snap
        actual_rf = (actual_lot * 100.0 * sl_dist) / eff_bal

        net_rr = float(row["realized_rr"]) - friction_rr
        pnl = eff_bal * actual_rf * net_rr

        # Close time based on actual M15 bars held (not signal index)
        hold_bars = max(1, int(row["bars_held"]))
        close_time = current_time + pd.Timedelta(minutes=hold_bars * m15_per_bar)
        open_positions.append({"close_time": close_time, "pnl": pnl})

    # Close any remaining open positions at end of fold
    for pos in open_positions:
        pnl = pos["pnl"]
        balance += pnl
        peak_bal = max(peak_bal, balance)
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

    dd_final = (peak_bal - balance) / peak_bal * 100 if peak_bal > 0 else 0.0
    n_trades = wins + losses
    wr = wins / max(n_trades, 1)
    ret_pct = (balance / start_bal - 1) * 100

    return {
        "end_bal": round(balance, 2),
        "pnl": round(balance - start_bal, 2),
        "ret_pct": round(ret_pct, 2),
        "trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wr, 4),
        "max_dd_pct": round(dd_final, 2),
    }


# ── Parameter sweep ───────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  FAST PARAM SWEEP — $70k target / 27 folds / $200/fold no-compound")
print("=" * 90)
print(f"  {'Risk%':>6}  {'MaxPos':>6}  {'Frict':>6}  {'Total$':>9}  {'Avg%/fold':>10}  {'AvgDD%':>7}  {'AvgWR%':>7}  {'Hit?':>5}")
print("  " + "-" * 80)

results = []

for friction_rr in FRICTION_VALUES:
    for risk_pct in RISK_VALUES:
        for max_pos in MAX_POS_VALUES:
            fold_pnls, fold_rets, fold_dds, fold_wrs = [], [], [], []

            for fold_id in folds:
                fold_df = merged[merged["fold"] == fold_id].copy()
                if len(fold_df) < 5:
                    continue
                r = simulate_fold(
                    fold_df,
                    start_bal=STARTING_BAL,
                    risk_per_trade=risk_pct,
                    max_positions=max_pos,
                    friction_rr=friction_rr,
                )
                fold_pnls.append(r["pnl"])
                fold_rets.append(r["ret_pct"])
                fold_dds.append(r["max_dd_pct"])
                fold_wrs.append(r["win_rate"] * 100)

            if not fold_pnls:
                continue
            total_pnl  = sum(fold_pnls)
            avg_ret    = sum(fold_rets) / len(fold_rets)
            avg_dd     = sum(fold_dds)  / len(fold_dds)
            avg_wr     = sum(fold_wrs)  / len(fold_wrs)
            hit_target = total_pnl >= TARGET_TOTAL_PNL

            results.append({
                "risk_pct": risk_pct,
                "max_pos": max_pos,
                "friction": friction_rr,
                "total_pnl": total_pnl,
                "avg_ret_pct": avg_ret,
                "avg_dd_pct": avg_dd,
                "avg_wr_pct": avg_wr,
                "hit": hit_target,
                "n_folds": len(fold_pnls),
            })

            # Print notable results: hits target, or interesting
            if hit_target or avg_dd < 35 and avg_ret > 200:
                flag = "✅" if hit_target else "  "
                print(
                    f"  {risk_pct*100:>5.0f}%"
                    f"  {max_pos:>6}"
                    f"  {friction_rr:.2f}"
                    f"  {total_pnl:>9,.0f}"
                    f"  {avg_ret:>+9.1f}%"
                    f"  {avg_dd:>6.1f}%"
                    f"  {avg_wr:>6.1f}%"
                    f"  {flag}"
                )

# ── Summary: best configs ────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  TOP RESULTS (highest total PnL, max_dd < 40%):")
print("=" * 90)
print(f"  {'Risk%':>6}  {'MaxPos':>6}  {'Frict':>6}  {'Total$':>9}  {'Avg%/fold':>10}  {'AvgDD%':>7}  {'AvgWR%':>7}  {'Hit?':>5}")
print("  " + "-" * 80)

results_df = pd.DataFrame(results)
top = results_df[results_df["avg_dd_pct"] < 40].nlargest(20, "total_pnl")
for _, r in top.iterrows():
    flag = "✅" if r["hit"] else "  "
    print(
        f"  {r['risk_pct']*100:>5.0f}%"
        f"  {r['max_pos']:>6.0f}"
        f"  {r['friction']:>6.2f}"
        f"  {r['total_pnl']:>9,.0f}"
        f"  {r['avg_ret_pct']:>+9.1f}%"
        f"  {r['avg_dd_pct']:>6.1f}%"
        f"  {r['avg_wr_pct']:>6.1f}%"
        f"  {flag}"
    )

# ── Best realistic config with 0.26R friction ────────────────────────────────
print("\n" + "─" * 90)
print("  CONFIGS WITH CURRENT FRICTION (0.26R) THAT HIT $70k:")
print("─" * 90)
hits_realistic = results_df[(results_df["friction"] == 0.26) & results_df["hit"]]
if hits_realistic.empty:
    print("  ❌ No combo hits $70k with 0.26R friction using existing predictions.")
    print("  → Need to relax friction or re-run WF with more aggressive risk/position params.")
    # Show closest
    close = results_df[results_df["friction"] == 0.26].nlargest(5, "total_pnl")
    print("  Closest with 0.26R friction:")
    for _, r in close.iterrows():
        print(f"    risk={r['risk_pct']*100:.0f}% maxpos={r['max_pos']:.0f}  total=${r['total_pnl']:,.0f}  avgdd={r['avg_dd_pct']:.1f}%")
else:
    for _, r in hits_realistic.nsmallest(5, "avg_dd_pct").iterrows():
        print(f"  risk={r['risk_pct']*100:.0f}% maxpos={r['max_pos']:.0f}  total=${r['total_pnl']:,.0f}  avgdd={r['avg_dd_pct']:.1f}%  avgret={r['avg_ret_pct']:+.1f}%")

# ── Recommendation ────────────────────────────────────────────────────────────
print("\n" + "─" * 90)
hits = results_df[results_df["hit"]]
if not hits.empty:
    best = hits.loc[hits["avg_dd_pct"].idxmin()]
    print(f"  RECOMMENDED CONFIG (min DD that hits $70k):")
    print(f"    risk_per_trade:   {best['risk_pct']:.0%}")
    print(f"    max_open_positions: {best['max_pos']:.0f}")
    print(f"    friction_rr:     {best['friction']:.2f}R")
    print(f"    Total P&L:       ${best['total_pnl']:,.0f}")
    print(f"    Avg return/fold: {best['avg_ret_pct']:+.1f}%")
    print(f"    Avg max DD:      {best['avg_dd_pct']:.1f}%")
else:
    print("  ⚠️  No config hits $70k target with these predictions.")
    print("  Need full WF rerun with more aggressive params OR lower friction.")
    best_overall = results_df.nlargest(1, "total_pnl").iloc[0]
    print(f"  Best found: risk={best_overall['risk_pct']:.0%} maxpos={best_overall['max_pos']:.0f} frict={best_overall['friction']:.2f}")
    print(f"              total=${best_overall['total_pnl']:,.0f} ({best_overall['total_pnl']/TARGET_TOTAL_PNL*100:.0f}% of target)")

print()
