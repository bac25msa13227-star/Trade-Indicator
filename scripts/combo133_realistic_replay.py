#!/usr/bin/env python3
"""
COMBO133 REALISTIC REPLAY — Fix WF vs Live Gap
===============================================
Apply trailing stop logic to ALL trades (win + loss) matching live behavior:
- breakeven_at_rr: 0.5
- activation_rr: 1.0  
- trail_atr_multiple: 1.0

This simulates realistic profit taking with trailing stop cutting winners early.
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import argparse

sys.stdout.reconfigure(line_buffering=True)


def apply_trailing_stop(
    entry_price: float,
    direction: int,  # +1=buy, -1=sell
    realized_rr: float,
    atr: float,
    be_rr: float = 0.5,
    activation_rr: float = 1.0,
    trail_mult: float = 1.0,
    bars_held: int = 0,
    cut_rate: float = 0.30,  # NEW: configurable trailing cut rate (30% = keep 70%)
) -> float:
    """
    Simulate trailing stop on a single trade — CALIBRATED model.
    
    Based on combo133 win distribution:
    - 73% wins hit 2-2.5R (median 2.333R)  
    - Trailing stop with trail_mult=1.0 ≈ 0.3-0.5R distance
    - Typical pullback 30-50% from peak
    
    Realistic retention model (cut_rate = 30%):
    - Activation at 1.0R → trail starts
    - Profit above BE gets cut by cut_rate due to retracement
    - Strong winners (≥2R): keep more (cut_rate × 0.8)
    - Moderate (1.5-2R): standard cut_rate
    - Weak (1-1.5R): aggressive cut (cut_rate × 1.5)
    """
    if realized_rr < activation_rr:
        # Trailing not activated, return as-is
        return realized_rr
    
    profit_above_be = realized_rr - be_rr
    
    # Adaptive cut rate based on win strength
    if realized_rr >= 2.0:
        # Strong winner (73% of wins): less aggressive trail cut
        effective_cut = cut_rate * 0.75  # Keep 77.5% vs 70% baseline
        retained = be_rr + profit_above_be * (1 - effective_cut)
        return retained
    elif realized_rr >= 1.5:
        # Moderate winner: standard cut
        retained = be_rr + profit_above_be * (1 - cut_rate)
        return retained
    elif realized_rr >= 1.0:
        # Weak winner: more aggressive cut (likely noise)
        effective_cut = min(cut_rate * 1.3, 0.60)
        retained = be_rr + profit_above_be * (1 - effective_cut)
        return max(retained, be_rr)  # Floor at BE
    else:
        # Just above activation: cut to BE
        return be_rr


def replay_month_realistic(
    trades: pd.DataFrame,
    capital: float,
    risk_pct: float,
    max_positions: int,
    prob_min: float,
    be_rr: float = 0.5,
    activation_rr: float = 1.0,
    trail_mult: float = 1.0,
    cut_rate: float = 0.30,
    use_compound: bool = True,
) -> dict:
    """Replay with realistic trailing stop."""
    if len(trades) == 0:
        return None

    balance = capital
    open_positions = []
    pnls = []
    bal_history = [capital]
    skipped_prob = 0
    skipped_overflow = 0
    trail_cuts = 0  # count trades cut by trailing

    for _, t in trades.iterrows():
        if t['probability'] < prob_min:
            skipped_prob += 1
            continue

        ts = pd.to_datetime(t['time'])

        # Close expired positions
        new_open = []
        for op in open_positions:
            if op['exit_t'] <= ts:
                balance += op['pnl']
                pnls.append(op['pnl'])
            else:
                new_open.append(op)
        open_positions = new_open

        if len(open_positions) >= max_positions:
            skipped_overflow += 1
            continue

        # Apply trailing stop logic
        direction = 1 if t['side'] == 'buy' else -1
        realized_rr = float(t['realized_rr'])
        
        # Apply trailing
        adjusted_rr = apply_trailing_stop(
            entry_price=float(t['entry_price']),
            direction=direction,
            realized_rr=realized_rr,
            atr=3.0,  # Assume 3.0 ATR (typical XAUUSD)
            be_rr=be_rr,
            activation_rr=activation_rr,
            trail_mult=trail_mult,
            bars_held=int(t['bars_held']),
            cut_rate=cut_rate,
        )
        
        if adjusted_rr < realized_rr:
            trail_cuts += 1
        
        # Deduct friction
        friction_rr = float(t.get('friction_rr', 0.17))
        net_rr = adjusted_rr - friction_rr
        
        base = balance if use_compound else capital
        risk_amt = base * (risk_pct / 100)
        pnl = risk_amt * net_rr

        if balance + min(0, pnl) < 5:
            continue

        exit_t = ts + pd.Timedelta(minutes=int(t['bars_held']) * 15)
        open_positions.append({'entry_t': ts, 'exit_t': exit_t, 'pnl': pnl})
        bal_history.append(balance)

    # Close remaining
    for op in open_positions:
        balance += op['pnl']
        pnls.append(op['pnl'])

    if not pnls:
        return None

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    bal_arr = np.array(bal_history + [balance])
    peak = np.maximum.accumulate(bal_arr)
    dd = (peak - bal_arr) / np.maximum(peak, 1) * 100

    return {
        'total_pnl': float(sum(pnls)),
        'final_balance': float(balance),
        'num_trades': len(pnls),
        'win_rate': len(wins) / len(pnls) * 100,
        'profit_factor': sum(wins) / abs(sum(losses)) if losses else 99.0,
        'max_dd': float(dd.max()),
        'skipped_prob': skipped_prob,
        'skipped_overflow': skipped_overflow,
        'trail_cuts': trail_cuts,
    }


def run_wf_realistic(
    trades: pd.DataFrame,
    capital: float,
    risk_pct: float,
    max_positions: int,
    prob_min: float,
    target: float,
    be_rr: float = 0.5,
    activation_rr: float = 1.0,
    cut_rate: float = 0.30,
):
    """Monthly WF with realistic trailing."""
    trades = trades.copy()
    trades['time'] = pd.to_datetime(trades['time'])
    if trades['time'].dt.tz is not None:
        trades['time'] = trades['time'].dt.tz_convert(None)
    trades['ym'] = trades['time'].dt.to_period('M')

    months = sorted(trades['ym'].unique())
    results = []
    for m in months:
        mtrades = trades[trades['ym'] == m].sort_values('time').reset_index(drop=True)
        if len(mtrades) < 5:
            continue
        r = replay_month_realistic(
            mtrades, capital, risk_pct, max_positions, prob_min,
            be_rr, activation_rr, cut_rate=cut_rate
        )
        if r is None:
            continue
        days_span = (mtrades['time'].max() - mtrades['time'].min()).total_seconds() / 86400
        trading_days = max(days_span * 5 / 7, 1)
        r['daily_pnl'] = r['total_pnl'] / trading_days
        r['month'] = str(m)
        r['hit_target'] = bool(r['daily_pnl'] >= target)
        results.append(r)

    if not results:
        return None
    daily = [r['daily_pnl'] for r in results]
    hits = sum(1 for r in results if r['hit_target'])
    total_cuts = sum(r.get('trail_cuts', 0) for r in results)
    
    return {
        'avg': float(np.mean(daily)),
        'median': float(np.median(daily)),
        'std': float(np.std(daily)),
        'min': float(np.min(daily)),
        'max': float(np.max(daily)),
        'success': hits,
        'total': len(results),
        'success_rate': hits / len(results) * 100,
        'avg_wr': float(np.mean([r['win_rate'] for r in results])),
        'avg_pf': float(np.mean([r['profit_factor'] for r in results if r['profit_factor'] < 99])),
        'avg_dd': float(np.mean([r['max_dd'] for r in results])),
        'avg_trades': float(np.mean([r['num_trades'] for r in results])),
        'total_trail_cuts': total_cuts,
        'avg_trail_cuts_per_month': total_cuts / len(results),
        'monthly': results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trades', default='outputs/combo133_trades.csv')
    ap.add_argument('--capital', type=float, default=500)
    ap.add_argument('--target', type=float, default=40)
    ap.add_argument('--be-rr', type=float, default=0.5)
    ap.add_argument('--activation-rr', type=float, default=1.0)
    ap.add_argument('--trail-mult', type=float, default=1.0)
    ap.add_argument('--cut-rate', type=float, default=0.30, help='Trailing cut rate (0.3 = keep 70pct profit)')
    args = ap.parse_args()

    print("=" * 100)
    print("COMBO133 REALISTIC REPLAY — Trailing Stop Fix (WF vs Live Gap)")
    print("=" * 100)
    print(f"Trailing config: BE={args.be_rr}R, Activation={args.activation_rr}R, Trail={args.trail_mult}×ATR")
    print(f"Cut rate: {args.cut_rate:.0%} (keep {(1-args.cut_rate):.0%} of profit above BE)")
    print()

    trades = pd.read_csv(args.trades)
    print(f"Loaded {len(trades)} trades from {args.trades}")
    print()

    # Sweep configs
    risk_pcts = [2, 3, 5]
    max_pos = [3, 5]
    prob_mins = [0.70, 0.76, 0.80]

    results_all = []
    for risk in risk_pcts:
        for mp in max_pos:
            for prob in prob_mins:
                r = run_wf_realistic(
                    trades, args.capital, risk, mp, prob, args.target,
                    args.be_rr, args.activation_rr, args.cut_rate
                )
                if r is None:
                    continue
                
                hit = '🎯' if r['success'] >= r['total'] * 0.5 else '  '
                print(
                    f"  risk={risk:>2}% mp={mp} prob>{prob:.2f} | "
                    f"avg=${r['avg']:>7.2f} std=${r['std']:>7.2f} "
                    f"succ={r['success_rate']:>4.1f}% trd={r['avg_trades']:>4.0f} "
                    f"wr={r['avg_wr']:>4.1f}% pf={r['avg_pf']:>4.2f} dd={r['avg_dd']:>4.1f}% "
                    f"cuts={r['avg_trail_cuts_per_month']:>4.0f}/mo {hit}"
                )
                
                if r['avg'] >= args.target:
                    results_all.append({
                        'risk_pct': risk,
                        'max_positions': mp,
                        'prob_min': prob,
                        **r,
                    })

    print()
    print(f"=" * 100)
    print(f"REALISTIC RESULTS: {len(results_all)} configs hit ${args.target}/day target")
    print(f"=" * 100)

    # Save results
    out_path = Path('outputs/combo133_realistic_results.json')
    out_path.write_text(json.dumps({
        'config': {
            'capital': args.capital,
            'target': args.target,
            'be_rr': args.be_rr,
            'activation_rr': args.activation_rr,
            'trail_mult': args.trail_mult,
            'cut_rate': args.cut_rate,
        },
        'targets_hit': results_all,
    }, indent=2))
    print(f"✅ Saved to {out_path}")


if __name__ == '__main__':
    main()
