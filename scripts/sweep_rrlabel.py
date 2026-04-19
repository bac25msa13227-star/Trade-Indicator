#!/usr/bin/env python3
"""
Sweep v4 — Uses realized_rr > 0 as training label (better model quality).
Hypothesis: Model trained on actual trade outcomes (RR>0) will have higher AUC,
better calibration, and allow more effective signal filtering.

This script REQUIRES the rrlabel fold cache to already be built:
  python -c "from wf_daily_reset_optimizer import *; ..."

Run after fold build completes:
  python scripts/sweep_rrlabel.py --config configs/acc1_v14pp_profit.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from wf_daily_reset_optimizer import (
    load_or_build_dataset,
    load_m1,
    build_fold_predictions,
    simulate_candidate_daily_reset,
    _candidate_fingerprint,
)
from xauusd_ai.config import load_settings


def score(s: dict) -> tuple[float, bool]:
    avg = s["avg_daily_net"]
    pf  = s["profit_factor"]
    dd  = s["worst_day_dd_pct"]
    ge60 = s["pct_days_ge_60"]
    neg  = s["pct_negative_days"]
    feasible = (pf >= 2.0) and (dd < 18.0) and (avg >= 20.0)
    if pf < 2.0:
        return avg * 0.3 - 100 + pf * 15, False
    elif dd >= 18.0:
        return avg * 0.5 + pf * 5 - (dd - 18.0) * 30, False
    else:
        return avg * 2.0 + pf * 5 + ge60 * 0.3 - neg * 0.3, feasible


def _exec():
    return {
        "close_opposite_on_signal": False,
        "trailing_sl": {
            "enabled": True,
            "breakeven_at_rr": 0.5,
            "activation_rr": 1.0,
            "trail_atr_multiple": 1.0,
        },
    }


def build_candidates() -> list[dict]:
    """
    Grid sweep on confidence × risk × positions × cooldown.
    With better model (rrlabel), lower confidence thresholds should work well.
    """
    cands = []

    def make(name, sc, risk, pos, dlim=0.17, dd_kill=0.15,
             cooldown_bars=4, pause_count=2, reentry_bars=4,
             amf=0.90, amr=2, sc_side=None, sc_vol=None) -> dict:
        sc_s = sc_side if sc_side is not None else sc
        sc_v = sc_vol  if sc_vol  is not None else sc
        return {
            "name": name,
            "strategy": {
                "sideway_min_confidence":  sc_s,
                "volatile_min_confidence": sc_v,
                "min_strategy_score": 0.0, "sideway_min_strategy_score": 0.0,
                "strong_volatility_min_strategy_score": 0.10,
                "blocked_hours_utc": [],
                "silver_bullet_confidence_boost": 0.05,
                "adx_gate_enabled": True, "adx_min_trend": 8.0,
            },
            "risk": {
                "risk_per_trade": risk, "risk_tier_floor": 0.01,
                "max_open_positions": pos,
                "min_confidence": sc,
                "take_profit_rr": 3.0,
                "volatile_take_profit_rr": 4.5, "sideway_take_profit_rr": 2.5,
                "daily_loss_limit_pct": dlim,
                "max_drawdown_kill_pct": dd_kill,
                "consecutive_loss_pause_count": pause_count,
                "consecutive_loss_cooldown_bars": cooldown_bars,
                "anti_martingale_factor": amf,
                "anti_martingale_max_reductions": amr,
                "sideway_risk_multiplier": 0.7, "normal_risk_multiplier": 1.0,
                "strong_volatility_risk_multiplier": 1.2,
                "reentry_cooldown_bars_after_sl": reentry_bars,
                "reentry_min_distance_atr": 0.15,
                "partial_tp_enabled": True, "partial_tp_rr": 0.8, "partial_tp_pct": 0.25,
                "max_risk_fraction": risk,
            },
            "execution": _exec(),
            "ml_threshold_offset": 0.0,
        }

    # Primary grid: rrlabel model has higher-quality probabilities → sweep wider range
    # With higher positive rate (37.6%), fold thresholds may be lower, so sweep from 0.50
    for sc, sc_l in [(0.50, "sc50"), (0.52, "sc52"), (0.54, "sc54"), (0.56, "sc56"),
                     (0.58, "sc58"), (0.60, "sc60"), (0.62, "sc62"), (0.65, "sc65"),
                     (0.68, "sc68"), (0.70, "sc70"), (0.72, "sc72"),
                     (0.74, "sc74"), (0.76, "sc76"), (0.78, "sc78"),
                     (0.80, "sc80"), (0.83, "sc83"), (0.85, "sc85")]:
        for r, rl in [(0.10, "r10"), (0.12, "r12"), (0.15, "r15"), (0.18, "r18")]:
            for p in [2, 3, 4]:
                if r * p > 0.36:
                    continue
                dlim = min(0.18, 0.13 + r)
                dd_kill = dlim - 0.02
                cands.append(make(
                    f"R_{sc_l}_{rl}_p{p}",
                    sc=sc, risk=r, pos=p, dlim=dlim, dd_kill=dd_kill,
                    cooldown_bars=4, pause_count=2, reentry_bars=4,
                ))

    # No-cooldown variants at best confidence levels
    for sc, sc_l in [(0.65, "sc65"), (0.70, "sc70"), (0.74, "sc74"), (0.78, "sc78")]:
        for r, rl in [(0.10, "r10"), (0.12, "r12"), (0.15, "r15")]:
            for p in [2, 3, 4]:
                if r * p > 0.36:
                    continue
                dlim = min(0.18, 0.13 + r)
                cands.append(make(
                    f"NC_{sc_l}_{rl}_p{p}",
                    sc=sc, risk=r, pos=p, dlim=dlim, dd_kill=dlim - 0.02,
                    cooldown_bars=0, pause_count=3, reentry_bars=2,
                    amf=1.0, amr=0,
                ))

    # High-risk short-burst configs
    for sc in [0.70, 0.74, 0.78, 0.83]:
        for r in [0.20, 0.25]:
            for p in [2, 3]:
                if r * p > 0.50:
                    continue
                sc_l = f"sc{int(sc*100)}"
                r_l = f"r{int(r*100)}"
                cands.append(make(
                    f"HR_{sc_l}_{r_l}_p{p}",
                    sc=sc, risk=r, pos=p, dlim=0.20, dd_kill=0.17,
                    cooldown_bars=0, pause_count=3, reentry_bars=2,
                    amf=1.0, amr=0,
                ))

    return cands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acc1_v14pp_profit.yaml")
    parser.add_argument("--starting-balance", type=float, default=200.0)
    parser.add_argument("--out-prefix", default="wf_rrlabel_sweep")
    parser.add_argument("--test-start", default="2024-08-14")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    base_settings = load_settings(config_path)

    print("=" * 95)
    print("  RRLABEL SWEEP — Model trained on realized_rr>0 label | Target: $60/day, DD<18%, PF≥2")
    print("=" * 95)

    full_ds = load_or_build_dataset(config_path, cache=True)
    m1_df = load_m1()

    # Load rrlabel fold cache (must be pre-built)
    fold_payloads = build_fold_predictions(
        config_path=config_path, full_ds=full_ds,
        use_cache=not args.no_cache, max_folds=None, test_start=args.test_start,
        use_realized_rr_label=True,
    )
    print(f"Folds: {len(fold_payloads)} | {fold_payloads[0]['test_start']} → {fold_payloads[-1]['test_end']}")
    aucs = [fi["roc_auc"] for fi in fold_payloads]
    precs = [fi["precision"] for fi in fold_payloads]
    import numpy as np
    print(f"Model quality: avg AUC={np.mean(aucs):.4f}, avg Precision={np.mean(precs):.4f}")
    print()

    candidates = build_candidates()
    print(f"Testing {len(candidates)} candidates...\n")

    hdr = (f"{'#':>4}  {'Name':<36} {'Res':>4}  {'Avg$/d':>7} {'Med$/d':>7} "
           f"{'PF':>5} {'DD%':>6} {'≥$60':>6} {'Neg%':>5} {'Trd/d':>5} {'Score':>7}")
    print(hdr)
    print("─" * 100)

    results = []
    seen = set()
    for idx, cand in enumerate(candidates, 1):
        fp = _candidate_fingerprint(cand)
        if fp in seen:
            continue
        seen.add(fp)
        name = cand.get("name", f"cand_{idx:03d}")
        result = simulate_candidate_daily_reset(
            name=name, base_settings=base_settings,
            candidate={k: v for k, v in cand.items() if not k.startswith("_")},
            fold_payloads=fold_payloads, m1_df=m1_df,
            starting_balance=args.starting_balance,
        )
        s = result["summary"]
        sc, feas = score(s)
        s["score"] = sc
        s["feasible"] = feas
        results.append(result)

        mark = "PASS" if feas else "FAIL"
        tp = "★ " if s["avg_daily_net"] >= 60.0 else "  "
        print(
            f"{idx:>4}  {tp}{name:<34} {mark:>4}  "
            f"${s['avg_daily_net']:>6.2f} ${s['median_daily_net']:>6.2f} "
            f"{s['profit_factor']:>5.2f} {s['worst_day_dd_pct']:>5.1f}% "
            f"{s['pct_days_ge_60']:>5.1f}% {s['pct_negative_days']:>4.1f}% "
            f"{s['avg_trades_per_day']:>5.2f} {sc:>7.3f}"
        )

    ranked = sorted(results, key=lambda r: r["summary"].get("score", -999), reverse=True)
    target60 = [r for r in ranked if r["summary"]["avg_daily_net"] >= 60.0]
    pass60 = [r for r in ranked if r["summary"].get("feasible") and r["summary"]["avg_daily_net"] >= 60.0]

    print("\n" + "=" * 100)
    print(f"CONFIGS WITH avg≥$60: {len(target60)} | STRICT PASS (DD<18%, PF≥2): {len(pass60)}")
    print(f"Best avg: ${max(r['summary']['avg_daily_net'] for r in results):.2f}/day")
    print()
    print("TOP 20:")
    print(hdr)
    print("─" * 100)
    for rank, r in enumerate(ranked[:20], 1):
        s = r["summary"]
        feas = s.get("feasible")
        mark = "PASS" if feas else "FAIL"
        tp = "★ " if s["avg_daily_net"] >= 60.0 else "  "
        print(
            f"{rank:>4}  {tp}{s['candidate']:<34} {mark:>4}  "
            f"${s['avg_daily_net']:>6.2f} ${s['median_daily_net']:>6.2f} "
            f"{s['profit_factor']:>5.2f} {s['worst_day_dd_pct']:>5.1f}% "
            f"{s['pct_days_ge_60']:>5.1f}% {s['pct_negative_days']:>4.1f}% "
            f"{s['avg_trades_per_day']:>5.2f} {s.get('score', 0):>7.3f}"
        )

    if pass60:
        print("\n" + "★" * 80)
        print(f"  FOUND {len(pass60)} CONFIG(S) MEETING TARGET (avg≥$60, DD<18%, PF≥2)!")
        for r in pass60:
            s = r["summary"]
            print(f"  → {s['candidate']}: avg=${s['avg_daily_net']:.2f}, PF={s['profit_factor']:.2f}, DD={s['worst_day_dd_pct']:.1f}%")
        print("★" * 80)

    out_dir = ROOT / "outputs"
    summaries = [r["summary"] for r in results]
    (out_dir / f"{args.out_prefix}_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(summaries).sort_values("score", ascending=False).to_csv(
        out_dir / f"{args.out_prefix}_summary.csv", index=False)
    print(f"\nSaved: {args.out_prefix}_summary.json / .csv")


if __name__ == "__main__":
    main()
