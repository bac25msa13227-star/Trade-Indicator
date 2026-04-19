#!/usr/bin/env python3
"""
Ceiling sweep — find if $60/day is achievable by disabling risk controls.
Tests 3 hypotheses:
  H1: Anti-martingale + cooldown kill returns → disable them
  H2: Need higher max_positions to capture more of 14 daily signals
  H3: Need aggressive risk per trade (15-20%) to boost PnL
Results will tell us if $60/day is achievable with current model,
or if we need a fundamentally better model.
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
    feasible = (pf >= 2.0) and (dd < 18.0) and (avg >= 20.0)
    if pf < 2.0:
        score_ = avg * 0.3 - 100 + pf * 15
    elif dd >= 18.0:
        score_ = avg * 0.5 + pf * 5 - (dd - 18.0) * 30
    else:
        score_ = avg * 2.0 + pf * 5 - s["pct_negative_days"] * 0.2
    return score_, feasible


def _exec(trailing=True):
    return {
        "close_opposite_on_signal": False,
        "trailing_sl": {
            "enabled": trailing,
            "breakeven_at_rr": 0.5,
            "activation_rr": 1.0,
            "trail_atr_multiple": 1.0,
        },
    }


def make(name, sc, risk, pos, dlim, dd_kill,
         cooldown_bars=0, pause_count=99, reentry_bars=0,
         amf=1.0, amr=0, sc_side=None, sc_vol=None,
         partial_pct=0.25, partial_rr=0.8) -> dict:
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
            "sideway_risk_multiplier": 0.8, "normal_risk_multiplier": 1.0,
            "strong_volatility_risk_multiplier": 1.2,
            "reentry_cooldown_bars_after_sl": reentry_bars,
            "reentry_min_distance_atr": 0.1,
            "partial_tp_enabled": True,
            "partial_tp_rr": partial_rr,
            "partial_tp_pct": partial_pct,
            "max_risk_fraction": risk,
        },
        "execution": _exec(),
        "ml_threshold_offset": 0.0,
    }


def build_ceiling_candidates() -> list[dict]:
    cands = []

    # ── Baseline (with controls) ──────────────────────────────────────────
    cands.append(make("ref_r147_tight_ctrl",
        sc=0.83, risk=0.12, pos=2, dlim=0.17, dd_kill=0.15,
        cooldown_bars=8, pause_count=2, reentry_bars=6, amf=0.85, amr=2))

    # ── H1: Same params, disable all throttling ───────────────────────────
    cands.append(make("h1_noCtrl_r12_p2_sc83",
        sc=0.83, risk=0.12, pos=2, dlim=0.30, dd_kill=0.17,
        cooldown_bars=0, pause_count=99, reentry_bars=0, amf=1.0, amr=0))

    cands.append(make("h1_noCtrl_r10_p2_sc83",
        sc=0.83, risk=0.10, pos=2, dlim=0.30, dd_kill=0.17,
        cooldown_bars=0, pause_count=99, reentry_bars=0, amf=1.0, amr=0))

    # ── H2: More positions, no controls ───────────────────────────────────
    for pos in [3, 4, 6, 8]:
        cands.append(make(f"h2_noCtrl_r10_p{pos}_sc83",
            sc=0.83, risk=0.10, pos=pos, dlim=0.30, dd_kill=0.17,
            cooldown_bars=0, pause_count=99, reentry_bars=0, amf=1.0, amr=0))

    for pos in [3, 4, 6]:
        cands.append(make(f"h2_noCtrl_r12_p{pos}_sc83",
            sc=0.83, risk=0.12, pos=pos, dlim=0.30, dd_kill=0.17,
            cooldown_bars=0, pause_count=99, reentry_bars=0, amf=1.0, amr=0))

    # ── H3: High risk, no controls ────────────────────────────────────────
    for risk in [0.15, 0.18, 0.20]:
        for pos in [2, 3, 4]:
            rl = f"r{int(risk*100)}"
            cands.append(make(f"h3_noCtrl_{rl}_p{pos}_sc83",
                sc=0.83, risk=risk, pos=pos, dlim=0.40, dd_kill=0.25,
                cooldown_bars=0, pause_count=99, reentry_bars=0, amf=1.0, amr=0))

    # ── H4: Combo — sc 0.80-0.85, max positions, no controls ─────────────
    for sc, sc_l in [(0.80, "sc80"), (0.83, "sc83"), (0.85, "sc85")]:
        for risk in [0.10, 0.12, 0.15]:
            for pos in [4, 6, 8]:
                rl = f"r{int(risk*100)}"
                cands.append(make(f"h4_{sc_l}_{rl}_p{pos}",
                    sc=sc, risk=risk, pos=pos, dlim=0.30, dd_kill=0.20,
                    cooldown_bars=0, pause_count=99, reentry_bars=0, amf=1.0, amr=0))

    # ── H5: Remove partial TP (let full position run to TP) ──────────────
    for risk in [0.10, 0.12, 0.15]:
        for pos in [3, 4]:
            rl = f"r{int(risk*100)}"
            cands.append(make(f"h5_noPartial_sc83_{rl}_p{pos}",
                sc=0.83, risk=risk, pos=pos, dlim=0.30, dd_kill=0.20,
                cooldown_bars=0, pause_count=99, reentry_bars=0, amf=1.0, amr=0,
                partial_pct=0.0, partial_rr=99.0))

    # ── H6: Mild controls + more positions ───────────────────────────────
    for pos in [4, 5, 6]:
        cands.append(make(f"h6_mildCtrl_r12_p{pos}_sc83",
            sc=0.83, risk=0.12, pos=pos, dlim=0.20, dd_kill=0.17,
            cooldown_bars=2, pause_count=3, reentry_bars=2, amf=0.95, amr=1))
        cands.append(make(f"h6_mildCtrl_r10_p{pos}_sc83",
            sc=0.83, risk=0.10, pos=pos, dlim=0.20, dd_kill=0.17,
            cooldown_bars=2, pause_count=3, reentry_bars=2, amf=0.95, amr=1))

    return cands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acc1_v14pp_profit.yaml")
    parser.add_argument("--starting-balance", type=float, default=200.0)
    parser.add_argument("--out-prefix", default="wf_ceiling_sweep")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    base_settings = load_settings(config_path)

    print("=" * 95)
    print("  CEILING SWEEP — Testing if $60/day is achievable by removing risk controls")
    print("=" * 95)

    full_ds = load_or_build_dataset(config_path, cache=True)
    m1_df = load_m1()
    fold_payloads = build_fold_predictions(config_path=config_path, full_ds=full_ds,
                                           use_cache=True, max_folds=None, test_start="2024-08-14")
    print(f"Folds: {len(fold_payloads)} | {fold_payloads[0]['test_start']} → {fold_payloads[-1]['test_end']}")

    candidates = build_ceiling_candidates()
    print(f"Testing {len(candidates)} ceiling candidates...\n")

    hdr = (f"{'#':>4}  {'Name':<38} {'Res':>4}  {'Avg$/d':>7} {'Med$/d':>7} "
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
        target = "★ " if s["avg_daily_net"] >= 60.0 else "  "
        print(
            f"{idx:>4}  {target}{name:<36} {mark:>4}  "
            f"${s['avg_daily_net']:>6.2f} ${s['median_daily_net']:>6.2f} "
            f"{s['profit_factor']:>5.2f} {s['worst_day_dd_pct']:>5.1f}% "
            f"{s['pct_days_ge_60']:>5.1f}% {s['pct_negative_days']:>4.1f}% "
            f"{s['avg_trades_per_day']:>5.2f} {sc:>7.3f}"
        )

    ranked = sorted(results, key=lambda r: r["summary"].get("score", -999), reverse=True)
    target60 = [r for r in ranked if r["summary"]["avg_daily_net"] >= 60.0]
    pass60 = [r for r in ranked if r["summary"].get("feasible") and r["summary"]["avg_daily_net"] >= 60.0]

    print("\n" + "=" * 100)
    print(f"CONFIGS WITH avg≥$60/day: {len(target60)} | STRICT PASS (DD<18%, PF≥2): {len(pass60)}")
    print(f"Best avg: ${max(r['summary']['avg_daily_net'] for r in results):.2f}/day")
    print()
    print("TOP 15:")
    print(hdr)
    print("─" * 100)
    for rank, r in enumerate(ranked[:15], 1):
        s = r["summary"]
        feas = s.get("feasible")
        mark = "PASS" if feas else "FAIL"
        tp = "★ " if s["avg_daily_net"] >= 60.0 else "  "
        print(
            f"{rank:>4}  {tp}{s['candidate']:<36} {mark:>4}  "
            f"${s['avg_daily_net']:>6.2f} ${s['median_daily_net']:>6.2f} "
            f"{s['profit_factor']:>5.2f} {s['worst_day_dd_pct']:>5.1f}% "
            f"{s['pct_days_ge_60']:>5.1f}% {s['pct_negative_days']:>4.1f}% "
            f"{s['avg_trades_per_day']:>5.2f} {s.get('score', 0):>7.3f}"
        )

    if not target60:
        print("\n  CONCLUSION: $60/day NOT achievable with current model quality.")
        print("  Even with no risk controls, max is below $60/day.")
        print("  NEED: Better model (higher AUC/precision) or new feature engineering.")
        best = ranked[0]["summary"]
        print(f"  Best found: {best['candidate']} → ${best['avg_daily_net']:.2f}/day, DD={best['worst_day_dd_pct']:.1f}%")
    else:
        print(f"\n  ★ $60/day IS achievable. Best: {target60[0]['summary']['candidate']}")
        if pass60:
            print(f"  ★ STRICT PASS (DD<18%, PF≥2): {pass60[0]['summary']['candidate']} → ${pass60[0]['summary']['avg_daily_net']:.2f}/day")

    out_dir = ROOT / "outputs"
    summaries = [r["summary"] for r in results]
    (out_dir / f"{args.out_prefix}_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(summaries).sort_values("score", ascending=False).to_csv(
        out_dir / f"{args.out_prefix}_summary.csv", index=False)
    print(f"\nSaved: {args.out_prefix}_summary.json / .csv")


if __name__ == "__main__":
    main()
