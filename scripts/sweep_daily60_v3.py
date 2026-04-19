#!/usr/bin/env python3
"""
Sweep v3 — Focused parameter sweep on ACTUAL simulation levers.
=================================================================
INSIGHT from v2: take_profit_rr has NO effect (realized_rr is pre-computed in dataset).
Only real levers:
  1. min_confidence / sideway_min_confidence / volatile_min_confidence → signal filter
  2. ml_threshold_offset → shift fold threshold (negative = more signals, lower precision)
  3. risk_per_trade / max_open_positions → position sizing
  4. cooldown settings → reentry frequency

This sweep systematically explores the space of levers that ACTUALLY work.
Target: avg_daily >= $60, DD < 18%, PF >= 2.0 with $200/day reset balance.
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
    build_fold_predictions_lgbm,
    simulate_candidate_daily_reset,
    _candidate_fingerprint,
)
from xauusd_ai.config import load_settings


def score_v3(s: dict) -> tuple[float, bool]:
    """
    Strict: PF >= 2.0, DD < 18%, avg >= $60.
    Score = avg_daily * PF_bonus * (1 - DD_penalty) * freq_bonus.
    """
    avg = s["avg_daily_net"]
    pf  = s["profit_factor"]
    dd  = s["worst_day_dd_pct"]
    ge60 = s["pct_days_ge_60"]
    neg  = s["pct_negative_days"]

    feasible = (pf >= 2.0) and (dd < 18.0) and (avg >= 20.0)

    if pf < 2.0:
        score = avg * 0.5 + pf * 10 - 100.0 - (dd - 18.0) * 10
    elif dd >= 18.0:
        score = avg * 0.5 + pf * 5 - (dd - 18.0) * 20
    else:
        # Feasible — maximize avg, reward ge60, penalize neg days
        score = avg * 1.5 + pf * 5 + ge60 * 0.3 - neg * 0.3
    return score, feasible


def _base_exec():
    return {
        "close_opposite_on_signal": False,
        "trailing_sl": {
            "enabled": True,
            "breakeven_at_rr": 0.5,
            "activation_rr": 1.0,
            "trail_atr_multiple": 1.0,
        },
    }


def build_candidates_v3() -> list[dict]:
    """
    Systematic grid on real levers:
    - min_confidence × ml_threshold_offset × risk × positions × cooldown
    """
    candidates = []

    # ─── Baseline references ─────────────────────────────────────────────
    candidates.append({
        "name": "ref_r200_mlm05",
        "strategy": {
            "sideway_min_confidence": 0.83,
            "volatile_min_confidence": 0.83,
            "adx_gate_enabled": True, "adx_min_trend": 10.0,
            "blocked_hours_utc": [],
            "silver_bullet_confidence_boost": 0.05,
            "min_strategy_score": 0.0, "sideway_min_strategy_score": 0.0,
            "strong_volatility_min_strategy_score": 0.20,
        },
        "risk": {
            "risk_per_trade": 0.12, "risk_tier_floor": 0.02, "max_open_positions": 2,
            "min_confidence": 0.83, "take_profit_rr": 3.0,
            "volatile_take_profit_rr": 4.5, "sideway_take_profit_rr": 2.5,
            "daily_loss_limit_pct": 0.20, "max_drawdown_kill_pct": 0.17,
            "consecutive_loss_pause_count": 2, "consecutive_loss_cooldown_bars": 8,
            "anti_martingale_factor": 0.85, "anti_martingale_max_reductions": 2,
            "sideway_risk_multiplier": 0.5, "normal_risk_multiplier": 1.0,
            "strong_volatility_risk_multiplier": 1.3,
            "reentry_cooldown_bars_after_sl": 6, "reentry_min_distance_atr": 0.25,
            "partial_tp_enabled": True, "partial_tp_rr": 0.8, "partial_tp_pct": 0.25,
            "max_risk_fraction": 0.12,
        },
        "execution": _base_exec(),
        "ml_threshold_offset": -0.05,
    })

    def make(name, sc_main, ml_offset, risk, pos,
             sc_side=None, sc_vol=None,
             dlim=0.16, dd_kill=0.14,
             cooldown_bars=6, pause_count=2, reentry_bars=6,
             amf=0.85, amr=2) -> dict:
        sc_s = sc_side if sc_side is not None else max(sc_main - 0.06, 0.55)
        sc_v = sc_vol  if sc_vol  is not None else min(sc_main + 0.04, 0.92)
        return {
            "name": name,
            "strategy": {
                "sideway_min_confidence":  sc_s,
                "volatile_min_confidence": sc_v,
                "min_strategy_score": 0.0, "sideway_min_strategy_score": 0.0,
                "strong_volatility_min_strategy_score": 0.20,
                "blocked_hours_utc": [],
                "silver_bullet_confidence_boost": 0.05,
                "adx_gate_enabled": True, "adx_min_trend": 10.0,
            },
            "risk": {
                "risk_per_trade": risk, "risk_tier_floor": 0.02,
                "max_open_positions": pos,
                "min_confidence": sc_main,
                "take_profit_rr": 3.0,
                "volatile_take_profit_rr": 4.5, "sideway_take_profit_rr": 2.5,
                "daily_loss_limit_pct": dlim, "max_drawdown_kill_pct": dd_kill,
                "consecutive_loss_pause_count": pause_count,
                "consecutive_loss_cooldown_bars": cooldown_bars,
                "anti_martingale_factor": amf,
                "anti_martingale_max_reductions": amr,
                "sideway_risk_multiplier": 0.5, "normal_risk_multiplier": 1.0,
                "strong_volatility_risk_multiplier": 1.3,
                "reentry_cooldown_bars_after_sl": reentry_bars,
                "reentry_min_distance_atr": 0.25,
                "partial_tp_enabled": True, "partial_tp_rr": 0.8, "partial_tp_pct": 0.25,
                "max_risk_fraction": risk,
            },
            "execution": _base_exec(),
            "ml_threshold_offset": ml_offset,
        }

    # ─── Group A: CONFIDENCE sweep (primary lever) ───────────────────────
    # Lower min_confidence → more signals → more daily PnL
    # Paired with slightly negative ml_offset to unlock probability gate too
    for sc, sc_l in [(0.83, "sc83"), (0.80, "sc80"), (0.78, "sc78"),
                     (0.76, "sc76"), (0.74, "sc74"), (0.72, "sc72"),
                     (0.70, "sc70"), (0.68, "sc68"), (0.65, "sc65")]:
        for ml, ml_l in [(0.0, "off0"), (-0.03, "off03"), (-0.06, "off06"), (-0.10, "off10")]:
            for r, rl in [(0.10, "r10"), (0.12, "r12"), (0.14, "r14")]:
                for p in [2, 3]:
                    if r * p > 0.28:
                        continue
                    n = f"A_{sc_l}_{ml_l}_{rl}_p{p}"
                    candidates.append(make(
                        n, sc_main=sc, ml_offset=ml, risk=r, pos=p,
                        dlim=0.17, dd_kill=0.15, cooldown_bars=4, pause_count=2,
                    ))

    # ─── Group B: RISK scaling (with best confidence regions) ────────────
    for sc, sc_l in [(0.76, "sc76"), (0.74, "sc74"), (0.72, "sc72"), (0.70, "sc70")]:
        for ml in [-0.05, -0.08, -0.12]:
            for r, rl in [(0.14, "r14"), (0.16, "r16"), (0.18, "r18"), (0.20, "r20")]:
                for p in [1, 2, 3, 4]:
                    if r * p > 0.40:
                        continue
                    ml_l = f"off{int(abs(ml)*100):02d}"
                    n = f"B_{sc_l}_{ml_l}_{rl}_p{p}"
                    # Tighter DD for high risk
                    dlim = max(0.15, 0.22 - r)
                    dd_kill = dlim - 0.02
                    candidates.append(make(
                        n, sc_main=sc, ml_offset=ml, risk=r, pos=p,
                        dlim=dlim, dd_kill=dd_kill,
                        cooldown_bars=2, pause_count=2,
                    ))

    # ─── Group C: COOLDOWN reduction (maximize reentry) ──────────────────
    for sc, sc_l in [(0.76, "sc76"), (0.74, "sc74"), (0.72, "sc72")]:
        for ml in [-0.03, -0.06, -0.10]:
            for r in [0.10, 0.12, 0.14]:
                for p in [2, 3]:
                    ml_l = f"off{int(abs(ml)*100):02d}"
                    r_l = f"r{int(r*100)}"
                    n = f"C_nocd_{sc_l}_{ml_l}_{r_l}_p{p}"
                    candidates.append(make(
                        n, sc_main=sc, ml_offset=ml, risk=r, pos=p,
                        dlim=0.17, dd_kill=0.15,
                        cooldown_bars=0, pause_count=1, reentry_bars=2,
                        amf=1.0, amr=0,
                    ))

    # ─── Group D: AGGRESSIVE offset (max signal unlock) ──────────────────
    for sc in [0.70, 0.72, 0.74]:
        for ml in [-0.12, -0.15, -0.18]:
            for r in [0.10, 0.12, 0.15]:
                for p in [2, 3, 4]:
                    if r * p > 0.36:
                        continue
                    sc_l = f"sc{int(sc*100)}"
                    ml_l = f"off{int(abs(ml)*100):02d}"
                    r_l = f"r{int(r*100)}"
                    n = f"D_agg_{sc_l}_{ml_l}_{r_l}_p{p}"
                    candidates.append(make(
                        n, sc_main=sc, ml_offset=ml, risk=r, pos=p,
                        dlim=0.17, dd_kill=0.15,
                        cooldown_bars=0, pause_count=1, reentry_bars=2,
                        amf=1.0, amr=0,
                    ))

    # Deduplicate by name
    seen_names = set()
    deduped = []
    for c in candidates:
        if c["name"] not in seen_names:
            seen_names.add(c["name"])
            deduped.append(c)
    return deduped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acc1_v14pp_profit.yaml")
    parser.add_argument("--use-lgbm-folds", action="store_true",
                        help="Use LGBM per-fold predictions (honest WF, ~20-30min build)")
    parser.add_argument("--test-start", default="2024-08-14")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--starting-balance", type=float, default=200.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out-prefix", default="wf_daily60_v3_sweep")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Limit candidates (for quick testing)")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    base_settings = load_settings(config_path)

    print("=" * 90)
    print("  SWEEP V3 — FOCUSED LEVER GRID | Target: avg≥$60/day, DD<18%, PF≥2.0 (HONEST WF)")
    print("=" * 90)
    print(f"  Config      : {config_path.name}")
    print(f"  Model type  : {'LGBM per-fold (honest)' if args.use_lgbm_folds else 'VotingClassifier per-fold (default)'}")
    print(f"  Test start  : {args.test_start}")
    print(f"  Balance/day : ${args.starting_balance}")
    print()

    print("Loading dataset + building fold predictions...")
    full_ds = load_or_build_dataset(config_path, cache=not args.no_cache)
    m1_df = load_m1()

    if args.use_lgbm_folds:
        fold_payloads = build_fold_predictions_lgbm(
            config_path=config_path, full_ds=full_ds,
            use_cache=not args.no_cache, max_folds=args.max_folds, test_start=args.test_start,
        )
    else:
        fold_payloads = build_fold_predictions(
            config_path=config_path, full_ds=full_ds,
            use_cache=not args.no_cache, max_folds=args.max_folds, test_start=args.test_start,
        )
    print(f"Folds: {len(fold_payloads)} | span: {fold_payloads[0]['test_start']} → {fold_payloads[-1]['test_end']}")

    candidates = build_candidates_v3()
    if args.max_candidates:
        candidates = candidates[:args.max_candidates]
    print(f"\nTesting {len(candidates)} candidates...\n")

    hdr = (f"{'#':>4}  {'Name':<38} {'Res':>4}  {'Avg$/d':>7} {'Med$/d':>7} "
           f"{'PF':>5} {'DD%':>6} {'≥$60':>6} {'Neg%':>5} {'Trd/d':>5} {'Score':>7}")
    print(hdr)
    print("─" * 100)

    results = []
    seen = set()
    pass_count = 0
    best_avg = 0.0
    for idx, cand in enumerate(candidates, 1):
        fp = _candidate_fingerprint(cand)
        if fp in seen:
            continue
        seen.add(fp)
        name = cand.get("name", f"cand_{idx:03d}")

        result = simulate_candidate_daily_reset(
            name=name,
            base_settings=base_settings,
            candidate={k: v for k, v in cand.items() if not k.startswith("_")},
            fold_payloads=fold_payloads,
            m1_df=m1_df,
            starting_balance=args.starting_balance,
        )
        s = result["summary"]
        custom_score, feas = score_v3(s)
        s["score_v3"] = custom_score
        s["feasible_strict"] = feas
        results.append(result)

        if feas:
            pass_count += 1
        if s["avg_daily_net"] > best_avg:
            best_avg = s["avg_daily_net"]

        mark = "PASS" if feas else "FAIL"
        print(
            f"{idx:>4}  {name:<38} {mark:>4}  "
            f"${s['avg_daily_net']:>6.2f} ${s['median_daily_net']:>6.2f} "
            f"{s['profit_factor']:>5.2f} {s['worst_day_dd_pct']:>5.1f}% "
            f"{s['pct_days_ge_60']:>5.1f}% {s['pct_negative_days']:>4.1f}% "
            f"{s['avg_trades_per_day']:>5.2f} {custom_score:>7.3f}"
        )
        # Progress report every 20 candidates
        if idx % 20 == 0:
            print(f"  -- [{idx}/{len(candidates)}] PASS so far: {pass_count} | best avg: ${best_avg:.2f}/day --")

    # ── Rankings ──────────────────────────────────────────────────────────
    ranked = sorted(results, key=lambda r: r["summary"].get("score_v3", -999), reverse=True)
    pass_results = [r for r in ranked if r["summary"].get("feasible_strict")]

    print("\n" + "=" * 100)
    print(f"STRICT PASS (PF≥2.0, DD<18%, avg≥$20): {len(pass_results)} / {len(results)}")
    target_pass = [r for r in pass_results if r["summary"]["avg_daily_net"] >= 60.0]
    print(f"TARGET PASS (avg≥$60): {len(target_pass)} / {len(results)}")
    print()
    print("TOP 20 (by score):")
    print(hdr)
    print("─" * 100)
    for rank, r in enumerate(ranked[:20], 1):
        s = r["summary"]
        feas = s.get("feasible_strict", False)
        mark = "PASS" if feas else "FAIL"
        tp = "★ " if s["avg_daily_net"] >= 60.0 else "  "
        print(
            f"{rank:>4}  {tp}{s['candidate']:<36} {mark:>4}  "
            f"${s['avg_daily_net']:>6.2f} ${s['median_daily_net']:>6.2f} "
            f"{s['profit_factor']:>5.2f} {s['worst_day_dd_pct']:>5.1f}% "
            f"{s['pct_days_ge_60']:>5.1f}% {s['pct_negative_days']:>4.1f}% "
            f"{s['avg_trades_per_day']:>5.2f} {s.get('score_v3', 0):>7.3f}"
        )

    if target_pass:
        print("\n" + "★" * 80)
        print(f"  FOUND {len(target_pass)} CONFIG(S) MEETING TARGET ($60/day, DD<18%, PF≥2.0)!")
        for r in target_pass:
            s = r["summary"]
            print(f"  → {s['candidate']}: avg=${s['avg_daily_net']:.2f}, PF={s['profit_factor']:.2f}, DD={s['worst_day_dd_pct']:.1f}%")
        print("★" * 80)
    else:
        print(f"\n  Best avg found: ${max(r['summary']['avg_daily_net'] for r in results):.2f}/day")
        print("  No config met $60/day target → need model retrain / new features")

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = [r["summary"] for r in results]
    suffix = "_lgbm" if args.use_lgbm_folds else ""
    out_json = out_dir / f"{args.out_prefix}{suffix}_summary.json"
    out_csv  = out_dir / f"{args.out_prefix}{suffix}_summary.csv"
    out_json.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(all_summaries).sort_values("score_v3", ascending=False).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_json.name} / {out_csv.name}")


if __name__ == "__main__":
    main()
