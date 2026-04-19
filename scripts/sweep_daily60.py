#!/usr/bin/env python3
"""
Targeted sweep to find configs achieving avg_daily_net >= $60 / high pct_days_ge_60.

Strategy:
- Test high-risk × low-confidence grids based on r147_tight as seed
- Relax feasibility: PF >= 1.8, DD < 25%
- Primary objective: pct_days_ge_60 × 2 + avg_daily_net

Usage:
  python scripts/sweep_daily60.py --config configs/acc1_r147tight.yaml
  python scripts/sweep_daily60.py --config configs/acc1_r147tight.yaml --random-samples 40
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
import pandas as pd

# Import infrastructure from the existing optimizer
sys.path.insert(0, str(ROOT / "scripts"))
from wf_daily_reset_optimizer import (
    load_or_build_dataset,
    load_m1,
    build_fold_predictions,
    simulate_candidate_daily_reset,
    _candidate_fingerprint,
    save_outputs,
)
from xauusd_ai.config import load_settings


# ── FEASIBILITY & OBJECTIVE (relaxed for $60/day search) ────────────────────
def score_60_target(s: dict) -> float:
    """Objective weighted toward daily-60 achievement.
    
    Primary:  pct_days_ge_60 (heavy weight)
    Secondary: avg_daily_net, median_daily_net, PF
    Penalty:  DD, negative days
    """
    score = (
        s["avg_daily_net"] * 0.8
        + 0.50 * s["median_daily_net"]        # median > 0 = more reliable
        + 0.35 * s["pct_days_ge_60"]           # ← main target, weighted high
        + 0.15 * s["pct_days_ge_100"]
        - 0.20 * s["pct_negative_days"]
        - 0.30 * s["worst_day_dd_pct"]
        + 0.30 * (s["avg_fold_profit_factor"] - 1.0)
    )
    # Relaxed feasibility: PF >= 1.8 and DD < 25%
    pf = s["profit_factor"]
    dd = s["worst_day_dd_pct"]
    feasible = pf >= 1.8 and dd < 25.0
    if not feasible:
        score -= 150.0
        if pf < 1.8:
            score -= (1.8 - pf) * 40.0
        if dd >= 25.0:
            score -= (dd - 25.0) * 8.0
    return round(score, 4), feasible


# ── MANUAL CANDIDATE GRID ──────────────────────────────────────────────────
def build_manual_candidates() -> list[dict]:
    """Build explicit grid of high-risk × low-confidence candidates."""
    
    # r147_tight base params
    base_strategy = {
        "sideway_min_confidence": 0.76,
        "volatile_min_confidence": 0.88,
        "min_strategy_score": 0.0,
        "sideway_min_strategy_score": 0.05,
        "strong_volatility_min_strategy_score": 0.30,
        "blocked_hours_utc": [],
        "silver_bullet_confidence_boost": 0.05,
        "adx_min_trend": 10.0,
    }
    base_risk = {
        "risk_per_trade": 0.147,
        "risk_tier_floor": 0.03,
        "max_open_positions": 2,
        "min_confidence": 0.85,
        "daily_loss_limit_pct": 0.14,
        "max_drawdown_kill_pct": 0.13,
        "consecutive_loss_pause_count": 2,
        "consecutive_loss_cooldown_bars": 0,
        "anti_martingale_factor": 0.85,
        "anti_martingale_max_reductions": 2,
        "sideway_risk_multiplier": 0.5,
        "normal_risk_multiplier": 1.0,
        "strong_volatility_risk_multiplier": 1.2,
        "reentry_cooldown_bars_after_sl": 8,
        "reentry_min_distance_atr": 0.25,
        "partial_tp_enabled": True,
        "partial_tp_rr": 0.8,
        "partial_tp_pct": 0.25,
    }
    base_execution = {
        "close_opposite_on_signal": False,
        "trailing_sl": {
            "enabled": True,
            "breakeven_at_rr": 0.5,
            "activation_rr": 1.0,
            "trail_atr_multiple": 1.0,
        },
    }
    base_ml_offset = -0.03

    def make(name, risk, sc_side, sc_main=None, pos=2, sc_vol=0.88,
             dlim=None, dd_kill=None, side_mult=0.5, norm_mult=1.0, vol_mult=1.2,
             ml_offset=-0.03, amf=0.85, partial_rr=0.8, cooldown_bars=0,
             reentry_bars=8):
        sc_m = sc_main if sc_main is not None else max(sc_side, 0.80)
        dl = dlim if dlim is not None else min(0.14 + (risk - 0.147) * 0.2, 0.22)
        dk = dd_kill if dd_kill is not None else min(0.13 + (risk - 0.147) * 0.3, 0.22)
        s = base_strategy.copy()
        s["sideway_min_confidence"] = sc_side
        s["volatile_min_confidence"] = sc_vol
        r = base_risk.copy()
        r["risk_per_trade"] = risk
        r["max_open_positions"] = pos
        r["min_confidence"] = sc_m
        r["daily_loss_limit_pct"] = dl
        r["max_drawdown_kill_pct"] = dk
        r["sideway_risk_multiplier"] = side_mult
        r["normal_risk_multiplier"] = norm_mult
        r["strong_volatility_risk_multiplier"] = vol_mult
        r["anti_martingale_factor"] = amf
        r["partial_tp_rr"] = partial_rr
        r["consecutive_loss_cooldown_bars"] = cooldown_bars
        r["reentry_cooldown_bars_after_sl"] = reentry_bars
        return {
            "name": name,
            "strategy": s,
            "risk": r,
            "execution": base_execution,
            "ml_threshold_offset": ml_offset,
        }

    candidates = []

    # ── Group A: r147_tight reference ──
    candidates.append(make("r147_tight_ref", 0.147, 0.76))

    # ── Group B: Pure risk increase (keep thresholds same as r147_tight) ──
    candidates.append(make("r200_tight",  0.20, 0.76, dlim=0.15, dd_kill=0.14))
    candidates.append(make("r250_tight",  0.25, 0.76, dlim=0.17, dd_kill=0.16))
    candidates.append(make("r300_tight",  0.30, 0.76, dlim=0.19, dd_kill=0.18))
    candidates.append(make("r350_tight",  0.35, 0.76, dlim=0.21, dd_kill=0.20))

    # ── Group C: Lower confidence (keep risk near r147) ──
    candidates.append(make("r147_sc70",   0.147, 0.70, sc_main=0.78))
    candidates.append(make("r147_sc65",   0.147, 0.65, sc_main=0.72))
    candidates.append(make("r147_sc60",   0.147, 0.60, sc_main=0.68))
    candidates.append(make("r147_sc55",   0.147, 0.55, sc_main=0.62, ml_offset=-0.05))

    # ── Group D: Combined higher risk + lower confidence ──
    candidates.append(make("r200_sc70",   0.20, 0.70, sc_main=0.78, dlim=0.16, dd_kill=0.15))
    candidates.append(make("r200_sc65",   0.20, 0.65, sc_main=0.72, dlim=0.16, dd_kill=0.15))
    candidates.append(make("r250_sc70",   0.25, 0.70, sc_main=0.78, dlim=0.18, dd_kill=0.17))
    candidates.append(make("r250_sc65",   0.25, 0.65, sc_main=0.72, dlim=0.18, dd_kill=0.17))
    candidates.append(make("r300_sc70",   0.30, 0.70, sc_main=0.78, dlim=0.20, dd_kill=0.19))

    # ── Group E: Max positions = 3 (more concurrent trades) ──
    candidates.append(make("r200_pos3_sc70", 0.20, 0.70, sc_main=0.78, pos=3, dlim=0.16, dd_kill=0.15))
    candidates.append(make("r250_pos3_sc70", 0.25, 0.70, sc_main=0.78, pos=3, dlim=0.18, dd_kill=0.17))
    candidates.append(make("r200_pos3_sc65", 0.20, 0.65, sc_main=0.72, pos=3, dlim=0.16, dd_kill=0.15))

    # ── Group F: Volatile-heavy (boost volatile multiplier, lower sideway risk) ──
    candidates.append(make("r200_volheavy", 0.20, 0.76, sc_vol=0.80, side_mult=0.3, vol_mult=1.5, dlim=0.15, dd_kill=0.14))
    candidates.append(make("r250_volheavy", 0.25, 0.76, sc_vol=0.80, side_mult=0.3, vol_mult=1.5, dlim=0.18, dd_kill=0.17))
    candidates.append(make("r300_volheavy", 0.30, 0.76, sc_vol=0.80, side_mult=0.3, vol_mult=1.5, dlim=0.20, dd_kill=0.19))

    # ── Group G: Aggressive ML threshold (more signals from model) ──
    candidates.append(make("r200_mlm05",  0.20, 0.76, ml_offset=-0.05, dlim=0.16, dd_kill=0.15))
    candidates.append(make("r250_mlm05",  0.25, 0.76, ml_offset=-0.05, dlim=0.18, dd_kill=0.17))
    candidates.append(make("r200_sc70_mlm05", 0.20, 0.70, sc_main=0.78, ml_offset=-0.05, dlim=0.16, dd_kill=0.15))
    candidates.append(make("r250_sc70_mlm05", 0.25, 0.70, sc_main=0.78, ml_offset=-0.05, dlim=0.18, dd_kill=0.17))

    # ── Group H: Ultra-aggressive (push limits) ──
    candidates.append(make("r400_tight",  0.40, 0.76, dlim=0.24, dd_kill=0.22, amf=0.70))
    candidates.append(make("r350_sc65",   0.35, 0.65, sc_main=0.72, dlim=0.22, dd_kill=0.20, amf=0.75))
    candidates.append(make("r300_sc60",   0.30, 0.60, sc_main=0.68, dlim=0.20, dd_kill=0.18, amf=0.75, ml_offset=-0.05))

    # ── Group I: Tight partial TP at earlier RR (lock profit early, more days positive) ──
    candidates.append(make("r200_tpE",    0.20, 0.70, partial_rr=0.6, dlim=0.16, dd_kill=0.15))
    candidates.append(make("r250_tpE",    0.25, 0.70, partial_rr=0.6, dlim=0.18, dd_kill=0.17))

    return candidates


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acc1_r147tight.yaml")
    parser.add_argument("--test-start", default="2024-08-14")
    parser.add_argument("--max-folds", type=int, default=None)  # None = all folds, hits the _all cache
    parser.add_argument("--starting-balance", type=float, default=200.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out-prefix", default="wf_daily60_sweep")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    base_settings = load_settings(config_path)

    print(f"Loading dataset for {config_path.name} ...")
    full_ds = load_or_build_dataset(config_path, cache=not args.no_cache)
    m1_df = load_m1()
    fold_payloads = build_fold_predictions(
        config_path=config_path,
        full_ds=full_ds,
        use_cache=not args.no_cache,
        max_folds=args.max_folds,
        test_start=args.test_start,
    )
    print(f"Folds: {len(fold_payloads)} | span: {fold_payloads[0]['test_start']} → {fold_payloads[-1]['test_end']}")

    candidates = build_manual_candidates()
    print(f"\nTesting {len(candidates)} candidates targeting $60/day ...\n")
    print(f"{'#':>3}  {'Name':<22} {'Res':>4}  {'Avg$/d':>7} {'Med$/d':>7} {'PF':>5} {'DD%':>6} {'≥$60':>6} {'≥$100':>6} {'Neg%':>5} {'Trd/d':>5} {'Score':>7}")
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
            name=name,
            base_settings=base_settings,
            candidate=cand,
            fold_payloads=fold_payloads,
            m1_df=m1_df,
            starting_balance=args.starting_balance,
        )
        s = result["summary"]
        custom_score, feas = score_60_target(s)
        s["score_60target"] = custom_score
        s["feasible_relaxed"] = feas
        results.append(result)

        mark = "PASS" if feas else "FAIL"
        print(
            f"{idx:>3}  {name:<22} {mark:>4}  "
            f"${s['avg_daily_net']:>6.2f} ${s['median_daily_net']:>6.2f} "
            f"{s['profit_factor']:>5.2f} {s['worst_day_dd_pct']:>5.1f}% "
            f"{s['pct_days_ge_60']:>5.1f}% {s['pct_days_ge_100']:>5.1f}% "
            f"{s['pct_negative_days']:>4.1f}% {s['avg_trades_per_day']:>5.2f} "
            f"{custom_score:>7.3f}"
        )

    # ── Rank and print top 10 ────────────────────────────────────────────────
    summaries = sorted(results, key=lambda r: r["summary"].get("score_60target", -999), reverse=True)
    print("\n" + "=" * 100)
    print("TOP 15 CANDIDATES (by $60/day score):")
    print(f"{'#':>3}  {'Name':<22} {'Res':>4}  {'Avg$/d':>7} {'Med$/d':>7} {'PF':>5} {'DD%':>6} {'≥$60':>6} {'≥$100':>6} {'Neg%':>5} {'Trd/d':>5} {'Score':>7}")
    print("─" * 100)
    for rank, r in enumerate(summaries[:15], 1):
        s = r["summary"]
        feas = s.get("feasible_relaxed", False)
        mark = "PASS" if feas else "FAIL"
        print(
            f"{rank:>3}  {s['candidate']:<22} {mark:>4}  "
            f"${s['avg_daily_net']:>6.2f} ${s['median_daily_net']:>6.2f} "
            f"{s['profit_factor']:>5.2f} {s['worst_day_dd_pct']:>5.1f}% "
            f"{s['pct_days_ge_60']:>5.1f}% {s['pct_days_ge_100']:>5.1f}% "
            f"{s['pct_negative_days']:>4.1f}% {s['avg_trades_per_day']:>5.2f} "
            f"{s.get('score_60target', 0):>7.3f}"
        )

    # ── Analysis ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    best = summaries[0]["summary"]
    print(f"BEST: {best['candidate']}")
    print(f"  Avg daily:    ${best['avg_daily_net']:.2f}")
    print(f"  Median daily: ${best['median_daily_net']:.2f}")
    print(f"  PF:           {best['profit_factor']:.3f}")
    print(f"  Max DD:       {best['worst_day_dd_pct']:.2f}%")
    print(f"  Days ≥ $60:   {best['pct_days_ge_60']:.2f}%")
    print(f"  Days ≥ $100:  {best['pct_days_ge_100']:.2f}%")
    print(f"  Negative days: {best['pct_negative_days']:.2f}%")
    print(f"  Total trades: {best['total_trades']}")
    print(f"  Score:        {best.get('score_60target', 0):.3f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = [r["summary"] for r in results]
    (out_dir / f"{args.out_prefix}_summary.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(all_summaries).sort_values("score_60target", ascending=False).to_csv(
        out_dir / f"{args.out_prefix}_summary.csv", index=False
    )
    print(f"\nSaved: outputs/{args.out_prefix}_summary.json")
    print(f"       outputs/{args.out_prefix}_summary.csv")


if __name__ == "__main__":
    main()
