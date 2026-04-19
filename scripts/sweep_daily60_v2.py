#!/usr/bin/env python3
"""
Sweep v2 — Target: avg_daily >= $60, DD < 18%, PF >= 2.0
============================================================
Chiến lược mới dựa trên phân tích bottleneck:
- Median = $0 do 43% zero-signal days (min_conf=0.85 quá cao)
- Precision=72% ở 0.85 → giảm xuống 0.70-0.75 vẫn duy trì PF≥2 nếu tăng RR
- Công thức toán học: RR=4-5:1 + WR≥40% + risk≤18%total → avg≥$60/day

Nhóm mới:
  J: High take_profit_rr (4, 5, 6) + lower risk
  K: Lower confidence + 3-4 positions + small risk
  L: Volatile-focus with high RR  
  M: ML offset aggressive với lower confidence
  N: LGBM model configs (nếu available)
  O: No-cooldown + no-anti-martingale (maximize signal count)
  P: Tight DD protection + high RR (asymmetric profit/loss)

Usage:
  python scripts/sweep_daily60_v2.py --config configs/acc1_v14pp_profit.yaml
  python scripts/sweep_daily60_v2.py --config configs/acc1_v14pp_profit.yaml --lgbm-config configs/acc1_lgbm.yaml
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

sys.path.insert(0, str(ROOT / "scripts"))
from wf_daily_reset_optimizer import (
    load_or_build_dataset,
    load_m1,
    build_fold_predictions,
    build_fold_predictions_lgbm,
    simulate_candidate_daily_reset,
    _candidate_fingerprint,
)
from xauusd_ai.config import load_settings


# ── STRICT FEASIBILITY & OBJECTIVE ──────────────────────────────────────────
def score_v2(s: dict) -> tuple[float, bool]:
    """
    Strict target: PF >= 2.0, DD < 18%, avg_daily >= 60 (aspirational).
    Score optimizes toward all three simultaneously.
    """
    avg  = s["avg_daily_net"]
    med  = s["median_daily_net"]
    pf   = s["profit_factor"]
    dd   = s["worst_day_dd_pct"]
    neg  = s["pct_negative_days"]
    ge60 = s["pct_days_ge_60"]
    ge100= s["pct_days_ge_100"]
    tpd  = s["avg_trades_per_day"]

    # Core score
    score = (
        avg  * 1.5              # avg daily profit (primary target)
        + med * 0.8             # median > 0 = reliable daily
        + ge60 * 0.5            # % days ≥$60
        + ge100 * 0.3           # % days ≥$100 (upside)
        - neg * 2.0             # heavy penalty for negative days
        - abs(dd) * 2.5         # heavy penalty for DD (target <18%)
        + (pf - 1.0) * 60       # PF bonus
    )

    # Feasibility: PF >= 2.0 AND DD < 18%
    feasible = pf >= 2.0 and dd < 18.0
    if not feasible:
        score -= 200.0
        if pf < 2.0:
            score -= (2.0 - pf) * 100.0
        if dd >= 18.0:
            score -= (dd - 18.0) * 15.0

    return round(score, 4), feasible


# ── CANDIDATE BUILDER ─────────────────────────────────────────────────────
def build_candidates_v2() -> list[dict]:
    """
    Build targeted candidates. Mathematical basis:
    - With DD<18%: max total_risk_exposure = 18% = $36 from $200
    - For avg=$60: expected_per_day = $60 → need high WR OR high RR
    - With RR=4: need WR≥53% AND PF auto=4×0.53/0.47=4.5✓
    - With RR=5: need WR≥45% AND PF=5×0.45/0.55=4.1✓
    Key: lower risk per trade + multiple positions + high RR = optimal combo
    """

    # Reference base (r147_tight params)
    def _base_execution():
        return {
            "close_opposite_on_signal": False,
            "trailing_sl": {
                "enabled": True,
                "breakeven_at_rr": 0.5,
                "activation_rr": 1.0,
                "trail_atr_multiple": 1.0,
            },
        }

    def make(
        name,
        risk,            # risk_per_trade
        sc_main,         # min_confidence (main)
        sc_side=None,    # sideway_min_confidence
        sc_vol=None,     # volatile_min_confidence
        pos=2,           # max_open_positions
        tp_rr=3.5,       # take_profit_rr
        tp_rr_vol=5.0,   # volatile_take_profit_rr
        tp_rr_side=2.5,  # sideway_take_profit_rr
        dlim=None,       # daily_loss_limit_pct
        dd_kill=None,    # max_drawdown_kill_pct
        side_mult=0.5,
        norm_mult=1.0,
        vol_mult=1.2,
        ml_offset=0.0,   # ml_threshold_offset
        amf=0.85,        # anti_martingale_factor
        amr=2,           # anti_martingale_max_reductions
        partial_rr=0.8,
        partial_pct=0.25,
        cooldown_bars=0,
        pause_count=2,
        reentry_bars=8,
        adx_min=10.0,
        model_path=None,  # override model file
    ) -> dict:
        sc_s = sc_side if sc_side is not None else max(sc_main - 0.10, 0.60)
        sc_v = sc_vol  if sc_vol  is not None else min(sc_main + 0.05, 0.92)
        dl   = dlim    if dlim    is not None else min(0.17, max(0.12, 0.13 + risk * 0.3))
        dk   = dd_kill if dd_kill is not None else min(0.16, max(0.11, dl - 0.02))

        strategy = {
            "sideway_min_confidence": sc_s,
            "volatile_min_confidence": sc_v,
            "min_strategy_score": 0.0,
            "sideway_min_strategy_score": 0.0,
            "strong_volatility_min_strategy_score": 0.20,
            "blocked_hours_utc": [],
            "silver_bullet_confidence_boost": 0.05,
            "adx_gate_enabled": True,
            "adx_min_trend": adx_min,
        }
        risk_cfg = {
            "risk_per_trade": risk,
            "risk_tier_floor": 0.02,
            "max_open_positions": pos,
            "min_confidence": sc_main,
            "take_profit_rr": tp_rr,
            "volatile_take_profit_rr": tp_rr_vol,
            "sideway_take_profit_rr": tp_rr_side,
            "daily_loss_limit_pct": dl,
            "max_drawdown_kill_pct": dk,
            "consecutive_loss_pause_count": pause_count,
            "consecutive_loss_cooldown_bars": cooldown_bars,
            "anti_martingale_factor": amf,
            "anti_martingale_max_reductions": amr,
            "sideway_risk_multiplier": side_mult,
            "normal_risk_multiplier": norm_mult,
            "strong_volatility_risk_multiplier": vol_mult,
            "reentry_cooldown_bars_after_sl": reentry_bars,
            "reentry_min_distance_atr": 0.25,
            "partial_tp_enabled": True,
            "partial_tp_rr": partial_rr,
            "partial_tp_pct": partial_pct,
            "max_risk_fraction": risk,
        }
        cand = {
            "name": name,
            "strategy": strategy,
            "risk": risk_cfg,
            "execution": _base_execution(),
            "ml_threshold_offset": ml_offset,
        }
        if model_path:
            cand["_model_path"] = str(model_path)
        return cand

    candidates = []

    # ─── Group R: Reference (r147_tight + r200_mlm05 from previous best) ───
    candidates.append(make("ref_r147_tight",  0.147, 0.85, sc_side=0.76, sc_vol=0.88,
                           tp_rr=3.5, tp_rr_vol=5.0, tp_rr_side=2.5,
                           dlim=0.14, dd_kill=0.13, ml_offset=-0.03))
    candidates.append(make("ref_r200_mlm05",  0.20, 0.85, sc_side=0.76, sc_vol=0.88,
                           tp_rr=3.5, tp_rr_vol=5.0, tp_rr_side=2.5,
                           dlim=0.15, dd_kill=0.15, ml_offset=-0.05))

    # ─── Group J: HIGH RR + low risk per trade ────────────────────────────
    # Math: RR=4, risk=9%, max_pos=2 → max_loss=18%, win=4×9%×200=$72
    # With WR=45%: E[trade]=0.45×72-0.55×18=$32.4-$9.9=$22.5
    # With 2 trades/day: $45/day → 3 trades: $67.5 ✓
    for tp, lbl in [(4.0, "tp4"), (5.0, "tp5"), (6.0, "tp6"), (8.0, "tp8")]:
        for r, rl in [(0.09, "r9"), (0.10, "r10"), (0.12, "r12")]:
            for p in [2, 3]:
                ml_off = -0.03 if tp <= 5 else 0.0
                cname = f"J_{lbl}_{rl}_p{p}"
                candidates.append(make(
                    cname, risk=r, sc_main=0.82, sc_side=0.75, sc_vol=0.85,
                    pos=p, tp_rr=tp, tp_rr_vol=tp+1.0, tp_rr_side=tp-1.0,
                    dlim=0.17, dd_kill=0.15, ml_offset=ml_off,
                    side_mult=0.5, vol_mult=1.2,
                ))

    # ─── Group K: Lower confidence + multi-position + small risk ─────────
    # At conf=0.70: 462/508 days have signals (91%) vs 119/508 (23%) at 0.85
    # With risk=6% per trade × pos=3 = 18% max DD
    for sc, scl in [(0.72, "sc72"), (0.75, "sc75"), (0.78, "sc78")]:
        for r, rl in [(0.06, "r6"), (0.08, "r8"), (0.10, "r10")]:
            for p in [2, 3]:
                if r * p > 0.20:  # skip if max exposure > 20%
                    continue
                cname = f"K_{scl}_{rl}_p{p}"
                candidates.append(make(
                    cname, risk=r, sc_main=sc, sc_side=sc-0.05, sc_vol=sc+0.05,
                    pos=p, tp_rr=4.0, tp_rr_vol=5.0, tp_rr_side=3.0,
                    dlim=0.16, dd_kill=0.14, ml_offset=-0.03,
                    side_mult=0.6, vol_mult=1.3,
                ))

    # ─── Group L: Volatile-session FOCUS with high RR ────────────────────
    # Volatile has largest moves → higher TP + lower threshold
    candidates.append(make("L_volfocus_tp6",  0.15, 0.82, sc_side=0.85, sc_vol=0.76,
                           pos=2, tp_rr=4.0, tp_rr_vol=6.0, tp_rr_side=3.0,
                           dlim=0.17, dd_kill=0.15, vol_mult=1.5, side_mult=0.3, ml_offset=-0.03))
    candidates.append(make("L_volfocus_tp8",  0.12, 0.80, sc_side=0.85, sc_vol=0.75,
                           pos=2, tp_rr=4.0, tp_rr_vol=8.0, tp_rr_side=3.0,
                           dlim=0.17, dd_kill=0.15, vol_mult=1.5, side_mult=0.3, ml_offset=-0.03))
    candidates.append(make("L_volfocus_tp5p3", 0.10, 0.78, sc_side=0.82, sc_vol=0.74,
                           pos=3, tp_rr=3.5, tp_rr_vol=5.0, tp_rr_side=2.5,
                           dlim=0.16, dd_kill=0.14, vol_mult=1.5, side_mult=0.3, ml_offset=-0.05))
    candidates.append(make("L_volonly_tp6p3",  0.08, 0.78, sc_side=0.88, sc_vol=0.72,
                           pos=3, tp_rr=3.5, tp_rr_vol=6.0, tp_rr_side=2.5,
                           dlim=0.15, dd_kill=0.13, vol_mult=1.8, side_mult=0.2, ml_offset=-0.05))

    # ─── Group M: ML offset aggressive + lower sc (compound effect) ──────
    for mloff, mol in [(-0.08, "m08"), (-0.10, "m10"), (-0.12, "m12")]:
        for sc, scl in [(0.78, "sc78"), (0.80, "sc80")]:
            for r in [0.12, 0.15]:
                cname = f"M_{scl}_{mol}_r{int(r*100)}"
                candidates.append(make(
                    cname, risk=r, sc_main=sc, sc_side=sc-0.08, sc_vol=sc+0.05,
                    pos=2, tp_rr=4.0, tp_rr_vol=5.0, tp_rr_side=3.0,
                    dlim=0.17, dd_kill=0.15, ml_offset=mloff,
                ))

    # ─── Group O: No-cooldown + fast re-entry (maximize signal count) ─────
    candidates.append(make("O_nocd_r10_tp4",  0.10, 0.80, sc_side=0.74, sc_vol=0.84,
                           pos=2, tp_rr=4.0, tp_rr_vol=5.5, tp_rr_side=3.0,
                           dlim=0.17, dd_kill=0.15, ml_offset=-0.03,
                           cooldown_bars=0, pause_count=1, reentry_bars=4, amf=1.0, amr=0))
    candidates.append(make("O_nocd_r8_tp5p3", 0.08, 0.78, sc_side=0.72, sc_vol=0.82,
                           pos=3, tp_rr=5.0, tp_rr_vol=6.0, tp_rr_side=3.5,
                           dlim=0.17, dd_kill=0.15, ml_offset=-0.05,
                           cooldown_bars=0, pause_count=1, reentry_bars=2, amf=1.0, amr=0))
    candidates.append(make("O_nocd_r12_tp4",  0.12, 0.80, sc_side=0.74, sc_vol=0.84,
                           pos=2, tp_rr=4.0, tp_rr_vol=5.5, tp_rr_side=3.0,
                           dlim=0.17, dd_kill=0.15, ml_offset=-0.03,
                           cooldown_bars=0, pause_count=1, reentry_bars=4, amf=1.0, amr=0))

    # ─── Group P: Asymmetric protection (tight DD + high RR) ─────────────
    # Tight daily kill → limits downside; high RR → big upside when signal fires
    candidates.append(make("P_asym_tp6_r12",  0.12, 0.83, sc_side=0.78, sc_vol=0.86,
                           pos=2, tp_rr=6.0, tp_rr_vol=8.0, tp_rr_side=4.0,
                           dlim=0.15, dd_kill=0.12, ml_offset=-0.03, vol_mult=1.3))
    candidates.append(make("P_asym_tp8_r9",   0.09, 0.82, sc_side=0.76, sc_vol=0.86,
                           pos=2, tp_rr=8.0, tp_rr_vol=10.0, tp_rr_side=5.0,
                           dlim=0.15, dd_kill=0.12, ml_offset=-0.03, vol_mult=1.3))
    candidates.append(make("P_asym_tp5p3_r8", 0.08, 0.80, sc_side=0.74, sc_vol=0.84,
                           pos=3, tp_rr=5.0, tp_rr_vol=7.0, tp_rr_side=3.5,
                           dlim=0.15, dd_kill=0.12, ml_offset=-0.05, vol_mult=1.4))
    candidates.append(make("P_asym_tp6p3_r6", 0.06, 0.78, sc_side=0.72, sc_vol=0.82,
                           pos=3, tp_rr=6.0, tp_rr_vol=8.0, tp_rr_side=4.0,
                           dlim=0.15, dd_kill=0.12, ml_offset=-0.05, vol_mult=1.5))

    # ─── Group N: LGBM configs — built separately via build_lgbm_candidates() ──
    # These use per-fold LGBM predictions (no data leakage).
    # Do NOT add them here — they use a separate fold_payloads built by build_fold_predictions_lgbm()

    return candidates


def build_lgbm_candidates() -> list[dict]:
    """
    LGBM-specific candidates: lower confidence levels that LGBM can handle well.
    These are designed for use with build_fold_predictions_lgbm() fold payloads.
    """
    def _base_execution():
        return {
            "close_opposite_on_signal": False,
            "trailing_sl": {
                "enabled": True,
                "breakeven_at_rr": 0.5,
                "activation_rr": 1.0,
                "trail_atr_multiple": 1.0,
            },
        }

    def make(name, risk, sc_main, sc_side=None, sc_vol=None, pos=2,
             tp_rr=4.0, tp_rr_vol=5.5, tp_rr_side=3.0,
             dlim=None, dd_kill=None, ml_offset=0.0) -> dict:
        sc_s = sc_side if sc_side is not None else max(sc_main - 0.08, 0.55)
        sc_v = sc_vol  if sc_vol  is not None else min(sc_main + 0.05, 0.90)
        dl   = dlim    if dlim    is not None else min(0.17, 0.13 + risk * 0.3)
        dk   = dd_kill if dd_kill is not None else min(0.15, dl - 0.02)
        return {
            "name": name,
            "strategy": {
                "sideway_min_confidence": sc_s,
                "volatile_min_confidence": sc_v,
                "min_strategy_score": 0.0,
                "sideway_min_strategy_score": 0.0,
                "strong_volatility_min_strategy_score": 0.20,
                "blocked_hours_utc": [],
                "silver_bullet_confidence_boost": 0.05,
                "adx_gate_enabled": True,
                "adx_min_trend": 10.0,
            },
            "risk": {
                "risk_per_trade": risk,
                "risk_tier_floor": 0.02,
                "max_open_positions": pos,
                "min_confidence": sc_main,
                "take_profit_rr": tp_rr,
                "volatile_take_profit_rr": tp_rr_vol,
                "sideway_take_profit_rr": tp_rr_side,
                "daily_loss_limit_pct": dl,
                "max_drawdown_kill_pct": dk,
                "consecutive_loss_pause_count": 2,
                "consecutive_loss_cooldown_bars": 0,
                "anti_martingale_factor": 0.85,
                "anti_martingale_max_reductions": 2,
                "sideway_risk_multiplier": 0.5,
                "normal_risk_multiplier": 1.0,
                "strong_volatility_risk_multiplier": 1.3,
                "reentry_cooldown_bars_after_sl": 6,
                "reentry_min_distance_atr": 0.25,
                "partial_tp_enabled": True,
                "partial_tp_rr": 0.8,
                "partial_tp_pct": 0.25,
                "max_risk_fraction": risk,
            },
            "execution": _base_execution(),
            "ml_threshold_offset": ml_offset,
        }

    candidates = []
    # Core grid: LGBM calibrates probabilities better → lower sc is meaningful
    for sc, scl in [(0.62, "sc62"), (0.65, "sc65"), (0.68, "sc68"), (0.70, "sc70"), (0.72, "sc72")]:
        for r, rl in [(0.08, "r8"), (0.10, "r10"), (0.12, "r12")]:
            for p in [2, 3]:
                if r * p > 0.20:
                    continue
                for tp in [4.0, 5.0]:
                    cname = f"LGBM_{scl}_{rl}_p{p}_tp{int(tp)}"
                    candidates.append(make(
                        cname, risk=r, sc_main=sc, pos=p,
                        tp_rr=tp, tp_rr_vol=tp+1.5, tp_rr_side=tp-1.0,
                        dlim=0.16, dd_kill=0.14,
                    ))
    # Volatile-focus with LGBM
    candidates.append(make("LGBM_vol_sc65_r10_tp5", 0.10, 0.65, sc_side=0.72, sc_vol=0.60,
                           pos=2, tp_rr=4.0, tp_rr_vol=5.0, tp_rr_side=3.0, dlim=0.16, dd_kill=0.14))
    candidates.append(make("LGBM_vol_sc68_r8_tp6p3", 0.08, 0.68, sc_side=0.74, sc_vol=0.62,
                           pos=3, tp_rr=4.0, tp_rr_vol=6.0, tp_rr_side=3.0, dlim=0.16, dd_kill=0.14))
    return candidates


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acc1_v14pp_profit.yaml",
                        help="Base config (must have fold prediction cache)")
    parser.add_argument("--use-lgbm-folds", action="store_true",
                        help="Build/use LGBM per-fold predictions (honest WF, slower ~20-30min)")
    parser.add_argument("--test-start", default="2024-08-14",
                        help="Must match cache: 2024-08-14 for _all cache")
    parser.add_argument("--max-folds", type=int, default=None,
                        help="None = all folds → hits _all cache")
    parser.add_argument("--starting-balance", type=float, default=200.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out-prefix", default="wf_daily60_v2_sweep")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    base_settings = load_settings(config_path)

    print("=" * 80)
    print("  SWEEP V2 — Target: $60/day avg, DD<18%, PF>=2.0 (HONEST WF)")
    print("=" * 80)
    print(f"  Base config  : {config_path.name}")
    print(f"  Model type   : {'LGBM per-fold (honest WF, no leakage)' if args.use_lgbm_folds else 'VotingClassifier per-fold (default)'}")
    print(f"  Test start   : {args.test_start}")
    print(f"  Balance/day  : ${args.starting_balance}")
    print()

    print(f"Loading dataset + building fold predictions...")
    full_ds = load_or_build_dataset(config_path, cache=not args.no_cache)
    m1_df = load_m1()

    if args.use_lgbm_folds:
        fold_payloads = build_fold_predictions_lgbm(
            config_path=config_path,
            full_ds=full_ds,
            use_cache=not args.no_cache,
            max_folds=args.max_folds,
            test_start=args.test_start,
        )
        lgbm_candidates = build_lgbm_candidates()
        print(f"  [LGBM] Additional {len(lgbm_candidates)} LGBM-specific candidates loaded")
    else:
        fold_payloads = build_fold_predictions(
            config_path=config_path,
            full_ds=full_ds,
            use_cache=not args.no_cache,
            max_folds=args.max_folds,
            test_start=args.test_start,
        )
        lgbm_candidates = []

    print(f"Folds: {len(fold_payloads)} | span: {fold_payloads[0]['test_start']} → {fold_payloads[-1]['test_end']}")

    candidates = build_candidates_v2()
    if lgbm_candidates:
        candidates.extend(lgbm_candidates)
    print(f"\nTesting {len(candidates)} candidates...\n")

    hdr = (f"{'#':>3}  {'Name':<28} {'Res':>4}  {'Avg$/d':>7} {'Med$/d':>7} "
           f"{'PF':>5} {'DD%':>6} {'≥$60':>6} {'≥$100':>6} {'Neg%':>5} {'Trd/d':>5} {'Score':>7}")
    print(hdr)
    print("─" * 110)

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
            candidate={k: v for k, v in cand.items() if not k.startswith("_")},
            fold_payloads=fold_payloads,
            m1_df=m1_df,
            starting_balance=args.starting_balance,
        )
        s = result["summary"]
        custom_score, feas = score_v2(s)
        s["score_v2"] = custom_score
        s["feasible_strict"] = feas
        results.append(result)

        mark = "PASS" if feas else "FAIL"
        print(
            f"{idx:>3}  {name:<28} {mark:>4}  "
            f"${s['avg_daily_net']:>6.2f} ${s['median_daily_net']:>6.2f} "
            f"{s['profit_factor']:>5.2f} {s['worst_day_dd_pct']:>5.1f}% "
            f"{s['pct_days_ge_60']:>5.1f}% {s['pct_days_ge_100']:>5.1f}% "
            f"{s['pct_negative_days']:>4.1f}% {s['avg_trades_per_day']:>5.2f} "
            f"{custom_score:>7.3f}"
        )

    # ── Results ──────────────────────────────────────────────────────────────
    summaries = sorted(results, key=lambda r: r["summary"].get("score_v2", -999), reverse=True)
    pass_results = [r for r in summaries if r["summary"].get("feasible_strict", False)]

    print("\n" + "=" * 110)
    print(f"STRICT PASS (PF≥2.0, DD<18%): {len(pass_results)} / {len(results)}")
    print()
    print("TOP 15 (by score):")
    print(hdr)
    print("─" * 110)
    for rank, r in enumerate(summaries[:15], 1):
        s = r["summary"]
        feas = s.get("feasible_strict", False)
        mark = "PASS" if feas else "FAIL"
        print(
            f"{rank:>3}  {s['candidate']:<28} {mark:>4}  "
            f"${s['avg_daily_net']:>6.2f} ${s['median_daily_net']:>6.2f} "
            f"{s['profit_factor']:>5.2f} {s['worst_day_dd_pct']:>5.1f}% "
            f"{s['pct_days_ge_60']:>5.1f}% {s['pct_days_ge_100']:>5.1f}% "
            f"{s['pct_negative_days']:>4.1f}% {s['avg_trades_per_day']:>5.2f} "
            f"{s.get('score_v2', 0):>7.3f}"
        )

    print()
    best = summaries[0]["summary"]
    print(f"{'='*60}")
    print(f"BEST: {best['candidate']}")
    print(f"  Avg daily:    ${best['avg_daily_net']:.2f}")
    print(f"  Median daily: ${best['median_daily_net']:.2f}")
    print(f"  PF:           {best['profit_factor']:.3f}")
    print(f"  Max DD:       {best['worst_day_dd_pct']:.2f}%")
    print(f"  Days ≥ $60:   {best['pct_days_ge_60']:.2f}%")
    print(f"  Days ≥ $100:  {best['pct_days_ge_100']:.2f}%")
    print(f"  Negative days: {best['pct_negative_days']:.2f}%")
    print(f"  Total trades: {best['total_trades']}")
    print(f"  Score:        {best.get('score_v2', 0):.3f}")
    if pass_results:
        best_pass = pass_results[0]["summary"]
        print(f"\nBEST STRICT PASS: {best_pass['candidate']}")
        print(f"  Avg daily: ${best_pass['avg_daily_net']:.2f}  PF={best_pass['profit_factor']:.3f}  DD={best_pass['worst_day_dd_pct']:.1f}%")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = [r["summary"] for r in results]
    suffix = "_lgbm" if args.use_lgbm_folds else ""
    out_json = out_dir / f"{args.out_prefix}{suffix}_summary.json"
    out_csv  = out_dir / f"{args.out_prefix}{suffix}_summary.csv"
    out_json.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(all_summaries).sort_values("score_v2", ascending=False).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_json.name}")
    print(f"       {out_csv.name}")


if __name__ == "__main__":
    main()
