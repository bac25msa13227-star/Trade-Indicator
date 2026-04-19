#!/usr/bin/env python3
"""
Sweep v5 — TARGETED FIX for circuit-breaker bottleneck.
Key insight: r=15% with dd_kill=16% → 48% of days stop after first loss.
Fix: r=5-7% + no strategy_score gate + dd_kill=18% → all 14 signals/day execute.

Theoretical at no_gate + 5%: $93.69/day
Theoretical at sc>=0 + 5%: $64.96/day
Target: avg≥$60/day, DD<18%, PF≥2.0
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np, pandas as pd
from wf_daily_reset_optimizer import (
    load_or_build_dataset, load_m1, build_fold_predictions,
    simulate_candidate_daily_reset, _candidate_fingerprint,
)
from xauusd_ai.config import load_settings


def score_candidate(s: dict) -> tuple[float, bool]:
    avg = s["avg_daily_net"]
    pf  = s["profit_factor"]
    dd  = s["worst_day_dd_pct"]
    ge60 = s["pct_days_ge_60"]
    neg  = s["pct_negative_days"]
    feasible = pf >= 2.0 and dd < 18.0 and avg >= 20.0
    if avg >= 60.0 and pf >= 2.0 and dd < 18.0:
        return avg * 3.0 + pf * 10 + ge60 * 0.5 - neg * 0.3, True
    elif pf < 2.0:
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
    Primary hypothesis: lower risk + remove strategy gate = more trades/day = higher profit.
    
    Key vars:
    - risk_per_trade: 0.04 to 0.10
    - min_strategy_score: -10.0 (no gate) vs 0.0 (current)
    - max_drawdown_kill_pct: 0.18 (wider to allow 3 losses @ 5% before stopping)
    - daily_loss_limit_pct: 0.20-0.22
    - min_confidence: 0.83 (keep quality)
    """
    cands = []

    def make(name, sc=0.83, risk=0.05, pos=3, dlim=0.22, dd_kill=0.18,
             cooldown_bars=0, pause_count=99, reentry_bars=0,
             amf=1.0, amr=0, score_gate=-10.0, score_sdw=-10.0, score_vol=-10.0,
             require_trend_align=False) -> dict:
        return {
            "name": name,
            "strategy": {
                "sideway_min_confidence": sc,
                "volatile_min_confidence": sc,
                "min_strategy_score": score_gate,
                "sideway_min_strategy_score": score_sdw,
                "strong_volatility_min_strategy_score": score_vol,
                "blocked_hours_utc": [],
                "silver_bullet_confidence_boost": 0.0,
                "adx_gate_enabled": False,  # DISABLE - not helping
                "adx_min_trend": 8.0,
                "require_trend_alignment": require_trend_align,
                "trend_bypass_confidence": None,
            },
            "risk": {
                "risk_per_trade": risk, "risk_tier_floor": 0.01,
                "max_open_positions": pos,
                "min_confidence": sc,
                "take_profit_rr": 3.5,
                "volatile_take_profit_rr": 3.5, "sideway_take_profit_rr": 3.5,
                "daily_loss_limit_pct": dlim,
                "max_drawdown_kill_pct": dd_kill,
                "consecutive_loss_pause_count": pause_count,
                "consecutive_loss_cooldown_bars": cooldown_bars,
                "anti_martingale_factor": amf,
                "anti_martingale_max_reductions": amr,
                "sideway_risk_multiplier": 1.0, "normal_risk_multiplier": 1.0,
                "strong_volatility_risk_multiplier": 1.0,
                "reentry_cooldown_bars_after_sl": reentry_bars,
                "reentry_min_distance_atr": 0.0,
                "partial_tp_enabled": True, "partial_tp_rr": 0.8, "partial_tp_pct": 0.25,
                "max_risk_fraction": risk,
            },
            "execution": _exec(),
            "ml_threshold_offset": 0.0,
        }

    # ═══════════════════════════════════════════════════════════════
    # GROUP A: LOW RISK + NO STRATEGY GATE (the key hypothesis)
    # Expected: 14 sigs/day × r × $200 × 0.664 → ~$93/day at r=5%
    # ═══════════════════════════════════════════════════════════════
    for r, rl in [(0.04, "r4"), (0.05, "r5"), (0.06, "r6"), (0.07, "r7"), (0.08, "r8")]:
        for sc_l, sc in [("sc83", 0.83), ("sc80", 0.80), ("sc78", 0.78)]:
            for p in [2, 3, 4]:
                cands.append(make(
                    f"A_nogate_{sc_l}_{rl}_p{p}",
                    sc=sc, risk=r, pos=p, dlim=0.22, dd_kill=0.18,
                    cooldown_bars=0, pause_count=99, reentry_bars=0,
                    amf=1.0, amr=0, score_gate=-10.0, score_sdw=-10.0, score_vol=-10.0,
                ))

    # ═══════════════════════════════════════════════════════════════
    # GROUP B: LOW RISK + SCORE >= 0 GATE (9.76 sigs/day)
    # Expected: 9.76 × r × $200 × 0.665 → ~$65/day at r=5%
    # ═══════════════════════════════════════════════════════════════
    for r, rl in [(0.04, "r4"), (0.05, "r5"), (0.06, "r6"), (0.07, "r7"), (0.08, "r8"), (0.10, "r10")]:
        for sc_l, sc in [("sc83", 0.83), ("sc80", 0.80)]:
            for p in [2, 3, 4]:
                cands.append(make(
                    f"B_sc0_{sc_l}_{rl}_p{p}",
                    sc=sc, risk=r, pos=p, dlim=0.22, dd_kill=0.18,
                    cooldown_bars=0, pause_count=99, reentry_bars=0,
                    amf=1.0, amr=0, score_gate=0.0, score_sdw=0.0, score_vol=0.0,
                ))

    # ═══════════════════════════════════════════════════════════════
    # GROUP C: WIDER DD KILL + MODERATE RISK (old-style but fixed)
    # dd_kill=0.18 + r=0.08-0.12 → survive 2-3 losses before stopping
    # ═══════════════════════════════════════════════════════════════
    for r, rl in [(0.08, "r8"), (0.10, "r10"), (0.12, "r12")]:
        for sc_l, sc in [("sc83", 0.83), ("sc80", 0.80)]:
            cands.append(make(
                f"C_wdd_{sc_l}_{rl}_p3",
                sc=sc, risk=r, pos=3, dlim=0.22, dd_kill=0.18,
                cooldown_bars=0, pause_count=99, reentry_bars=0,
                amf=1.0, amr=0, score_gate=-10.0, score_sdw=-10.0, score_vol=-10.0,
            ))

    # ═══════════════════════════════════════════════════════════════
    # GROUP D: TIGHTEST VALID — exact $60 target configs
    # r=5-6%, sc=0.83, no_gate → should hit ~$93/day theoretical
    # Test with actual constraints
    # ═══════════════════════════════════════════════════════════════
    for r in [0.05, 0.06]:
        # With conservative drawdown management
        cands.append(make(
            f"D_conservative_r{int(r*100)}_nogate",
            sc=0.83, risk=r, pos=4, dlim=0.20, dd_kill=0.17,
            cooldown_bars=2, pause_count=3, reentry_bars=1,
            amf=0.95, amr=2, score_gate=-10.0, score_sdw=-10.0, score_vol=-10.0,
        ))
        # With strict but fair drawdown management
        cands.append(make(
            f"D_fair_r{int(r*100)}_nogate",
            sc=0.83, risk=r, pos=4, dlim=0.21, dd_kill=0.18,
            cooldown_bars=0, pause_count=99, reentry_bars=0,
            amf=1.0, amr=0, score_gate=-10.0, score_sdw=-10.0, score_vol=-10.0,
        ))

    # ═══════════════════════════════════════════════════════════════
    # GROUP E: REFERENCE — old config (to verify hypothesis by comparison)
    # r=15%, score_gate=0, dd_kill=0.16 → should give ~$20/day as before
    # ═══════════════════════════════════════════════════════════════
    cands.append(make(
        "E_old_r15_sc83_gate0",
        sc=0.83, risk=0.15, pos=3, dlim=0.18, dd_kill=0.16,
        cooldown_bars=4, pause_count=2, reentry_bars=4,
        amf=0.90, amr=2, score_gate=0.0, score_sdw=0.0, score_vol=0.10,
        require_trend_align=True,  # old config
    ))
    cands.append(make(
        "E_old_r10_sc83_gate0",
        sc=0.83, risk=0.10, pos=3, dlim=0.18, dd_kill=0.16,
        cooldown_bars=4, pause_count=2, reentry_bars=4,
        amf=0.90, amr=2, score_gate=0.0, score_sdw=0.0, score_vol=0.10,
        require_trend_align=True,  # old config
    ))

    return cands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acc1_v14pp_profit.yaml")
    parser.add_argument("--starting-balance", type=float, default=200.0)
    parser.add_argument("--out-prefix", default="wf_sweep_v5_lowrisk")
    parser.add_argument("--test-start", default="2024-08-14")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    base_settings = load_settings(config_path)

    print("=" * 95)
    print("  SWEEP V5 — LOW RISK + NO STRATEGY GATE FIX | Target: $60/day, DD<18%, PF≥2")
    print("  Hypothesis: r=5% + no_gate + dd_kill=18% → 14 sigs/day all execute → ~$93/day")
    print("=" * 95)

    full_ds = load_or_build_dataset(config_path, cache=True)
    m1_df = load_m1()

    fold_payloads = build_fold_predictions(
        config_path=config_path, full_ds=full_ds,
        use_cache=True, max_folds=None, test_start=args.test_start,
        use_realized_rr_label=False,  # USE OLD MODEL (AUC=0.67)
    )
    print(f"Folds: {len(fold_payloads)} | {fold_payloads[0]['test_start']} → {fold_payloads[-1]['test_end']}")
    aucs = [fi["roc_auc"] for fi in fold_payloads]
    print(f"Model: avg AUC={np.mean(aucs):.4f}")
    print()

    candidates = build_candidates()
    # Deduplicate by fingerprint
    seen, unique = set(), []
    for c in candidates:
        fp = _candidate_fingerprint(c)
        if fp not in seen:
            seen.add(fp)
            unique.append(c)
    candidates = unique
    print(f"Testing {len(candidates)} unique candidates...\n")

    hdr = (f"{'#':>4}  {'Name':<42} {'Res':>4}  {'Avg$/d':>7} {'Med$/d':>7} "
           f"{'PF':>5} {'DD%':>6} {'≥$60':>6} {'Neg%':>5} {'Trd/d':>5}")
    print(hdr)
    print("─" * 100)

    results = []
    for idx, cand in enumerate(candidates, 1):
        name = cand.get("name", f"cand_{idx:03d}")
        result = simulate_candidate_daily_reset(
            name=name, base_settings=base_settings,
            candidate={k: v for k, v in cand.items() if not k.startswith("_")},
            fold_payloads=fold_payloads, m1_df=m1_df,
            starting_balance=args.starting_balance,
        )
        s = result["summary"]
        sc, feas = score_candidate(s)
        s["score"] = sc
        s["feasible"] = feas
        results.append(result)

        mark = "PASS" if feas else "FAIL"
        star = "★" if s["avg_daily_net"] >= 60.0 and s["profit_factor"] >= 2.0 and s["worst_day_dd_pct"] < 18.0 else " "
        print(
            f"{idx:>4}  {star}{name:<41} {mark:>4}  "
            f"${s['avg_daily_net']:>6.2f} ${s['median_daily_net']:>6.2f} "
            f"{s['profit_factor']:>5.2f} {s['worst_day_dd_pct']:>5.1f}% "
            f"{s['pct_days_ge_60']:>5.1f}% {s['pct_negative_days']:>4.1f}% "
            f"{s['avg_trades_per_day']:>5.2f}"
        )

    ranked = sorted(results, key=lambda r: r["summary"].get("score", -999), reverse=True)
    pass60 = [r for r in ranked if
              r["summary"]["avg_daily_net"] >= 60.0 and
              r["summary"]["profit_factor"] >= 2.0 and
              r["summary"]["worst_day_dd_pct"] < 18.0]

    print("\n" + "=" * 100)
    print(f"STRICT PASS (avg≥$60, DD<18%, PF≥2): {len(pass60)}")
    print(f"Best avg $/day: ${max(r['summary']['avg_daily_net'] for r in results):.2f}")
    print()
    print("TOP 20:")
    print(hdr)
    print("─" * 100)
    for rank, r in enumerate(ranked[:20], 1):
        s = r["summary"]
        feas = s.get("feasible")
        mark = "PASS" if feas else "FAIL"
        star = "★" if s["avg_daily_net"] >= 60.0 and s["profit_factor"] >= 2.0 and s["worst_day_dd_pct"] < 18.0 else " "
        print(
            f"{rank:>4}  {star}{s['candidate']:<41} {mark:>4}  "
            f"${s['avg_daily_net']:>6.2f} ${s['median_daily_net']:>6.2f} "
            f"{s['profit_factor']:>5.2f} {s['worst_day_dd_pct']:>5.1f}% "
            f"{s['pct_days_ge_60']:>5.1f}% {s['pct_negative_days']:>4.1f}% "
            f"{s['avg_trades_per_day']:>5.2f}"
        )

    if pass60:
        print("\n" + "★" * 80)
        print(f"  {len(pass60)} CONFIGS MEET TARGET (avg≥$60, DD<18%, PF≥2)!")
        for r in pass60:
            s = r["summary"]
            print(f"  → {s['candidate']}: avg=${s['avg_daily_net']:.2f}, "
                  f"PF={s['profit_factor']:.2f}, DD={s['worst_day_dd_pct']:.1f}%, "
                  f"Trd/d={s['avg_trades_per_day']:.2f}")
        print("★" * 80)

    out = ROOT / "outputs"
    sums = [r["summary"] for r in results]
    (out / f"{args.out_prefix}_summary.json").write_text(
        json.dumps(sums, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(sums).sort_values("score", ascending=False).to_csv(
        out / f"{args.out_prefix}_summary.csv", index=False)
    print(f"\nSaved: {args.out_prefix}_summary.json / .csv")


if __name__ == "__main__":
    main()
