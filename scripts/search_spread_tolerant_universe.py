from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    print("RUN", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(results_csv: Path, stress_json: Path, target: float, deposit: float, min_dd: float) -> dict[str, Any]:
    results = pd.read_csv(results_csv)
    results["final_balance"] = pd.to_numeric(results["final_balance"], errors="coerce")
    results["max_dd_pct"] = pd.to_numeric(results["max_dd_pct"], errors="coerce")
    raw_strict = (results["final_balance"] >= target) & (results["max_dd_pct"] >= min_dd) & (results["final_balance"] >= deposit)
    stress = read_json(stress_json).get("summary", {})
    return {
        "raw_strict_pass": int(raw_strict.sum()),
        "raw_target_pass": int((results["final_balance"] >= target).sum()),
        "raw_dd_pass": int((results["max_dd_pct"] >= min_dd).sum()),
        "raw_loss_folds": int((results["final_balance"] < deposit).sum()),
        "raw_min_final": round(float(results["final_balance"].min()), 2),
        "raw_median_final": round(float(results["final_balance"].median()), 2),
        "raw_worst_dd": round(float(results["max_dd_pct"].min()), 2),
        "raw_fail_folds": results.loc[~raw_strict, "fold"].astype(int).tolist(),
        "stress_all_pass": int(stress.get("all_pass_folds", 0) or 0),
        "stress_target_pass": int(stress.get("target_pass_folds", 0) or 0),
        "stress_dd_pass": int(stress.get("dd_pass_folds", 0) or 0),
        "stress_loss_folds": int(stress.get("loss_folds", 0) or 0),
        "stress_min_final": float(stress.get("min_final_balance", 0.0) or 0.0),
        "stress_median_final": float(stress.get("median_final_balance", 0.0) or 0.0),
        "stress_worst_dd": float(stress.get("worst_dd_pct", 0.0) or 0.0),
        "stress_gate_pass": bool(stress.get("realistic_gate_pass", False)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Search spread-tolerant signal universes for high-cost XAUUSD broker.")
    parser.add_argument("--base-manifest", type=Path, default=ROOT / "outputs/target1500_regime_rule_family_selector_v1_buffer_20260515/manifest.json")
    parser.add_argument("--feedback", type=Path, default=ROOT / "outputs/target1500_regime_rule_family_selector_v1_buffer_20260515/mt5_trade_feedback_detailed.csv")
    parser.add_argument("--features", type=Path, default=ROOT / "outputs/mt5_full_ict_wyckoff_features_20210318_20260514_xauusdm.csv")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs/spread_tolerant_universe_20260518")
    parser.add_argument("--portable-root", type=Path, default=ROOT / ".mt5_tester_portable/MetaTrader5EXNESS")
    parser.add_argument("--start-fold", type=int, default=1)
    parser.add_argument("--end-fold", type=int, default=30)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--target-stop-balance", type=float, default=1200.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--stress-points", type=float, default=462.0)
    parser.add_argument("--config-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()

    configs = [
        {"name": "h48_sl3_tp25_top120_s025_r35", "tp": 2.5, "sl": 3.0, "h": 48, "top": 120, "score": 0.25, "risk": 3.5, "stride": 3},
        {"name": "h72_sl4_tp30_top100_s030_r35", "tp": 3.0, "sl": 4.0, "h": 72, "top": 100, "score": 0.30, "risk": 3.5, "stride": 3},
        {"name": "h96_sl5_tp35_top80_s035_r30", "tp": 3.5, "sl": 5.0, "h": 96, "top": 80, "score": 0.35, "risk": 3.0, "stride": 6},
        {"name": "h144_sl6_tp40_top60_s040_r25", "tp": 4.0, "sl": 6.0, "h": 144, "top": 60, "score": 0.40, "risk": 2.5, "stride": 6},
        {"name": "h72_sl4_tp30_top220_s015_r60", "tp": 3.0, "sl": 4.0, "h": 72, "top": 220, "score": 0.15, "risk": 6.0, "stride": 2},
        {"name": "h96_sl5_tp35_top180_s020_r65", "tp": 3.5, "sl": 5.0, "h": 96, "top": 180, "score": 0.20, "risk": 6.5, "stride": 3},
        {"name": "h144_sl6_tp40_top160_s020_r70", "tp": 4.0, "sl": 6.0, "h": 144, "top": 160, "score": 0.20, "risk": 7.0, "stride": 3},
        {"name": "h192_sl7_tp45_top140_s025_r70", "tp": 4.5, "sl": 7.0, "h": 192, "top": 140, "score": 0.25, "risk": 7.0, "stride": 4},
    ][int(args.config_offset) : int(args.config_offset) + int(args.limit)]

    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for cfg in configs:
        out_dir = out_root / cfg["name"]
        manifest = out_dir / "manifest.json"
        if not manifest.exists():
            proc = run(
                [
                    sys.executable,
                    "scripts/export_mt5_profitr_model_manifest.py",
                    "--base-manifest",
                    str(args.base_manifest),
                    "--feedback",
                    str(args.feedback),
                    "--features",
                    str(args.features),
                    "--out-dir",
                    str(out_dir),
                    "--candidate-source",
                    "feature_bars",
                    "--model-type",
                    "extra_trees",
                    "--min-train-rows",
                    "800",
                    "--positive-r-threshold",
                    "0.5",
                    "--min-score",
                    str(cfg["score"]),
                    "--top-k",
                    str(cfg["top"]),
                    "--min-keep",
                    "20",
                    "--direction-mode",
                    "trade_side",
                    "--tp-rr",
                    str(cfg["tp"]),
                    "--sl-atr-mult",
                    str(cfg["sl"]),
                    "--horizon-bars",
                    str(cfg["h"]),
                    "--candidate-stride",
                    str(cfg["stride"]),
                    "--exclude-hours",
                    "0,1,2,3,4,5,6,22,23",
                    "--exclude-news-blackout",
                    "--dedupe-open-time",
                    "--cold-start-mode",
                    "no_trade",
                    "--risk-pct",
                    str(cfg["risk"]),
                    "--max-risk-pct",
                    str(cfg["risk"]),
                    "--max-positions",
                    "1",
                    "--extra-roundtrip-points",
                    str(args.stress_points),
                    "--thin-hour-extra-points",
                    "0",
                    "--friday-extra-points",
                    "0",
                ],
                900,
            )
            if proc.returncode != 0:
                summaries.append({"name": cfg["name"], "stage": "export_failed", "output": proc.stdout[-3000:]})
                continue

        results_csv = out_dir / "mt5_wf_results.csv"
        if not results_csv.exists():
            proc = run(
                [
                    "powershell",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    "scripts/run_mt5_wf_folds.ps1",
                    "-Manifest",
                    str(manifest),
                    "-StartFold",
                    str(args.start_fold),
                    "-EndFold",
                    str(args.end_fold),
                    "-PortableRoot",
                    str(args.portable_root),
                    "-TargetBalanceStop",
                    str(args.target_stop_balance),
                    "-MaxDailyLoss",
                    "100",
                    "-MaxDDKillPct",
                    str(args.max_dd_pct),
                    "-MaxPeakDDKillPct",
                    "0",
                    "-TrailingEnabled",
                    "0",
                    "-WaitSeconds",
                    "360",
                    "-Symbol",
                    "XAUUSDm",
                    "-Period",
                    "M5",
                    "-Deposit",
                    str(args.deposit),
                ],
                3600000,
            )
            if proc.returncode != 0:
                summaries.append({"name": cfg["name"], "stage": "mt5_failed", "output": proc.stdout[-5000:]})
                (out_root / "search_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
                continue

        feedback = out_dir / "mt5_trade_feedback_detailed.csv"
        if not feedback.exists():
            proc = run(
                [
                    sys.executable,
                    "scripts/export_mt5_trade_feedback.py",
                    "--manifest",
                    str(manifest),
                    "--results",
                    str(results_csv),
                    "--out",
                    str(feedback),
                ],
                600,
            )
            if proc.returncode != 0:
                summaries.append({"name": cfg["name"], "stage": "feedback_failed", "output": proc.stdout[-3000:]})
                continue

        stress = out_dir / "stress_broker_native.json"
        stress_folds = out_dir / "stress_broker_native_folds.csv"
        proc = run(
            [
                sys.executable,
                "scripts/mt5_realistic_stress_gate.py",
                "--feedback",
                str(feedback),
                "--out",
                str(stress),
                "--fold-report-out",
                str(stress_folds),
                "--deposit",
                str(args.deposit),
                "--target-balance",
                str(args.target_balance),
                "--max-dd-pct",
                str(args.max_dd_pct),
                "--max-loss-folds",
                "0",
                "--extra-roundtrip-points",
                str(args.stress_points),
                "--thin-hour-extra-points",
                "0",
                "--friday-extra-points",
                "0",
            ],
            600,
        )
        if proc.returncode != 0:
            summaries.append({"name": cfg["name"], "stage": "stress_failed", "output": proc.stdout[-3000:]})
            continue
        row = {"name": cfg["name"], "out_dir": str(out_dir), "config": cfg, "stage": "complete"}
        row.update(summarize(results_csv, stress, args.target_balance, args.deposit, -float(args.max_dd_pct)))
        summaries.append(row)
        (out_root / "search_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(json.dumps(row, indent=2), flush=True)

    summaries_sorted = sorted(
        summaries,
        key=lambda item: (
            int(item.get("stress_all_pass", -1)),
            int(item.get("stress_target_pass", -1)),
            int(item.get("stress_dd_pass", -1)),
            float(item.get("stress_median_final", -1.0)),
        ),
        reverse=True,
    )
    result = {"best": summaries_sorted[:10], "runs": len(summaries)}
    (out_root / "final_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
