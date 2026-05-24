from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    print("RUN", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except (UnicodeError, json.JSONDecodeError):
            continue
    return {}


def summarize_raw(results_csv: Path, target: float, min_dd: float, deposit: float) -> dict[str, Any]:
    df = pd.read_csv(results_csv)
    df["final_balance"] = pd.to_numeric(df["final_balance"], errors="coerce")
    df["max_dd_pct"] = pd.to_numeric(df["max_dd_pct"], errors="coerce")
    strict = (df["final_balance"] >= target) & (df["max_dd_pct"] >= min_dd) & (df["final_balance"] >= deposit)
    return {
        "raw_strict_pass": int(strict.sum()),
        "raw_target_pass": int((df["final_balance"] >= target).sum()),
        "raw_dd_pass": int((df["max_dd_pct"] >= min_dd).sum()),
        "raw_loss_folds": int((df["final_balance"] < deposit).sum()),
        "raw_min_final": float(df["final_balance"].min()),
        "raw_median_final": float(df["final_balance"].median()),
        "raw_worst_dd": float(df["max_dd_pct"].min()),
        "raw_fail_folds": df.loc[~strict, "fold"].astype(int).tolist(),
    }


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError(f"empty float list: {text!r}")
    return values


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError(f"empty int list: {text!r}")
    return values


def config_grid(
    limit: int,
    candidate_sources: tuple[str, ...],
    *,
    model_types: tuple[str, ...],
    risk_pcts: tuple[float, ...],
    top_ks: tuple[int, ...],
    min_scores: tuple[float, ...],
    recipes: tuple[tuple[float, float, int], ...],
    min_train_rows: tuple[int, ...],
    min_keep: tuple[int, ...],
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for candidate_source in candidate_sources:
        for model_type in model_types:
            for risk_pct in risk_pcts:
                for top_k in top_ks:
                    for min_score in min_scores:
                        for keep in min_keep:
                            for train_rows in min_train_rows:
                                for tp_rr, sl_mult, horizon in recipes:
                                    configs.append(
                                        {
                                            "candidate_source": candidate_source,
                                            "model_type": model_type,
                                            "risk_pct": risk_pct,
                                            "top_k": top_k,
                                            "min_score": min_score,
                                            "min_keep": keep,
                                            "min_train_rows": train_rows,
                                            "tp_rr": tp_rr,
                                            "sl_mult": sl_mult,
                                            "horizon": horizon,
                                        }
                                    )
    return configs[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated clean universe search with MT5 + realistic stress gate.")
    parser.add_argument("--base-manifest", type=Path, default=ROOT / "outputs/target1500_regime_rule_family_selector_v1_buffer_20260515/manifest.json")
    parser.add_argument("--feedback", type=Path, default=ROOT / "outputs/target1500_regime_rule_family_selector_v1_buffer_20260515/mt5_trade_feedback_detailed.csv")
    parser.add_argument("--features", type=Path, default=ROOT / "outputs/mt5_full_ict_wyckoff_features_20210318_20260514_xauusdm.csv")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs/auto_clean_live_gate_search_20260517")
    parser.add_argument("--portable-root", type=Path, default=ROOT / ".mt5_tester_portable/MetaTrader5EXNESS")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--candidate-sources", default="feature_bars,base_manifest")
    parser.add_argument("--start-fold", type=int, default=1)
    parser.add_argument("--end-fold", type=int, default=30)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--min-dd-pct", type=float, default=-20.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--max-loss-folds", type=int, default=1)
    parser.add_argument("--target-pass-required", type=int, default=28)
    parser.add_argument("--strict-pass-required", type=int, default=28)
    parser.add_argument("--mt5-timeout", type=int, default=3600000)
    parser.add_argument("--model-types", default="extra_trees,hgb")
    parser.add_argument("--risk-pcts", default="3.5,4.0,4.5")
    parser.add_argument("--top-ks", default="500,800,1200")
    parser.add_argument("--min-scores", default="-0.10,0.0,0.10")
    parser.add_argument("--min-train-rows-list", default="600")
    parser.add_argument("--min-keep-list", default="80")
    parser.add_argument(
        "--recipes",
        default="2.0:0.8:24,2.5:1.0:18,3.0:1.0:12",
        help="Comma-separated tp:sl:horizon tuples.",
    )
    parser.add_argument("--target-stop-balance", type=float, default=1500.0)
    args = parser.parse_args()

    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    passed: dict[str, Any] | None = None

    candidate_sources = tuple(part.strip() for part in str(args.candidate_sources).split(",") if part.strip())
    bad_sources = sorted(set(candidate_sources).difference({"feature_bars", "base_manifest", "hybrid_base_feature_bars"}))
    if bad_sources:
        raise ValueError(f"Unsupported candidate sources: {bad_sources}")
    model_types = tuple(part.strip() for part in str(args.model_types).split(",") if part.strip())
    bad_models = sorted(set(model_types).difference({"extra_trees", "hgb"}))
    if bad_models:
        raise ValueError(f"Unsupported model types: {bad_models}")
    recipes = tuple(
        (float(parts[0]), float(parts[1]), int(parts[2]))
        for parts in (part.strip().split(":") for part in str(args.recipes).split(",") if part.strip())
    )
    for idx, cfg in enumerate(
        config_grid(
            int(args.limit),
            candidate_sources,
            model_types=model_types,
            risk_pcts=parse_float_list(args.risk_pcts),
            top_ks=parse_int_list(args.top_ks),
            min_scores=parse_float_list(args.min_scores),
            recipes=recipes,
            min_train_rows=parse_int_list(args.min_train_rows_list),
            min_keep=parse_int_list(args.min_keep_list),
        ),
        start=1,
    ):
        name = (
            f"cfg{idx:03d}_{cfg['candidate_source']}_{cfg['model_type']}"
            f"_r{str(cfg['risk_pct']).replace('.', 'p')}_k{cfg['top_k']}"
            f"_s{str(cfg['min_score']).replace('-', 'm').replace('.', 'p')}"
            f"_mk{cfg['min_keep']}_mt{cfg['min_train_rows']}"
            f"_tp{str(cfg['tp_rr']).replace('.', 'p')}_sl{str(cfg['sl_mult']).replace('.', 'p')}_h{cfg['horizon']}"
        )
        out_dir = out_root / name
        if not (out_dir / "manifest.json").exists():
            cmd = [
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
                str(cfg["candidate_source"]),
                "--model-type",
                str(cfg["model_type"]),
                "--min-train-rows",
                str(cfg["min_train_rows"]),
                "--positive-r-threshold",
                "0.05",
                "--min-score",
                str(cfg["min_score"]),
                "--top-k",
                str(cfg["top_k"]),
                "--min-keep",
                str(cfg["min_keep"]),
                "--direction-mode",
                "trade_side",
                "--tp-rr",
                str(cfg["tp_rr"]),
                "--sl-atr-mult",
                str(cfg["sl_mult"]),
                "--candidate-stride",
                "1",
                "--exclude-hours",
                "0,1,2,3,4,5,6,22,23",
                "--exclude-news-blackout",
                "--dedupe-open-time",
                "--cold-start-mode",
                "source_signals",
                "--risk-pct",
                str(cfg["risk_pct"]),
                "--max-risk-pct",
                str(cfg["risk_pct"]),
                "--max-positions",
                "1",
                "--extra-roundtrip-points",
                "30",
                "--thin-hour-extra-points",
                "50",
                "--friday-extra-points",
                "100",
            ]
            proc = run(cmd, timeout=900)
            if proc.returncode != 0:
                summaries.append({"name": name, "stage": "export_failed", "stderr": proc.stderr[-2000:]})
                continue

        results_csv = out_dir / "mt5_wf_results.csv"
        if not results_csv.exists():
            cmd = [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                "scripts/run_mt5_wf_folds.ps1",
                "-Manifest",
                str(out_dir / "manifest.json"),
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
            ]
            proc = run(cmd, timeout=int(args.mt5_timeout))
            if proc.returncode != 0:
                summaries.append({"name": name, "stage": "mt5_failed", "stderr": proc.stderr[-2000:], "stdout": proc.stdout[-2000:]})
                continue

        feedback_csv = out_dir / "mt5_trade_feedback_detailed.csv"
        if not feedback_csv.exists():
            proc = run(
                [
                    sys.executable,
                    "scripts/export_mt5_trade_feedback.py",
                    "--manifest",
                    str(out_dir / "manifest.json"),
                    "--results",
                    str(results_csv),
                    "--out",
                    str(feedback_csv),
                ],
                timeout=300,
            )
            if proc.returncode != 0:
                summaries.append({"name": name, "stage": "feedback_failed", "stderr": proc.stderr[-2000:]})
                continue

        stress_json = out_dir / "stress_full.json"
        stress_csv = out_dir / "stress_full.csv"
        proc = run(
            [
                sys.executable,
                "scripts/mt5_realistic_stress_gate.py",
                "--feedback",
                str(feedback_csv),
                "--out",
                str(stress_json),
                "--fold-report-out",
                str(stress_csv),
                "--deposit",
                str(args.deposit),
                "--target-balance",
                str(args.target_balance),
                "--max-dd-pct",
                str(args.max_dd_pct),
                "--max-loss-folds",
                str(args.max_loss_folds),
                "--extra-roundtrip-points",
                "30",
                "--thin-hour-extra-points",
                "50",
                "--friday-extra-points",
                "100",
            ],
            timeout=300,
        )
        if proc.returncode != 0:
            summaries.append({"name": name, "stage": "stress_failed", "stderr": proc.stderr[-2000:]})
            continue

        stress = read_json(stress_json)
        summary = dict(stress.get("summary", {}))
        summary.update(summarize_raw(results_csv, args.target_balance, args.min_dd_pct, args.deposit))
        summary.update({"name": name, "out_dir": str(out_dir), "config": cfg, "stage": "complete"})
        summaries.append(summary)
        (out_root / "search_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        if (
            int(summary.get("all_pass_folds", 0)) >= int(args.strict_pass_required)
            and int(summary.get("target_pass_folds", 0)) >= int(args.target_pass_required)
            and int(summary.get("loss_folds", 99)) <= int(args.max_loss_folds)
            and float(summary.get("worst_dd_pct", -999.0)) >= float(args.min_dd_pct)
        ):
            passed = summary
            break

    result = {"passed": passed is not None, "winner": passed, "runs": len(summaries)}
    (out_root / "final_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if passed is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
