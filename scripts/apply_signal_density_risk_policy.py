from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except (UnicodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not parse JSON: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a deterministic signal-density risk policy to a WF manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-risk-pct", type=float, default=3.5)
    parser.add_argument("--low-signal-risk-pct", type=float, default=6.0)
    parser.add_argument("--low-signal-threshold", type=int, default=250)
    parser.add_argument("--very-low-signal-risk-pct", type=float, default=None)
    parser.add_argument("--very-low-signal-threshold", type=int, default=0)
    parser.add_argument("--min-train-rows-for-boost", type=int, default=0)
    parser.add_argument("--score-boost-risk-pct", type=float, default=None)
    parser.add_argument("--score-boost-max-signals", type=int, default=0)
    parser.add_argument("--score-boost-min-mean-score", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=1)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    folds = manifest.get("folds") or []
    changed = []
    for fold in folds:
        signals = int(fold.get("signals") or 0)
        train_rows = int(float(fold.get("profitr_model_train_rows") or 0))
        mean_score_raw = fold.get("selected_mean_score")
        mean_score = float(mean_score_raw) if mean_score_raw is not None else None
        can_boost = train_rows >= int(args.min_train_rows_for_boost)
        risk = float(args.base_risk_pct)
        reason = "base"
        if (
            can_boost
            and args.very_low_signal_risk_pct is not None
            and int(args.very_low_signal_threshold) > 0
            and signals <= int(args.very_low_signal_threshold)
        ):
            risk = float(args.very_low_signal_risk_pct)
            reason = "very_low_signal"
        elif can_boost and signals <= int(args.low_signal_threshold):
            risk = float(args.low_signal_risk_pct)
            reason = "low_signal"
        elif (
            can_boost
            and args.score_boost_risk_pct is not None
            and int(args.score_boost_max_signals) > 0
            and args.score_boost_min_mean_score is not None
            and mean_score is not None
            and signals <= int(args.score_boost_max_signals)
            and mean_score >= float(args.score_boost_min_mean_score)
        ):
            risk = float(args.score_boost_risk_pct)
            reason = "score_boost"
        fold["risk_pct"] = round(risk, 4)
        fold["max_risk_pct"] = round(risk, 4)
        fold["max_positions"] = int(args.max_positions)
        fold["max_exposure_pct"] = round(risk * int(args.max_positions), 4)
        fold["signal_density_risk_policy"] = {
            "base_risk_pct": float(args.base_risk_pct),
            "low_signal_risk_pct": float(args.low_signal_risk_pct),
            "low_signal_threshold": int(args.low_signal_threshold),
            "very_low_signal_risk_pct": args.very_low_signal_risk_pct,
            "very_low_signal_threshold": int(args.very_low_signal_threshold),
            "min_train_rows_for_boost": int(args.min_train_rows_for_boost),
            "score_boost_risk_pct": args.score_boost_risk_pct,
            "score_boost_max_signals": int(args.score_boost_max_signals),
            "score_boost_min_mean_score": args.score_boost_min_mean_score,
            "uses_current_fold_performance": False,
            "uses_current_fold_signal_count": True,
        }
        if risk != float(args.base_risk_pct):
            changed.append({"fold": int(fold.get("fold")), "signals": signals, "risk_pct": risk, "reason": reason})

    manifest["signal_density_risk_policy"] = {
        "base_risk_pct": float(args.base_risk_pct),
        "low_signal_risk_pct": float(args.low_signal_risk_pct),
        "low_signal_threshold": int(args.low_signal_threshold),
        "very_low_signal_risk_pct": args.very_low_signal_risk_pct,
        "very_low_signal_threshold": int(args.very_low_signal_threshold),
        "min_train_rows_for_boost": int(args.min_train_rows_for_boost),
        "score_boost_risk_pct": args.score_boost_risk_pct,
        "score_boost_max_signals": int(args.score_boost_max_signals),
        "score_boost_min_mean_score": args.score_boost_min_mean_score,
        "changed_folds": changed,
        "uses_current_fold_performance": False,
        "uses_current_fold_signal_count": True,
    }
    manifest["adaptive_per_fold"] = False
    manifest["research_oracle_fold_selection"] = False
    manifest["selection_uses_current_fold_metrics"] = False

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "changed_folds": changed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
