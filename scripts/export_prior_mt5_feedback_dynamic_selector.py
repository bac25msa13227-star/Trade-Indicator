from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeError:
            continue
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON file: {path}")


def resolve_csv(manifest_path: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw
    for candidate in [ROOT / raw, manifest_path.parent / raw.name, manifest_path.parent / raw]:
        if candidate.exists():
            return candidate
    return ROOT / raw


def parse_bootstrap_map(values: list[str] | None) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"--bootstrap-map must be FOLD=DIR, got {value!r}")
        fold_text, dir_text = value.split("=", 1)
        mapping[int(fold_text.strip())] = Path(dir_text.strip())
    return mapping


def find_fold(manifest: dict[str, Any], fold_id: int) -> dict[str, Any]:
    for fold in manifest.get("folds", []) or []:
        if int(fold["fold"]) == int(fold_id):
            return dict(fold)
    raise ValueError(f"fold {fold_id} not found")


def copy_bootstrap_fold(source_dir: Path, fold_id: int, out_dir: Path) -> dict[str, Any]:
    manifest_path = source_dir / "manifest.json"
    source_manifest = read_json(manifest_path)
    source_fold = find_fold(source_manifest, fold_id)
    source_csv = resolve_csv(manifest_path, str(source_fold["csv"]))
    target_csv = out_dir / f"fold_{fold_id:02d}_signals.csv"
    shutil.copy2(source_csv, target_csv)
    signals = max(0, sum(1 for _ in target_csv.open("r", encoding="utf-8")) - 1)
    source_fold["csv"] = str(target_csv)
    source_fold["signals"] = int(signals)
    source_fold["selection_mode"] = "bootstrap_declared_map"
    source_fold["selection_uses_current_fold_metrics"] = False
    source_fold["selected_candidate_dir"] = str(source_dir)
    for key in list(source_fold):
        if key.startswith("selected_candidate_"):
            source_fold.pop(key, None)
    return source_fold


def choose_from_history(history: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    if history.empty:
        return {"exclude_hours": [], "min_probability": None, "history_rows": 0}

    history = history.copy()
    if "signal_hour" not in history.columns:
        if "signal_open_time" in history.columns:
            history["signal_hour"] = pd.to_datetime(
                history["signal_open_time"],
                format="%Y.%m.%d %H:%M",
                errors="coerce",
            ).dt.hour
        else:
            history["signal_hour"] = history.get("hour")
    history["signal_hour"] = pd.to_numeric(history["signal_hour"], errors="coerce")
    history["profit"] = pd.to_numeric(history["profit"], errors="coerce").fillna(0.0)
    history["signal_probability"] = pd.to_numeric(history["signal_probability"], errors="coerce")

    hour_stats = (
        history.dropna(subset=["signal_hour"])
        .groupby("signal_hour")
        .agg(n=("profit", "size"), pnl=("profit", "sum"), avg=("profit", "mean"))
        .reset_index()
    )
    bad_hours = hour_stats[
        (hour_stats["n"] >= int(args.min_hour_trades))
        & (hour_stats["pnl"] < float(args.bad_hour_max_pnl))
    ].sort_values(["pnl", "avg"], ascending=[True, True])
    exclude_hours = [int(x) for x in bad_hours["signal_hour"].head(int(args.max_exclude_hours)).tolist()]

    best_threshold: float | None = None
    best_score = -1.0e18
    raw_thresholds = str(args.prob_thresholds or "").strip()
    if raw_thresholds.lower() in {"", "none", "null", "off"}:
        thresholds = [None]
    else:
        thresholds = [None] + [float(x) for x in raw_thresholds.split(",") if x.strip()]
    total_rows = max(1, len(history))
    for threshold in thresholds:
        filtered = history
        if exclude_hours:
            filtered = filtered[~filtered["signal_hour"].isin(exclude_hours)]
        if threshold is not None:
            filtered = filtered[filtered["signal_probability"] >= threshold]
        if len(filtered) < max(int(args.min_filtered_trades), int(total_rows * float(args.min_history_keep_rate))):
            continue
        pnl = float(filtered["profit"].sum())
        worst_hour = (
            filtered.groupby("signal_hour")["profit"].sum().min()
            if not filtered.empty and "signal_hour" in filtered
            else 0.0
        )
        score = pnl + 0.25 * float(worst_hour) + 0.01 * len(filtered)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return {
        "exclude_hours": exclude_hours,
        "min_probability": best_threshold,
        "history_rows": int(len(history)),
        "history_profit": float(history["profit"].sum()),
        "history_bad_hour_count": int(len(exclude_hours)),
    }


def filter_signals(signals: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    frame = signals.copy()
    hours = pd.to_datetime(frame["open_time"], format="%Y.%m.%d %H:%M", errors="raise").dt.hour
    exclude_hours = set(int(x) for x in rule.get("exclude_hours", []) or [])
    if exclude_hours:
        frame = frame[~hours.isin(exclude_hours)].copy()
    threshold = rule.get("min_probability")
    if threshold is not None:
        frame = frame[pd.to_numeric(frame["probability"], errors="coerce") >= float(threshold)].copy()
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a rolling manifest from prior MT5 trade feedback only.")
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-map", action="append", default=None)
    parser.add_argument("--cold-start-folds", type=int, default=5)
    parser.add_argument("--lookback-folds", type=int, default=0)
    parser.add_argument("--risk-pct", type=float, default=None)
    parser.add_argument("--max-risk-pct", type=float, default=None)
    parser.add_argument("--max-exposure-pct", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--max-exclude-hours", type=int, default=4)
    parser.add_argument("--min-hour-trades", type=int, default=8)
    parser.add_argument("--bad-hour-max-pnl", type=float, default=0.0)
    parser.add_argument("--prob-thresholds", default="0.58,0.60,0.62,0.65,0.70")
    parser.add_argument("--min-filtered-trades", type=int, default=30)
    parser.add_argument("--min-history-keep-rate", type=float, default=0.25)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.base_manifest
    base_manifest = read_json(manifest_path)
    feedback = pd.read_csv(args.feedback)
    feedback["fold"] = pd.to_numeric(feedback["fold"], errors="coerce").astype("Int64")
    bootstrap = parse_bootstrap_map(args.bootstrap_map)

    folds: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    total_signals = 0
    for source_fold in base_manifest.get("folds", []) or []:
        fold_id = int(source_fold["fold"])
        if fold_id <= int(args.cold_start_folds):
            if fold_id not in bootstrap:
                raise ValueError(f"cold-start fold {fold_id} needs --bootstrap-map")
            fold = copy_bootstrap_fold(bootstrap[fold_id], fold_id, args.out_dir)
            folds.append(fold)
            total_signals += int(fold["signals"])
            rules.append({"fold": fold_id, "selection_mode": "bootstrap_declared_map"})
            continue

        history = feedback[feedback["fold"] < fold_id].copy()
        if int(args.lookback_folds) > 0:
            history = history[history["fold"] >= fold_id - int(args.lookback_folds)].copy()
        rule = choose_from_history(history, args)
        source_csv = resolve_csv(manifest_path, str(source_fold["csv"]))
        signals = pd.read_csv(source_csv)
        export = filter_signals(signals, rule)
        target_csv = args.out_dir / f"fold_{fold_id:02d}_signals.csv"
        export.to_csv(target_csv, index=False)

        risk_pct = float(args.risk_pct if args.risk_pct is not None else source_fold.get("risk_pct", 0.0))
        max_positions = int(args.max_positions if args.max_positions is not None else source_fold.get("max_positions", 1))
        max_risk_pct = float(args.max_risk_pct if args.max_risk_pct is not None else risk_pct)
        max_exposure_pct = float(
            args.max_exposure_pct if args.max_exposure_pct is not None else max_risk_pct * max_positions
        )
        fold = dict(source_fold)
        for key in list(fold):
            if key.startswith("selected_candidate_"):
                fold.pop(key, None)
        fold.update(
            {
            "signals": int(len(export)),
            "csv": str(target_csv),
            "risk_pct": round(risk_pct, 4),
            "max_risk_pct": round(max_risk_pct, 4),
            "max_exposure_pct": round(max_exposure_pct, 4),
            "max_positions": max_positions,
            "selection_mode": "prior_mt5_trade_feedback",
            "selection_uses_current_fold_metrics": False,
            "selection_history_folds": ",".join(str(x) for x in range(1, fold_id)),
            "exclude_hours_from_prior_feedback": ",".join(str(x) for x in rule["exclude_hours"]),
            "min_probability_from_prior_feedback": rule.get("min_probability"),
            "history_rows": rule.get("history_rows", 0),
            }
        )
        folds.append(fold)
        total_signals += int(len(export))
        rules.append({"fold": fold_id, **rule, "signals": int(len(export)), "risk_pct": risk_pct})

    manifest = {
        "all_signals": None,
        "folds": folds,
        "total_signals": int(total_signals),
        "live_protocol": True,
        "prior_mt5_feedback_dynamic_selector": True,
        "selection_uses_current_fold_metrics": False,
        "research_oracle_fold_selection": False,
        "adaptive_per_fold": False,
        "base_manifest": str(args.base_manifest),
        "feedback": str(args.feedback),
        "cold_start_folds": int(args.cold_start_folds),
        "lookback_folds": int(args.lookback_folds),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(rules).to_csv(args.out_dir / "dynamic_selector_rules.csv", index=False)
    print(json.dumps({"manifest": str(args.out_dir / "manifest.json"), "folds": len(folds), "signals": total_signals}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
