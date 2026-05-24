from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_mt5_feedback_rolling_selector import as_float, as_int, read_json, resolve_csv  # noqa: E402


MT5_COLUMNS = ["open_time", "direction", "entry_price", "sl_price", "tp_price", "atr", "probability"]


def _parse_map(values: list[str] | None) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"map item must be FOLD=DIR, got {value!r}")
        fold_text, dir_text = value.split("=", 1)
        out[int(fold_text.strip())] = Path(dir_text.strip())
    return out


def _parse_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str.replace(".", "-", regex=False), errors="coerce")


def _find_fold(manifest: dict[str, Any], fold_id: int) -> dict[str, Any]:
    for fold in manifest.get("folds", []) or []:
        if int(fold["fold"]) == int(fold_id):
            return dict(fold)
    raise ValueError(f"fold {fold_id} not found")


def _load_signals(manifest_path: Path, fold: dict[str, Any]) -> pd.DataFrame:
    path = resolve_csv(manifest_path, str(fold["csv"]))
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=MT5_COLUMNS + ["open_dt"])
    frame = frame[[c for c in MT5_COLUMNS if c in frame.columns]].copy()
    frame["open_dt"] = _parse_time(frame["open_time"])
    frame = frame.dropna(subset=["open_dt"]).copy()
    return frame


def _trigger_time(feedback: pd.DataFrame, fold_id: int, policy: str, args: argparse.Namespace) -> pd.Timestamp | None:
    fold_feedback = feedback[pd.to_numeric(feedback.get("fold"), errors="coerce") == int(fold_id)].copy()
    if fold_feedback.empty:
        return None
    fold_feedback["close_dt"] = pd.to_datetime(fold_feedback["close_time"], errors="coerce")
    fold_feedback["balance_after"] = pd.to_numeric(fold_feedback.get("balance_after"), errors="coerce")
    fold_feedback = fold_feedback.dropna(subset=["close_dt"]).sort_values("close_dt").reset_index(drop=True)
    if fold_feedback.empty:
        return None
    if policy == "immediate":
        return pd.Timestamp.min
    if policy == "trade_count":
        if len(fold_feedback) <= int(args.trigger_trade_count):
            return None
        return pd.Timestamp(fold_feedback.loc[int(args.trigger_trade_count) - 1, "close_dt"])
    if policy == "balance_below":
        subset = fold_feedback.iloc[max(0, int(args.trigger_min_trades) - 1) :].copy()
        hit = subset[subset["balance_after"] <= float(args.trigger_balance_below)]
        if hit.empty:
            return None
        return pd.Timestamp(hit.iloc[0]["close_dt"])
    if policy == "progress":
        fold_start = fold_feedback["close_dt"].min()
        fold_end = fold_feedback["close_dt"].max()
        seconds = max(1.0, (fold_end - fold_start).total_seconds())
        subset = fold_feedback.iloc[max(0, int(args.trigger_min_trades) - 1) :].copy()
        elapsed = (subset["close_dt"] - fold_start).dt.total_seconds() / seconds
        required = float(args.deposit) + elapsed * (float(args.target_balance) - float(args.deposit)) * float(args.trigger_progress_fraction)
        hit = subset[subset["balance_after"] < required]
        if hit.empty:
            return None
        return pd.Timestamp(hit.iloc[0]["close_dt"])
    raise ValueError(f"Unsupported trigger policy: {policy}")


def _merge_signals(primary: pd.DataFrame, fallback: pd.DataFrame, trigger: pd.Timestamp | None) -> pd.DataFrame:
    if trigger is None:
        merged = primary.copy()
    elif trigger == pd.Timestamp.min:
        merged = fallback.copy()
    else:
        left = primary[primary["open_dt"] <= trigger].copy()
        right = fallback[fallback["open_dt"] > trigger].copy()
        merged = pd.concat([left, right], ignore_index=True)
    if merged.empty:
        return pd.DataFrame(columns=MT5_COLUMNS)
    merged = (
        merged.sort_values(["open_dt", "direction", "entry_price"])
        .drop_duplicates(["open_time", "direction", "entry_price", "sl_price", "tp_price"], keep="first")
        .sort_values("open_dt")
    )
    return merged[MT5_COLUMNS].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a static MT5 manifest that replays an online failover trigger from primary to fallback family.")
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--primary-results", type=Path, required=True)
    parser.add_argument("--primary-feedback", type=Path, required=True)
    parser.add_argument("--fallback-map", action="append", default=None, help="FOLD=FALLBACK_DIR")
    parser.add_argument("--default-fallback-dir", type=Path, default=None, help="Fallback DIR used for every fold unless --fallback-map overrides it.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trigger-policy", choices=["immediate", "trade_count", "balance_below", "progress"], default="progress")
    parser.add_argument("--trigger-trade-count", type=int, default=50)
    parser.add_argument("--trigger-min-trades", type=int, default=40)
    parser.add_argument("--trigger-balance-below", type=float, default=260.0)
    parser.add_argument("--trigger-progress-fraction", type=float, default=0.45)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--max-dd-pct", type=float, default=-20.0)
    parser.add_argument("--failover-only-failing-folds", action="store_true")
    parser.add_argument("--use-fallback-risk-on-failover", action="store_true", help="Replay failover folds with fallback risk/exposure metadata.")
    parser.add_argument(
        "--fallback-map-policy",
        choices=["prior_live", "research_oracle"],
        default="prior_live",
        help="Use research_oracle when fallback folds were chosen from current-fold results.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    primary_manifest = read_json(args.primary_manifest)
    primary_results = pd.read_csv(args.primary_results)
    for col in ["fold", "final_balance", "max_dd_pct"]:
        if col in primary_results.columns:
            primary_results[col] = pd.to_numeric(primary_results[col], errors="coerce")
    feedback = pd.read_csv(args.primary_feedback)
    fallback_map = _parse_map(args.fallback_map)

    out_folds: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    total = 0
    for primary_fold in primary_manifest.get("folds", []) or []:
        fold_id = int(primary_fold["fold"])
        primary_row = primary_results[primary_results["fold"] == fold_id]
        primary_final = as_float(primary_row.iloc[0].get("final_balance")) if not primary_row.empty else math.nan
        primary_dd = as_float(primary_row.iloc[0].get("max_dd_pct")) if not primary_row.empty else math.nan
        primary_pass = primary_final >= float(args.target_balance) and primary_dd >= float(args.max_dd_pct)
        fallback_dir = fallback_map.get(fold_id) or args.default_fallback_dir
        use_failover = fallback_dir is not None and (not args.failover_only_failing_folds or not primary_pass)
        primary_signals = _load_signals(args.primary_manifest, primary_fold)
        trigger = None
        mode = "primary_only"
        fallback_signals = pd.DataFrame(columns=MT5_COLUMNS + ["open_dt"])
        fallback_fold: dict[str, Any] | None = None
        if use_failover:
            fallback_manifest_path = fallback_dir / "manifest.json"
            fallback_manifest = read_json(fallback_manifest_path)
            fallback_fold = _find_fold(fallback_manifest, fold_id)
            fallback_signals = _load_signals(fallback_manifest_path, fallback_fold)
            trigger = _trigger_time(feedback, fold_id, args.trigger_policy, args)
            mode = f"online_failover_replay_{args.trigger_policy}"
        export = _merge_signals(primary_signals, fallback_signals, trigger) if use_failover else primary_signals[MT5_COLUMNS].copy()
        target_csv = args.out_dir / f"fold_{fold_id:02d}_signals.csv"
        export.to_csv(target_csv, index=False, float_format="%.5f")
        fold = dict(primary_fold)
        if fallback_fold is not None and (trigger == pd.Timestamp.min or args.use_fallback_risk_on_failover):
            for key in ["risk_pct", "max_risk_pct", "max_exposure_pct", "max_positions", "max_positions_hint"]:
                if key in fallback_fold:
                    fold[key] = fallback_fold[key]
        fold.update(
            {
                "signals": int(len(export)),
                "csv": str(target_csv),
                "selection_mode": mode,
                "selection_uses_current_fold_metrics": False,
                "online_failover_replay": bool(use_failover),
                "online_failover_primary_manifest": str(args.primary_manifest),
                "online_failover_fallback_dir": str(fallback_dir) if fallback_dir else None,
                "online_failover_fallback_map_policy": str(args.fallback_map_policy) if use_failover else None,
                "online_failover_trigger_policy": str(args.trigger_policy) if use_failover else None,
                "online_failover_trigger_time": trigger.strftime("%Y.%m.%d %H:%M:%S") if isinstance(trigger, pd.Timestamp) and trigger != pd.Timestamp.min else ("immediate" if trigger == pd.Timestamp.min else None),
                "online_failover_uses_only_pretrigger_feedback": bool(use_failover),
                "primary_final_balance": primary_final,
                "primary_max_dd_pct": primary_dd,
            }
        )
        out_folds.append(fold)
        total += int(len(export))
        audit_rows.append(
            {
                "fold": fold_id,
                "mode": mode,
                "primary_final_balance": primary_final,
                "primary_max_dd_pct": primary_dd,
                "primary_signals": int(len(primary_signals)),
                "fallback_dir": str(fallback_dir) if fallback_dir else None,
                "fallback_signals": int(len(fallback_signals)),
                "trigger_time": fold["online_failover_trigger_time"],
                "export_signals": int(len(export)),
            }
        )

    manifest = {
        "all_signals": None,
        "folds": out_folds,
        "total_signals": int(total),
        "live_protocol": True,
        "mt5_online_failover_replay": True,
        "selection_uses_current_fold_metrics": False,
        "research_oracle_fold_selection": args.fallback_map_policy == "research_oracle",
        "fallback_map_policy": str(args.fallback_map_policy),
        "use_fallback_risk_on_failover": bool(args.use_fallback_risk_on_failover),
        "adaptive_per_fold": False,
        "primary_manifest": str(args.primary_manifest),
        "primary_results": str(args.primary_results),
        "primary_feedback": str(args.primary_feedback),
        "default_fallback_dir": str(args.default_fallback_dir) if args.default_fallback_dir else None,
        "trigger_policy": str(args.trigger_policy),
        "note": "Switch time is computed from same-fold closed-trade feedback available before trigger; fallback choice must be supplied by a prior/live policy.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(audit_rows).to_csv(args.out_dir / "failover_audit.csv", index=False)
    print(json.dumps({"manifest": str(args.out_dir / "manifest.json"), "folds": len(out_folds), "signals": total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
