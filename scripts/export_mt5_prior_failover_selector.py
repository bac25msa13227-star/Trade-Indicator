from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MT5_COLUMNS = ["open_time", "direction", "entry_price", "sl_price", "tp_price", "atr", "probability"]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le"):
        try:
            return json.loads(path.read_text(encoding=encoding).replace("NaN", "null"))
        except (UnicodeError, json.JSONDecodeError):
            continue
    return {}


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def resolve_csv(manifest_path: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw
    candidates = [
        ROOT / raw,
        manifest_path.parent / raw,
        manifest_path.parent / raw.name,
        Path.cwd() / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / raw).resolve()


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "" or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "" or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def parse_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str.replace(".", "-", regex=False), errors="coerce")


def find_fold(manifest: dict[str, Any], fold_id: int) -> dict[str, Any]:
    for fold in manifest.get("folds", []) or []:
        if as_int(fold.get("fold")) == int(fold_id):
            return dict(fold)
    raise ValueError(f"fold {fold_id} not found")


def load_signals(manifest_path: Path, fold: dict[str, Any]) -> pd.DataFrame:
    csv_key = str(fold.get("csv") or fold.get("signals_csv") or "")
    if not csv_key:
        return pd.DataFrame(columns=MT5_COLUMNS + ["open_dt"])
    path = resolve_csv(manifest_path, csv_key)
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=MT5_COLUMNS + ["open_dt"])
    missing = sorted(set(MT5_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    frame = frame[MT5_COLUMNS].copy()
    frame["open_dt"] = parse_time(frame["open_time"])
    return frame.dropna(subset=["open_dt"]).copy()


def dirty_manifest(manifest: dict[str, Any]) -> bool:
    return any(
        is_true(manifest.get(key))
        for key in ("adaptive_per_fold", "research_oracle_fold_selection", "selection_uses_current_fold_metrics")
    )


def dirty_fold(fold: dict[str, Any]) -> bool:
    return any(
        is_true(fold.get(key))
        for key in (
            "selection_uses_current_fold_metrics",
            "selected_candidate_adaptive_per_fold",
            "selected_candidate_research_oracle_fold_selection",
            "selected_candidate_selection_uses_current_fold_metrics",
        )
    )


def load_source_meta(candidate_dir: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    manifest = read_json(candidate_dir / "manifest.json")
    folds: dict[int, dict[str, Any]] = {}
    for fold in manifest.get("folds", []) or []:
        fold_id = as_int(fold.get("fold"), -1)
        if fold_id >= 0:
            folds[fold_id] = dict(fold)
    return manifest, folds


def build_clean_index(args: argparse.Namespace) -> pd.DataFrame:
    stress = pd.read_csv(resolve(args.stress_folds))
    for col in ("target_hit", "dd_pass", "loss_fold"):
        if col in stress.columns:
            stress[col] = bool_series(stress[col])
    stress["fold"] = pd.to_numeric(stress["fold"], errors="coerce").astype(int)
    include_re = re.compile(args.include_label_regex) if args.include_label_regex else None
    exclude_re = re.compile(args.exclude_label_regex) if args.exclude_label_regex else None

    meta_cache: dict[str, tuple[dict[str, Any], dict[int, dict[str, Any]]]] = {}
    rows: list[dict[str, Any]] = []
    for _, row in stress.iterrows():
        label = str(row["candidate_label"])
        if include_re and not include_re.search(label):
            continue
        if exclude_re and exclude_re.search(label):
            continue
        candidate_dir = resolve(Path(str(row["candidate_dir"])))
        key = str(candidate_dir)
        if key not in meta_cache:
            meta_cache[key] = load_source_meta(candidate_dir)
        manifest, source_folds = meta_cache[key]
        source_fold = source_folds.get(int(row["fold"]), {})
        manifest_dirty = dirty_manifest(manifest)
        fold_dirty_flag = dirty_fold(source_fold)
        source_risk_pct = as_float(source_fold.get("risk_pct"))
        risk_ok = math.isnan(source_risk_pct) or source_risk_pct <= float(args.max_source_risk_pct)
        live_ok = bool(args.allow_non_live_protocol) or is_true(manifest.get("live_protocol"))
        clean = bool((not manifest_dirty) and (not fold_dirty_flag) and risk_ok and live_ok)
        row_out = row.to_dict()
        row_out.update(
            {
                "candidate_dir": str(candidate_dir),
                "source_manifest_dirty": bool(manifest_dirty),
                "source_fold_dirty": bool(fold_dirty_flag),
                "source_live_protocol": is_true(manifest.get("live_protocol")),
                "source_risk_pct": source_risk_pct,
                "source_clean": clean,
                "stress_strict_pass": bool(row.get("target_hit")) and bool(row.get("dd_pass")) and not bool(row.get("loss_fold")),
            }
        )
        rows.append(row_out)
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No stress rows remain after label filters.")
    return out


def score_history(history: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    if history.empty:
        return {
            "score": -1.0e18,
            "hist_rows": 0,
            "hist_strict_rate": 0.0,
            "hist_target_rate": 0.0,
            "hist_dd_rate": 0.0,
            "hist_loss_rate": 1.0,
            "hist_median_final": math.nan,
            "hist_min_final": math.nan,
            "hist_worst_dd": math.nan,
            "hist_avg_trades": math.nan,
        }
    frame = history.copy().sort_values("fold")
    if int(args.history_lookback) > 0:
        current_max = int(frame["fold"].max())
        frame = frame[frame["fold"] >= current_max - int(args.history_lookback) + 1].copy()
    strict = frame["stress_strict_pass"].astype(bool)
    target = frame["target_hit"].astype(bool)
    dd = frame["dd_pass"].astype(bool)
    loss = frame["loss_fold"].astype(bool)
    rows = int(len(frame))
    finals = pd.to_numeric(frame["final_balance"], errors="coerce")
    dds = pd.to_numeric(frame["max_dd_pct"], errors="coerce")
    trades = pd.to_numeric(frame.get("trades_used"), errors="coerce")
    strict_rate = float(strict.mean())
    target_rate = float(target.mean())
    dd_rate = float(dd.mean())
    loss_rate = float(loss.mean())
    median_final = float(finals.median())
    min_final = float(finals.min())
    worst_dd = float(dds.min())
    avg_trades = float(trades.mean())
    final_component = max(-2.5, min(2.5, (median_final - float(args.target_balance)) / float(args.target_balance)))
    min_component = max(-2.5, min(2.5, (min_final - float(args.target_balance)) / float(args.target_balance)))
    dd_component = max(-2.5, min(2.5, (worst_dd - float(args.min_dd_pct)) / abs(float(args.min_dd_pct))))
    trade_component = -max(0.0, avg_trades - float(args.max_preferred_trades)) / 500.0
    low_history_penalty = max(0, int(args.min_history_rows) - rows) * float(args.low_history_penalty)
    score = (
        float(args.strict_weight) * strict_rate
        + float(args.target_weight) * target_rate
        + float(args.dd_weight) * dd_rate
        - float(args.loss_weight) * loss_rate
        + float(args.median_final_weight) * final_component
        + float(args.min_final_weight) * min_component
        + float(args.worst_dd_weight) * dd_component
        + float(args.trades_penalty_weight) * trade_component
        - low_history_penalty
    )
    return {
        "score": float(score),
        "hist_rows": rows,
        "hist_strict_rate": strict_rate,
        "hist_target_rate": target_rate,
        "hist_dd_rate": dd_rate,
        "hist_loss_rate": loss_rate,
        "hist_median_final": median_final,
        "hist_min_final": min_final,
        "hist_worst_dd": worst_dd,
        "hist_avg_trades": avg_trades,
    }


def trigger_time(feedback: pd.DataFrame, fold_id: int, args: argparse.Namespace) -> pd.Timestamp | None:
    fold_feedback = feedback[pd.to_numeric(feedback.get("fold"), errors="coerce") == int(fold_id)].copy()
    if fold_feedback.empty:
        return None
    fold_feedback["close_dt"] = pd.to_datetime(fold_feedback.get("close_time"), errors="coerce")
    fold_feedback["balance_after"] = pd.to_numeric(fold_feedback.get("balance_after"), errors="coerce")
    fold_feedback = fold_feedback.dropna(subset=["close_dt"]).sort_values("close_dt").reset_index(drop=True)
    if fold_feedback.empty:
        return None
    if args.trigger_policy == "immediate":
        return pd.Timestamp.min
    if args.trigger_policy == "trade_count":
        if len(fold_feedback) < int(args.trigger_trade_count):
            return None
        return pd.Timestamp(fold_feedback.loc[int(args.trigger_trade_count) - 1, "close_dt"])
    subset = fold_feedback.iloc[max(0, int(args.trigger_min_trades) - 1) :].copy()
    if args.trigger_policy == "balance_below":
        hit = subset[subset["balance_after"] <= float(args.trigger_balance_below)]
        return None if hit.empty else pd.Timestamp(hit.iloc[0]["close_dt"])
    if args.trigger_policy == "progress":
        fold_start = fold_feedback["close_dt"].min()
        fold_end = fold_feedback["close_dt"].max()
        seconds = max(1.0, (fold_end - fold_start).total_seconds())
        elapsed = (subset["close_dt"] - fold_start).dt.total_seconds() / seconds
        required = float(args.deposit) + elapsed * (float(args.target_balance) - float(args.deposit)) * float(
            args.trigger_progress_fraction
        )
        hit = subset[subset["balance_after"] < required]
        return None if hit.empty else pd.Timestamp(hit.iloc[0]["close_dt"])
    raise ValueError(f"Unsupported trigger policy: {args.trigger_policy}")


def merge_signals(primary: pd.DataFrame, fallback: pd.DataFrame, trigger: pd.Timestamp | None) -> pd.DataFrame:
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


def choose_fallback(clean_index: pd.DataFrame, fold_id: int, args: argparse.Namespace) -> tuple[str | None, dict[str, Any]]:
    choices = clean_index[(clean_index["fold"].eq(int(fold_id))) & clean_index["source_clean"].astype(bool)].copy()
    if choices.empty:
        return None, {"score": -1.0e18, "hist_rows": 0, "reason": "no_clean_current_candidate"}
    scored: list[dict[str, Any]] = []
    for label in sorted(choices["candidate_label"].astype(str).unique()):
        history = clean_index[
            (clean_index["candidate_label"].astype(str).eq(label))
            & (clean_index["fold"] < int(fold_id))
            & clean_index["source_clean"].astype(bool)
        ].copy()
        score = score_history(history, args)
        current = choices[choices["candidate_label"].astype(str).eq(label)].iloc[0].to_dict()
        scored.append({"candidate_label": label, **score, **{f"actual_{k}": v for k, v in current.items()}})
    scored_df = pd.DataFrame(scored)
    scored_df = scored_df[scored_df["hist_rows"] >= int(args.min_history_rows)].copy()
    if scored_df.empty:
        return None, {"score": -1.0e18, "hist_rows": 0, "reason": "not_enough_prior_history"}
    scored_df = scored_df.sort_values(
        ["score", "hist_strict_rate", "hist_dd_rate", "hist_median_final", "candidate_label"],
        ascending=[False, False, False, False, True],
    )
    row = scored_df.iloc[0].to_dict()
    return str(row["candidate_label"]), row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a clean online failover replay: fallback is selected from prior MT5 feedback only; trigger uses only closed trades before trigger."
    )
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--primary-results", type=Path, required=True)
    parser.add_argument("--primary-feedback", type=Path, required=True)
    parser.add_argument("--stress-folds", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=30)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--min-dd-pct", type=float, default=-20.0)
    parser.add_argument("--max-source-risk-pct", type=float, default=4.0)
    parser.add_argument("--max-output-risk-pct", type=float, default=4.0)
    parser.add_argument("--allow-non-live-protocol", action="store_true")
    parser.add_argument("--include-label-regex", default="")
    parser.add_argument("--exclude-label-regex", default="")
    parser.add_argument("--history-lookback", type=int, default=0)
    parser.add_argument("--min-history-rows", type=int, default=4)
    parser.add_argument("--strict-weight", type=float, default=1500.0)
    parser.add_argument("--target-weight", type=float, default=900.0)
    parser.add_argument("--dd-weight", type=float, default=1000.0)
    parser.add_argument("--loss-weight", type=float, default=1500.0)
    parser.add_argument("--median-final-weight", type=float, default=250.0)
    parser.add_argument("--min-final-weight", type=float, default=350.0)
    parser.add_argument("--worst-dd-weight", type=float, default=300.0)
    parser.add_argument("--trades-penalty-weight", type=float, default=40.0)
    parser.add_argument("--max-preferred-trades", type=float, default=260.0)
    parser.add_argument("--low-history-penalty", type=float, default=150.0)
    parser.add_argument("--trigger-policy", choices=["immediate", "trade_count", "balance_below", "progress"], default="progress")
    parser.add_argument("--trigger-trade-count", type=int, default=50)
    parser.add_argument("--trigger-min-trades", type=int, default=40)
    parser.add_argument("--trigger-balance-below", type=float, default=260.0)
    parser.add_argument("--trigger-progress-fraction", type=float, default=0.45)
    parser.add_argument("--failover-only-if-primary-triggered", action="store_true", default=True)
    parser.add_argument("--use-fallback-risk-on-trigger", action="store_true")
    args = parser.parse_args()

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_manifest_path = resolve(args.primary_manifest)
    primary_manifest = read_json(primary_manifest_path)
    primary_results = pd.read_csv(resolve(args.primary_results))
    primary_feedback = pd.read_csv(resolve(args.primary_feedback))
    for col in ("fold", "final_balance", "max_dd_pct"):
        if col in primary_results.columns:
            primary_results[col] = pd.to_numeric(primary_results[col], errors="coerce")

    clean_index = build_clean_index(args)
    clean_index.to_csv(out_dir / "clean_candidate_index.csv", index=False)

    candidate_dirs = {
        str(row["candidate_label"]): resolve(Path(str(row["candidate_dir"])))
        for _, row in clean_index[clean_index["source_clean"].astype(bool)][["candidate_label", "candidate_dir"]]
        .drop_duplicates()
        .iterrows()
    }
    manifest_cache = {label: read_json(path / "manifest.json") for label, path in candidate_dirs.items()}

    out_folds: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    total_signals = 0
    for fold_id in range(int(args.fold_start), int(args.fold_end) + 1):
        primary_fold = find_fold(primary_manifest, fold_id)
        primary_row = primary_results[primary_results["fold"].eq(fold_id)]
        primary_final = as_float(primary_row.iloc[0].get("final_balance")) if not primary_row.empty else math.nan
        primary_dd = as_float(primary_row.iloc[0].get("max_dd_pct")) if not primary_row.empty else math.nan
        primary_signals = load_signals(primary_manifest_path, primary_fold)
        fallback_label, selected = choose_fallback(clean_index, fold_id, args)
        trigger = trigger_time(primary_feedback, fold_id, args) if fallback_label else None
        fallback_signals = pd.DataFrame(columns=MT5_COLUMNS + ["open_dt"])
        fallback_fold: dict[str, Any] | None = None
        fallback_dir: Path | None = None
        if fallback_label and (trigger is not None or not bool(args.failover_only_if_primary_triggered)):
            fallback_dir = candidate_dirs[fallback_label]
            fallback_manifest = manifest_cache[fallback_label]
            fallback_fold = find_fold(fallback_manifest, fold_id)
            fallback_signals = load_signals(fallback_dir / "manifest.json", fallback_fold)
        else:
            fallback_label = None
        export = merge_signals(primary_signals, fallback_signals, trigger if fallback_label else None)
        target_csv = out_dir / f"fold_{fold_id:02d}_signals.csv"
        export.to_csv(target_csv, index=False, float_format="%.5f")

        fold = dict(primary_fold)
        if fallback_fold and args.use_fallback_risk_on_trigger:
            for key in ("risk_pct", "max_risk_pct", "max_exposure_pct", "max_positions", "max_positions_hint"):
                if key in fallback_fold:
                    fold[key] = fallback_fold[key]
        for key in ("risk_pct", "max_risk_pct"):
            if key in fold and as_float(fold.get(key)) > float(args.max_output_risk_pct):
                fold[key] = float(args.max_output_risk_pct)
        max_positions = max(1, as_int(fold.get("max_positions"), 1))
        risk = as_float(fold.get("risk_pct"), float(args.max_output_risk_pct))
        fold["max_exposure_pct"] = min(as_float(fold.get("max_exposure_pct"), risk * max_positions), risk * max_positions)
        fold.update(
            {
                "fold": int(fold_id),
                "csv": str(target_csv),
                "signals": int(len(export)),
                "selection_mode": "clean_prior_online_failover" if fallback_label else "primary_only_no_clean_failover",
                "selection_uses_current_fold_metrics": False,
                "research_oracle_fold_selection": False,
                "adaptive_per_fold": False,
                "online_failover_uses_only_pretrigger_feedback": bool(fallback_label),
                "online_failover_trigger_policy": str(args.trigger_policy) if fallback_label else None,
                "online_failover_trigger_time": (
                    "immediate"
                    if trigger == pd.Timestamp.min
                    else trigger.strftime("%Y.%m.%d %H:%M:%S")
                    if isinstance(trigger, pd.Timestamp)
                    else None
                ),
                "selected_fallback_label": fallback_label,
                "selected_fallback_dir": str(fallback_dir) if fallback_dir else None,
                "selected_fallback_from_prior_folds": ",".join(str(x) for x in range(1, fold_id)),
                "selected_candidate_adaptive_per_fold": False,
                "selected_candidate_research_oracle_fold_selection": False,
                "selected_candidate_selection_uses_current_fold_metrics": False,
                "primary_manifest": str(primary_manifest_path),
                "primary_final_balance": primary_final,
                "primary_max_dd_pct": primary_dd,
            }
        )
        for key, value in selected.items():
            if key.startswith("hist_") or key == "score" or key == "reason":
                fold[f"fallback_{key}"] = value
        out_folds.append(fold)
        total_signals += int(len(export))
        audit_rows.append(
            {
                "fold": fold_id,
                "mode": fold["selection_mode"],
                "primary_final_balance": primary_final,
                "primary_max_dd_pct": primary_dd,
                "primary_signals": int(len(primary_signals)),
                "trigger_time": fold["online_failover_trigger_time"],
                "fallback_label": fallback_label,
                "fallback_dir": str(fallback_dir) if fallback_dir else None,
                "fallback_signals": int(len(fallback_signals)),
                "export_signals": int(len(export)),
                "score": as_float(selected.get("score")),
                "hist_rows": as_int(selected.get("hist_rows")),
                "hist_strict_rate": as_float(selected.get("hist_strict_rate")),
                "actual_fallback_final_balance": as_float(selected.get("actual_final_balance")),
                "actual_fallback_max_dd_pct": as_float(selected.get("actual_max_dd_pct")),
                "actual_fallback_strict_pass": bool(selected.get("actual_stress_strict_pass", False)),
                "selection_reason": selected.get("reason"),
            }
        )

    manifest = {
        "all_signals": None,
        "folds": out_folds,
        "total_signals": int(total_signals),
        "live_protocol": False,
        "clean_prior_online_failover_selector": True,
        "selection_uses_current_fold_metrics": False,
        "research_oracle_fold_selection": False,
        "adaptive_per_fold": False,
        "primary_manifest": str(primary_manifest_path),
        "primary_results": str(resolve(args.primary_results)),
        "primary_feedback": str(resolve(args.primary_feedback)),
        "stress_folds": str(resolve(args.stress_folds)),
        "selector_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "note": "Research replay. Fallback choice uses only clean prior folds; switch trigger uses only same-fold closed trades before trigger. Must pass MT5 + paper/canary before live.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out_dir / "prior_failover_selection_audit.csv", index=False)
    summary = {
        "manifest": str(out_dir / "manifest.json"),
        "folds": len(out_folds),
        "failover_folds": int(audit["fallback_label"].notna().sum()),
        "primary_only_folds": int(audit["fallback_label"].isna().sum()),
        "signals": int(total_signals),
        "selected_failover_folds": audit.loc[audit["fallback_label"].notna(), "fold"].astype(int).tolist(),
        "primary_only_folds_list": audit.loc[audit["fallback_label"].isna(), "fold"].astype(int).tolist(),
    }
    (out_dir / "prior_failover_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
