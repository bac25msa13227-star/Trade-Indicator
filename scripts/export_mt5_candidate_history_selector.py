from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGIME_COLUMNS = [
    "atr",
    "atr_ratio",
    "range_efficiency",
    "adx",
    "tick_volume_zscore",
    "strategy_score",
    "regime_score",
    "regime_trending",
    "regime_sideway",
    "regime_volatile",
    "trend_strength_score",
    "multi_tf_consensus",
    "h4_ict_confluence",
    "bb_position",
    "stoch_k",
    "rsi",
    "macd_hist",
    "price_roc",
    "volatility_regime",
    "trade_side",
    "expected_direction",
    "daily_bias",
    "hourly_bias",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeError:
            continue
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON file: {path}")


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Empty label in {value!r}")
    return label, Path(raw_path.strip())


def resolve_path(path: Path) -> Path:
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


def find_fold(manifest: dict[str, Any], fold_id: int) -> dict[str, Any]:
    for fold in manifest.get("folds", []) or []:
        if as_int(fold.get("fold")) == int(fold_id):
            return dict(fold)
    raise ValueError(f"fold {fold_id} not found in manifest")


def count_csv_rows(path: Path) -> int:
    return max(0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1)


def load_results(
    *,
    label: str,
    source_dir: Path,
    split: str,
    fold_shift: int,
    deposit: float,
    target_balance: float,
    min_dd_pct: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_dir = resolve_path(source_dir)
    manifest_path = source_dir / "manifest.json"
    results_path = source_dir / "mt5_wf_results.csv"
    manifest = read_json(manifest_path)
    if not results_path.exists():
        return pd.DataFrame(), manifest

    results = pd.read_csv(results_path)
    rows: list[dict[str, Any]] = []
    for _, result in results.iterrows():
        fold = as_int(result.get("fold"))
        try:
            fold_meta = find_fold(manifest, fold)
        except ValueError:
            fold_meta = {}
        final_balance = as_float(result.get("final_balance"))
        max_dd_pct = as_float(result.get("max_dd_pct"))
        signals = as_int(result.get("signals"))
        loaded_signals = as_int(result.get("loaded_signals"))
        target_pass = bool(final_balance >= target_balance)
        dd_pass = bool(max_dd_pct >= min_dd_pct)
        loss_fold = bool(final_balance < deposit)
        rows.append(
            {
                "candidate_label": label,
                "source_dir": str(source_dir),
                "manifest": str(manifest_path),
                "split": split,
                "fold": fold,
                "history_fold": int(fold + fold_shift),
                "train_start": result.get("train_start") or fold_meta.get("train_start"),
                "train_end": result.get("train_end") or fold_meta.get("train_end"),
                "test_start": result.get("test_start"),
                "test_end": result.get("test_end"),
                "signals": signals,
                "loaded_signals": loaded_signals,
                "loaded_signal_gap": int(loaded_signals - signals),
                "final_balance": final_balance,
                "net_pct": as_float(result.get("net_pct")),
                "max_dd_pct": max_dd_pct,
                "trades": as_int(result.get("trades")),
                "win_rate_pct": as_float(result.get("win_rate_pct")),
                "target_pass": target_pass,
                "dd_pass": dd_pass,
                "loss_fold": loss_fold,
                "strict_pass": bool(target_pass and dd_pass and not loss_fold),
                "target_gap": float(final_balance - target_balance),
                "dd_buffer": float(max_dd_pct - min_dd_pct),
            }
        )
    return pd.DataFrame(rows), manifest


def parse_regime_columns(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return list(DEFAULT_REGIME_COLUMNS)
    return [part.strip() for part in value.split(",") if part.strip()]


def period_key(row: pd.Series | dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("split", "")),
            str(as_int(row.get("fold"))),
            str(row.get("train_start", "")),
            str(row.get("train_end", "")),
        ]
    )


def build_regime_vectors(
    *,
    features_csv: Path,
    rows: pd.DataFrame,
    requested_columns: list[str],
) -> tuple[pd.DataFrame, list[str], pd.Series]:
    if rows.empty:
        return pd.DataFrame(), [], pd.Series(dtype=float)

    header = pd.read_csv(features_csv, nrows=0)
    available_columns = [col for col in requested_columns if col in header.columns]
    if not available_columns:
        return pd.DataFrame(), [], pd.Series(dtype=float)

    usecols = ["time", *available_columns]
    features = pd.read_csv(features_csv, usecols=usecols)
    features["time"] = pd.to_datetime(features["time"], errors="coerce")
    features = features.dropna(subset=["time"]).sort_values("time")
    for col in available_columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    periods = rows[["split", "fold", "train_start", "train_end"]].drop_duplicates().copy()
    vector_rows: list[dict[str, Any]] = []
    for _, period in periods.iterrows():
        start = pd.to_datetime(period.get("train_start"), errors="coerce")
        end = pd.to_datetime(period.get("train_end"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        window = features[(features["time"] >= start) & (features["time"] < end)]
        if window.empty:
            continue
        row: dict[str, Any] = {
            "regime_period_key": period_key(period),
        }
        for col in available_columns:
            values = pd.to_numeric(window[col], errors="coerce").dropna()
            if values.empty:
                row[f"regime_{col}_mean"] = math.nan
                row[f"regime_{col}_std"] = math.nan
                row[f"regime_{col}_last"] = math.nan
            else:
                row[f"regime_{col}_mean"] = float(values.mean())
                row[f"regime_{col}_std"] = float(values.std(ddof=0))
                row[f"regime_{col}_last"] = float(values.iloc[-1])
        vector_rows.append(row)

    vectors = pd.DataFrame(vector_rows)
    if vectors.empty:
        return vectors, [], pd.Series(dtype=float)
    regime_feature_cols = [col for col in vectors.columns if col.startswith("regime_") and col != "regime_period_key"]
    medians = vectors[regime_feature_cols].median(numeric_only=True)
    vectors[regime_feature_cols] = vectors[regime_feature_cols].fillna(medians)
    scales = vectors[regime_feature_cols].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return vectors, regime_feature_cols, scales


def attach_regime_vectors(frame: pd.DataFrame, vectors: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or vectors.empty:
        return frame
    out = frame.copy()
    out["regime_period_key"] = out.apply(period_key, axis=1)
    return out.merge(vectors, on="regime_period_key", how="left")


def weighted_rate(values: pd.Series, weights: pd.Series, prior_rate: float, prior_n: float) -> float:
    if values.empty:
        return float(prior_rate)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return float(prior_rate)
    observed = float((values.astype(float) * weights).sum())
    return float((observed + prior_rate * prior_n) / (total_weight + prior_n))


def score_history(
    history: pd.DataFrame,
    args: argparse.Namespace,
    target_regime: pd.Series | None = None,
    regime_feature_cols: list[str] | None = None,
    regime_scales: pd.Series | None = None,
) -> dict[str, Any]:
    if history.empty:
        return {
            "score": -1.0e12,
            "hist_rows": 0,
            "hist_strict_rate": 0.0,
            "hist_target_rate": 0.0,
            "hist_dd_rate": 0.0,
            "hist_loss_rate": 1.0,
            "hist_median_final": math.nan,
            "hist_min_final": math.nan,
            "hist_worst_dd": math.nan,
            "hist_total_trades": 0,
            "hist_regime_similarity": math.nan,
        }

    frame = history.copy().sort_values("history_fold")
    if int(args.history_lookback) > 0:
        max_fold = int(frame["history_fold"].max())
        frame = frame[frame["history_fold"] >= max_fold - int(args.history_lookback) + 1].copy()

    weights = pd.Series(float(args.calibration_weight), index=frame.index)
    weights.loc[frame["split"].eq("actual")] = float(args.actual_weight)
    regime_similarity = pd.Series(0.0, index=frame.index)
    if target_regime is not None and regime_feature_cols and regime_scales is not None:
        valid_cols = [
            col
            for col in regime_feature_cols
            if col in frame.columns and col in target_regime.index and col in regime_scales.index
        ]
        if valid_cols:
            hist_values = frame[valid_cols].apply(pd.to_numeric, errors="coerce")
            target_values = pd.to_numeric(target_regime[valid_cols], errors="coerce")
            fill_values = hist_values.median(numeric_only=True)
            hist_values = hist_values.fillna(fill_values)
            target_values = target_values.fillna(fill_values)
            scale_values = regime_scales[valid_cols].replace(0.0, 1.0).fillna(1.0)
            diff = (hist_values - target_values) / scale_values
            distances = (diff.pow(2).mean(axis=1).pow(0.5)).replace([math.inf, -math.inf], math.nan).fillna(10.0)
            regime_similarity = (-0.5 * distances.pow(2)).apply(math.exp)
            if int(args.regime_top_k) > 0 and len(frame) > int(args.regime_top_k):
                keep_index = regime_similarity.sort_values(ascending=False).head(int(args.regime_top_k)).index
                frame = frame.loc[keep_index].sort_values("history_fold")
                weights = weights.loc[frame.index]
                regime_similarity = regime_similarity.loc[frame.index]
            weights = weights * (1.0 + float(args.regime_weight) * regime_similarity)

    strict_rate = weighted_rate(frame["strict_pass"], weights, float(args.prior_strict_rate), float(args.prior_n))
    target_rate = weighted_rate(frame["target_pass"], weights, float(args.prior_target_rate), float(args.prior_n))
    dd_rate = weighted_rate(frame["dd_pass"], weights, float(args.prior_dd_rate), float(args.prior_n))
    loss_rate = weighted_rate(frame["loss_fold"], weights, float(args.prior_loss_rate), float(args.prior_n))

    recent = frame.tail(int(args.recent_folds)) if int(args.recent_folds) > 0 else frame.iloc[0:0]
    recent_weights = weights.loc[recent.index] if not recent.empty else pd.Series(dtype=float)
    recent_strict_rate = weighted_rate(
        recent["strict_pass"], recent_weights, float(args.prior_strict_rate), float(args.prior_n)
    )
    recent_dd_rate = weighted_rate(recent["dd_pass"], recent_weights, float(args.prior_dd_rate), float(args.prior_n))

    median_final = float(pd.to_numeric(frame["final_balance"], errors="coerce").median())
    min_final = float(pd.to_numeric(frame["final_balance"], errors="coerce").min())
    worst_dd = float(pd.to_numeric(frame["max_dd_pct"], errors="coerce").min())
    avg_trades = float(pd.to_numeric(frame["trades"], errors="coerce").mean())

    final_component = max(-3.0, min(3.0, (median_final - float(args.target_balance)) / 1000.0))
    min_final_component = max(-3.0, min(3.0, (min_final - float(args.target_balance)) / 1000.0))
    dd_component = max(-3.0, min(3.0, (worst_dd - float(args.min_dd_pct)) / 10.0))
    trades_component = max(0.0, min(2.0, avg_trades / 500.0))

    score = (
        float(args.strict_weight) * strict_rate
        + float(args.target_weight) * target_rate
        + float(args.dd_weight) * dd_rate
        - float(args.loss_weight) * loss_rate
        + float(args.recent_strict_weight) * recent_strict_rate
        + float(args.recent_dd_weight) * recent_dd_rate
        + float(args.median_final_weight) * final_component
        + float(args.min_final_weight) * min_final_component
        + float(args.worst_dd_weight) * dd_component
        + float(args.trades_weight) * trades_component
    )
    if len(frame) < int(args.min_history_rows):
        score -= float(args.low_history_penalty) * float(int(args.min_history_rows) - len(frame))

    return {
        "score": float(score),
        "hist_rows": int(len(frame)),
        "hist_strict_pass": int(frame["strict_pass"].sum()),
        "hist_target_pass": int(frame["target_pass"].sum()),
        "hist_dd_pass": int(frame["dd_pass"].sum()),
        "hist_loss": int(frame["loss_fold"].sum()),
        "hist_strict_rate": float(strict_rate),
        "hist_target_rate": float(target_rate),
        "hist_dd_rate": float(dd_rate),
        "hist_loss_rate": float(loss_rate),
        "hist_recent_strict_rate": float(recent_strict_rate),
        "hist_recent_dd_rate": float(recent_dd_rate),
        "hist_median_final": median_final,
        "hist_min_final": min_final,
        "hist_worst_dd": worst_dd,
        "hist_total_trades": int(pd.to_numeric(frame["trades"], errors="coerce").fillna(0).sum()),
        "hist_regime_similarity": float(regime_similarity.mean()) if len(regime_similarity) else math.nan,
    }


def copy_selected_fold(
    *,
    label: str,
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
    source_dir: Path,
    fold_id: int,
    out_dir: Path,
    score: dict[str, Any],
) -> dict[str, Any]:
    source_fold = find_fold(source_manifest, fold_id)
    source_csv = resolve_csv(source_manifest_path, str(source_fold["csv"]))
    target_csv = out_dir / f"fold_{fold_id:02d}_signals.csv"
    shutil.copy2(source_csv, target_csv)
    signals = count_csv_rows(target_csv)

    fold = dict(source_fold)
    for key in list(fold.keys()):
        if key.startswith("selected_candidate_") or key.startswith("hist_"):
            fold.pop(key, None)
    fold.update(
        {
            "fold": int(fold_id),
            "csv": str(target_csv),
            "signals": int(signals),
            "selection_mode": "rolling_candidate_history_mt5",
            "selection_uses_current_fold_metrics": False,
            "selection_history_folds": ",".join(str(x) for x in range(1, fold_id)),
            "selected_candidate_label": label,
            "selected_candidate_dir": str(source_dir),
            "selected_candidate_live_protocol": is_true(source_manifest.get("live_protocol")),
            "selected_candidate_adaptive_per_fold": is_true(source_manifest.get("adaptive_per_fold")),
            "selected_candidate_research_oracle_fold_selection": is_true(
                source_manifest.get("research_oracle_fold_selection")
            ),
            "selected_candidate_selection_uses_current_fold_metrics": is_true(
                source_manifest.get("selection_uses_current_fold_metrics")
            ),
            "selected_score_from_past_mt5": round(float(score["score"]), 6),
        }
    )
    for key, value in score.items():
        if key == "score":
            continue
        fold[f"hist_{key}"] = value
    return fold


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a rolling manifest from per-candidate prior MT5 WF feedback.")
    parser.add_argument("--actual-candidate", action="append", required=True, help="LABEL=DIR. May be repeated.")
    parser.add_argument("--calibration-candidate", action="append", default=None, help="LABEL=DIR. May be repeated.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--min-dd-pct", type=float, default=-20.0)
    parser.add_argument("--calibration-fold-shift", type=int, default=-29)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=30)
    parser.add_argument("--history-lookback", type=int, default=0)
    parser.add_argument("--recent-folds", type=int, default=8)
    parser.add_argument("--min-history-rows", type=int, default=4)
    parser.add_argument("--calibration-weight", type=float, default=0.75)
    parser.add_argument("--actual-weight", type=float, default=1.0)
    parser.add_argument("--prior-n", type=float, default=4.0)
    parser.add_argument("--prior-strict-rate", type=float, default=0.45)
    parser.add_argument("--prior-target-rate", type=float, default=0.55)
    parser.add_argument("--prior-dd-rate", type=float, default=0.80)
    parser.add_argument("--prior-loss-rate", type=float, default=0.05)
    parser.add_argument("--strict-weight", type=float, default=1200.0)
    parser.add_argument("--target-weight", type=float, default=650.0)
    parser.add_argument("--dd-weight", type=float, default=900.0)
    parser.add_argument("--loss-weight", type=float, default=800.0)
    parser.add_argument("--recent-strict-weight", type=float, default=500.0)
    parser.add_argument("--recent-dd-weight", type=float, default=300.0)
    parser.add_argument("--median-final-weight", type=float, default=220.0)
    parser.add_argument("--min-final-weight", type=float, default=180.0)
    parser.add_argument("--worst-dd-weight", type=float, default=220.0)
    parser.add_argument("--trades-weight", type=float, default=20.0)
    parser.add_argument("--low-history-penalty", type=float, default=200.0)
    parser.add_argument("--features-csv", type=Path, default=None)
    parser.add_argument("--regime-columns", type=str, default=None)
    parser.add_argument("--regime-weight", type=float, default=0.0)
    parser.add_argument("--regime-top-k", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    actual_dirs: dict[str, Path] = {}
    calibration_dirs: dict[str, Path] = {}
    for item in args.actual_candidate:
        label, path = parse_label_path(item)
        actual_dirs[label] = resolve_path(path)
    for item in args.calibration_candidate or []:
        label, path = parse_label_path(item)
        calibration_dirs[label] = resolve_path(path)

    history_frames: list[pd.DataFrame] = []
    actual_frames: list[pd.DataFrame] = []
    manifests: dict[str, dict[str, Any]] = {}
    manifest_paths: dict[str, Path] = {}

    for label, source_dir in actual_dirs.items():
        frame, manifest = load_results(
            label=label,
            source_dir=source_dir,
            split="actual",
            fold_shift=0,
            deposit=args.deposit,
            target_balance=args.target_balance,
            min_dd_pct=args.min_dd_pct,
        )
        if frame.empty:
            raise ValueError(f"Actual candidate {label} has no mt5_wf_results.csv rows: {source_dir}")
        actual_frames.append(frame)
        history_frames.append(frame)
        manifests[label] = manifest
        manifest_paths[label] = source_dir / "manifest.json"

    for label, source_dir in calibration_dirs.items():
        frame, _manifest = load_results(
            label=label,
            source_dir=source_dir,
            split="calibration",
            fold_shift=int(args.calibration_fold_shift),
            deposit=args.deposit,
            target_balance=args.target_balance,
            min_dd_pct=args.min_dd_pct,
        )
        if not frame.empty:
            history_frames.append(frame)

    history = pd.concat(history_frames, ignore_index=True)
    actual = pd.concat(actual_frames, ignore_index=True)
    regime_feature_cols: list[str] = []
    regime_scales = pd.Series(dtype=float)
    if args.features_csv is not None and float(args.regime_weight) > 0:
        requested_columns = parse_regime_columns(args.regime_columns)
        combined_periods = pd.concat([history, actual], ignore_index=True)
        regime_vectors, regime_feature_cols, regime_scales = build_regime_vectors(
            features_csv=resolve_path(args.features_csv),
            rows=combined_periods,
            requested_columns=requested_columns,
        )
        history = attach_regime_vectors(history, regime_vectors)
        actual = attach_regime_vectors(actual, regime_vectors)

    selected_folds: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    total_signals = 0

    for fold_id in range(int(args.fold_start), int(args.fold_end) + 1):
        choices: list[dict[str, Any]] = []
        target_rows = actual[pd.to_numeric(actual["fold"], errors="coerce") == int(fold_id)]
        target_regime = target_rows.iloc[0] if len(target_rows) and regime_feature_cols else None
        for label in sorted(actual_dirs):
            candidate_history = history[
                (history["candidate_label"] == label)
                & (pd.to_numeric(history["history_fold"], errors="coerce") < int(fold_id))
            ].copy()
            score = score_history(candidate_history, args, target_regime, regime_feature_cols, regime_scales)
            actual_row = actual[
                (actual["candidate_label"] == label) & (pd.to_numeric(actual["fold"], errors="coerce") == int(fold_id))
            ]
            if actual_row.empty:
                continue
            actual_record = actual_row.iloc[0].to_dict()
            choice = {"candidate_label": label, **score, **{f"actual_{k}": v for k, v in actual_record.items()}}
            choices.append(choice)
            score_rows.append({"fold": fold_id, "candidate_label": label, **score})

        if not choices:
            raise ValueError(f"No candidate has actual fold {fold_id}")
        choices_df = pd.DataFrame(choices)
        choices_df = choices_df.sort_values(
            ["score", "hist_strict_rate", "hist_dd_rate", "hist_median_final", "candidate_label"],
            ascending=[False, False, False, False, True],
        )
        selected = choices_df.iloc[0].to_dict()
        label = str(selected["candidate_label"])
        fold = copy_selected_fold(
            label=label,
            source_manifest=manifests[label],
            source_manifest_path=manifest_paths[label],
            source_dir=actual_dirs[label],
            fold_id=fold_id,
            out_dir=args.out_dir,
            score={k: selected[k] for k in selected if k.startswith("hist_") or k == "score"},
        )
        selected_folds.append(fold)
        total_signals += int(fold["signals"])

        audit_rows.append(
            {
                "fold": fold_id,
                "selected_candidate_label": label,
                "selected_candidate_dir": str(actual_dirs[label]),
                "selected_score_from_past_mt5": selected["score"],
                "hist_rows": selected["hist_rows"],
                "hist_strict_rate": selected["hist_strict_rate"],
                "hist_target_rate": selected["hist_target_rate"],
                "hist_dd_rate": selected["hist_dd_rate"],
                "hist_loss_rate": selected["hist_loss_rate"],
                "actual_final_balance": selected["actual_final_balance"],
                "actual_max_dd_pct": selected["actual_max_dd_pct"],
                "actual_trades": selected["actual_trades"],
                "actual_target_pass": selected["actual_target_pass"],
                "actual_dd_pass": selected["actual_dd_pass"],
                "actual_loss_fold": selected["actual_loss_fold"],
                "actual_strict_pass": selected["actual_strict_pass"],
            }
        )

    manifest = {
        "all_signals": None,
        "folds": selected_folds,
        "total_signals": int(total_signals),
        "live_protocol": True,
        "rolling_candidate_history_selector": True,
        "selection_uses_current_fold_metrics": False,
        "research_oracle_fold_selection": False,
        "adaptive_per_fold": False,
        "actual_candidates": {label: str(path) for label, path in sorted(actual_dirs.items())},
        "calibration_candidates": {label: str(path) for label, path in sorted(calibration_dirs.items())},
        "selector_args": {
            key: json_safe(value)
            for key, value in vars(args).items()
            if key not in {"actual_candidate", "calibration_candidate", "out_dir"}
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(audit_rows).to_csv(args.out_dir / "selection_audit.csv", index=False)
    pd.DataFrame(score_rows).to_csv(args.out_dir / "candidate_scores_by_fold.csv", index=False)
    history.to_csv(args.out_dir / "candidate_history_matrix.csv", index=False)

    audit = pd.DataFrame(audit_rows)
    summary = {
        "manifest": str(args.out_dir / "manifest.json"),
        "folds": int(len(audit)),
        "signals": int(total_signals),
        "strict_pass": int(audit["actual_strict_pass"].sum()),
        "target_pass": int(audit["actual_target_pass"].sum()),
        "dd_pass": int(audit["actual_dd_pass"].sum()),
        "loss_folds": int(audit["actual_loss_fold"].sum()),
        "min_final": float(pd.to_numeric(audit["actual_final_balance"], errors="coerce").min()),
        "median_final": float(pd.to_numeric(audit["actual_final_balance"], errors="coerce").median()),
        "worst_dd": float(pd.to_numeric(audit["actual_max_dd_pct"], errors="coerce").min()),
        "total_trades": int(pd.to_numeric(audit["actual_trades"], errors="coerce").fillna(0).sum()),
        "fail_folds": audit.loc[~audit["actual_strict_pass"], "fold"].astype(int).tolist(),
    }
    (args.out_dir / "selector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
