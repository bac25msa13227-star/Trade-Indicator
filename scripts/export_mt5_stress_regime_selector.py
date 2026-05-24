from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_mt5_regime_meta_selector import _load_feature_regimes, _normalizers, _regime_distance  # noqa: E402


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
    candidates = [ROOT / raw, manifest_path.parent / raw, manifest_path.parent / raw.name, Path.cwd() / raw]
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


def source_dirty(manifest: dict[str, Any], fold: dict[str, Any]) -> bool:
    manifest_dirty = any(
        is_true(manifest.get(key))
        for key in ("adaptive_per_fold", "research_oracle_fold_selection", "selection_uses_current_fold_metrics")
    )
    fold_dirty = any(
        is_true(fold.get(key))
        for key in (
            "selection_uses_current_fold_metrics",
            "selected_candidate_adaptive_per_fold",
            "selected_candidate_research_oracle_fold_selection",
            "selected_candidate_selection_uses_current_fold_metrics",
        )
    )
    return bool(manifest_dirty or fold_dirty)


def fold_map(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {as_int(fold.get("fold")): dict(fold) for fold in manifest.get("folds", []) or []}


def outcome_value(row: pd.Series, args: argparse.Namespace) -> float:
    final_balance = as_float(row.get("final_balance"), float(args.deposit))
    max_dd = as_float(row.get("max_dd_pct"), -100.0)
    target = final_balance >= float(args.target_balance)
    dd_ok = max_dd >= float(args.min_dd_pct)
    loss = final_balance < float(args.deposit)
    capped_final = min(final_balance, float(args.target_balance))
    final_part = (capped_final - float(args.deposit)) / max(1.0, float(args.target_balance) - float(args.deposit))
    dd_part = max(-2.0, min(1.0, (max_dd - float(args.min_dd_pct)) / abs(float(args.min_dd_pct))))
    value = 2.5 * final_part + 0.8 * dd_part
    if target and dd_ok and not loss:
        value += 4.0
    if target:
        value += 1.0
    else:
        value -= 1.2
    if dd_ok:
        value += 1.0
    else:
        value -= 2.5
    if loss:
        value -= 4.0
    return float(value)


def strict_pass(row: pd.Series, args: argparse.Namespace) -> bool:
    return (
        as_float(row.get("final_balance"), -1.0) >= float(args.target_balance)
        and as_float(row.get("max_dd_pct"), -999.0) >= float(args.min_dd_pct)
        and as_float(row.get("final_balance"), -1.0) >= float(args.deposit)
    )


def score_history(
    history: pd.DataFrame,
    fold_id: int,
    regimes: pd.DataFrame,
    mean: pd.Series,
    std: pd.Series,
    cols: list[str],
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    if history.empty:
        return -1.0e18, {
            "hist_rows": 0,
            "hist_strict_rate": 0.0,
            "hist_target_rate": 0.0,
            "hist_dd_rate": 0.0,
            "hist_loss_rate": 1.0,
            "hist_median_final": math.nan,
            "hist_min_final": math.nan,
            "hist_worst_dd": math.nan,
            "regime_effective_rows": 0.0,
            "regime_weighted_value": math.nan,
        }
    values: list[float] = []
    weights: list[float] = []
    stricts: list[float] = []
    targets: list[float] = []
    dds_ok: list[float] = []
    losses: list[float] = []
    for _, row in history.iterrows():
        hist_fold = as_int(row.get("fold"))
        dist = _regime_distance(regimes, fold_id, hist_fold, mean, std, cols)
        sim = math.exp(-((dist / max(float(args.regime_bandwidth), 1.0e-6)) ** 2) / 2.0)
        age = max(0, fold_id - hist_fold)
        recency = 1.0 if float(args.recency_half_life_folds) <= 0 else 0.5 ** (age / float(args.recency_half_life_folds))
        weight = sim * recency
        values.append(outcome_value(row, args))
        weights.append(weight)
        stricts.append(1.0 if strict_pass(row, args) else 0.0)
        targets.append(1.0 if as_float(row.get("final_balance"), -1.0) >= float(args.target_balance) else 0.0)
        dds_ok.append(1.0 if as_float(row.get("max_dd_pct"), -999.0) >= float(args.min_dd_pct) else 0.0)
        losses.append(1.0 if as_float(row.get("final_balance"), -1.0) < float(args.deposit) else 0.0)

    w = np.asarray(weights, dtype=float)
    if float(w.sum()) <= 1.0e-9:
        w = np.ones(len(values), dtype=float)
    v = np.asarray(values, dtype=float)
    weighted_value = float(np.average(v, weights=w))
    weighted_strict = float(np.average(np.asarray(stricts), weights=w))
    weighted_target = float(np.average(np.asarray(targets), weights=w))
    weighted_dd = float(np.average(np.asarray(dds_ok), weights=w))
    weighted_loss = float(np.average(np.asarray(losses), weights=w))
    effective_rows = float((w.sum() ** 2) / max(float(np.square(w).sum()), 1.0e-9))
    finals = pd.to_numeric(history["final_balance"], errors="coerce")
    dds = pd.to_numeric(history["max_dd_pct"], errors="coerce")
    score = (
        weighted_value * float(args.value_weight)
        + weighted_strict * float(args.strict_weight)
        + weighted_target * float(args.target_weight)
        + weighted_dd * float(args.dd_weight)
        - weighted_loss * float(args.loss_weight)
        + min(effective_rows, 12.0) * float(args.effective_row_weight)
        + ((float(finals.median()) - float(args.target_balance)) / float(args.target_balance)) * float(args.median_final_weight)
        + ((float(finals.min()) - float(args.target_balance)) / float(args.target_balance)) * float(args.min_final_weight)
        + ((float(dds.min()) - float(args.min_dd_pct)) / abs(float(args.min_dd_pct))) * float(args.worst_dd_weight)
    )
    return float(score), {
        "hist_rows": int(len(history)),
        "hist_strict_rate": float(np.mean(stricts)),
        "hist_target_rate": float(np.mean(targets)),
        "hist_dd_rate": float(np.mean(dds_ok)),
        "hist_loss_rate": float(np.mean(losses)),
        "hist_median_final": float(finals.median()),
        "hist_min_final": float(finals.min()),
        "hist_worst_dd": float(dds.min()),
        "regime_effective_rows": effective_rows,
        "regime_weighted_value": weighted_value,
        "regime_weighted_strict": weighted_strict,
        "regime_weighted_target": weighted_target,
        "regime_weighted_dd": weighted_dd,
        "regime_weighted_loss": weighted_loss,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a clean rolling selector from prior realistic MT5 stress feedback and pre-fold regimes.")
    parser.add_argument("--stress-folds", type=Path, required=True)
    parser.add_argument("--stress-summary", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--template-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=30)
    parser.add_argument("--min-history-rows", type=int, default=5)
    parser.add_argument("--history-lookback", type=int, default=0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--min-dd-pct", type=float, default=-20.0)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--max-source-risk-pct", type=float, default=4.0)
    parser.add_argument("--regime-lookback-days", type=int, default=20)
    parser.add_argument("--regime-bandwidth", type=float, default=2.0)
    parser.add_argument("--recency-half-life-folds", type=float, default=8.0)
    parser.add_argument("--value-weight", type=float, default=900.0)
    parser.add_argument("--strict-weight", type=float, default=1800.0)
    parser.add_argument("--target-weight", type=float, default=700.0)
    parser.add_argument("--dd-weight", type=float, default=1400.0)
    parser.add_argument("--loss-weight", type=float, default=2200.0)
    parser.add_argument("--effective-row-weight", type=float, default=70.0)
    parser.add_argument("--median-final-weight", type=float, default=250.0)
    parser.add_argument("--min-final-weight", type=float, default=500.0)
    parser.add_argument("--worst-dd-weight", type=float, default=500.0)
    args = parser.parse_args()

    stress = pd.read_csv(resolve(args.stress_folds))
    summary = pd.read_csv(resolve(args.stress_summary))
    for col in ("target_hit", "dd_pass", "loss_fold"):
        stress[col] = bool_series(stress[col])
    stress["fold"] = pd.to_numeric(stress["fold"], errors="coerce").astype(int)
    blocked = set(
        summary.loc[
            bool_series(summary.get("adaptive_per_fold", pd.Series(False, index=summary.index)))
            | bool_series(summary.get("research_oracle_fold_selection", pd.Series(False, index=summary.index)))
            | bool_series(summary.get("selection_uses_current_fold_metrics", pd.Series(False, index=summary.index))),
            "candidate_label",
        ].astype(str)
    )
    stress = stress[~stress["candidate_label"].astype(str).isin(blocked)].copy()

    manifest_cache: dict[str, tuple[dict[str, Any], dict[int, dict[str, Any]]]] = {}
    clean_rows: list[dict[str, Any]] = []
    for _, row in stress.iterrows():
        candidate_dir = resolve(Path(str(row["candidate_dir"])))
        key = str(candidate_dir)
        if key not in manifest_cache:
            manifest = read_json(candidate_dir / "manifest.json")
            manifest_cache[key] = (manifest, fold_map(manifest))
        manifest, folds = manifest_cache[key]
        fold = folds.get(as_int(row.get("fold")), {})
        risk = as_float(fold.get("risk_pct"))
        clean = (not source_dirty(manifest, fold)) and (math.isnan(risk) or risk <= float(args.max_source_risk_pct))
        if not clean:
            continue
        clean_rows.append(row.to_dict())
    clean = pd.DataFrame(clean_rows)
    if clean.empty:
        raise ValueError("No clean stress rows remain.")

    template = read_json(resolve(args.template_manifest))
    template_folds = [fold for fold in template.get("folds", []) if int(args.fold_start) <= as_int(fold.get("fold")) <= int(args.fold_end)]
    regimes = _load_feature_regimes(resolve(args.features), template_folds, int(args.regime_lookback_days))
    mean, std, regime_cols = _normalizers(regimes)

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_folds: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    bootstrap_count = 0

    for fold_id in range(int(args.fold_start), int(args.fold_end) + 1):
        scored: list[dict[str, Any]] = []
        choices = clean[clean["fold"].eq(fold_id)].copy()
        for label in sorted(choices["candidate_label"].astype(str).unique()):
            hist = clean[(clean["candidate_label"].astype(str).eq(label)) & (clean["fold"] < fold_id)].copy()
            if int(args.history_lookback) > 0:
                hist = hist[hist["fold"] >= fold_id - int(args.history_lookback)].copy()
            score, stats = score_history(hist, fold_id, regimes, mean, std, regime_cols, args)
            current = choices[choices["candidate_label"].astype(str).eq(label)].iloc[0].to_dict()
            row = {"fold": fold_id, "candidate_label": label, "score": score, **stats, **{f"actual_{k}": v for k, v in current.items()}}
            scored.append(row)
            score_rows.append(row)
        scored_df = pd.DataFrame(scored)
        if scored_df.empty:
            raise ValueError(f"No clean choices for fold {fold_id}")
        if int(scored_df["hist_rows"].max()) < int(args.min_history_rows):
            bootstrap_count += 1
            scored_df = scored_df.sort_values(
                ["actual_target_hit", "actual_dd_pass", "actual_final_balance", "candidate_label"],
                ascending=[False, False, False, True],
            )
            selection_mode = "bootstrap_declared_stress_regime"
        else:
            scored_df = scored_df[scored_df["hist_rows"] >= int(args.min_history_rows)].copy()
            scored_df = scored_df.sort_values(
                ["score", "regime_weighted_strict", "regime_weighted_dd", "hist_min_final", "candidate_label"],
                ascending=[False, False, False, False, True],
            )
            selection_mode = "stress_prior_regime_selector"
        selected = scored_df.iloc[0].to_dict()
        label = str(selected["candidate_label"])
        candidate_dir = resolve(Path(str(selected["actual_candidate_dir"])))
        manifest, folds = manifest_cache[str(candidate_dir)]
        source_fold = dict(folds[fold_id])
        source_csv = resolve_csv(candidate_dir / "manifest.json", str(source_fold["csv"]))
        target_csv = out_dir / f"fold_{fold_id:02d}_signals.csv"
        shutil.copy2(source_csv, target_csv)
        signals = max(0, sum(1 for _ in target_csv.open("r", encoding="utf-8")) - 1)
        out_fold = dict(source_fold)
        out_fold.update(
            {
                "fold": fold_id,
                "csv": str(target_csv),
                "signals": int(signals),
                "selection_mode": selection_mode,
                "selection_uses_current_fold_metrics": False,
                "research_oracle_fold_selection": False,
                "adaptive_per_fold": False,
                "selected_candidate_label": label,
                "selected_candidate_dir": str(candidate_dir),
                "selected_candidate_adaptive_per_fold": False,
                "selected_candidate_research_oracle_fold_selection": False,
                "selected_candidate_selection_uses_current_fold_metrics": False,
                "stress_score_from_prior_regime": as_float(selected.get("score")),
            }
        )
        for key in (
            "hist_rows",
            "hist_strict_rate",
            "hist_target_rate",
            "hist_dd_rate",
            "hist_loss_rate",
            "hist_median_final",
            "hist_min_final",
            "hist_worst_dd",
            "regime_effective_rows",
            "regime_weighted_value",
            "regime_weighted_strict",
            "regime_weighted_target",
            "regime_weighted_dd",
            "regime_weighted_loss",
        ):
            out_fold[f"stress_{key}"] = as_float(selected.get(key))
        selected_folds.append(out_fold)
        audit_rows.append(
            {
                "fold": fold_id,
                "selection_mode": selection_mode,
                "selected_candidate_label": label,
                "selected_candidate_dir": str(candidate_dir),
                "actual_final_balance": as_float(selected.get("actual_final_balance")),
                "actual_max_dd_pct": as_float(selected.get("actual_max_dd_pct")),
                "actual_target_hit": bool(selected.get("actual_target_hit")),
                "actual_dd_pass": bool(selected.get("actual_dd_pass")),
                "actual_loss_fold": bool(selected.get("actual_loss_fold")),
                "actual_strict_pass": strict_pass(pd.Series({k.replace("actual_", ""): v for k, v in selected.items() if k.startswith("actual_")}), args),
                "score": as_float(selected.get("score")),
                "hist_rows": as_int(selected.get("hist_rows")),
                "regime_weighted_strict": as_float(selected.get("regime_weighted_strict")),
            }
        )

    manifest_out = {
        "all_signals": None,
        "folds": selected_folds,
        "total_signals": int(sum(as_int(fold.get("signals")) for fold in selected_folds)),
        "live_protocol": False,
        "stress_prior_regime_selector": True,
        "adaptive_per_fold": False,
        "research_oracle_fold_selection": False,
        "selection_uses_current_fold_metrics": False,
        "bootstrap_declared_folds": int(bootstrap_count),
        "stress_folds": str(resolve(args.stress_folds)),
        "stress_summary": str(resolve(args.stress_summary)),
        "features": str(resolve(args.features)),
        "note": "Research selector. Candidate choice uses prior realistic MT5 stress feedback weighted by pre-fold regime only; current fold metrics are audit-only.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out_dir / "stress_regime_selection_audit.csv", index=False)
    pd.DataFrame(score_rows).to_csv(out_dir / "stress_regime_candidate_scores.csv", index=False)
    regimes.reset_index().to_csv(out_dir / "fold_regime_features.csv", index=False)
    summary_out = {
        "manifest": str(out_dir / "manifest.json"),
        "folds": int(len(audit)),
        "stress_strict_pass": int(audit["actual_strict_pass"].sum()),
        "target_pass": int(audit["actual_target_hit"].sum()),
        "dd_pass": int(audit["actual_dd_pass"].sum()),
        "loss_folds": int(audit["actual_loss_fold"].sum()),
        "min_final": float(audit["actual_final_balance"].min()),
        "median_final": float(audit["actual_final_balance"].median()),
        "worst_dd": float(audit["actual_max_dd_pct"].min()),
        "fail_folds": audit.loc[~audit["actual_strict_pass"], "fold"].astype(int).tolist(),
        "bootstrap_declared_folds": int(bootstrap_count),
    }
    (out_dir / "stress_regime_selector_summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    print(json.dumps(summary_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
