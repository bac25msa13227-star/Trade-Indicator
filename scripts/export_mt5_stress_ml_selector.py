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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_mt5_regime_meta_selector import _load_feature_regimes  # noqa: E402


DIRTY_KEYS = ("adaptive_per_fold", "research_oracle_fold_selection", "selection_uses_current_fold_metrics")
FOLD_DIRTY_KEYS = (
    "selection_uses_current_fold_metrics",
    "selected_candidate_adaptive_per_fold",
    "selected_candidate_research_oracle_fold_selection",
    "selected_candidate_selection_uses_current_fold_metrics",
)


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


def fold_map(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {as_int(fold.get("fold")): dict(fold) for fold in manifest.get("folds", []) or []}


def dirty_source(manifest: dict[str, Any], fold: dict[str, Any]) -> bool:
    return any(is_true(manifest.get(key)) for key in DIRTY_KEYS) or any(is_true(fold.get(key)) for key in FOLD_DIRTY_KEYS)


def candidate_family(label: str, candidate_dir: str) -> str:
    text = f"{label} {candidate_dir}".lower().replace("\\", "/")
    for token in (
        "actual_research_f07",
        "actual_research_f12",
        "actual_research_f14",
        "actual_rule_f02",
        "target1500_regime",
        "feedback_model",
        "prior_mt5_feedback",
        "profitr",
        "fixed_library",
        "rule_f01",
        "rule_f02",
        "rule_f03",
        "rule_f04",
        "rule_f05",
        "rule_f06",
        "rule_f07",
        "rule_f08",
        "rule_f09",
        "rule_f10",
        "rule_f12",
        "rule_f13",
        "rule_f15",
        "rule_f21",
        "rule_f26",
    ):
        if token in text:
            return token
    return Path(candidate_dir).name


def strict_pass(row: pd.Series | dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        as_float(row.get("final_balance"), -1.0) >= float(args.target_balance)
        and as_float(row.get("max_dd_pct"), -999.0) >= float(args.min_dd_pct)
        and as_float(row.get("final_balance"), -1.0) >= float(args.deposit)
    )


def outcome(row: pd.Series | dict[str, Any], args: argparse.Namespace) -> float:
    final_balance = as_float(row.get("final_balance"), float(args.deposit))
    max_dd = as_float(row.get("max_dd_pct"), -100.0)
    target = final_balance >= float(args.target_balance)
    dd_ok = max_dd >= float(args.min_dd_pct)
    loss = final_balance < float(args.deposit)
    final_score = (min(final_balance, float(args.target_balance)) - float(args.deposit)) / max(
        1.0, float(args.target_balance) - float(args.deposit)
    )
    dd_score = max(-2.0, min(1.0, (max_dd - float(args.min_dd_pct)) / abs(float(args.min_dd_pct))))
    value = 2.4 * final_score + 0.8 * dd_score
    if target and dd_ok and not loss:
        value += 4.0
    if target:
        value += 1.0
    else:
        value -= 1.4
    if dd_ok:
        value += 0.8
    else:
        value -= 2.6
    if loss:
        value -= 4.0
    return float(value)


def history_stats(frame: pd.DataFrame, fold_id: int, args: argparse.Namespace) -> dict[str, Any]:
    hist = frame[frame["fold"] < fold_id].copy()
    if int(args.history_lookback) > 0:
        hist = hist[hist["fold"] >= fold_id - int(args.history_lookback)].copy()
    if hist.empty:
        return {
            "hist_rows": 0,
            "hist_strict_rate": 0.0,
            "hist_target_rate": 0.0,
            "hist_dd_rate": 0.0,
            "hist_loss_rate": 1.0,
            "hist_median_final": math.nan,
            "hist_min_final": math.nan,
            "hist_worst_dd": math.nan,
            "hist_final_trend": 0.0,
            "hist_recent_strict": 0.0,
        }
    finals = pd.to_numeric(hist["final_balance"], errors="coerce")
    dds = pd.to_numeric(hist["max_dd_pct"], errors="coerce")
    strict = hist.apply(lambda row: strict_pass(row, args), axis=1).astype(float)
    target = (finals >= float(args.target_balance)).astype(float)
    dd_ok = (dds >= float(args.min_dd_pct)).astype(float)
    loss = (finals < float(args.deposit)).astype(float)
    trend = 0.0
    if len(finals.dropna()) >= 2:
        x = np.arange(len(finals), dtype=float)
        trend = float(np.polyfit(x, finals.fillna(float(args.deposit)).to_numpy(dtype=float), deg=1)[0])
    recent = strict.tail(max(1, int(args.recent_folds))).mean()
    return {
        "hist_rows": int(len(hist)),
        "hist_strict_rate": float(strict.mean()),
        "hist_target_rate": float(target.mean()),
        "hist_dd_rate": float(dd_ok.mean()),
        "hist_loss_rate": float(loss.mean()),
        "hist_median_final": float(finals.median()),
        "hist_min_final": float(finals.min()),
        "hist_worst_dd": float(dds.min()),
        "hist_final_trend": trend / max(1.0, float(args.target_balance)),
        "hist_recent_strict": float(recent),
    }


def prepare_clean_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, tuple[dict[str, Any], dict[int, dict[str, Any]]]]]:
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
    cache: dict[str, tuple[dict[str, Any], dict[int, dict[str, Any]]]] = {}
    rows: list[dict[str, Any]] = []
    for _, row in stress.iterrows():
        candidate_dir = resolve(Path(str(row["candidate_dir"])))
        key = str(candidate_dir)
        if key not in cache:
            manifest = read_json(candidate_dir / "manifest.json")
            cache[key] = (manifest, fold_map(manifest))
        manifest, folds = cache[key]
        fold = folds.get(as_int(row.get("fold")), {})
        risk = as_float(fold.get("risk_pct"))
        if dirty_source(manifest, fold):
            continue
        if not math.isnan(risk) and risk > float(args.max_source_risk_pct):
            continue
        out = row.to_dict()
        out["candidate_dir"] = key
        out["candidate_family"] = candidate_family(str(row["candidate_label"]), key)
        out["source_risk_pct"] = risk
        out["source_signals"] = as_int(fold.get("signals"), as_int(row.get("trades_available")))
        out["source_max_positions"] = as_int(fold.get("max_positions", fold.get("max_positions_hint")), 1)
        out["test_start"] = fold.get("test_start")
        out["test_end"] = fold.get("test_end")
        out["strict_pass"] = strict_pass(out, args)
        out["label_outcome"] = outcome(out, args)
        rows.append(out)
    clean = pd.DataFrame(rows)
    if clean.empty:
        raise ValueError("No clean stress rows remain.")
    return clean, cache


def feature_rows(clean: pd.DataFrame, regimes: pd.DataFrame, fold_id: int, args: argparse.Namespace, include_labels: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, group in clean.groupby("candidate_label", sort=True):
        current = group[group["fold"].eq(fold_id)]
        if current.empty:
            continue
        row = current.iloc[0].to_dict()
        stats = history_stats(group.sort_values("fold"), fold_id, args)
        out = {
            "fold": fold_id,
            "fold_sin": math.sin(2.0 * math.pi * float(fold_id) / 12.0),
            "fold_cos": math.cos(2.0 * math.pi * float(fold_id) / 12.0),
            "candidate_label": label,
            "candidate_dir": row["candidate_dir"],
            "candidate_family": row["candidate_family"],
            "source_risk_pct": row.get("source_risk_pct"),
            "source_signals": row.get("source_signals"),
            "source_max_positions": row.get("source_max_positions"),
            "signal_risk_density": as_float(row.get("source_risk_pct"), 0.0) * as_int(row.get("source_signals"), 0),
            **stats,
        }
        if fold_id in regimes.index:
            for col, value in regimes.loc[fold_id].items():
                out[col] = value
        test_start = pd.NaT
        if "test_start" in row:
            test_start = pd.to_datetime(row.get("test_start"), errors="coerce")
        if pd.isna(test_start):
            # Stress matrix rows do not always carry date columns; derive month from the template regime fold order.
            test_start = pd.NaT
        if not pd.isna(test_start):
            month = int(test_start.month)
            out["test_month"] = month
            out["test_month_sin"] = math.sin(2.0 * math.pi * month / 12.0)
            out["test_month_cos"] = math.cos(2.0 * math.pi * month / 12.0)
        if include_labels:
            out.update(
                {
                    "label_strict": int(bool(row["strict_pass"])),
                    "label_target": int(bool(row["target_hit"])),
                    "label_dd": int(bool(row["dd_pass"])),
                    "label_loss": int(bool(row["loss_fold"])),
                    "label_outcome": as_float(row["label_outcome"]),
                    "label_final_balance": as_float(row["final_balance"]),
                    "label_max_dd_pct": as_float(row["max_dd_pct"]),
                    "label_trades_used": as_int(row.get("trades_used")),
                }
            )
        rows.append(out)
    return pd.DataFrame(rows)


def make_models(seed: int) -> tuple[Pipeline, Pipeline]:
    clf = ExtraTreesClassifier(
        n_estimators=700,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    reg = ExtraTreesRegressor(
        n_estimators=700,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed + 10_000,
        n_jobs=-1,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", clf)]), Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("model", reg)]
    )


def score_current(train: pd.DataFrame, current: pd.DataFrame, fold_id: int, args: argparse.Namespace) -> pd.DataFrame:
    out = current.copy()
    if train.empty or train["label_strict"].nunique() < 2:
        out["ml_p_strict"] = out["hist_strict_rate"]
        out["ml_expected_outcome"] = out["hist_strict_rate"] * 4.0
    else:
        drop_cols = {
            "candidate_label",
            "candidate_dir",
            "label_strict",
            "label_target",
            "label_dd",
            "label_loss",
            "label_outcome",
            "label_final_balance",
            "label_max_dd_pct",
            "label_trades_used",
        }
        combined = pd.concat(
            [
                train.drop(columns=[c for c in drop_cols if c in train], errors="ignore"),
                out.drop(columns=[c for c in drop_cols if c in out], errors="ignore"),
            ],
            ignore_index=True,
        )
        combined = pd.get_dummies(combined, columns=["candidate_family"], dummy_na=True)
        x_train = combined.iloc[: len(train)].copy()
        x_current = combined.iloc[len(train) :].copy()
        for col in x_train.columns:
            x_train[col] = pd.to_numeric(x_train[col], errors="coerce")
            x_current[col] = pd.to_numeric(x_current[col], errors="coerce")
        clf, reg = make_models(seed=fold_id)
        clf.fit(x_train, train["label_strict"].astype(int))
        reg.fit(x_train, pd.to_numeric(train["label_outcome"], errors="coerce").fillna(0.0))
        out["ml_p_strict"] = clf.predict_proba(x_current)[:, 1]
        out["ml_expected_outcome"] = reg.predict(x_current)
    out["ml_score"] = (
        pd.to_numeric(out["ml_p_strict"], errors="coerce").fillna(0.0) * float(args.p_strict_weight)
        + pd.to_numeric(out["ml_expected_outcome"], errors="coerce").fillna(0.0) * float(args.outcome_weight)
        + pd.to_numeric(out["hist_strict_rate"], errors="coerce").fillna(0.0) * float(args.hist_strict_weight)
        + pd.to_numeric(out["hist_recent_strict"], errors="coerce").fillna(0.0) * float(args.recent_strict_weight)
        + pd.to_numeric(out["hist_dd_rate"], errors="coerce").fillna(0.0) * float(args.hist_dd_weight)
        - pd.to_numeric(out["hist_loss_rate"], errors="coerce").fillna(1.0) * float(args.hist_loss_weight)
        + pd.to_numeric(out["hist_min_final"], errors="coerce").fillna(float(args.deposit)) / float(args.target_balance) * float(args.min_final_weight)
        + pd.to_numeric(out["hist_worst_dd"], errors="coerce").fillna(-100.0) / abs(float(args.min_dd_pct)) * float(args.worst_dd_weight)
        + np.minimum(pd.to_numeric(out["signal_risk_density"], errors="coerce").fillna(0.0), float(args.max_density_cap))
        * float(args.density_weight)
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Chronological ML selector trained on realistic MT5 stress feedback.")
    parser.add_argument("--stress-folds", type=Path, required=True)
    parser.add_argument("--stress-summary", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--template-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=30)
    parser.add_argument("--min-history-rows", type=int, default=5)
    parser.add_argument("--history-lookback", type=int, default=0)
    parser.add_argument("--recent-folds", type=int, default=3)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--min-dd-pct", type=float, default=-20.0)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--max-source-risk-pct", type=float, default=4.0)
    parser.add_argument("--regime-lookback-days", type=int, default=20)
    parser.add_argument("--p-strict-weight", type=float, default=160000.0)
    parser.add_argument("--outcome-weight", type=float, default=45000.0)
    parser.add_argument("--hist-strict-weight", type=float, default=35000.0)
    parser.add_argument("--recent-strict-weight", type=float, default=25000.0)
    parser.add_argument("--hist-dd-weight", type=float, default=14000.0)
    parser.add_argument("--hist-loss-weight", type=float, default=90000.0)
    parser.add_argument("--min-final-weight", type=float, default=8000.0)
    parser.add_argument("--worst-dd-weight", type=float, default=6000.0)
    parser.add_argument("--density-weight", type=float, default=8.0)
    parser.add_argument("--max-density-cap", type=float, default=2200.0)
    parser.add_argument("--bootstrap-candidate-label", default="")
    args = parser.parse_args()
    if int(args.history_lookback) > 0 and int(args.min_history_rows) > int(args.history_lookback):
        raise ValueError("--min-history-rows cannot exceed --history-lookback when lookback is positive.")

    clean, cache = prepare_clean_rows(args)
    template = read_json(resolve(args.template_manifest))
    template_folds = [fold for fold in template.get("folds", []) if int(args.fold_start) <= as_int(fold.get("fold")) <= int(args.fold_end)]
    regimes = _load_feature_regimes(resolve(args.features), template_folds, int(args.regime_lookback_days))
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_label = str(args.bootstrap_candidate_label).strip()
    if not bootstrap_label:
        summary = pd.read_csv(resolve(args.stress_summary))
        clean_labels = set(clean["candidate_label"].astype(str))
        ranked = summary[summary["candidate_label"].astype(str).isin(clean_labels)].copy()
        ranked = ranked.sort_values(
            ["all_pass_folds", "target_pass_folds", "dd_pass_folds", "median_final_balance"],
            ascending=[False, False, False, False],
        )
        bootstrap_label = str(ranked.iloc[0]["candidate_label"])

    selected_folds: list[dict[str, Any]] = []
    choices: list[dict[str, Any]] = []
    audit_parts: list[pd.DataFrame] = []
    train_parts: list[pd.DataFrame] = []
    bootstrap_count = 0

    for fold_id in range(int(args.fold_start), int(args.fold_end) + 1):
        current = feature_rows(clean, regimes, fold_id, args, include_labels=True)
        if current.empty:
            raise ValueError(f"No current rows for fold {fold_id}")
        train_rows = []
        for hist_fold in range(int(args.fold_start), fold_id):
            hist_features = feature_rows(clean, regimes, hist_fold, args, include_labels=True)
            if not hist_features.empty:
                hist_features = hist_features[hist_features["hist_rows"] >= int(args.min_history_rows)].copy()
                train_rows.append(hist_features)
        train = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
        eligible = current[current["hist_rows"] >= int(args.min_history_rows)].copy()
        selection_mode = "stress_ml_prior_selector"
        if eligible.empty:
            bootstrap_count += 1
            scored = current.copy()
            scored["ml_p_strict"] = scored["hist_strict_rate"]
            scored["ml_expected_outcome"] = scored["hist_strict_rate"] * 4.0
            scored["ml_score"] = -1.0e18
            scored = scored[scored["candidate_label"].astype(str).eq(bootstrap_label)].copy()
            if scored.empty:
                raise ValueError(f"Bootstrap candidate label not available for fold {fold_id}: {bootstrap_label}")
            selection_mode = "bootstrap_declared_stress_ml"
        else:
            scored = score_current(train, eligible, fold_id, args)
            scored = scored.sort_values(["ml_score", "ml_p_strict", "hist_recent_strict", "hist_strict_rate"], ascending=False)
        chosen = scored.iloc[0].to_dict()
        candidate_dir = str(chosen["candidate_dir"])
        manifest, folds = cache[candidate_dir]
        source_fold = dict(folds[fold_id])
        source_csv = resolve_csv(Path(candidate_dir) / "manifest.json", str(source_fold["csv"]))
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
                "selected_candidate_label": str(chosen["candidate_label"]),
                "selected_candidate_dir": candidate_dir,
                "selected_candidate_adaptive_per_fold": False,
                "selected_candidate_research_oracle_fold_selection": False,
                "selected_candidate_selection_uses_current_fold_metrics": False,
                "stress_ml_score_from_past": as_float(chosen.get("ml_score")),
                "stress_ml_p_strict_from_past": as_float(chosen.get("ml_p_strict")),
                "stress_ml_expected_outcome_from_past": as_float(chosen.get("ml_expected_outcome")),
                "hist_rows": as_int(chosen.get("hist_rows")),
                "hist_strict_rate": as_float(chosen.get("hist_strict_rate")),
                "hist_target_rate": as_float(chosen.get("hist_target_rate")),
                "hist_dd_rate": as_float(chosen.get("hist_dd_rate")),
                "hist_loss_rate": as_float(chosen.get("hist_loss_rate")),
            }
        )
        selected_folds.append(out_fold)
        choice = {
            **out_fold,
            "audit_current_final_not_used": as_float(chosen.get("label_final_balance")),
            "audit_current_dd_not_used": as_float(chosen.get("label_max_dd_pct")),
            "audit_current_strict_not_used": bool(chosen.get("label_strict")),
        }
        choices.append(choice)
        audit_parts.append(scored.assign(selection_fold=fold_id))
        if not train.empty:
            train_parts.append(train.assign(selection_fold=fold_id))

    choices_df = pd.DataFrame(choices)
    manifest_out = {
        "all_signals": None,
        "folds": selected_folds,
        "total_signals": int(sum(as_int(fold.get("signals")) for fold in selected_folds)),
        "live_protocol": False,
        "stress_ml_prior_selector": True,
        "adaptive_per_fold": False,
        "research_oracle_fold_selection": False,
        "selection_uses_current_fold_metrics": False,
        "bootstrap_declared_folds": int(bootstrap_count),
        "bootstrap_candidate_label": bootstrap_label,
        "stress_folds": str(resolve(args.stress_folds)),
        "stress_summary": str(resolve(args.stress_summary)),
        "features": str(resolve(args.features)),
        "note": "Research selector. ML training labels are realistic MT5 stress rows from folds before the selected fold only; selected fold metrics are audit-only.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    choices_df.to_csv(out_dir / "stress_ml_selector_choices.csv", index=False)
    pd.concat(audit_parts, ignore_index=True).to_csv(out_dir / "stress_ml_candidate_audit.csv", index=False)
    if train_parts:
        pd.concat(train_parts, ignore_index=True).to_csv(out_dir / "stress_ml_training_rows.csv", index=False)
    regimes.reset_index().to_csv(out_dir / "fold_regime_features.csv", index=False)
    summary = {
        "manifest": str(out_dir / "manifest.json"),
        "folds": int(len(choices_df)),
        "stress_strict_pass": int(choices_df["audit_current_strict_not_used"].sum()),
        "target_pass": int((choices_df["audit_current_final_not_used"] >= float(args.target_balance)).sum()),
        "dd_pass": int((choices_df["audit_current_dd_not_used"] >= float(args.min_dd_pct)).sum()),
        "loss_folds": int((choices_df["audit_current_final_not_used"] < float(args.deposit)).sum()),
        "min_final": float(choices_df["audit_current_final_not_used"].min()),
        "median_final": float(choices_df["audit_current_final_not_used"].median()),
        "worst_dd": float(choices_df["audit_current_dd_not_used"].min()),
        "fail_folds": choices_df.loc[~choices_df["audit_current_strict_not_used"], "fold"].astype(int).tolist(),
        "bootstrap_declared_folds": int(bootstrap_count),
    }
    (out_dir / "stress_ml_selector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
