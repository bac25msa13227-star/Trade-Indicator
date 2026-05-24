from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding).replace("NaN", "null"))
        except (UnicodeError, json.JSONDecodeError):
            continue
    return {}


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def resolve_csv(manifest_path: Path, csv_text: str) -> Path:
    raw = Path(csv_text)
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
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def find_fold(manifest: dict[str, Any], fold_id: int) -> dict[str, Any]:
    for fold in manifest.get("folds", []) or []:
        if as_int(fold.get("fold")) == int(fold_id):
            return dict(fold)
    raise ValueError(f"fold {fold_id} not found")


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
        }
    strict = history["stress_strict_pass"].astype(bool)
    target = history["target_hit"].astype(bool)
    dd = history["dd_pass"].astype(bool)
    loss = history["loss_fold"].astype(bool)
    rows = len(history)
    strict_rate = float(strict.sum()) / rows
    target_rate = float(target.sum()) / rows
    dd_rate = float(dd.sum()) / rows
    loss_rate = float(loss.sum()) / rows
    finals = pd.to_numeric(history["final_balance"], errors="coerce")
    dds = pd.to_numeric(history["max_dd_pct"], errors="coerce")
    median_final = float(finals.median())
    min_final = float(finals.min())
    worst_dd = float(dds.min())
    trades = float(pd.to_numeric(history.get("trades_used"), errors="coerce").fillna(0).median())
    score = (
        float(args.strict_weight) * strict_rate
        + float(args.target_weight) * target_rate
        + float(args.dd_weight) * dd_rate
        - float(args.loss_weight) * loss_rate
        + float(args.median_final_weight) * ((median_final - float(args.target_balance)) / float(args.target_balance))
        + float(args.min_final_weight) * ((min_final - float(args.target_balance)) / float(args.target_balance))
        + float(args.worst_dd_weight) * ((worst_dd + float(args.max_dd_pct)) / float(args.max_dd_pct))
        - float(args.trades_penalty_weight) * max(0.0, trades - float(args.max_preferred_trades)) / 1000.0
    )
    return {
        "score": score,
        "hist_rows": int(rows),
        "hist_strict_rate": strict_rate,
        "hist_target_rate": target_rate,
        "hist_dd_rate": dd_rate,
        "hist_loss_rate": loss_rate,
        "hist_median_final": median_final,
        "hist_min_final": min_final,
        "hist_worst_dd": worst_dd,
    }


def copy_selected_fold(
    *,
    candidate_dir: Path,
    candidate_label: str,
    fold_id: int,
    out_dir: Path,
    score: dict[str, Any],
    bootstrap: bool,
) -> dict[str, Any]:
    manifest_path = candidate_dir / "manifest.json"
    manifest = read_json(manifest_path)
    source_fold = find_fold(manifest, fold_id)
    source_csv = resolve_csv(manifest_path, str(source_fold["csv"]))
    target_csv = out_dir / f"fold_{fold_id:02d}_signals.csv"
    shutil.copy2(source_csv, target_csv)
    signal_count = max(0, sum(1 for _ in target_csv.open("r", encoding="utf-8")) - 1)

    out = dict(source_fold)
    out.update(
        {
            "fold": int(fold_id),
            "csv": str(target_csv),
            "signals": int(signal_count),
            "selection_mode": "stress_prior_rolling_selector_bootstrap" if bootstrap else "stress_prior_rolling_selector",
            "selection_uses_current_fold_metrics": False,
            "selected_candidate_label": candidate_label,
            "selected_candidate_dir": str(candidate_dir),
            "selected_candidate_live_protocol": is_true(manifest.get("live_protocol")),
            "selected_candidate_adaptive_per_fold": is_true(manifest.get("adaptive_per_fold")),
            "selected_candidate_research_oracle_fold_selection": is_true(manifest.get("research_oracle_fold_selection")),
            "selected_candidate_selection_uses_current_fold_metrics": is_true(
                manifest.get("selection_uses_current_fold_metrics")
            ),
            "bootstrap_declared": bool(bootstrap),
        }
    )
    for key, value in score.items():
        out[f"stress_hist_{key}"] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a rolling selector from prior realistic MT5 stress feedback.")
    parser.add_argument("--stress-folds", type=Path, required=True)
    parser.add_argument("--stress-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=30)
    parser.add_argument("--history-lookback", type=int, default=0)
    parser.add_argument("--min-history-rows", type=int, default=4)
    parser.add_argument("--bootstrap-candidate-label", default="")
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--strict-weight", type=float, default=1400.0)
    parser.add_argument("--target-weight", type=float, default=900.0)
    parser.add_argument("--dd-weight", type=float, default=900.0)
    parser.add_argument("--loss-weight", type=float, default=1600.0)
    parser.add_argument("--median-final-weight", type=float, default=300.0)
    parser.add_argument("--min-final-weight", type=float, default=450.0)
    parser.add_argument("--worst-dd-weight", type=float, default=300.0)
    parser.add_argument("--trades-penalty-weight", type=float, default=40.0)
    parser.add_argument("--max-preferred-trades", type=float, default=250.0)
    args = parser.parse_args()

    folds = pd.read_csv(resolve(args.stress_folds))
    summary = pd.read_csv(resolve(args.stress_summary))
    for col in ["target_hit", "dd_pass", "loss_fold"]:
        folds[col] = bool_series(folds[col])
    folds["stress_strict_pass"] = folds["target_hit"] & folds["dd_pass"] & ~folds["loss_fold"]
    folds["fold"] = pd.to_numeric(folds["fold"], errors="coerce").astype(int)

    blocked_labels = set(
        summary.loc[
            bool_series(summary.get("adaptive_per_fold", pd.Series(False, index=summary.index)))
            | bool_series(summary.get("research_oracle_fold_selection", pd.Series(False, index=summary.index)))
            | bool_series(summary.get("selection_uses_current_fold_metrics", pd.Series(False, index=summary.index))),
            "candidate_label",
        ].astype(str)
    )
    folds = folds[~folds["candidate_label"].astype(str).isin(blocked_labels)].copy()
    candidate_dirs = {
        str(row["candidate_label"]): resolve(Path(str(row["candidate_dir"])))
        for _, row in folds[["candidate_label", "candidate_dir"]].drop_duplicates().iterrows()
    }
    if not candidate_dirs:
        raise ValueError("No non-oracle candidates remain.")

    args.out_dir = resolve(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selected_folds: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    bootstrap_count = 0

    default_bootstrap = str(args.bootstrap_candidate_label).strip()
    if not default_bootstrap:
        summary_clean = summary[~summary["candidate_label"].astype(str).isin(blocked_labels)].copy()
        summary_clean = summary_clean.sort_values(
            ["all_pass_folds", "target_pass_folds", "dd_pass_folds", "median_final_balance"],
            ascending=[False, False, False, False],
        )
        default_bootstrap = str(summary_clean.iloc[0]["candidate_label"])

    for fold_id in range(int(args.fold_start), int(args.fold_end) + 1):
        choices = folds[folds["fold"].eq(fold_id)].copy()
        if choices.empty:
            raise ValueError(f"No stress rows for fold {fold_id}")
        scored: list[dict[str, Any]] = []
        for label in sorted(candidate_dirs):
            history = folds[(folds["candidate_label"].astype(str) == label) & (folds["fold"] < fold_id)].copy()
            if int(args.history_lookback) > 0:
                history = history[history["fold"] >= fold_id - int(args.history_lookback)].copy()
            score = score_history(history, args)
            current = choices[choices["candidate_label"].astype(str) == label]
            if current.empty:
                continue
            row = current.iloc[0].to_dict()
            scored_row = {"fold": fold_id, "candidate_label": label, **score}
            score_rows.append(scored_row)
            scored.append({**scored_row, **{f"actual_{k}": v for k, v in row.items()}})

        if not scored:
            raise ValueError(f"No candidate choices for fold {fold_id}")
        scored_df = pd.DataFrame(scored)
        bootstrap = False
        if int(scored_df["hist_rows"].max()) < int(args.min_history_rows):
            label = default_bootstrap
            bootstrap = True
            bootstrap_count += 1
            score = score_history(
                folds[(folds["candidate_label"].astype(str) == label) & (folds["fold"] < fold_id)].copy(),
                args,
            )
            selected_row = scored_df[scored_df["candidate_label"].astype(str) == label].iloc[0].to_dict()
        else:
            scored_df = scored_df.sort_values(
                ["score", "hist_strict_rate", "hist_dd_rate", "hist_median_final", "candidate_label"],
                ascending=[False, False, False, False, True],
            )
            selected_row = scored_df.iloc[0].to_dict()
            label = str(selected_row["candidate_label"])
            score = {k: selected_row[k] for k in selected_row if k in {
                "score",
                "hist_rows",
                "hist_strict_rate",
                "hist_target_rate",
                "hist_dd_rate",
                "hist_loss_rate",
                "hist_median_final",
                "hist_min_final",
                "hist_worst_dd",
            }}

        fold = copy_selected_fold(
            candidate_dir=candidate_dirs[label],
            candidate_label=label,
            fold_id=fold_id,
            out_dir=args.out_dir,
            score=score,
            bootstrap=bootstrap,
        )
        selected_folds.append(fold)
        audit_rows.append(
            {
                "fold": fold_id,
                "selected_candidate_label": label,
                "bootstrap_declared": bootstrap,
                "actual_final_balance": as_float(selected_row.get("actual_final_balance")),
                "actual_max_dd_pct": as_float(selected_row.get("actual_max_dd_pct")),
                "actual_target_hit": bool(selected_row.get("actual_target_hit")),
                "actual_dd_pass": bool(selected_row.get("actual_dd_pass")),
                "actual_loss_fold": bool(selected_row.get("actual_loss_fold")),
                "actual_stress_strict_pass": bool(selected_row.get("actual_stress_strict_pass")),
                "score": as_float(score.get("score")),
                "hist_rows": as_int(score.get("hist_rows")),
                "hist_strict_rate": as_float(score.get("hist_strict_rate")),
                "hist_target_rate": as_float(score.get("hist_target_rate")),
                "hist_dd_rate": as_float(score.get("hist_dd_rate")),
                "hist_loss_rate": as_float(score.get("hist_loss_rate")),
            }
        )

    manifest = {
        "all_signals": None,
        "folds": selected_folds,
        "total_signals": int(sum(as_int(fold.get("signals")) for fold in selected_folds)),
        "live_protocol": False,
        "adaptive_per_fold": False,
        "research_oracle_fold_selection": False,
        "selection_uses_current_fold_metrics": False,
        "stress_prior_rolling_selector": True,
        "bootstrap_declared_folds": int(bootstrap_count),
        "bootstrap_candidate_label": default_bootstrap,
        "stress_folds": str(resolve(args.stress_folds)),
        "stress_summary": str(resolve(args.stress_summary)),
        "note": "Research shadow selector. It uses only prior stress feedback for non-bootstrap folds; bootstrap needs pre-actual calibration before live use.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.out_dir / "stress_selection_audit.csv", index=False)
    pd.DataFrame(score_rows).to_csv(args.out_dir / "stress_candidate_scores_by_fold.csv", index=False)
    summary_out = {
        "manifest": str(args.out_dir / "manifest.json"),
        "folds": int(len(audit)),
        "stress_strict_pass": int(audit["actual_stress_strict_pass"].sum()),
        "target_pass": int(audit["actual_target_hit"].sum()),
        "dd_pass": int(audit["actual_dd_pass"].sum()),
        "loss_folds": int(audit["actual_loss_fold"].sum()),
        "min_final": float(audit["actual_final_balance"].min()),
        "median_final": float(audit["actual_final_balance"].median()),
        "worst_dd": float(audit["actual_max_dd_pct"].min()),
        "fail_folds": audit.loc[~audit["actual_stress_strict_pass"], "fold"].astype(int).tolist(),
        "bootstrap_declared_folds": int(bootstrap_count),
        "bootstrap_candidate_label": default_bootstrap,
    }
    (args.out_dir / "stress_selector_summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    print(json.dumps(summary_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
