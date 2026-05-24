from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xauusd_ai.features.dataset import FEATURE_COLUMNS  # noqa: E402


MT5_COLUMNS = ["open_time", "direction", "entry_price", "sl_price", "tp_price", "atr", "probability"]


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except (UnicodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not parse JSON file: {path}")


def resolve_csv(manifest_path: Path, csv_text: str) -> Path:
    raw = Path(csv_text)
    if raw.is_absolute():
        return raw
    for candidate in [ROOT / raw, manifest_path.parent / raw.name, manifest_path.parent / raw]:
        if candidate.exists():
            return candidate
    return ROOT / raw


def load_features(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    cols = [c for c in ["time", "open", "high", "low", "close", "trade_side"] + FEATURE_COLUMNS if c in header]
    frame = pd.read_csv(path, usecols=cols)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"]).sort_values("time").drop_duplicates("time", keep="last")
    return frame.set_index("time")


def add_join_time_from_signal(frame: pd.DataFrame, column: str = "open_time") -> pd.DataFrame:
    out = frame.copy()
    out["join_time"] = pd.to_datetime(out[column], format="%Y.%m.%d %H:%M", errors="coerce")
    return out


def build_model_frame(feedback: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    fb = feedback.copy()
    fb = fb[fb.get("signal_matched", True).astype(bool)].copy()
    fb["join_time"] = pd.to_datetime(fb["signal_open_time"], format="%Y.%m.%d %H:%M", errors="coerce")
    fb = fb.dropna(subset=["join_time", "profit"])
    joined = fb.join(features, on="join_time", how="inner", rsuffix="_feature")
    joined["label"] = (pd.to_numeric(joined["profit"], errors="coerce").fillna(0.0) > 0.0).astype(int)
    joined["signal_probability"] = pd.to_numeric(joined.get("signal_probability"), errors="coerce")
    joined["signal_atr"] = pd.to_numeric(joined.get("signal_atr"), errors="coerce")
    joined["direction"] = pd.to_numeric(joined.get("direction"), errors="coerce")
    joined["hour"] = pd.to_numeric(joined.get("hour"), errors="coerce")
    joined["weekday"] = pd.to_numeric(joined.get("weekday"), errors="coerce")
    return joined


def feature_columns(frame: pd.DataFrame) -> list[str]:
    extras = ["signal_probability", "signal_atr", "direction", "hour", "weekday"]
    cols = [c for c in FEATURE_COLUMNS + extras if c in frame.columns]
    return cols


def make_classifier(kind: str, seed: int) -> Pipeline:
    if kind == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])
    clf = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=seed,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)])


def score_signals(signals: pd.DataFrame, features: pd.DataFrame, model: Pipeline, cols: list[str]) -> pd.DataFrame:
    sig = add_join_time_from_signal(signals)
    sig["signal_probability"] = pd.to_numeric(sig.get("probability"), errors="coerce")
    sig["signal_atr"] = pd.to_numeric(sig.get("atr"), errors="coerce")
    sig["direction"] = pd.to_numeric(sig.get("direction"), errors="coerce")
    sig["hour"] = sig["join_time"].dt.hour
    sig["weekday"] = sig["join_time"].dt.weekday
    joined = sig.join(features, on="join_time", how="inner", rsuffix="_feature")
    if joined.empty:
        return joined
    joined["feedback_model_probability"] = model.predict_proba(joined[cols])[:, 1]
    return joined


def select_rows(scored: pd.DataFrame, min_probability: float, top_k: int, min_keep: int) -> pd.DataFrame:
    if scored.empty:
        return scored
    filtered = scored[scored["feedback_model_probability"] >= float(min_probability)].copy()
    if len(filtered) < int(min_keep):
        filtered = scored.sort_values("feedback_model_probability", ascending=False).head(int(min_keep)).copy()
    if int(top_k) > 0 and len(filtered) > int(top_k):
        filtered = filtered.sort_values("feedback_model_probability", ascending=False).head(int(top_k)).copy()
    filtered = filtered.sort_values("join_time").copy()
    export = filtered[MT5_COLUMNS].copy()
    export["probability"] = filtered["feedback_model_probability"].astype(float)
    return export


def main() -> int:
    parser = argparse.ArgumentParser(description="Export rolling signals filtered by a model trained on prior MT5 trade feedback.")
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=["extra_trees", "hgb"], default="extra_trees")
    parser.add_argument("--min-train-rows", type=int, default=800)
    parser.add_argument("--min-probability", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=600)
    parser.add_argument("--min-keep", type=int, default=80)
    parser.add_argument("--risk-multiplier", type=float, default=1.0)
    parser.add_argument("--max-risk-pct", type=float, default=8.0)
    parser.add_argument("--max-positions", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.base_manifest
    manifest = read_json(manifest_path)
    feedback = pd.read_csv(args.feedback)
    feedback["fold"] = pd.to_numeric(feedback["fold"], errors="coerce")
    features = load_features(args.features)

    folds: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    total = 0
    for source_fold in manifest.get("folds", []) or []:
        fold_id = int(source_fold["fold"])
        hist = feedback[feedback["fold"] < fold_id].copy()
        train = build_model_frame(hist, features)
        source_csv = resolve_csv(manifest_path, str(source_fold["csv"]))
        signals = pd.read_csv(source_csv)
        export = pd.DataFrame(columns=MT5_COLUMNS)
        mode = "no_trade_insufficient_feedback"
        train_rows = int(len(train))
        positive_rate = float(train["label"].mean()) if train_rows else float("nan")
        if train_rows >= int(args.min_train_rows) and train["label"].nunique() == 2:
            cols = feature_columns(train)
            model = make_classifier(args.model_type, seed=fold_id)
            model.fit(train[cols], train["label"])
            scored = score_signals(signals, features, model, cols)
            export = select_rows(scored, args.min_probability, args.top_k, args.min_keep)
            mode = "prior_mt5_feedback_model"
        target_csv = args.out_dir / f"fold_{fold_id:02d}_signals.csv"
        export.to_csv(target_csv, index=False, float_format="%.5f")
        risk_pct = min(float(source_fold.get("risk_pct", 0.0)) * float(args.risk_multiplier), float(args.max_risk_pct))
        max_positions = int(args.max_positions if args.max_positions is not None else source_fold.get("max_positions", 1))
        fold = dict(source_fold)
        for key in list(fold):
            if key.startswith("selected_candidate_"):
                fold.pop(key, None)
        fold.update(
            {
                "signals": int(len(export)),
                "csv": str(target_csv),
                "risk_pct": round(risk_pct, 4),
                "max_risk_pct": round(risk_pct, 4),
                "max_exposure_pct": round(risk_pct * max_positions, 4),
                "max_positions": max_positions,
                "selection_mode": mode,
                "selection_uses_current_fold_metrics": False,
                "feedback_model_train_rows": train_rows,
                "feedback_model_positive_rate": positive_rate,
                "feedback_model_min_probability": float(args.min_probability),
                "feedback_model_top_k": int(args.top_k),
                "feedback_model_min_keep": int(args.min_keep),
            }
        )
        folds.append(fold)
        total += int(len(export))
        rules.append(
            {
                "fold": fold_id,
                "signals": int(len(export)),
                "selection_mode": mode,
                "train_rows": train_rows,
                "positive_rate": positive_rate,
                "risk_pct": risk_pct,
            }
        )

    out_manifest = {
        "all_signals": None,
        "folds": folds,
        "total_signals": int(total),
        "live_protocol": True,
        "mt5_feedback_model_selector": True,
        "selection_uses_current_fold_metrics": False,
        "research_oracle_fold_selection": False,
        "base_manifest": str(args.base_manifest),
        "feedback": str(args.feedback),
        "features": str(args.features),
        "model_type": args.model_type,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    pd.DataFrame(rules).to_csv(args.out_dir / "feedback_model_rules.csv", index=False)
    print(json.dumps({"manifest": str(args.out_dir / "manifest.json"), "folds": len(folds), "signals": total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
