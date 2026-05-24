from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_mt5_feedback_model_manifest import (  # noqa: E402
    build_model_frame,
    feature_columns as feedback_feature_columns,
    load_features as load_feedback_features,
    make_classifier,
)
from scripts.train_target1200_signal_universe import (  # noqa: E402
    available_feature_columns,
    first_hit_outcomes,
    fit_predict_oos,
    load_feature_frame,
    parse_hours,
)


MT5_COLUMNS = ["open_time", "direction", "entry_price", "sl_price", "tp_price", "atr", "probability"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def utc_ts(value: str | pd.Timestamp | datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def iso(value: pd.Timestamp | datetime) -> str:
    return utc_ts(value).isoformat()


def resolve_path(raw: Any, base: Path | None = None) -> Path:
    value = Path(str(raw))
    if value.is_absolute():
        return value
    candidates = [ROOT / value]
    if base is not None:
        candidates.extend([base.parent / value.name, base.parent / value])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / value).resolve()


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def bridge_get_json(bridge_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
    query = ""
    if params:
        query = "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(bridge_url.rstrip("/") + path + query, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def latest_closed_bar_from_bridge(bridge_url: str, symbol: str) -> pd.Timestamp:
    rates = bridge_get_json(bridge_url, "/rates", {"symbol": symbol, "timeframe": "M5", "bars": 6})
    if not isinstance(rates, list) or len(rates) < 2:
        raise RuntimeError("bridge returned fewer than 2 M5 rates")
    times = sorted(int(row["time"]) for row in rates)
    return pd.Timestamp(datetime.fromtimestamp(times[-2], tz=timezone.utc))


def current_fold(folds: list[dict[str, Any]], ts: pd.Timestamp) -> dict[str, Any]:
    ts = utc_ts(ts)
    for fold in folds:
        if utc_ts(fold["test_start"]) <= ts < utc_ts(fold["test_end"]):
            return dict(fold)
    raise RuntimeError(f"no fold covers {ts.isoformat()}")


def empty_signal_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MT5_COLUMNS)
        writer.writeheader()


def write_signal_csv(path: Path, row: dict[str, Any] | None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MT5_COLUMNS)
        writer.writeheader()
        if row is not None:
            writer.writerow({key: row[key] for key in MT5_COLUMNS})
            return 1
    return 0


def load_combined_feedback_features(training_features: Path, live_features: Path) -> pd.DataFrame:
    train = load_feedback_features(training_features)
    live = load_feedback_features(live_features)
    combined = pd.concat([train, live]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def base_candidate_for_bar(
    *,
    features_path: Path,
    fold: dict[str, Any],
    target_bar: pd.Timestamp,
    tp_rr: float,
    sl_mult: float,
    horizon_bars: int,
    min_probability: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    feature_cols = available_feature_columns(features_path)
    frame = load_feature_frame(
        features_path=features_path,
        feature_cols=feature_cols,
        folds=[fold],
        broker_gmt=args.broker_gmt,
        include_hours=parse_hours(args.include_broker_hours),
        exclude_hours=parse_hours(args.exclude_broker_hours),
        side=args.side,
        direction_mode=args.direction_mode,
        min_atr_percentile=args.min_atr_percentile,
        max_atr_percentile=args.max_atr_percentile,
        exclude_news_blackout=args.exclude_news_blackout,
        candidate_stride=args.candidate_stride,
    )
    rr_all, _ = first_hit_outcomes(
        open_=frame["open"].to_numpy(np.float64),
        high=frame["high"].to_numpy(np.float64),
        low=frame["low"].to_numpy(np.float64),
        close=frame["close"].to_numpy(np.float64),
        atr=frame["atr"].to_numpy(np.float64),
        direction=frame["direction"].to_numpy(np.int8),
        tp_rr=float(tp_rr),
        sl_atr_mult=float(sl_mult),
        horizon_bars=int(horizon_bars),
        entry_delay_bars=int(args.entry_delay_bars),
    )
    scored = fit_predict_oos(
        frame=frame,
        folds=[fold],
        feature_cols=feature_cols,
        labels=rr_all,
        horizon_bars=int(horizon_bars),
        args=SimpleNamespace(**vars(args)),
    )
    if scored.empty:
        return {"status": "no_trade", "reason": "base model produced no scored rows"}

    target_ns = utc_ts(target_bar).value
    scored_time = pd.to_datetime(scored["time"], utc=True).astype("int64")
    exact = scored[scored_time == target_ns].copy()
    if exact.empty:
        return {"status": "no_trade", "reason": "target bar absent after base feature filters"}

    row = exact.sort_values("probability", ascending=False).iloc[0]
    probability = float(row["probability"])
    direction = int(row["direction"])
    entry_price = float(row["close"])
    atr = float(row["atr"])
    sl_dist = atr * float(sl_mult)
    if direction not in {-1, 1} or not np.isfinite(entry_price) or not np.isfinite(atr) or atr <= 0:
        return {"status": "no_trade", "reason": "base row has invalid trade plan"}
    is_buy = direction == 1
    sl_price = entry_price - sl_dist if is_buy else entry_price + sl_dist
    tp_price = entry_price + sl_dist * float(tp_rr) if is_buy else entry_price - sl_dist * float(tp_rr)
    signal_row = {
        "open_time": (utc_ts(target_bar) + pd.Timedelta(hours=int(args.broker_gmt))).strftime("%Y.%m.%d %H:%M"),
        "direction": direction,
        "entry_price": f"{entry_price:.5f}",
        "sl_price": f"{sl_price:.5f}",
        "tp_price": f"{tp_price:.5f}",
        "atr": f"{atr:.5f}",
        "probability": f"{probability:.5f}",
    }
    return {
        "status": "candidate" if probability >= float(min_probability) else "no_trade",
        "reason": "base pass threshold" if probability >= float(min_probability) else "base probability below threshold",
        "probability": probability,
        "direction": direction,
        "side": "buy" if is_buy else "sell",
        "entry_price": entry_price,
        "atr": atr,
        "sl_price": float(sl_price),
        "tp_price": float(tp_price),
        "trade_side_feature": str(row.get("trade_side", "")),
        "model_side": str(row.get("model_side", "")),
        "signal_row": signal_row,
    }


def feedback_score_signal(
    *,
    signal_row: dict[str, Any],
    feedback_path: Path,
    feedback_features: pd.DataFrame,
    fold_id: int,
    model_type: str,
    min_train_rows: int,
) -> dict[str, Any]:
    feedback = pd.read_csv(feedback_path)
    feedback["fold"] = pd.to_numeric(feedback["fold"], errors="coerce")
    hist = feedback[feedback["fold"] < int(fold_id)].copy()
    train = build_model_frame(hist, feedback_features)
    if len(train) < int(min_train_rows) or train["label"].nunique() < 2:
        return {
            "status": "no_trade",
            "reason": "insufficient prior MT5 feedback for feedback model",
            "train_rows": int(len(train)),
        }
    cols = feedback_feature_columns(train)
    model = make_classifier(model_type, seed=int(fold_id))
    model.fit(train[cols], train["label"])

    sig = pd.DataFrame([signal_row])
    sig["join_time"] = pd.to_datetime(sig["open_time"], format="%Y.%m.%d %H:%M", errors="coerce")
    sig["signal_probability"] = pd.to_numeric(sig["probability"], errors="coerce")
    sig["signal_atr"] = pd.to_numeric(sig["atr"], errors="coerce")
    sig["direction"] = pd.to_numeric(sig["direction"], errors="coerce")
    sig["hour"] = sig["join_time"].dt.hour
    sig["weekday"] = sig["join_time"].dt.weekday
    joined = sig.join(feedback_features, on="join_time", how="inner", rsuffix="_feature")
    if joined.empty:
        return {"status": "no_trade", "reason": "feedback feature join missing for target bar", "train_rows": int(len(train))}
    probability = float(model.predict_proba(joined[cols])[:, 1][0])
    return {
        "status": "candidate",
        "probability": probability,
        "train_rows": int(len(train)),
        "positive_rate": float(train["label"].mean()),
        "feature_columns": len(cols),
    }


def result_summary(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    for col in ["final_balance", "max_dd_pct", "loaded_signals", "signals", "trades"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return {
        "results_path": str(path),
        "rows": int(len(df)),
        "target_pass_folds": int((df["final_balance"] >= 1200.0).sum()),
        "dd_pass_folds": int((df["max_dd_pct"] >= -20.0).sum()),
        "loss_folds": int((df["final_balance"] < 200.0).sum()),
        "min_final_balance": float(df["final_balance"].min()),
        "worst_dd_pct": float(df["max_dd_pct"].min()),
        "loaded_signal_max_abs_gap": float((df["loaded_signals"] - df["signals"]).abs().max()),
        "total_trades": int(df["trades"].fillna(0).sum()),
    }


def write_online_preflight(out_dir: Path, manifest: dict[str, Any], results: Path) -> None:
    blockers: list[str] = []
    source_flags = manifest.get("source_manifest_flags") if isinstance(manifest.get("source_manifest_flags"), dict) else {}
    for key in ["adaptive_per_fold", "research_oracle_fold_selection", "selection_uses_current_fold_metrics"]:
        if is_true(source_flags.get(key)):
            blockers.append(f"source flag {key}=true")
    summary = result_summary(results)
    if summary["target_pass_folds"] < summary["rows"]:
        blockers.append("source MT5 results have target failures")
    if summary["dd_pass_folds"] < summary["rows"]:
        blockers.append("source MT5 results have DD failures")
    if summary["loss_folds"] > 0:
        blockers.append("source MT5 results have loss folds")
    if summary["loaded_signal_max_abs_gap"] > 0:
        blockers.append("source MT5 loaded signal gap is nonzero")
    report = {
        "manifest": str(out_dir / "manifest.json"),
        "live_allowed": len(blockers) == 0,
        "blockers": blockers,
        "warnings": ["online_signal_stream: MT5 WF validates selector family, live bars are paper/canary gated"],
        "manifest_flags": {
            "live_protocol": bool(manifest.get("live_protocol")),
            "online_signal_stream": bool(manifest.get("online_signal_stream")),
            "adaptive_per_fold": bool(manifest.get("adaptive_per_fold")),
            "research_oracle_fold_selection": bool(manifest.get("research_oracle_fold_selection")),
            "selection_uses_current_fold_metrics": bool(manifest.get("selection_uses_current_fold_metrics")),
            "source_manifest_flags": source_flags,
        },
        "results": summary,
    }
    write_json(out_dir / "live_preflight_report.json", report)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one online signal from a clean MT5 feedback selector.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--selector-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/target1200_live_rolling_v3_current_selected")
    parser.add_argument("--bridge-url", default="http://localhost:5601")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--target-bar-utc", default="")
    parser.add_argument("--now", default="")
    parser.add_argument("--paper-risk-pct", type=float, default=1.0)
    parser.add_argument("--paper-max-exposure-pct", type=float, default=1.0)
    parser.add_argument("--broker-gmt", type=int, default=0)
    parser.add_argument("--include-broker-hours", default=None)
    parser.add_argument("--exclude-broker-hours", default=None)
    parser.add_argument("--side", choices=["all", "buy", "sell"], default="all")
    parser.add_argument("--direction-mode", choices=["trade_side", "reverse_trade_side", "buy", "sell"], default="trade_side")
    parser.add_argument("--min-atr-percentile", type=float, default=None)
    parser.add_argument("--max-atr-percentile", type=float, default=None)
    parser.add_argument("--exclude-news-blackout", action="store_true")
    parser.add_argument("--candidate-stride", type=int, default=1)
    parser.add_argument("--entry-delay-bars", type=int, default=1)
    parser.add_argument("--model-type", choices=["hgb", "extra_trees"], default="hgb")
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.05)
    parser.add_argument("--positive-weight", type=float, default=1.5)
    parser.add_argument("--recency-half-life-days", type=float, default=0.0)
    parser.add_argument("--min-train-rows", type=int, default=1000)
    args = parser.parse_args()

    now_utc = utc_ts(args.now) if args.now else pd.Timestamp.now(tz="UTC")
    target_bar = utc_ts(args.target_bar_utc) if args.target_bar_utc else latest_closed_bar_from_bridge(args.bridge_url, args.symbol)
    selector_manifest = read_json(args.selector_manifest)
    fold = current_fold(selector_manifest.get("folds", []) or [], target_bar)
    if is_true(fold.get("selected_candidate_adaptive_per_fold")):
        raise RuntimeError("selected candidate is adaptive_per_fold")
    if is_true(fold.get("selected_candidate_research_oracle_fold_selection")):
        raise RuntimeError("selected candidate is research_oracle")
    if is_true(fold.get("selected_candidate_selection_uses_current_fold_metrics")):
        raise RuntimeError("selected candidate uses current fold metrics")

    candidate_dir = resolve_path(fold["selected_candidate_dir"], args.selector_manifest)
    candidate_manifest_path = candidate_dir / "manifest.json"
    candidate_manifest = read_json(candidate_manifest_path)
    candidate_fold = current_fold(candidate_manifest.get("folds", []) or [], target_bar)

    tp_rr = float(candidate_fold["selected_tp_rr"])
    sl_mult = float(candidate_fold["selected_sl_mult"])
    horizon_bars = int(candidate_fold["selected_horizon_bars"])
    base_min_probability = float(candidate_fold.get("selected_min_probability", 0.5))
    feedback_threshold = float(candidate_fold.get("feedback_model_min_probability", 0.5))
    feedback_path = resolve_path(candidate_manifest["feedback"], candidate_manifest_path)
    training_features_path = resolve_path(candidate_manifest["features"], candidate_manifest_path)
    feedback_features = load_combined_feedback_features(training_features_path, args.features)

    base = base_candidate_for_bar(
        features_path=args.features,
        fold=fold,
        target_bar=target_bar,
        tp_rr=tp_rr,
        sl_mult=sl_mult,
        horizon_bars=horizon_bars,
        min_probability=base_min_probability,
        args=args,
    )
    signal_row: dict[str, Any] | None = None
    feedback: dict[str, Any] = {"status": "not_scored"}
    should_trade = False
    reason = str(base.get("reason", ""))
    if base.get("status") == "candidate":
        feedback = feedback_score_signal(
            signal_row=dict(base["signal_row"]),
            feedback_path=feedback_path,
            feedback_features=feedback_features,
            fold_id=int(fold["fold"]),
            model_type=str(candidate_manifest.get("model_type") or "extra_trees"),
            min_train_rows=int(candidate_fold.get("feedback_model_min_train_rows", 800) or 800),
        )
        if feedback.get("status") == "candidate" and float(feedback["probability"]) >= feedback_threshold:
            signal_row = dict(base["signal_row"])
            signal_row["probability"] = f"{float(feedback['probability']):.5f}"
            should_trade = True
            reason = "base and MT5 feedback model pass thresholds"
        else:
            reason = feedback.get("reason") or "feedback model probability below threshold"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    signal_path = args.out_dir / f"fold_{int(fold['fold']):02d}_signals.csv"
    signal_count = write_signal_csv(signal_path, signal_row)
    decision = {
        "generated_at_utc": iso(pd.Timestamp.now(tz="UTC")),
        "target_bar_utc": iso(target_bar),
        "now_utc": iso(now_utc),
        "fold": int(fold["fold"]),
        "online_signal_stream": True,
        "selector_manifest": str(args.selector_manifest),
        "selected_candidate_dir": str(candidate_dir),
        "signal_csv": str(signal_path),
        "should_trade": should_trade,
        "reason": reason,
        "base_model": {key: value for key, value in base.items() if key != "signal_row"},
        "feedback_model": feedback,
        "thresholds": {
            "base_min_probability": base_min_probability,
            "feedback_min_probability": feedback_threshold,
        },
        "signal_written": bool(signal_count),
    }
    decision_path = args.out_dir / "online_decision.json"
    write_json(decision_path, decision)

    manifest = {
        "all_signals": str(signal_path),
        "folds": [
            {
                "fold": int(fold["fold"]),
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "signals": int(signal_count),
                "csv": str(signal_path),
                "risk_pct": float(args.paper_risk_pct),
                "max_risk_pct": float(args.paper_risk_pct),
                "max_exposure_pct": float(args.paper_max_exposure_pct),
                "max_positions": 1,
                "validated_risk_pct": float(fold.get("risk_pct", 0.0)),
                "validated_max_exposure_pct": float(fold.get("max_exposure_pct", 0.0)),
                "selected_tp_rr": tp_rr,
                "selected_sl_mult": sl_mult,
                "selected_horizon_bars": horizon_bars,
                "selected_min_probability": base_min_probability,
                "feedback_model_min_probability": feedback_threshold,
                "selection_mode": "online_prior_mt5_feedback_model",
                "selection_uses_current_fold_metrics": False,
                "selected_candidate_dir": str(candidate_dir),
                "selected_candidate_adaptive_per_fold": False,
                "selected_candidate_research_oracle_fold_selection": False,
                "selected_candidate_selection_uses_current_fold_metrics": False,
            }
        ],
        "total_signals": int(signal_count),
        "live_protocol": True,
        "current_campaign": True,
        "online_signal_stream": True,
        "selection_uses_current_fold_metrics": False,
        "research_oracle_fold_selection": False,
        "adaptive_per_fold": False,
        "source_policy": "online scoring with clean rolling selector and prior MT5 trade feedback only",
        "source_manifest_flags": {
            "adaptive_per_fold": False,
            "research_oracle_fold_selection": False,
            "selection_uses_current_fold_metrics": False,
            "live_protocol": bool(selector_manifest.get("live_protocol")),
        },
        "source_validation": {
            "selector_manifest": str(args.selector_manifest),
            "source_manifest": str(args.selector_manifest),
            "source_results": str(args.selector_results),
            "selected_candidate_manifest": str(candidate_manifest_path),
        },
        "online_decision": str(decision_path),
        "last_online_bar_utc": iso(target_bar),
    }
    write_json(args.out_dir / "manifest.json", manifest)
    write_online_preflight(args.out_dir, manifest, args.selector_results)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
