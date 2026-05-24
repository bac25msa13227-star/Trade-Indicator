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

from scripts.export_rolling_v3_current_campaign import (  # noqa: E402
    _apply_recipe_overrides,
    _generate_missing_folds,
    _latest_validated_recipe,
    _read_json,
    _utc,
)
from scripts.train_target1200_signal_universe import (  # noqa: E402
    available_feature_columns,
    first_hit_outcomes,
    fit_predict_oos,
    load_feature_frame,
    parse_hours,
)


MT5_COLUMNS = ["open_time", "direction", "entry_price", "sl_price", "tp_price", "atr", "probability"]


def _iso(ts: pd.Timestamp | datetime) -> str:
    if isinstance(ts, datetime):
        dt = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    return ts.tz_convert("UTC").isoformat() if ts.tzinfo else ts.tz_localize("UTC").isoformat()


def _bridge_get_json(bridge_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
    query = ""
    if params:
        query = "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(bridge_url.rstrip("/") + path + query, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _latest_closed_bar_from_bridge(bridge_url: str, symbol: str) -> pd.Timestamp:
    rates = _bridge_get_json(bridge_url, "/rates", {"symbol": symbol, "timeframe": "M5", "bars": 6})
    if not isinstance(rates, list) or len(rates) < 2:
        raise RuntimeError("bridge returned fewer than 2 M5 rates")
    times = sorted(int(row["time"]) for row in rates)
    return pd.Timestamp(datetime.fromtimestamp(times[-2], tz=timezone.utc))


def _current_fold(folds: list[dict[str, Any]], now_utc: pd.Timestamp) -> dict[str, Any]:
    for fold in folds:
        if _utc(fold["test_start"]) <= now_utc < _utc(fold["test_end"]):
            return dict(fold)
    return dict(folds[-1])


def _empty_signal_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MT5_COLUMNS)
        writer.writeheader()


def _write_signal_csv(path: Path, row: dict[str, Any] | None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MT5_COLUMNS)
        writer.writeheader()
        if row is not None:
            writer.writerow({key: row[key] for key in MT5_COLUMNS})
            return 1
    return 0


def _explicit_recipe_from_args(args: argparse.Namespace, source_fold: int) -> dict[str, Any]:
    required = {
        "tp_rr": args.tp_rr,
        "sl_mult": args.sl_mult,
        "horizon_bars": args.horizon_bars,
        "min_probability": args.min_probability,
        "top_k_per_fold": args.top_k,
        "risk_pct": args.risk_pct,
        "max_risk_pct": args.max_risk_pct,
        "max_exposure_pct": args.max_exposure_pct,
        "max_positions": args.max_positions,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(f"source manifest has no selected_* recipe and explicit recipe is incomplete: {missing}")
    return {
        "source_fold": int(source_fold),
        "tp_rr": float(args.tp_rr),
        "sl_mult": float(args.sl_mult),
        "horizon_bars": int(args.horizon_bars),
        "min_probability": float(args.min_probability),
        "top_k_per_fold": int(args.top_k),
        "risk_pct": float(args.risk_pct),
        "max_risk_pct": float(args.max_risk_pct),
        "max_exposure_pct": float(args.max_exposure_pct),
        "max_positions": int(args.max_positions),
        "mt5_validated_final_balance": None,
        "mt5_validated_max_dd_pct": None,
        "explicit_recipe_not_source_selected": True,
    }


def _write_manifest(
    out_dir: Path,
    base_manifest_path: Path,
    source_manifest_path: Path,
    source_manifest: dict[str, Any],
    source_results_path: Path,
    fold: dict[str, Any],
    recipe: dict[str, Any],
    signal_csv: Path,
    signal_count: int,
    decision_path: Path,
    target_bar: pd.Timestamp,
) -> dict[str, Any]:
    manifest = {
        "all_signals": str(signal_csv),
        "folds": [
            {
                **fold,
                "signals": int(signal_count),
                "csv": str(signal_csv),
                "risk_pct": float(recipe["risk_pct"]),
                "max_risk_pct": float(recipe["max_risk_pct"]),
                "max_exposure_pct": float(recipe["max_exposure_pct"]),
                "max_positions": int(recipe["max_positions"]),
                "selected_tp_rr": float(recipe["tp_rr"]),
                "selected_sl_mult": float(recipe["sl_mult"]),
                "selected_horizon_bars": int(recipe["horizon_bars"]),
                "selected_min_probability": float(recipe["min_probability"]),
                "selected_top_k_per_fold": int(recipe["top_k_per_fold"]),
                "selected_candidate_adaptive_per_fold": bool(source_manifest.get("adaptive_per_fold")),
                "selected_candidate_research_oracle_fold_selection": bool(
                    source_manifest.get("research_oracle_fold_selection")
                ),
                "selected_candidate_selection_uses_current_fold_metrics": bool(
                    source_manifest.get("selection_uses_current_fold_metrics")
                ),
            }
        ],
        "total_signals": int(signal_count),
        "live_protocol": True,
        "current_campaign": True,
        "online_signal_stream": True,
        "selection_uses_current_fold_metrics": False,
        "research_oracle_fold_selection": False,
        "adaptive_per_fold": False,
        "source_policy": "online per-M5 scoring using MT5-validated rolling_v3 recipe; no current-fold outcome metrics",
        "source_recipe": recipe,
        "source_manifest_flags": {
            "adaptive_per_fold": bool(source_manifest.get("adaptive_per_fold")),
            "research_oracle_fold_selection": bool(source_manifest.get("research_oracle_fold_selection")),
            "selection_uses_current_fold_metrics": bool(source_manifest.get("selection_uses_current_fold_metrics")),
            "live_protocol": bool(source_manifest.get("live_protocol")),
        },
        "source_validation": {
            "base_manifest": str(base_manifest_path),
            "source_manifest": str(source_manifest_path),
            "source_results": str(source_results_path),
        },
        "online_decision": str(decision_path),
        "last_online_bar_utc": _iso(target_bar),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _write_online_preflight(out_dir: Path, manifest: dict[str, Any], results_path: Path) -> None:
    results = {}
    try:
        df = pd.read_csv(results_path)
        for col in ["final_balance", "max_dd_pct", "loaded_signals", "signals", "trades"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        results = {
            "results_path": str(results_path),
            "rows": int(len(df)),
            "target_pass_folds": int((df["final_balance"] >= 1200.0).sum()) if "final_balance" in df else 0,
            "dd_pass_folds": int((df["max_dd_pct"] >= -20.0).sum()) if "max_dd_pct" in df else 0,
            "loss_folds": int((df["final_balance"] < 200.0).sum()) if "final_balance" in df else 0,
            "min_final_balance": float(df["final_balance"].min()) if "final_balance" in df and len(df) else None,
            "worst_dd_pct": float(df["max_dd_pct"].min()) if "max_dd_pct" in df and len(df) else None,
            "total_trades": int(df["trades"].fillna(0).sum()) if "trades" in df else None,
        }
    except Exception as exc:  # noqa: BLE001
        results = {"results_path": str(results_path), "error": str(exc)}

    blockers: list[str] = []
    warnings = ["online_signal_stream: MT5 WF validates source recipe, not each dynamic live bar"]
    source_flags = manifest.get("source_manifest_flags") or {}
    if bool(source_flags.get("adaptive_per_fold")):
        blockers.append("source manifest is adaptive_per_fold; this is research/hindsight, not live protocol")
    if bool(source_flags.get("research_oracle_fold_selection")):
        blockers.append("source manifest is research_oracle_fold_selection")
    if bool(source_flags.get("selection_uses_current_fold_metrics")):
        blockers.append("source manifest selection uses current fold metrics")

    report = {
        "manifest": str(out_dir / "manifest.json"),
        "live_allowed": len(blockers) == 0,
        "blockers": blockers,
        "warnings": warnings,
        "manifest_flags": {
            "live_protocol": bool(manifest.get("live_protocol")),
            "online_signal_stream": bool(manifest.get("online_signal_stream")),
            "adaptive_per_fold": bool(manifest.get("adaptive_per_fold")),
            "research_oracle_fold_selection": bool(manifest.get("research_oracle_fold_selection")),
            "selection_uses_current_fold_metrics": bool(manifest.get("selection_uses_current_fold_metrics")),
            "source_manifest_flags": source_flags,
        },
        "results": results,
    }
    (out_dir / "live_preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one fresh rolling_v3 online decision/signal for latest closed M5 bar.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/target1200_live_rolling_v3_current_selected")
    parser.add_argument("--base-manifest", type=Path, default=ROOT / "outputs/target1200_live_rolling_v3/manifest.json")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "outputs/target1200_mt5full_ict_adaptive_combo_mt5v2/manifest.json")
    parser.add_argument("--source-results", type=Path, default=ROOT / "outputs/target1200_live_rolling_v3_xauusdm_risk_sweep/f30_r60/mt5_wf_results.csv")
    parser.add_argument("--bridge-url", default="http://localhost:5601")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--target-bar-utc", default="")
    parser.add_argument("--now", default="")
    parser.add_argument("--source-fold", type=int, default=None)
    parser.add_argument("--tp-rr", type=float, default=2.5)
    parser.add_argument("--sl-mult", type=float, default=1.0)
    parser.add_argument("--horizon-bars", type=int, default=12)
    parser.add_argument("--min-probability", type=float, default=0.50)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--risk-pct", type=float, default=6.0)
    parser.add_argument("--max-risk-pct", type=float, default=6.0)
    parser.add_argument("--max-exposure-pct", type=float, default=6.0)
    parser.add_argument("--max-positions", type=int, default=1)
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

    now_utc = _utc(args.now) if args.now else pd.Timestamp.now(tz="UTC")
    target_bar = _utc(args.target_bar_utc) if args.target_bar_utc else _latest_closed_bar_from_bridge(args.bridge_url, args.symbol)

    base_manifest = _read_json(args.base_manifest)
    source_manifest = _read_json(args.source_manifest)
    base_folds = sorted(base_manifest.get("folds", []) or [], key=lambda row: int(row["fold"]))
    current_folds = _generate_missing_folds(base_folds, now_utc)
    if not current_folds:
        raise RuntimeError("base manifest already covers now; no current folds generated")
    fold = _current_fold(current_folds, target_bar)
    max_base_fold = max(int(row["fold"]) for row in base_folds)
    try:
        recipe = _latest_validated_recipe(source_manifest, max_fold=max_base_fold, source_fold=args.source_fold)
    except ValueError:
        recipe = _explicit_recipe_from_args(args, source_fold=args.source_fold or max_base_fold)
    recipe = _apply_recipe_overrides(recipe, args)

    feature_cols = available_feature_columns(args.features)
    frame = load_feature_frame(
        features_path=args.features,
        feature_cols=feature_cols,
        folds=current_folds,
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

    rr_all, _close_idx_all = first_hit_outcomes(
        open_=frame["open"].to_numpy(np.float64),
        high=frame["high"].to_numpy(np.float64),
        low=frame["low"].to_numpy(np.float64),
        close=frame["close"].to_numpy(np.float64),
        atr=frame["atr"].to_numpy(np.float64),
        direction=frame["direction"].to_numpy(np.int8),
        tp_rr=float(recipe["tp_rr"]),
        sl_atr_mult=float(recipe["sl_mult"]),
        horizon_bars=int(recipe["horizon_bars"]),
        entry_delay_bars=int(args.entry_delay_bars),
    )
    scored = fit_predict_oos(
        frame=frame,
        folds=current_folds,
        feature_cols=feature_cols,
        labels=rr_all,
        horizon_bars=int(recipe["horizon_bars"]),
        args=SimpleNamespace(**vars(args)),
    )

    target_ns = target_bar.value
    scored_time = pd.to_datetime(scored["time"], utc=True).astype("int64")
    exact = scored[scored_time == target_ns].copy()
    decision_path = args.out_dir / "online_decision.json"
    signal_path = args.out_dir / f"fold_{int(fold['fold']):02d}_signals.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    decision: dict[str, Any] = {
        "generated_at_utc": _iso(pd.Timestamp.now(tz="UTC")),
        "target_bar_utc": _iso(target_bar),
        "now_utc": _iso(now_utc),
        "fold": int(fold["fold"]),
        "threshold": float(recipe["min_probability"]),
        "should_trade": False,
        "reason": "",
        "signal_csv": str(signal_path),
        "features": str(args.features),
        "online_signal_stream": True,
    }
    signal_row: dict[str, Any] | None = None
    if exact.empty:
        decision["reason"] = "target bar not present after feature/candidate filters"
    else:
        row = exact.sort_values("probability", ascending=False).iloc[0]
        probability = float(row["probability"])
        direction = int(row["direction"])
        entry_price = float(row["close"])
        atr = float(row["atr"])
        sl_dist = atr * float(recipe["sl_mult"])
        is_buy = direction == 1
        sl_price = entry_price - sl_dist if is_buy else entry_price + sl_dist
        tp_price = entry_price + sl_dist * float(recipe["tp_rr"]) if is_buy else entry_price - sl_dist * float(recipe["tp_rr"])
        should_trade = (
            probability >= float(recipe["min_probability"])
            and direction in {-1, 1}
            and np.isfinite(entry_price)
            and np.isfinite(atr)
            and atr > 0
        )
        decision.update(
            {
                "probability": probability,
                "side": "buy" if is_buy else "sell",
                "direction": direction,
                "entry_price": entry_price,
                "atr": atr,
                "sl_price": float(sl_price),
                "tp_price": float(tp_price),
                "trade_side_feature": str(row.get("trade_side", "")),
                "model_side": str(row.get("model_side", "")),
                "should_trade": bool(should_trade),
                "reason": "pass threshold" if should_trade else "probability below threshold",
            }
        )
        if should_trade:
            signal_row = {
                "open_time": (target_bar + pd.Timedelta(hours=int(args.broker_gmt))).strftime("%Y.%m.%d %H:%M"),
                "direction": direction,
                "entry_price": f"{entry_price:.5f}",
                "sl_price": f"{sl_price:.5f}",
                "tp_price": f"{tp_price:.5f}",
                "atr": f"{atr:.5f}",
                "probability": f"{probability:.5f}",
            }

    signal_count = _write_signal_csv(signal_path, signal_row)
    decision["signal_written"] = bool(signal_count)
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    manifest = _write_manifest(
        out_dir=args.out_dir,
        base_manifest_path=args.base_manifest,
        source_manifest_path=args.source_manifest,
        source_manifest=source_manifest,
        source_results_path=args.source_results,
        fold=fold,
        recipe=recipe,
        signal_csv=signal_path,
        signal_count=signal_count,
        decision_path=decision_path,
        target_bar=target_bar,
    )
    _write_online_preflight(args.out_dir, manifest, args.source_results)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
