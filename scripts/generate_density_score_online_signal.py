from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_mt5_profitr_model_manifest import (  # noqa: E402
    MT5_COLUMNS,
    add_realized_r,
    add_stressed_profit,
    build_feature_bar_candidates,
    build_model_frame,
    feature_columns,
    load_features,
    make_models,
    parse_hours,
    score_signals,
)
from scripts.mt5_prior_sequence_bucket_throttle_sweep import (  # noqa: E402
    build_stats as build_bucket_stats,
    enrich_buckets,
    lookup_stats as lookup_bucket_stats,
    row_with_buckets,
    state_features,
)


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except (UnicodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not parse JSON: {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")


def resolve_path(raw: Any, base: Path | None = None) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    candidates = [ROOT / path]
    if base is not None:
        candidates.extend([base.parent / path.name, base.parent / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / path).resolve()


def utc_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


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


def current_fold(folds: list[dict[str, Any]], bar_time: pd.Timestamp) -> dict[str, Any]:
    ts = utc_ts(bar_time)
    for fold in folds:
        if utc_ts(fold["test_start"]) <= ts < utc_ts(fold["test_end"]):
            return fold
    raise RuntimeError(f"no manifest fold covers {ts.isoformat()}")


def write_signal_csv(path: Path, row: dict[str, Any] | None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MT5_COLUMNS)
        writer.writeheader()
        if row is None:
            return 0
        writer.writerow({column: row[column] for column in MT5_COLUMNS})
        return 1


def combine_features(training_path: Path, live_path: Path) -> pd.DataFrame:
    training = load_features(training_path)
    live = load_features(live_path)
    combined = pd.concat([training, live]).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def make_live_fold(source_fold: dict[str, Any], target_bar: pd.Timestamp) -> dict[str, Any]:
    target = utc_ts(target_bar).tz_localize(None)
    return {
        **source_fold,
        "test_start": target.strftime("%Y-%m-%d %H:%M:%S"),
        "test_end": (target + pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
    }


def choose_signal(
    scored: pd.DataFrame,
    *,
    min_score: float,
    min_probability: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if scored.empty:
        return None, {"decision": "idle", "reason": "no scored signal for latest closed M5 bar"}
    ranked = scored.sort_values(["profitr_score", "profitr_expected_r", "profitr_positive_probability"], ascending=False)
    row = ranked.iloc[0]
    score = float(row["profitr_score"])
    probability = float(row["profitr_positive_probability"])
    decision = {
        "decision": "trade" if score >= float(min_score) and probability >= float(min_probability) else "idle",
        "reason": "score pass" if score >= float(min_score) and probability >= float(min_probability) else "score/probability below frozen threshold",
        "profitr_score": score,
        "profitr_expected_r": float(row["profitr_expected_r"]),
        "profitr_positive_probability": probability,
        "min_score": float(min_score),
        "min_probability": float(min_probability),
        "direction": int(row["direction"]),
        "entry_price": float(row["entry_price"]),
        "sl_price": float(row["sl_price"]),
        "tp_price": float(row["tp_price"]),
        "atr": float(row["atr"]),
    }
    if decision["decision"] != "trade":
        return None, decision
    signal = {column: row[column] for column in MT5_COLUMNS}
    signal["probability"] = probability
    return signal, decision


def apply_live_spread_gate(
    *,
    signal_row: dict[str, Any] | None,
    decision: dict[str, Any],
    bridge_url: str,
    symbol: str,
    max_spread_points: float,
    max_spread_to_sl_r: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if signal_row is None:
        return None, {"enabled": False, "reason": "no signal row"}
    try:
        tick = bridge_get_json(bridge_url, "/tick", {"symbol": symbol})
        symbol_info = bridge_get_json(bridge_url, "/symbol/info", {"symbol": symbol})
    except Exception as exc:  # noqa: BLE001
        decision.update({"decision": "idle", "reason": f"live spread gate failed: {exc}"})
        return None, {"enabled": True, "action": "skip", "reason": "tick_or_symbol_fetch_failed", "error": str(exc)}
    point = float(symbol_info.get("point") or symbol_info.get("trade_tick_size") or 0.0)
    bid = float(tick.get("bid") or 0.0)
    ask = float(tick.get("ask") or 0.0)
    spread_price = ask - bid
    spread_points = spread_price / point if point > 0 and ask > 0 and bid > 0 else float("inf")
    entry = float(signal_row["entry_price"])
    sl = float(signal_row["sl_price"])
    sl_distance = abs(entry - sl)
    spread_to_sl_r = spread_price / sl_distance if sl_distance > 0 else float("inf")
    gate = {
        "enabled": True,
        "action": "pass",
        "tick": tick,
        "symbol_info": {
            "symbol": symbol_info.get("symbol") or symbol_info.get("name"),
            "point": point,
            "digits": symbol_info.get("digits"),
        },
        "spread_price": spread_price,
        "spread_points": spread_points,
        "max_spread_points": float(max_spread_points),
        "spread_to_sl_r": spread_to_sl_r,
        "max_spread_to_sl_r": float(max_spread_to_sl_r),
        "sl_distance_price": sl_distance,
    }
    if float(max_spread_points) > 0 and spread_points > float(max_spread_points):
        decision.update({"decision": "idle", "reason": "live broker spread exceeds cap"})
        gate.update({"action": "skip", "reason": "spread_points_exceeds_cap"})
        return None, gate
    if float(max_spread_to_sl_r) > 0 and spread_to_sl_r > float(max_spread_to_sl_r):
        decision.update({"decision": "idle", "reason": "live broker spread too large vs SL"})
        gate.update({"action": "skip", "reason": "spread_to_sl_r_exceeds_cap"})
        return None, gate
    return signal_row, gate


def epoch_seconds(value: Any) -> float:
    if not value:
        return 0.0
    try:
        return float(pd.Timestamp(value).timestamp())
    except Exception:
        return 0.0


def load_live_sequence_state(
    *,
    bridge_url: str,
    symbol: str,
    magic: int,
    state_path: Path | None,
    initial_balance: float,
    risk_pct: float,
) -> dict[str, Any]:
    state = read_json(state_path) if state_path and state_path.exists() else {}
    account = bridge_get_json(bridge_url, "/account")
    equity = float(account.get("equity") or account.get("balance") or initial_balance)
    peak = max(float(state.get("peak_equity") or 0.0), equity, float(initial_balance))
    since = epoch_seconds(state.get("created_at")) or (pd.Timestamp.utcnow() - pd.Timedelta(days=14)).timestamp()
    try:
        deals = bridge_get_json(bridge_url, "/history/closed", {"symbol": symbol, "since": since, "magic": int(magic)})
    except Exception:
        deals = []
    deals = sorted(deals if isinstance(deals, list) else [], key=lambda item: float(item.get("close_time") or 0.0))
    risk_unit = max(float(initial_balance) * max(float(risk_pct), 0.1) / 100.0, 1.0)
    recent_r = [float(item.get("profit") or 0.0) / risk_unit for item in deals[-20:]]
    loss_streak = 0
    win_streak = 0
    for value in recent_r:
        if value < 0:
            loss_streak += 1
            win_streak = 0
        elif value > 0:
            win_streak += 1
            loss_streak = 0
    features = state_features(
        balance=equity,
        peak=peak,
        recent_r=recent_r,
        loss_streak=loss_streak,
        win_streak=win_streak,
        trade_index=len(deals) + 1,
    )
    return {
        **features,
        "account": account,
        "closed_deals": len(deals),
        "risk_unit": risk_unit,
        "peak_equity": peak,
    }


def apply_prior_sequence_bucket_gate(
    *,
    signal_row: dict[str, Any] | None,
    decision: dict[str, Any],
    feedback: pd.DataFrame,
    fold_id: int,
    policy_report: Path | None,
    bridge_url: str,
    symbol: str,
    magic: int,
    state_path: Path | None,
    initial_balance: float,
    base_risk_pct: float,
) -> tuple[dict[str, Any] | None, dict[str, Any], float]:
    if signal_row is None or not policy_report or not policy_report.exists():
        return signal_row, {"enabled": False, "reason": "no bucket throttle policy"}, 1.0
    report = read_json(policy_report)
    best = (report.get("best") or [{}])[0]
    if not best:
        return signal_row, {"enabled": False, "reason": "empty bucket throttle policy"}, 1.0

    train_rows = build_model_frame(
        add_realized_r(feedback[feedback["fold"] < fold_id].copy(), 100.0, -3.0, 5.0),
        combine_features(Path(decision["training_features"]), Path(decision["live_features"])),
        0.0,
    ) if False else None

    from scripts.mt5_prior_sequence_regime_throttle_sweep import build_training_rows, add_realized_r as seq_add_realized_r

    prior = feedback[feedback["fold"] < fold_id].copy()
    prior = seq_add_realized_r(prior, 100.0)
    training_rows = build_training_rows(prior, initial_balance)
    training_rows, atr_edges = enrich_buckets(training_rows, report.get("atr_edges"))
    stats = build_bucket_stats(training_rows, int(best.get("min_count") or 40))
    live_state = load_live_sequence_state(
        bridge_url=bridge_url,
        symbol=symbol,
        magic=magic,
        state_path=state_path,
        initial_balance=initial_balance,
        risk_pct=base_risk_pct,
    )
    if float(live_state["pre_dd_pct"]) <= -float(best.get("pre_dd_hard_skip_pct") or 20.0):
        decision.update({"decision": "idle", "reason": "prior sequence hard DD gate"})
        return None, {"enabled": True, "action": "skip", "reason": "pre_dd_hard_skip", "live_state": live_state, "policy": best}, 0.0

    row = pd.Series(
        {
            "hour": utc_ts(decision["latest_closed_bar_utc"]).hour,
            "weekday": utc_ts(decision["latest_closed_bar_utc"]).weekday(),
            "direction": int(signal_row["direction"]),
            "signal_probability": float(signal_row["probability"]),
            "signal_atr": float(signal_row["atr"]),
            "rr": abs(float(signal_row["tp_price"]) - float(signal_row["entry_price"]))
            / max(abs(float(signal_row["entry_price"]) - float(signal_row["sl_price"])), 1e-9),
        }
    )
    enriched = row_with_buckets(row, live_state, atr_edges)
    stat = lookup_bucket_stats(enriched, stats)
    gate = {"enabled": True, "action": "pass", "policy": best, "live_state": live_state, "bucket": enriched, "bucket_stat": stat}
    if stat is not None and (
        float(stat["mean_r"]) < float(best.get("mean_r_skip") or -0.2)
        or float(stat["positive_rate"]) < float(best.get("pos_rate_skip") or 0.35)
    ):
        decision.update({"decision": "idle", "reason": "prior sequence bucket skip"})
        gate.update({"action": "skip", "reason": "bucket_bad_expectancy"})
        return None, gate, 0.0
    if stat is not None and float(stat["mean_r"]) < float(best.get("reduce_mean_r") or 0.2):
        mult = float(best.get("reduce_mult") or 0.5)
        gate.update({"action": "reduce", "risk_multiplier": mult, "reason": "bucket_low_expectancy"})
        decision["prior_sequence_risk_multiplier"] = mult
        return signal_row, gate, mult
    if stat is not None and (
        float(stat["mean_r"]) >= float(best.get("boost_mean_r") or 999.0)
        and float(stat["positive_rate"]) >= float(best.get("boost_pos_rate") or 0.55)
        and float(live_state["pre_dd_pct"]) >= -float(best.get("boost_pre_dd_floor_pct") or 20.0)
    ):
        mult = float(best.get("boost_mult") or 1.0)
        if mult > 1.0:
            gate.update({"action": "boost", "risk_multiplier": mult, "reason": "bucket_high_expectancy"})
            decision["prior_sequence_risk_multiplier"] = mult
            return signal_row, gate, mult
    return signal_row, gate, 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate fresh M5 signal for frozen density-score ACC2 demo policy.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True, help="Fresh live feature CSV built from MT5 bridge rates.")
    parser.add_argument("--bridge-url", default="http://localhost:5601")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--magic", type=int, default=23018)
    parser.add_argument("--bridge-state", type=Path, default=None)
    parser.add_argument("--initial-balance", type=float, default=200.0)
    parser.add_argument("--sequence-throttle-report", type=Path, default=None)
    parser.add_argument("--max-spread-points", type=float, default=120.0)
    parser.add_argument("--max-spread-to-sl-r", type=float, default=0.10)
    parser.add_argument("--feedback", type=Path, default=None)
    parser.add_argument("--training-features", type=Path, default=None)
    parser.add_argument("--min-train-rows", type=int, default=600)
    parser.add_argument("--min-probability", type=float, default=0.0)
    parser.add_argument("--positive-r-threshold", type=float, default=0.0)
    parser.add_argument("--model-type", choices=["extra_trees", "hgb"], default=None)
    parser.add_argument("--contract-value-per-lot", type=float, default=100.0)
    parser.add_argument("--clip-r-min", type=float, default=-1.5)
    parser.add_argument("--clip-r-max", type=float, default=4.0)
    parser.add_argument("--extra-roundtrip-points", type=float, default=30.0)
    parser.add_argument("--thin-hours", default="0,1,2,3,4,5,6,22,23")
    parser.add_argument("--thin-hour-extra-points", type=float, default=50.0)
    parser.add_argument("--friday-cutoff-hour", type=int, default=20)
    parser.add_argument("--friday-extra-points", type=float, default=100.0)
    parser.add_argument("--point-size", type=float, default=0.001)
    parser.add_argument("--commission-per-lot-roundtrip", type=float, default=0.0)
    parser.add_argument("--include-hours", default=None)
    parser.add_argument("--exclude-hours", default=None)
    parser.add_argument("--min-atr-percentile", type=float, default=None)
    parser.add_argument("--max-atr-percentile", type=float, default=None)
    parser.add_argument("--exclude-news-blackout", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    target_bar = latest_closed_bar_from_bridge(args.bridge_url, args.symbol)
    fold = current_fold(manifest.get("folds", []), target_bar)
    fold_id = int(fold["fold"])

    training_features_path = args.training_features or resolve_path(manifest.get("features"), manifest_path)
    feedback_path = args.feedback or (out_dir / "mt5_trade_feedback_detailed.csv")
    if not feedback_path.exists():
        feedback_path = resolve_path(manifest.get("feedback"), manifest_path)
    features = combine_features(training_features_path, args.features.resolve())

    feedback = pd.read_csv(feedback_path)
    feedback["fold"] = pd.to_numeric(feedback["fold"], errors="coerce")
    feedback = add_stressed_profit(
        feedback,
        point_size=float(args.point_size),
        contract_value_per_lot=float(args.contract_value_per_lot),
        extra_roundtrip_points=float(args.extra_roundtrip_points),
        thin_hours=parse_hours(args.thin_hours),
        thin_hour_extra_points=float(args.thin_hour_extra_points),
        friday_cutoff_hour=int(args.friday_cutoff_hour),
        friday_extra_points=float(args.friday_extra_points),
        commission_per_lot_roundtrip=float(args.commission_per_lot_roundtrip),
    )
    feedback = add_realized_r(feedback, args.contract_value_per_lot, args.clip_r_min, args.clip_r_max)
    hist = feedback[feedback["fold"] < fold_id].copy()
    train = build_model_frame(hist, features, args.positive_r_threshold)

    reason = ""
    signal_row: dict[str, Any] | None = None
    decision: dict[str, Any]
    if len(train) < int(args.min_train_rows):
        decision = {
            "decision": "idle",
            "reason": "insufficient prior MT5 feedback",
            "train_rows": int(len(train)),
            "min_train_rows": int(args.min_train_rows),
        }
    elif train["label_positive_r"].nunique() < 2:
        decision = {
            "decision": "idle",
            "reason": "prior MT5 feedback has only one class",
            "train_rows": int(len(train)),
        }
    else:
        cols = feature_columns(train)
        if not cols:
            decision = {"decision": "idle", "reason": "no model feature columns available", "train_rows": int(len(train))}
        else:
            model_type = args.model_type or str(manifest.get("model_type") or "extra_trees")
            reg, clf = make_models(model_type, seed=fold_id)
            reg.fit(train[cols], train["realized_r"])
            clf.fit(train[cols], train["label_positive_r"])
            live_fold = make_live_fold(fold, target_bar)
            candidates = build_feature_bar_candidates(
                features=features,
                fold=live_fold,
                direction_mode=str(fold.get("profitr_direction_mode") or "trade_side"),
                tp_rr=float(fold.get("profitr_tp_rr") or 2.5),
                sl_atr_mult=float(fold.get("profitr_sl_atr_mult") or 1.0),
                entry_price_source=str(fold.get("profitr_entry_price_source") or "close"),
                candidate_stride=int(fold.get("profitr_candidate_stride") or 1),
                include_hours=parse_hours(args.include_hours),
                exclude_hours=parse_hours(args.exclude_hours),
                min_atr_percentile=args.min_atr_percentile,
                max_atr_percentile=args.max_atr_percentile,
                exclude_news_blackout=bool(args.exclude_news_blackout),
            )
            scored = score_signals(candidates, features, reg, clf, cols)
            signal_row, decision = choose_signal(
                scored,
                min_score=float(fold.get("profitr_model_min_score") or 0.1),
                min_probability=float(args.min_probability),
            )
            decision.update(
                {
                    "train_rows": int(len(train)),
                    "train_mean_r": float(train["realized_r"].mean()),
                    "train_positive_rate": float(train["label_positive_r"].mean()),
                    "model_type": model_type,
                    "feature_columns": len(cols),
                }
            )

    base_risk_pct = float(fold.get("frozen_policy_risk_pct") or fold.get("risk_pct") or 0.0)
    if "frozen_policy_risk_pct" not in fold:
        fold["frozen_policy_risk_pct"] = base_risk_pct
    sequence_gate: dict[str, Any] = {"enabled": False}
    risk_multiplier = 1.0
    if decision.get("decision") == "trade":
        signal_row, sequence_gate, risk_multiplier = apply_prior_sequence_bucket_gate(
            signal_row=signal_row,
            decision={
                **decision,
                "latest_closed_bar_utc": utc_ts(target_bar).isoformat(),
                "training_features": str(training_features_path),
                "live_features": str(args.features.resolve()),
            },
            feedback=feedback,
            fold_id=fold_id,
            policy_report=args.sequence_throttle_report,
            bridge_url=args.bridge_url,
            symbol=args.symbol,
            magic=int(args.magic),
            state_path=args.bridge_state,
            initial_balance=float(args.initial_balance),
            base_risk_pct=base_risk_pct,
        )
    spread_gate: dict[str, Any] = {"enabled": False}
    if decision.get("decision") == "trade":
        signal_row, spread_gate = apply_live_spread_gate(
            signal_row=signal_row,
            decision=decision,
            bridge_url=args.bridge_url,
            symbol=args.symbol,
            max_spread_points=float(args.max_spread_points),
            max_spread_to_sl_r=float(args.max_spread_to_sl_r),
        )

    effective_risk_pct = round(base_risk_pct * float(risk_multiplier), 4)
    csv_path = resolve_path(fold.get("csv"), manifest_path)
    written = write_signal_csv(csv_path, signal_row)
    for target_fold in manifest.get("folds", []):
        if int(target_fold["fold"]) == fold_id:
            target_fold["csv"] = str(csv_path)
            target_fold["signals"] = int(written)
            target_fold["frozen_policy_risk_pct"] = base_risk_pct
            target_fold["risk_pct"] = effective_risk_pct
            target_fold["max_risk_pct"] = effective_risk_pct
            target_fold["max_exposure_pct"] = effective_risk_pct * float(target_fold.get("max_positions") or 1)
            target_fold["online_signal_generator"] = "density_score_live_m5"
            target_fold["online_signal_generated_at_utc"] = pd.Timestamp.utcnow().isoformat()
            target_fold["online_signal_latest_closed_bar_utc"] = utc_ts(target_bar).isoformat()
            target_fold["online_signal_decision"] = str(decision.get("decision"))
            target_fold["online_signal_reason"] = str(decision.get("reason"))
            target_fold["prior_sequence_gate"] = sequence_gate
            target_fold["live_spread_gate"] = spread_gate
            break
    manifest["online_signal_generator"] = "density_score_live_m5"
    manifest["online_signal_source"] = "fresh_mt5_bridge_m5_bar"
    manifest["online_signal_last_bar_utc"] = utc_ts(target_bar).isoformat()
    write_json(manifest_path, manifest)

    now_utc = pd.Timestamp.utcnow().isoformat()
    decision_payload = {
        **decision,
        "latest_closed_bar_utc": utc_ts(target_bar).isoformat(),
        "fold": fold_id,
        "signal_csv": str(csv_path),
        "signals_written": int(written),
        "risk_pct": effective_risk_pct,
        "base_risk_pct": base_risk_pct,
        "risk_multiplier": float(risk_multiplier),
        "max_risk_pct": effective_risk_pct,
        "prior_sequence_gate": sequence_gate,
        "live_spread_gate": spread_gate,
        "feedback": str(feedback_path),
        "training_features": str(training_features_path),
        "live_features": str(args.features.resolve()),
        "generated_at_utc": now_utc,
        "candidate_source": "live_feature_bar_scored_by_frozen_density_profitr_model",
    }
    write_json(out_dir / "online_density_score_decision.json", decision_payload)

    guard_path = out_dir / "live_canary_guard_report.json"
    guard = read_json(guard_path) if guard_path.exists() else {}
    guard.update(
        {
            "live_start_allowed": True,
            "now_utc": now_utc,
            "current_signal": {
                "fold": fold_id,
                "csv": str(csv_path),
                "latest_closed_bar_utc": utc_ts(target_bar).isoformat(),
                "decision": decision.get("decision"),
                "reason": decision.get("reason"),
                "signals_written": int(written),
            },
            "warnings": sorted(
                set((guard.get("warnings") or []) + ["ACC2 DEMO ONLY: fresh density-score online signal, real-money live remains blocked."])
            ),
        }
    )
    write_json(guard_path, guard)

    print(
        "online_density_signal "
        f"fold={fold_id} bar={utc_ts(target_bar).isoformat()} decision={decision.get('decision')} "
        f"signals={written} reason={decision.get('reason')} csv={csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
