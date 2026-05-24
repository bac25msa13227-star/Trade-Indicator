from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
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


def parse_time(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%Y.%m.%d %H:%M", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], errors="coerce")
    return parsed


def numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def boolish(frame: pd.DataFrame, column: str, default: bool = True) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    raw = frame[column]
    if raw.dtype == bool:
        return raw.fillna(default).astype(bool)
    text = raw.astype(str).str.strip().str.lower()
    return text.isin(["1", "true", "yes", "y", "matched"])


def parse_hours(text: str | None) -> set[int] | None:
    if not text:
        return None
    hours = {int(part.strip()) for part in text.split(",") if part.strip()}
    bad = sorted(hour for hour in hours if hour < 0 or hour > 23)
    if bad:
        raise ValueError(f"hours must be 0..23, got {bad}")
    return hours


def parse_feature_conditions(text: str | None) -> list[tuple[str, str, float]]:
    if not text:
        return []
    conditions: list[tuple[str, str, float]] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        for op in [">=", "<=", "==", "!=", ">", "<"]:
            if op in item:
                left, right = item.split(op, 1)
                conditions.append((left.strip(), op, float(right.strip())))
                break
        else:
            raise ValueError(f"Unsupported feature condition: {item}")
    return conditions


def apply_feature_conditions(frame: pd.DataFrame, conditions: list[tuple[str, str, float]]) -> pd.DataFrame:
    if not conditions or frame.empty:
        return frame
    mask = pd.Series(True, index=frame.index)
    for column, op, value in conditions:
        if column not in frame.columns:
            mask &= False
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if op == ">=":
            mask &= series >= value
        elif op == "<=":
            mask &= series <= value
        elif op == "==":
            mask &= series == value
        elif op == "!=":
            mask &= series != value
        elif op == ">":
            mask &= series > value
        elif op == "<":
            mask &= series < value
    return frame[mask].copy()


def load_features(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    base = ["time", "open", "high", "low", "close", "atr", "trade_side", "atr_percentile", "news_is_blackout"]
    cols = [c for c in base + FEATURE_COLUMNS if c in header]
    frame = pd.read_csv(path, usecols=cols)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"]).sort_values("time").drop_duplicates("time", keep="last")
    return frame.set_index("time")


def add_realized_r(feedback: pd.DataFrame, contract_value_per_lot: float, clip_min: float, clip_max: float) -> pd.DataFrame:
    out = feedback.copy()
    profit = numeric(out, "profit", 0.0).fillna(0.0)
    entry = numeric(out, "entry_price")
    sl = numeric(out, "sl_price")
    volume = numeric(out, "volume")
    rr = numeric(out, "rr")
    risk_usd = (entry - sl).abs() * volume * float(contract_value_per_lot)
    realized_r = profit / risk_usd.where(risk_usd > 0.0)

    fallback = pd.Series(np.nan, index=out.index, dtype=float)
    outcome = out.get("outcome", pd.Series("", index=out.index)).astype(str).str.lower()
    fallback.loc[outcome.str.contains("tp", na=False)] = rr.where(rr > 0.0, 1.0)
    fallback.loc[outcome.str.contains("sl", na=False)] = -1.0
    fallback.loc[fallback.isna() & (profit > 0.0)] = 1.0
    fallback.loc[fallback.isna() & (profit < 0.0)] = -1.0
    fallback.loc[fallback.isna()] = 0.0

    realized_r = realized_r.where(realized_r.notna(), fallback)
    realized_r = realized_r.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(float(clip_min), float(clip_max))
    out["realized_r"] = realized_r.astype(float)
    out["risk_usd_est"] = risk_usd.replace([np.inf, -np.inf], np.nan)
    return out


def add_stressed_profit(
    feedback: pd.DataFrame,
    *,
    point_size: float,
    contract_value_per_lot: float,
    extra_roundtrip_points: float,
    thin_hours: set[int] | None,
    thin_hour_extra_points: float,
    friday_cutoff_hour: int,
    friday_extra_points: float,
    commission_per_lot_roundtrip: float,
) -> pd.DataFrame:
    out = feedback.copy()
    if float(extra_roundtrip_points) == 0.0 and float(thin_hour_extra_points) == 0.0 and float(friday_extra_points) == 0.0 and float(commission_per_lot_roundtrip) == 0.0:
        out["profit_raw"] = numeric(out, "profit", 0.0).fillna(0.0)
        out["profit_extra_cost"] = 0.0
        return out

    hour = numeric(out, "hour", -1).fillna(-1).astype(int)
    weekday = numeric(out, "weekday", -1).fillna(-1).astype(int)
    volume = numeric(out, "volume", 0.0).fillna(0.0)
    points = pd.Series(float(extra_roundtrip_points), index=out.index, dtype=float)
    if thin_hours is not None and float(thin_hour_extra_points) != 0.0:
        points += hour.isin(thin_hours).astype(float) * float(thin_hour_extra_points)
    if float(friday_extra_points) != 0.0:
        points += ((weekday == 4) & (hour >= int(friday_cutoff_hour))).astype(float) * float(friday_extra_points)
    extra_cost = volume * float(contract_value_per_lot) * float(point_size) * points
    if float(commission_per_lot_roundtrip) != 0.0:
        extra_cost += volume * float(commission_per_lot_roundtrip)
    profit_raw = numeric(out, "profit", 0.0).fillna(0.0)
    out["profit_raw"] = profit_raw
    out["profit_extra_cost"] = extra_cost.astype(float)
    out["profit"] = profit_raw - extra_cost
    return out


def add_join_time_from_signal(frame: pd.DataFrame, column: str = "open_time") -> pd.DataFrame:
    out = frame.copy()
    out["join_time"] = parse_time(out[column])
    return out


def apply_base_direction_mode(signals: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = signals.copy()
    if out.empty or "direction" not in out.columns or mode == "source":
        return out
    direction = numeric(out, "direction").fillna(0.0)
    if mode == "reverse":
        out["direction"] = np.where(direction >= 0, -1, 1)
    elif mode == "buy":
        out["direction"] = 1
    elif mode == "sell":
        out["direction"] = -1
    else:
        raise ValueError(f"Unsupported base direction mode: {mode}")
    if {"entry_price", "sl_price", "tp_price"}.issubset(out.columns):
        entry = numeric(out, "entry_price")
        sl = numeric(out, "sl_price")
        tp = numeric(out, "tp_price")
        sl_dist = (entry - sl).abs()
        tp_dist = (tp - entry).abs()
        is_buy = numeric(out, "direction") > 0
        out["sl_price"] = np.where(is_buy, entry - sl_dist, entry + sl_dist)
        out["tp_price"] = np.where(is_buy, entry + tp_dist, entry - tp_dist)
    return out


def apply_signal_time_filters(
    signals: pd.DataFrame,
    include_hours: set[int] | None,
    exclude_hours: set[int] | None,
    exclude_news_blackout: bool,
    min_signal_probability: float | None = None,
    max_signal_probability: float | None = None,
    min_signal_atr: float | None = None,
    max_signal_atr: float | None = None,
) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    out = add_join_time_from_signal(signals)
    hour = out["join_time"].dt.hour
    mask = pd.Series(True, index=out.index)
    if include_hours is not None:
        mask &= hour.isin(include_hours)
    if exclude_hours is not None:
        mask &= ~hour.isin(exclude_hours)
    if exclude_news_blackout:
        minute = out["join_time"].dt.minute
        mask &= ~((hour == 12) & (minute >= 25) & (minute <= 35))
        mask &= ~((hour == 13) & (minute >= 25) & (minute <= 35))
    if "probability" in out.columns:
        probability = numeric(out, "probability")
        if min_signal_probability is not None:
            mask &= probability >= float(min_signal_probability)
        if max_signal_probability is not None:
            mask &= probability <= float(max_signal_probability)
    if "atr" in out.columns:
        atr = numeric(out, "atr")
        if min_signal_atr is not None:
            mask &= atr >= float(min_signal_atr)
        if max_signal_atr is not None:
            mask &= atr <= float(max_signal_atr)
    return out.loc[mask].copy()


def build_model_frame(feedback: pd.DataFrame, features: pd.DataFrame, positive_r_threshold: float) -> pd.DataFrame:
    fb = feedback[boolish(feedback, "signal_matched", True)].copy()
    if "signal_open_time" not in fb.columns:
        raise ValueError("feedback CSV must include signal_open_time")
    fb["join_time"] = parse_time(fb["signal_open_time"])
    fb = fb.dropna(subset=["join_time", "realized_r"])
    joined = fb.join(features, on="join_time", how="inner", rsuffix="_feature")
    joined["label_positive_r"] = (pd.to_numeric(joined["realized_r"], errors="coerce").fillna(0.0) > float(positive_r_threshold)).astype(int)
    joined["signal_probability"] = numeric(joined, "signal_probability")
    joined["signal_atr"] = numeric(joined, "signal_atr")
    joined["direction"] = numeric(joined, "direction")
    joined["hour"] = numeric(joined, "hour")
    joined["weekday"] = numeric(joined, "weekday")
    joined["profit"] = numeric(joined, "profit", 0.0)
    return joined


def feature_columns(frame: pd.DataFrame) -> list[str]:
    extras = ["signal_probability", "signal_atr", "direction", "hour", "weekday"]
    return [c for c in FEATURE_COLUMNS + extras if c in frame.columns]


def make_models(kind: str, seed: int) -> tuple[Pipeline, Pipeline]:
    if kind == "extra_trees":
        reg = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
        clf = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed + 1000,
            n_jobs=-1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", reg)]), Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", clf)]
        )
    reg = HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.035,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=seed,
    )
    clf = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.035,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=seed + 1000,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", reg)]), Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", clf)]
    )


def score_signals(signals: pd.DataFrame, features: pd.DataFrame, reg: Pipeline, clf: Pipeline, cols: list[str]) -> pd.DataFrame:
    sig = add_join_time_from_signal(signals)
    sig["signal_probability"] = numeric(sig, "probability")
    sig["signal_atr"] = numeric(sig, "atr")
    sig["direction"] = numeric(sig, "direction")
    sig["hour"] = sig["join_time"].dt.hour
    sig["weekday"] = sig["join_time"].dt.weekday
    joined = sig.join(features, on="join_time", how="inner", rsuffix="_feature")
    if joined.empty:
        return joined
    x = joined[cols]
    joined["profitr_expected_r"] = reg.predict(x)
    joined["profitr_positive_probability"] = clf.predict_proba(x)[:, 1]
    joined["profitr_score"] = joined["profitr_expected_r"] + 0.45 * (joined["profitr_positive_probability"] - 0.5)
    return joined


def window_dd_pct(profits: list[float], start_balance: float) -> float:
    balance = float(start_balance)
    peak = balance
    worst = 0.0
    for profit in profits:
        balance += float(profit)
        peak = max(peak, balance)
        if peak > 0:
            worst = min(worst, (balance - peak) / peak * 100.0)
    return worst


def build_sequence_frame(
    feedback: pd.DataFrame,
    features: pd.DataFrame,
    *,
    deposit: float,
    window_trades: int,
    bad_window_dd_pct: float,
    min_future_r_label: float,
) -> pd.DataFrame:
    if feedback.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for _, group in feedback.groupby("fold", sort=True):
        ordered = group.sort_values(["close_time", "open_time"], na_position="last").reset_index(drop=True).copy()
        future_sum_r: list[float] = []
        future_dd: list[float] = []
        for idx, _row in ordered.iterrows():
            future = ordered.iloc[idx : idx + int(window_trades)]
            profits = pd.to_numeric(future["profit"], errors="coerce").fillna(0.0).tolist()
            rs = pd.to_numeric(future["realized_r"], errors="coerce").fillna(0.0).tolist()
            future_sum_r.append(float(np.sum(rs)))
            future_dd.append(window_dd_pct(profits, float(deposit)))
        ordered["future_sum_r"] = future_sum_r
        ordered["future_window_dd_pct"] = future_dd
        rows.append(ordered)
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if frame.empty or "signal_open_time" not in frame.columns:
        return pd.DataFrame()
    frame["join_time"] = parse_time(frame["signal_open_time"])
    joined = frame.join(features, on="join_time", how="inner", rsuffix="_feature")
    joined["signal_probability"] = numeric(joined, "signal_probability")
    joined["signal_atr"] = numeric(joined, "signal_atr")
    joined["direction"] = numeric(joined, "direction")
    joined["hour"] = joined["join_time"].dt.hour
    joined["weekday"] = joined["join_time"].dt.weekday
    joined["label_sequence_bad"] = (
        (pd.to_numeric(joined["future_sum_r"], errors="coerce").fillna(0.0) < float(min_future_r_label))
        | (pd.to_numeric(joined["future_window_dd_pct"], errors="coerce").fillna(0.0) < -float(bad_window_dd_pct))
    ).astype(int)
    return joined


def apply_sequence_penalty(scored: pd.DataFrame, train: pd.DataFrame, cols: list[str], args: argparse.Namespace, seed: int) -> pd.DataFrame:
    if scored.empty or float(args.sequence_penalty_weight) <= 0.0:
        return scored
    out = scored.copy()
    out["seq_bad_probability"] = 0.0
    out["seq_expected_future_r"] = 0.0
    out["profitr_score_raw"] = out["profitr_score"]
    if len(train) < int(args.sequence_min_train_rows) or train["label_sequence_bad"].nunique() != 2 or not cols:
        out["profitr_score"] = out["profitr_score_raw"]
        return out
    reg = ExtraTreesRegressor(
        n_estimators=int(args.sequence_n_estimators),
        min_samples_leaf=int(args.sequence_min_samples_leaf),
        max_features="sqrt",
        random_state=seed + 9000,
        n_jobs=-1,
    )
    clf = ExtraTreesClassifier(
        n_estimators=int(args.sequence_n_estimators),
        min_samples_leaf=int(args.sequence_min_samples_leaf),
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed + 10000,
        n_jobs=-1,
    )
    reg.fit(train[cols], train["future_sum_r"])
    clf.fit(train[cols], train["label_sequence_bad"])
    out["seq_expected_future_r"] = reg.predict(out[cols])
    out["seq_bad_probability"] = clf.predict_proba(out[cols])[:, 1]
    out["profitr_score"] = (
        out["profitr_score_raw"]
        + float(args.sequence_future_r_weight) * out["seq_expected_future_r"]
        - float(args.sequence_penalty_weight) * out["seq_bad_probability"]
    )
    return out


def build_feature_bar_candidates(
    features: pd.DataFrame,
    fold: dict[str, Any],
    direction_mode: str,
    tp_rr: float,
    sl_atr_mult: float,
    entry_price_source: str,
    candidate_stride: int,
    include_hours: set[int] | None,
    exclude_hours: set[int] | None,
    min_atr_percentile: float | None,
    max_atr_percentile: float | None,
    exclude_news_blackout: bool,
    feature_conditions: list[tuple[str, str, float]] | None = None,
) -> pd.DataFrame:
    frame = features.reset_index().copy()
    start = pd.Timestamp(str(fold["test_start"]))
    end = pd.Timestamp(str(fold["test_end"]))
    frame = frame[(frame["time"] >= start) & (frame["time"] < end)].copy()
    frame = frame.dropna(subset=["time", entry_price_source, "atr"]).copy()
    frame = frame[pd.to_numeric(frame["atr"], errors="coerce") > 0.0].copy()
    if include_hours is not None:
        frame = frame[frame["time"].dt.hour.isin(include_hours)].copy()
    if exclude_hours is not None:
        frame = frame[~frame["time"].dt.hour.isin(exclude_hours)].copy()
    if min_atr_percentile is not None and "atr_percentile" in frame.columns:
        frame = frame[pd.to_numeric(frame["atr_percentile"], errors="coerce") >= float(min_atr_percentile)].copy()
    if max_atr_percentile is not None and "atr_percentile" in frame.columns:
        frame = frame[pd.to_numeric(frame["atr_percentile"], errors="coerce") <= float(max_atr_percentile)].copy()
    if exclude_news_blackout and "news_is_blackout" in frame.columns:
        frame = frame[pd.to_numeric(frame["news_is_blackout"], errors="coerce").fillna(0.0) <= 0.0].copy()
    frame = apply_feature_conditions(frame, feature_conditions or [])
    if int(candidate_stride) > 1:
        frame = frame.iloc[:: int(candidate_stride)].copy()
    if frame.empty:
        return pd.DataFrame(columns=MT5_COLUMNS)

    base_direction = frame.get("trade_side", pd.Series("buy", index=frame.index)).map({"buy": 1, "sell": -1}).fillna(1).astype(int)
    if direction_mode == "trade_side":
        frame["direction"] = base_direction
    elif direction_mode == "reverse_trade_side":
        frame["direction"] = -base_direction
    elif direction_mode == "buy":
        frame["direction"] = 1
    elif direction_mode == "sell":
        frame["direction"] = -1
    elif direction_mode == "both":
        buy = frame.copy()
        sell = frame.copy()
        buy["direction"] = 1
        sell["direction"] = -1
        frame = pd.concat([buy, sell], ignore_index=True)
    else:
        raise ValueError(f"Unsupported direction_mode: {direction_mode}")

    entry = pd.to_numeric(frame[entry_price_source], errors="coerce")
    atr = pd.to_numeric(frame["atr"], errors="coerce")
    sl_dist = atr * float(sl_atr_mult)
    direction = pd.to_numeric(frame["direction"], errors="coerce").fillna(1).astype(int)
    is_buy = direction == 1
    out = pd.DataFrame(
        {
            "open_time": pd.to_datetime(frame["time"]).dt.strftime("%Y.%m.%d %H:%M"),
            "direction": direction,
            "entry_price": entry,
            "sl_price": np.where(is_buy, entry - sl_dist, entry + sl_dist),
            "tp_price": np.where(is_buy, entry + sl_dist * float(tp_rr), entry - sl_dist * float(tp_rr)),
            "atr": atr,
            "probability": 0.5,
        }
    )
    return out.dropna(subset=["entry_price", "sl_price", "tp_price", "atr"]).copy()


def select_rows(
    scored: pd.DataFrame,
    min_score: float,
    top_k: int,
    min_keep: int,
    dedupe_open_time: bool,
    max_per_day: int,
    min_spacing_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scored.empty:
        return scored, scored
    ranked = scored.sort_values(["profitr_score", "profitr_expected_r", "profitr_positive_probability"], ascending=False).copy()
    if dedupe_open_time:
        ranked = ranked.drop_duplicates("join_time", keep="first").copy()
    filtered = ranked[ranked["profitr_score"] >= float(min_score)].copy()
    if len(filtered) < int(min_keep):
        filtered = ranked.head(int(min_keep)).copy()
    if int(top_k) > 0 and len(filtered) > int(top_k):
        filtered = filtered.head(int(top_k)).copy()
    filtered = apply_path_spacing(filtered, max_per_day=max_per_day, min_spacing_minutes=min_spacing_minutes)
    filtered = filtered.sort_values("join_time").copy()
    export = filtered[MT5_COLUMNS].copy()
    export["probability"] = filtered["profitr_positive_probability"].clip(0.0, 1.0).astype(float)
    audit_cols = [
        c
        for c in MT5_COLUMNS
        + [
            "profitr_expected_r",
            "profitr_positive_probability",
            "profitr_score_raw",
            "seq_expected_future_r",
            "seq_bad_probability",
            "profitr_score",
        ]
        if c in filtered.columns
    ]
    return export, filtered[audit_cols].copy()


def apply_path_spacing(frame: pd.DataFrame, max_per_day: int, min_spacing_minutes: int) -> pd.DataFrame:
    if frame.empty or (int(max_per_day) <= 0 and int(min_spacing_minutes) <= 0):
        return frame.copy()
    work = frame.copy()
    if "join_time" not in work.columns:
        work = add_join_time_from_signal(work)
    selected: list[int] = []
    per_day: dict[str, int] = {}
    selected_times: list[pd.Timestamp] = []
    for idx, row in work.sort_values(
        ["profitr_score", "profitr_expected_r", "profitr_positive_probability"],
        ascending=False,
    ).iterrows():
        ts = row.get("join_time")
        if pd.isna(ts):
            continue
        ts = pd.Timestamp(ts)
        day = ts.strftime("%Y-%m-%d")
        if int(max_per_day) > 0 and per_day.get(day, 0) >= int(max_per_day):
            continue
        if int(min_spacing_minutes) > 0:
            too_close = any(abs((ts - other).total_seconds()) < int(min_spacing_minutes) * 60 for other in selected_times)
            if too_close:
                continue
        selected.append(idx)
        selected_times.append(ts)
        per_day[day] = per_day.get(day, 0) + 1
    return work.loc[selected].copy()


def fold_summary(export: pd.DataFrame, audit: pd.DataFrame) -> dict[str, Any]:
    if export.empty or audit.empty:
        return {
            "signals": int(len(export)),
            "selected_min_score": None,
            "selected_mean_score": None,
            "selected_mean_expected_r": None,
            "selected_mean_positive_probability": None,
        }
    return {
        "signals": int(len(export)),
        "selected_min_score": float(audit["profitr_score"].min()),
        "selected_mean_score": float(audit["profitr_score"].mean()),
        "selected_mean_expected_r": float(audit["profitr_expected_r"].mean()),
        "selected_mean_positive_probability": float(audit["profitr_positive_probability"].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export rolling signal universe selected by prior MT5 trade feedback realized profit/R.")
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate-source", choices=["base_manifest", "feature_bars", "hybrid_base_feature_bars"], default="base_manifest")
    parser.add_argument("--model-type", choices=["extra_trees", "hgb"], default="extra_trees")
    parser.add_argument("--min-train-rows", type=int, default=800)
    parser.add_argument("--positive-r-threshold", type=float, default=0.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--min-keep", type=int, default=120)
    parser.add_argument("--direction-mode", choices=["trade_side", "reverse_trade_side", "buy", "sell", "both"], default="trade_side")
    parser.add_argument("--base-direction-mode", choices=["source", "reverse", "buy", "sell"], default="source")
    parser.add_argument("--tp-rr", type=float, default=2.0)
    parser.add_argument("--sl-atr-mult", type=float, default=0.6)
    parser.add_argument("--horizon-bars", type=int, default=0)
    parser.add_argument("--entry-price-source", choices=["close", "open"], default="close")
    parser.add_argument("--candidate-stride", type=int, default=1)
    parser.add_argument("--include-hours", default=None)
    parser.add_argument("--exclude-hours", default=None)
    parser.add_argument("--min-atr-percentile", type=float, default=None)
    parser.add_argument("--max-atr-percentile", type=float, default=None)
    parser.add_argument("--min-signal-probability", type=float, default=None)
    parser.add_argument("--max-signal-probability", type=float, default=None)
    parser.add_argument("--min-signal-atr", type=float, default=None)
    parser.add_argument("--max-signal-atr", type=float, default=None)
    parser.add_argument("--exclude-news-blackout", action="store_true")
    parser.add_argument("--feature-conditions", default=None, help="Comma conditions on feature columns, e.g. h4_market_structure_bias==1")
    parser.add_argument("--dedupe-open-time", action="store_true")
    parser.add_argument("--max-per-day", type=int, default=0)
    parser.add_argument("--min-spacing-minutes", type=int, default=0)
    parser.add_argument("--cold-start-mode", choices=["no_trade", "source_signals"], default="no_trade")
    parser.add_argument("--risk-pct", type=float, default=None)
    parser.add_argument("--risk-multiplier", type=float, default=1.0)
    parser.add_argument("--max-risk-pct", type=float, default=4.0)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--contract-value-per-lot", type=float, default=100.0)
    parser.add_argument("--clip-r-min", type=float, default=-1.5)
    parser.add_argument("--clip-r-max", type=float, default=4.0)
    parser.add_argument("--extra-roundtrip-points", type=float, default=0.0)
    parser.add_argument("--thin-hours", default="0,1,2,3,4,5,6,22,23")
    parser.add_argument("--thin-hour-extra-points", type=float, default=0.0)
    parser.add_argument("--friday-cutoff-hour", type=int, default=20)
    parser.add_argument("--friday-extra-points", type=float, default=0.0)
    parser.add_argument("--point-size", type=float, default=0.001)
    parser.add_argument("--commission-per-lot-roundtrip", type=float, default=0.0)
    parser.add_argument("--sequence-penalty-weight", type=float, default=0.0)
    parser.add_argument("--sequence-future-r-weight", type=float, default=0.0)
    parser.add_argument("--sequence-window-trades", type=int, default=10)
    parser.add_argument("--sequence-bad-window-dd-pct", type=float, default=18.0)
    parser.add_argument("--sequence-min-future-r-label", type=float, default=-1.0)
    parser.add_argument("--sequence-min-train-rows", type=int, default=600)
    parser.add_argument("--sequence-n-estimators", type=int, default=300)
    parser.add_argument("--sequence-min-samples-leaf", type=int, default=12)
    args = parser.parse_args()
    if args.candidate_stride < 1:
        raise ValueError("--candidate-stride must be >= 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.base_manifest
    manifest = read_json(manifest_path)
    feedback = pd.read_csv(args.feedback)
    feedback["fold"] = pd.to_numeric(feedback["fold"], errors="coerce")
    thin_hours = parse_hours(args.thin_hours)
    feedback = add_stressed_profit(
        feedback,
        point_size=float(args.point_size),
        contract_value_per_lot=float(args.contract_value_per_lot),
        extra_roundtrip_points=float(args.extra_roundtrip_points),
        thin_hours=thin_hours,
        thin_hour_extra_points=float(args.thin_hour_extra_points),
        friday_cutoff_hour=int(args.friday_cutoff_hour),
        friday_extra_points=float(args.friday_extra_points),
        commission_per_lot_roundtrip=float(args.commission_per_lot_roundtrip),
    )
    feedback = add_realized_r(feedback, args.contract_value_per_lot, args.clip_r_min, args.clip_r_max)
    features = load_features(args.features)
    sequence_frame = build_sequence_frame(
        feedback,
        features,
        deposit=200.0,
        window_trades=int(args.sequence_window_trades),
        bad_window_dd_pct=float(args.sequence_bad_window_dd_pct),
        min_future_r_label=float(args.sequence_min_future_r_label),
    ) if float(args.sequence_penalty_weight) > 0.0 or float(args.sequence_future_r_weight) != 0.0 else pd.DataFrame()
    include_hours = parse_hours(args.include_hours)
    exclude_hours = parse_hours(args.exclude_hours)
    feature_conditions = parse_feature_conditions(args.feature_conditions)

    folds: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    total = 0
    for source_fold in manifest.get("folds", []) or []:
        fold_id = int(source_fold["fold"])
        hist = feedback[feedback["fold"] < fold_id].copy()
        train = build_model_frame(hist, features, args.positive_r_threshold)
        source_csv = resolve_csv(manifest_path, str(source_fold["csv"]))
        source_signals = apply_base_direction_mode(pd.read_csv(source_csv), str(args.base_direction_mode))
        source_signals = apply_signal_time_filters(
            source_signals,
            include_hours=include_hours,
            exclude_hours=exclude_hours,
            exclude_news_blackout=bool(args.exclude_news_blackout),
            min_signal_probability=args.min_signal_probability,
            max_signal_probability=args.max_signal_probability,
            min_signal_atr=args.min_signal_atr,
            max_signal_atr=args.max_signal_atr,
        )
        if args.candidate_source == "base_manifest":
            signals = source_signals
        elif args.candidate_source == "feature_bars":
            signals = build_feature_bar_candidates(
                features=features,
                fold=source_fold,
                direction_mode=args.direction_mode,
                tp_rr=args.tp_rr,
                sl_atr_mult=args.sl_atr_mult,
                entry_price_source=args.entry_price_source,
                candidate_stride=args.candidate_stride,
                include_hours=include_hours,
                exclude_hours=exclude_hours,
                min_atr_percentile=args.min_atr_percentile,
                max_atr_percentile=args.max_atr_percentile,
                exclude_news_blackout=bool(args.exclude_news_blackout),
                feature_conditions=feature_conditions,
            )
        else:
            feature_signals = build_feature_bar_candidates(
                features=features,
                fold=source_fold,
                direction_mode=args.direction_mode,
                tp_rr=args.tp_rr,
                sl_atr_mult=args.sl_atr_mult,
                entry_price_source=args.entry_price_source,
                candidate_stride=args.candidate_stride,
                include_hours=include_hours,
                exclude_hours=exclude_hours,
                min_atr_percentile=args.min_atr_percentile,
                max_atr_percentile=args.max_atr_percentile,
                exclude_news_blackout=bool(args.exclude_news_blackout),
                feature_conditions=feature_conditions,
            )
            signals = pd.concat([source_signals, feature_signals], ignore_index=True)
            if not signals.empty:
                signals = signals.drop_duplicates(["open_time", "direction"], keep="first").copy()
        export = pd.DataFrame(columns=MT5_COLUMNS)
        audit = pd.DataFrame()
        mode = "no_trade_insufficient_profitr_feedback"
        train_rows = int(len(train))
        train_mean_r = float(train["realized_r"].mean()) if train_rows else float("nan")
        train_positive_rate = float(train["label_positive_r"].mean()) if train_rows else float("nan")
        if train_rows >= int(args.min_train_rows) and train["label_positive_r"].nunique() == 2 and len(feature_columns(train)) > 0:
            cols = feature_columns(train)
            reg, clf = make_models(args.model_type, seed=fold_id)
            reg.fit(train[cols], train["realized_r"])
            clf.fit(train[cols], train["label_positive_r"])
            scored = score_signals(signals, features, reg, clf, cols)
            if not sequence_frame.empty:
                seq_train = sequence_frame[sequence_frame["fold"] < fold_id].copy()
                seq_cols = feature_columns(seq_train)
                scored = apply_sequence_penalty(scored, seq_train, seq_cols, args, seed=fold_id)
            scored = apply_feature_conditions(scored, feature_conditions)
            export, audit = select_rows(
                scored,
                args.min_score,
                args.top_k,
                args.min_keep,
                bool(args.dedupe_open_time),
                int(args.max_per_day),
                int(args.min_spacing_minutes),
            )
            mode = "prior_mt5_profitr_model"
        elif args.cold_start_mode == "source_signals":
            export = source_signals[[c for c in MT5_COLUMNS if c in source_signals.columns]].copy()
            if feature_conditions:
                export = add_join_time_from_signal(export)
                export = export.join(features, on="join_time", how="inner", rsuffix="_feature")
                export = apply_feature_conditions(export, feature_conditions)
                export = export[[c for c in MT5_COLUMNS if c in export.columns]].copy()
            for col in MT5_COLUMNS:
                if col not in export.columns:
                    export[col] = np.nan
            export = export[MT5_COLUMNS].copy()
            if int(args.max_per_day) > 0 or int(args.min_spacing_minutes) > 0:
                export = add_join_time_from_signal(export)
                export["profitr_score"] = pd.to_numeric(export.get("probability", 0.5), errors="coerce").fillna(0.5)
                export["profitr_expected_r"] = 0.0
                export["profitr_positive_probability"] = export["profitr_score"]
                export = apply_path_spacing(
                    export,
                    max_per_day=int(args.max_per_day),
                    min_spacing_minutes=int(args.min_spacing_minutes),
                )
                export = export[MT5_COLUMNS].copy()
            mode = "cold_start_source_signals"
        target_csv = args.out_dir / f"fold_{fold_id:02d}_signals.csv"
        target_audit_csv = args.out_dir / f"fold_{fold_id:02d}_score_audit.csv"
        export.to_csv(target_csv, index=False, float_format="%.5f")
        audit.to_csv(target_audit_csv, index=False, float_format="%.5f")

        if args.risk_pct is not None:
            risk_pct = min(float(args.risk_pct), float(args.max_risk_pct))
        else:
            risk_pct = min(float(source_fold.get("risk_pct", 0.0)) * float(args.risk_multiplier), float(args.max_risk_pct))
        max_positions = int(args.max_positions if args.max_positions is not None else source_fold.get("max_positions", 1))
        fold = dict(source_fold)
        for key in list(fold):
            if key.startswith("selected_candidate_"):
                fold.pop(key, None)
        summary = fold_summary(export, audit)
        fold.update(
            {
                "signals": int(len(export)),
                "csv": str(target_csv),
                "score_audit_csv": str(target_audit_csv),
                "risk_pct": round(risk_pct, 4),
                "max_risk_pct": round(risk_pct, 4),
                "max_exposure_pct": round(risk_pct * max_positions, 4),
                "max_positions": max_positions,
                "selection_mode": mode,
                "selection_uses_current_fold_metrics": False,
                "research_oracle_fold_selection": False,
                "mt5_profitr_model_selector": True,
                "profitr_model_train_rows": train_rows,
                "profitr_model_train_mean_r": train_mean_r,
                "profitr_model_train_positive_rate": train_positive_rate,
                "profitr_model_min_score": float(args.min_score),
                "profitr_model_top_k": int(args.top_k),
                "profitr_model_min_keep": int(args.min_keep),
                "profitr_candidate_source": str(args.candidate_source),
                "profitr_direction_mode": str(args.direction_mode),
                "profitr_base_direction_mode": str(args.base_direction_mode),
                "profitr_tp_rr": float(args.tp_rr),
                "profitr_sl_atr_mult": float(args.sl_atr_mult),
                "selected_horizon_bars": int(args.horizon_bars) if hasattr(args, "horizon_bars") else int(source_fold.get("selected_horizon_bars", 0) or 0),
                "profitr_entry_price_source": str(args.entry_price_source),
                "profitr_candidate_stride": int(args.candidate_stride),
                "profitr_feature_conditions": str(args.feature_conditions or ""),
                "profitr_dedupe_open_time": bool(args.dedupe_open_time),
                "profitr_max_per_day": int(args.max_per_day),
                "profitr_min_spacing_minutes": int(args.min_spacing_minutes),
                "profitr_min_signal_probability": args.min_signal_probability,
                "profitr_max_signal_probability": args.max_signal_probability,
                "profitr_min_signal_atr": args.min_signal_atr,
                "profitr_max_signal_atr": args.max_signal_atr,
                "profitr_cold_start_mode": str(args.cold_start_mode),
                "sequence_penalty_weight": float(args.sequence_penalty_weight),
                "sequence_future_r_weight": float(args.sequence_future_r_weight),
                **summary,
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
                "train_mean_r": train_mean_r,
                "train_positive_rate": train_positive_rate,
                "risk_pct": risk_pct,
                **summary,
            }
        )

    out_manifest = {
        "all_signals": None,
        "folds": folds,
        "total_signals": int(total),
        "live_protocol": True,
        "mt5_profitr_model_selector": True,
        "mt5_feedback_model_selector": True,
        "selection_uses_current_fold_metrics": False,
        "research_oracle_fold_selection": False,
        "base_manifest": str(args.base_manifest),
        "feedback": str(args.feedback),
        "features": str(args.features),
        "model_type": args.model_type,
        "candidate_source": str(args.candidate_source),
        "feature_conditions": str(args.feature_conditions or ""),
        "direction_mode": str(args.direction_mode),
        "base_direction_mode": str(args.base_direction_mode),
        "tp_rr": float(args.tp_rr),
        "sl_atr_mult": float(args.sl_atr_mult),
        "entry_price_source": str(args.entry_price_source),
        "candidate_stride": int(args.candidate_stride),
        "dedupe_open_time": bool(args.dedupe_open_time),
        "max_per_day": int(args.max_per_day),
        "min_spacing_minutes": int(args.min_spacing_minutes),
        "min_signal_probability": args.min_signal_probability,
        "max_signal_probability": args.max_signal_probability,
        "min_signal_atr": args.min_signal_atr,
        "max_signal_atr": args.max_signal_atr,
        "cold_start_mode": str(args.cold_start_mode),
        "target_basis": "prior_mt5_trade_feedback_realized_profit_r",
        "contract_value_per_lot": float(args.contract_value_per_lot),
        "clip_r_min": float(args.clip_r_min),
        "clip_r_max": float(args.clip_r_max),
        "cost_aware_training": {
            "extra_roundtrip_points": float(args.extra_roundtrip_points),
            "thin_hours": sorted(thin_hours) if thin_hours is not None else None,
            "thin_hour_extra_points": float(args.thin_hour_extra_points),
            "friday_cutoff_hour": int(args.friday_cutoff_hour),
            "friday_extra_points": float(args.friday_extra_points),
            "point_size": float(args.point_size),
            "commission_per_lot_roundtrip": float(args.commission_per_lot_roundtrip),
        },
        "sequence_hybrid_score": {
            "penalty_weight": float(args.sequence_penalty_weight),
            "future_r_weight": float(args.sequence_future_r_weight),
            "window_trades": int(args.sequence_window_trades),
            "bad_window_dd_pct": float(args.sequence_bad_window_dd_pct),
            "min_future_r_label": float(args.sequence_min_future_r_label),
            "min_train_rows": int(args.sequence_min_train_rows),
            "n_estimators": int(args.sequence_n_estimators),
            "min_samples_leaf": int(args.sequence_min_samples_leaf),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    pd.DataFrame(rules).to_csv(args.out_dir / "profitr_model_rules.csv", index=False)
    print(json.dumps({"manifest": str(args.out_dir / "manifest.json"), "folds": len(folds), "signals": total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
