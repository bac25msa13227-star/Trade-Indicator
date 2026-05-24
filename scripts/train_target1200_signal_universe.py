from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is available in the trading env.
    njit = None

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from xauusd_ai.features.dataset import FEATURE_COLUMNS
except Exception:  # pragma: no cover - fallback for standalone use.
    FEATURE_COLUMNS = []


BASE_COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "atr",
    "trade_side",
    "atr_percentile",
    "volatility_regime",
    "news_is_blackout",
]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_hours(text: str | None) -> set[int] | None:
    if not text:
        return None
    hours = {int(part.strip()) for part in text.split(",") if part.strip()}
    bad = sorted(hour for hour in hours if hour < 0 or hour > 23)
    if bad:
        raise ValueError(f"broker hours must be 0..23, got {bad}")
    return hours


def load_folds(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    folds = manifest.get("folds", [])
    if not folds:
        raise ValueError(f"No folds found in {path}")
    return sorted(folds, key=lambda row: int(row["fold"]))


def available_feature_columns(features_path: Path) -> list[str]:
    header = pd.read_csv(features_path, nrows=0)
    columns = set(header.columns)
    if FEATURE_COLUMNS:
        features = [column for column in FEATURE_COLUMNS if column in columns]
    else:
        blocked = set(BASE_COLUMNS) | {
            "expected_direction",
            "tick_volume",
            "tick_volume_delta",
            "volume_imbalance",
            "returns",
            "price_momentum",
        }
        features = [column for column in header.columns if column not in blocked]
    if not features:
        raise ValueError(f"No usable feature columns found in {features_path}")
    return features


def load_feature_frame(
    features_path: Path,
    feature_cols: list[str],
    folds: list[dict[str, Any]],
    broker_gmt: int,
    include_hours: set[int] | None,
    exclude_hours: set[int] | None,
    side: str,
    direction_mode: str,
    min_atr_percentile: float | None,
    max_atr_percentile: float | None,
    exclude_news_blackout: bool,
    candidate_stride: int,
) -> pd.DataFrame:
    all_columns = set(pd.read_csv(features_path, nrows=0).columns)
    usecols = [column for column in dict.fromkeys(BASE_COLUMNS + feature_cols) if column in all_columns]
    frame = pd.read_csv(features_path, usecols=usecols)
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["time", "close", "high", "low", "atr", "trade_side"]).copy()
    frame = frame.sort_values("time").reset_index(drop=True)

    start = min(pd.Timestamp(fold["train_start"], tz="UTC") for fold in folds)
    end = max(pd.Timestamp(fold["test_end"], tz="UTC") for fold in folds)
    frame = frame[(frame["time"] >= start) & (frame["time"] < end)].copy()

    base_direction = frame["trade_side"].map({"buy": 1, "sell": -1}).fillna(0).astype(np.int8)
    if direction_mode == "trade_side":
        frame["direction"] = base_direction
    elif direction_mode == "reverse_trade_side":
        frame["direction"] = -base_direction
    elif direction_mode == "buy":
        frame["direction"] = np.int8(1)
    elif direction_mode == "sell":
        frame["direction"] = np.int8(-1)
    else:
        raise ValueError(f"Unsupported direction_mode: {direction_mode}")
    frame["model_side"] = np.where(frame["direction"].astype(int) == 1, "buy", "sell")
    frame = frame[frame["direction"] != 0].copy()
    frame = frame[frame["atr"].astype(float) > 0].copy()

    broker_time = frame["time"] + pd.Timedelta(hours=broker_gmt)
    frame["broker_hour"] = broker_time.dt.hour.astype(np.int16)
    if include_hours is not None:
        frame = frame[frame["broker_hour"].isin(include_hours)].copy()
    if exclude_hours is not None:
        frame = frame[~frame["broker_hour"].isin(exclude_hours)].copy()
    if side in {"buy", "sell"}:
        frame = frame[frame["model_side"] == side].copy()
    if min_atr_percentile is not None and "atr_percentile" in frame.columns:
        frame = frame[frame["atr_percentile"].astype(float) >= min_atr_percentile].copy()
    if max_atr_percentile is not None and "atr_percentile" in frame.columns:
        frame = frame[frame["atr_percentile"].astype(float) <= max_atr_percentile].copy()
    if exclude_news_blackout and "news_is_blackout" in frame.columns:
        frame = frame[frame["news_is_blackout"].fillna(0).astype(float) <= 0].copy()
    if candidate_stride > 1:
        frame = frame.iloc[::candidate_stride].copy()

    frame["side_num"] = frame["direction"].astype(np.float32)
    if "broker_hour_sin" not in frame.columns:
        radians = 2.0 * np.pi * frame["broker_hour"].astype(float) / 24.0
        frame["broker_hour_sin"] = np.sin(radians).astype(np.float32)
        frame["broker_hour_cos"] = np.cos(radians).astype(np.float32)
    return frame.reset_index(drop=True)


if njit is not None:

    @njit(cache=True)
    def first_hit_outcomes(
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr: np.ndarray,
        direction: np.ndarray,
        tp_rr: float,
        sl_atr_mult: float,
        horizon_bars: int,
        entry_delay_bars: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = close.shape[0]
        rr = np.zeros(n, dtype=np.float64)
        close_idx = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            side = direction[i]
            if side == 0:
                continue
            entry_idx = i + entry_delay_bars
            if entry_idx >= n:
                continue
            entry = open_[entry_idx]
            sl_dist = atr[i] * sl_atr_mult
            if sl_dist <= 0.0 or not np.isfinite(sl_dist) or not np.isfinite(entry):
                continue
            sl = entry - side * sl_dist
            tp = entry + side * sl_dist * tp_rr
            stop = entry_idx + horizon_bars + 1
            if stop > n:
                stop = n
            for j in range(entry_idx, stop):
                if side > 0:
                    hit_sl = low[j] <= sl
                    hit_tp = high[j] >= tp
                else:
                    hit_sl = high[j] >= sl
                    hit_tp = low[j] <= tp
                if hit_sl or hit_tp:
                    close_idx[i] = j
                    rr[i] = -1.0 if hit_sl else tp_rr
                    break
        return rr, close_idx


    @njit(cache=True)
    def simulate_selected(
        selected: np.ndarray,
        folds: np.ndarray,
        start_idx: np.ndarray,
        close_idx: np.ndarray,
        rr: np.ndarray,
        risk_pct: float,
        max_positions: int,
        max_dd_pct: float,
        fold_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        final_balance = np.full(fold_count, 200.0, dtype=np.float64)
        net_pct = np.zeros(fold_count, dtype=np.float64)
        max_dd = np.zeros(fold_count, dtype=np.float64)
        trades = np.zeros(fold_count, dtype=np.int64)
        wins = np.zeros(fold_count, dtype=np.int64)
        losses = np.zeros(fold_count, dtype=np.int64)

        cursor = 0
        n_selected = selected.shape[0]
        for fold_id in range(1, fold_count + 1):
            balance = 200.0
            peak = 200.0
            fold_dd = 0.0
            fold_trades = 0
            fold_wins = 0
            fold_losses = 0

            open_exit = np.empty(max_positions, dtype=np.int64)
            open_pnl = np.empty(max_positions, dtype=np.float64)
            open_count = 0

            while cursor < n_selected and folds[selected[cursor]] < fold_id:
                cursor += 1

            local_cursor = cursor
            while local_cursor < n_selected and folds[selected[local_cursor]] == fold_id:
                row = selected[local_cursor]
                current_idx = start_idx[row]

                while True:
                    min_pos = -1
                    min_exit = 9223372036854775807
                    for pos in range(open_count):
                        if open_exit[pos] <= current_idx and open_exit[pos] < min_exit:
                            min_exit = open_exit[pos]
                            min_pos = pos
                    if min_pos < 0:
                        break
                    pnl = open_pnl[min_pos]
                    balance += pnl
                    if balance > peak:
                        peak = balance
                    if peak > 0.0:
                        dd = (balance - peak) / peak * 100.0
                        if dd < fold_dd:
                            fold_dd = dd
                    if pnl > 0.0:
                        fold_wins += 1
                    elif pnl < 0.0:
                        fold_losses += 1
                    open_count -= 1
                    if min_pos < open_count:
                        open_exit[min_pos] = open_exit[open_count]
                        open_pnl[min_pos] = open_pnl[open_count]

                if open_count < max_positions and fold_dd > -max_dd_pct:
                    outcome = rr[row]
                    exit_idx = close_idx[row]
                    if outcome != 0.0 and exit_idx >= current_idx:
                        pnl = max(balance, 0.0) * risk_pct * outcome
                        open_exit[open_count] = exit_idx
                        open_pnl[open_count] = pnl
                        open_count += 1
                        fold_trades += 1
                local_cursor += 1

            while open_count > 0:
                min_pos = 0
                min_exit = open_exit[0]
                for pos in range(1, open_count):
                    if open_exit[pos] < min_exit:
                        min_exit = open_exit[pos]
                        min_pos = pos
                pnl = open_pnl[min_pos]
                balance += pnl
                if balance > peak:
                    peak = balance
                if peak > 0.0:
                    dd = (balance - peak) / peak * 100.0
                    if dd < fold_dd:
                        fold_dd = dd
                if pnl > 0.0:
                    fold_wins += 1
                elif pnl < 0.0:
                    fold_losses += 1
                open_count -= 1
                if min_pos < open_count:
                    open_exit[min_pos] = open_exit[open_count]
                    open_pnl[min_pos] = open_pnl[open_count]

            idx = fold_id - 1
            final_balance[idx] = balance
            net_pct[idx] = (balance - 200.0) / 200.0 * 100.0
            max_dd[idx] = fold_dd
            trades[idx] = fold_trades
            wins[idx] = fold_wins
            losses[idx] = fold_losses
            cursor = local_cursor

        return final_balance, net_pct, max_dd, trades, wins, losses


    @njit(cache=True)
    def simulate_selected_mt5_minlot(
        selected: np.ndarray,
        folds: np.ndarray,
        start_idx: np.ndarray,
        close_idx: np.ndarray,
        rr: np.ndarray,
        sl_dist: np.ndarray,
        risk_pct: float,
        max_positions: int,
        max_dd_pct: float,
        fold_count: int,
        min_lot_risk_per_price: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        final_balance = np.full(fold_count, 200.0, dtype=np.float64)
        net_pct = np.zeros(fold_count, dtype=np.float64)
        max_dd = np.zeros(fold_count, dtype=np.float64)
        trades = np.zeros(fold_count, dtype=np.int64)
        wins = np.zeros(fold_count, dtype=np.int64)
        losses = np.zeros(fold_count, dtype=np.int64)

        cursor = 0
        n_selected = selected.shape[0]
        for fold_id in range(1, fold_count + 1):
            balance = 200.0
            peak = 200.0
            fold_dd = 0.0
            fold_trades = 0
            fold_wins = 0
            fold_losses = 0

            open_exit = np.empty(max_positions, dtype=np.int64)
            open_pnl = np.empty(max_positions, dtype=np.float64)
            open_count = 0

            while cursor < n_selected and folds[selected[cursor]] < fold_id:
                cursor += 1

            local_cursor = cursor
            while local_cursor < n_selected and folds[selected[local_cursor]] == fold_id:
                row = selected[local_cursor]
                current_idx = start_idx[row]

                while True:
                    min_pos = -1
                    min_exit = 9223372036854775807
                    for pos in range(open_count):
                        if open_exit[pos] <= current_idx and open_exit[pos] < min_exit:
                            min_exit = open_exit[pos]
                            min_pos = pos
                    if min_pos < 0:
                        break
                    pnl = open_pnl[min_pos]
                    balance += pnl
                    if balance > peak:
                        peak = balance
                    if peak > 0.0:
                        dd = (balance - peak) / peak * 100.0
                        if dd < fold_dd:
                            fold_dd = dd
                    if pnl > 0.0:
                        fold_wins += 1
                    elif pnl < 0.0:
                        fold_losses += 1
                    open_count -= 1
                    if min_pos < open_count:
                        open_exit[min_pos] = open_exit[open_count]
                        open_pnl[min_pos] = open_pnl[open_count]

                if open_count < max_positions and fold_dd > -max_dd_pct:
                    outcome = rr[row]
                    exit_idx = close_idx[row]
                    one_step_risk = sl_dist[row] * min_lot_risk_per_price
                    max_order_risk = max(balance, 0.0) * risk_pct
                    if (
                        outcome != 0.0
                        and exit_idx >= current_idx
                        and one_step_risk > 0.0
                        and one_step_risk <= max_order_risk * 1.001
                    ):
                        lot_steps = np.floor(max_order_risk / one_step_risk)
                        if lot_steps >= 1.0:
                            pnl = lot_steps * one_step_risk * outcome
                            open_exit[open_count] = exit_idx
                            open_pnl[open_count] = pnl
                            open_count += 1
                            fold_trades += 1
                local_cursor += 1

            while open_count > 0:
                min_pos = 0
                min_exit = open_exit[0]
                for pos in range(1, open_count):
                    if open_exit[pos] < min_exit:
                        min_exit = open_exit[pos]
                        min_pos = pos
                pnl = open_pnl[min_pos]
                balance += pnl
                if balance > peak:
                    peak = balance
                if peak > 0.0:
                    dd = (balance - peak) / peak * 100.0
                    if dd < fold_dd:
                        fold_dd = dd
                if pnl > 0.0:
                    fold_wins += 1
                elif pnl < 0.0:
                    fold_losses += 1
                open_count -= 1
                if min_pos < open_count:
                    open_exit[min_pos] = open_exit[open_count]
                    open_pnl[min_pos] = open_pnl[open_count]

            idx = fold_id - 1
            final_balance[idx] = balance
            net_pct[idx] = (balance - 200.0) / 200.0 * 100.0
            max_dd[idx] = fold_dd
            trades[idx] = fold_trades
            wins[idx] = fold_wins
            losses[idx] = fold_losses
            cursor = local_cursor

        return final_balance, net_pct, max_dd, trades, wins, losses

else:

    def first_hit_outcomes(
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr: np.ndarray,
        direction: np.ndarray,
        tp_rr: float,
        sl_atr_mult: float,
        horizon_bars: int,
        entry_delay_bars: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rr = np.zeros(len(close), dtype=float)
        close_idx = np.full(len(close), -1, dtype=np.int64)
        for i in range(len(close)):
            side = int(direction[i])
            entry_idx = i + entry_delay_bars
            if entry_idx >= len(close):
                continue
            sl_dist = atr[i] * sl_atr_mult
            if side == 0 or sl_dist <= 0:
                continue
            entry = open_[entry_idx]
            sl = entry - side * sl_dist
            tp = entry + side * sl_dist * tp_rr
            for j in range(entry_idx, min(len(close), entry_idx + horizon_bars + 1)):
                hit_sl = low[j] <= sl if side > 0 else high[j] >= sl
                hit_tp = high[j] >= tp if side > 0 else low[j] <= tp
                if hit_sl or hit_tp:
                    rr[i] = -1.0 if hit_sl else tp_rr
                    close_idx[i] = j
                    break
        return rr, close_idx


def make_config_key(args: argparse.Namespace, tp_rr: float, sl_mult: float, horizon_bars: int) -> str:
    payload = {
        "features": str(args.features.resolve()),
        "fold_manifest": str(args.fold_manifest.resolve()),
        "tp_rr": tp_rr,
        "sl_mult": sl_mult,
        "horizon_bars": horizon_bars,
        "side": args.side,
        "direction_mode": args.direction_mode,
        "include_broker_hours": args.include_broker_hours,
        "exclude_broker_hours": args.exclude_broker_hours,
        "min_atr_percentile": args.min_atr_percentile,
        "max_atr_percentile": args.max_atr_percentile,
        "exclude_news_blackout": args.exclude_news_blackout,
        "candidate_stride": args.candidate_stride,
        "entry_delay_bars": args.entry_delay_bars,
        "mt5_min_lot_sizing": args.mt5_min_lot_sizing,
        "mt5_min_lot_risk_per_price": args.mt5_min_lot_risk_per_price,
        "mt5_select_tradeable_first": args.mt5_select_tradeable_first,
        "min_signal_gap_bars": args.min_signal_gap_bars,
        "model": {
            "model_type": args.model_type,
            "max_iter": args.max_iter,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "learning_rate": args.learning_rate,
            "max_leaf_nodes": args.max_leaf_nodes,
            "l2_regularization": args.l2_regularization,
            "positive_weight": args.positive_weight,
            "recency_half_life_days": args.recency_half_life_days,
        },
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def fold_timestamps(fold: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    return (
        pd.Timestamp(fold["train_start"], tz="UTC"),
        pd.Timestamp(fold["train_end"], tz="UTC"),
        pd.Timestamp(fold["test_start"], tz="UTC"),
        pd.Timestamp(fold["test_end"], tz="UTC"),
    )


def fit_predict_oos(
    frame: pd.DataFrame,
    folds: list[dict[str, Any]],
    feature_cols: list[str],
    labels: np.ndarray,
    horizon_bars: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    model_features = [column for column in feature_cols if column in frame.columns]
    for extra in ["side_num", "broker_hour_sin", "broker_hour_cos"]:
        if extra in frame.columns and extra not in model_features:
            model_features.append(extra)

    x_all = frame[model_features].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    y_all = (labels > 0.0).astype(np.int8)
    time = frame["time"]

    scored_parts: list[pd.DataFrame] = []
    purge = pd.Timedelta(minutes=5 * horizon_bars)
    for fold in folds:
        fold_id = int(fold["fold"])
        train_start, train_end, test_start, test_end = fold_timestamps(fold)
        train_cutoff = train_end - purge
        train_mask = (time >= train_start) & (time < train_cutoff)
        test_mask = (time >= test_start) & (time < test_end)

        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(test_idx) == 0 or len(train_idx) < args.min_train_rows:
            continue

        y_train = y_all[train_idx]
        if y_train.min() == y_train.max():
            proba = np.full(len(test_idx), float(y_train[0]), dtype=np.float64)
        else:
            sample_weight = np.ones(len(train_idx), dtype=np.float64)
            if args.positive_weight != 1.0:
                sample_weight[y_train == 1] *= args.positive_weight
            if args.recency_half_life_days > 0:
                age_days = (
                    (train_end - time.iloc[train_idx]).dt.total_seconds().to_numpy()
                    / 86400.0
                )
                sample_weight *= np.power(0.5, age_days / args.recency_half_life_days)
            if args.model_type == "hgb":
                model = HistGradientBoostingClassifier(
                    max_iter=args.max_iter,
                    learning_rate=args.learning_rate,
                    max_leaf_nodes=args.max_leaf_nodes,
                    l2_regularization=args.l2_regularization,
                    random_state=fold_id,
                )
                model.fit(x_all.iloc[train_idx], y_train, sample_weight=sample_weight)
            elif args.model_type == "extra_trees":
                model = ExtraTreesClassifier(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth if args.max_depth > 0 else None,
                    min_samples_leaf=args.min_samples_leaf,
                    max_features="sqrt",
                    bootstrap=False,
                    random_state=fold_id,
                    n_jobs=-1,
                )
                fit_x = x_all.iloc[train_idx].fillna(-999.0)
                model.fit(fit_x, y_train, sample_weight=sample_weight)
            else:
                raise ValueError(f"Unsupported model_type: {args.model_type}")
            pred_x = x_all.iloc[test_idx]
            if args.model_type == "extra_trees":
                pred_x = pred_x.fillna(-999.0)
            proba = model.predict_proba(pred_x)[:, 1]

        part = frame.iloc[test_idx][
            ["time", "high", "low", "close", "atr", "trade_side", "model_side", "direction", "broker_hour"]
        ].copy()
        part["fold"] = fold_id
        part["test_start"] = fold["test_start"]
        part["test_end"] = fold["test_end"]
        part["probability"] = proba
        part["label_win"] = y_all[test_idx]
        part["source_idx"] = test_idx.astype(np.int64)
        scored_parts.append(part)
        print(
            f"fold {fold_id:02d}: train={len(train_idx):5d} test={len(test_idx):5d} "
            f"pos_rate={y_train.mean():.3f} score_p95={np.quantile(proba, 0.95):.3f}"
        )

    if not scored_parts:
        raise RuntimeError("No OOS scored rows were produced")
    return pd.concat(scored_parts, ignore_index=True)


def build_selected_indices(
    folds_array: np.ndarray,
    proba: np.ndarray,
    start_idx: np.ndarray,
    min_probability: float,
    top_k: int,
    fold_count: int,
    candidate_mask: np.ndarray | None = None,
    min_signal_gap_bars: int = 0,
) -> np.ndarray:
    selected_parts: list[np.ndarray] = []
    for fold_id in range(1, fold_count + 1):
        mask = (folds_array == fold_id) & (proba >= min_probability)
        if candidate_mask is not None:
            mask &= candidate_mask
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        order = np.lexsort((start_idx[idx], -proba[idx]))
        idx = idx[order]
        if min_signal_gap_bars > 0 and idx.size > 1:
            kept: list[int] = []
            kept_starts: list[int] = []
            for row in idx:
                row_start = int(start_idx[row])
                if all(abs(row_start - seen) >= min_signal_gap_bars for seen in kept_starts):
                    kept.append(int(row))
                    kept_starts.append(row_start)
                    if top_k > 0 and len(kept) >= top_k:
                        break
            idx = np.asarray(kept, dtype=np.int64)
        if top_k > 0:
            idx = idx[:top_k]
        selected_parts.append(idx)
    if not selected_parts:
        return np.empty(0, dtype=np.int64)
    selected = np.concatenate(selected_parts).astype(np.int64)
    return selected[np.lexsort((start_idx[selected], folds_array[selected]))]


def summarize_result(
    final_balance: np.ndarray,
    net_pct: np.ndarray,
    max_dd: np.ndarray,
    trades: np.ndarray,
    wins: np.ndarray,
    losses: np.ndarray,
) -> dict[str, float | int]:
    return {
        "target_folds": int((final_balance >= 1200.0).sum()),
        "loss_folds": int((net_pct < 0.0).sum()),
        "dd_fail_folds": int((max_dd < -20.0).sum()),
        "min_final_balance": round(float(final_balance.min()), 2),
        "worst_dd_pct": round(float(max_dd.min()), 2),
        "worst_net_pct": round(float(net_pct.min()), 2),
        "median_final_balance": round(float(np.median(final_balance)), 2),
        "mean_final_balance": round(float(np.mean(final_balance)), 2),
        "best_final_balance": round(float(final_balance.max()), 2),
        "total_trades": int(trades.sum()),
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "win_rate_pct": round(float(wins.sum() / max(1, wins.sum() + losses.sum()) * 100.0), 2),
    }


def fold_result_frame(
    final_balance: np.ndarray,
    net_pct: np.ndarray,
    max_dd: np.ndarray,
    trades: np.ndarray,
    wins: np.ndarray,
    losses: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": np.arange(1, len(final_balance) + 1, dtype=int),
            "final_balance": np.round(final_balance, 2),
            "net_pct": np.round(net_pct, 2),
            "max_dd_pct": np.round(max_dd, 2),
            "trades": trades.astype(int),
            "wins": wins.astype(int),
            "losses": losses.astype(int),
        }
    )


def write_mt5_manifest(
    selected_signals: pd.DataFrame,
    folds: list[dict[str, Any]],
    out_dir: Path,
    broker_gmt: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mt5_cols = ["open_time", "direction", "entry_price", "sl_price", "tp_price", "atr", "probability"]
    export = selected_signals.copy()
    signal_time = export["feature_time"] if "feature_time" in export.columns else export["time"]
    # The EA fires on a new M5 bar and matches CSV time against the previous bar.
    # Therefore CSV open_time must be the feature bar, while entry_price is the next-open reference.
    export["open_time"] = (
        pd.to_datetime(signal_time, utc=True) + pd.Timedelta(hours=broker_gmt)
    ).dt.strftime("%Y.%m.%d %H:%M")
    export[mt5_cols].to_csv(out_dir / "all_signals.csv", index=False, float_format="%.5f")

    manifest: dict[str, Any] = {
        "all_signals": str(out_dir / "all_signals.csv"),
        "folds": [],
        "total_signals": int(len(export)),
    }
    empty = pd.DataFrame(columns=mt5_cols)
    for fold in folds:
        fold_id = int(fold["fold"])
        fold_rows = export[export["fold"].astype(int) == fold_id].copy()
        fold_path = out_dir / f"fold_{fold_id:02d}_signals.csv"
        if fold_rows.empty:
            empty.to_csv(fold_path, index=False)
        else:
            fold_rows[mt5_cols].to_csv(fold_path, index=False, float_format="%.5f")
        manifest["folds"].append(
            {
                "fold": fold_id,
                "train_start": fold.get("train_start"),
                "train_end": fold.get("train_end"),
                "test_start": fold.get("test_start"),
                "test_end": fold.get("test_end"),
                "signals": int(len(fold_rows)),
                "csv": str(fold_path),
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def score_one_label_config(
    frame: pd.DataFrame,
    folds: list[dict[str, Any]],
    feature_cols: list[str],
    tp_rr: float,
    sl_mult: float,
    horizon_bars: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    print(f"\n=== label config tp_rr={tp_rr} sl_mult={sl_mult} horizon={horizon_bars} ===")
    open_ = frame["open"].to_numpy(np.float64)
    high = frame["high"].to_numpy(np.float64)
    low = frame["low"].to_numpy(np.float64)
    close = frame["close"].to_numpy(np.float64)
    atr = frame["atr"].to_numpy(np.float64)
    direction = frame["direction"].to_numpy(np.int8)
    rr_all, close_idx_all = first_hit_outcomes(
        open_=open_,
        high=high,
        low=low,
        close=close,
        atr=atr,
        direction=direction,
        tp_rr=tp_rr,
        sl_atr_mult=sl_mult,
        horizon_bars=horizon_bars,
        entry_delay_bars=args.entry_delay_bars,
    )
    print(
        f"labels: win_rate={(rr_all > 0).sum() / max(1, (rr_all != 0).sum()):.3f} "
        f"resolved={(rr_all != 0).sum()} rows={len(frame)}"
    )

    scored = fit_predict_oos(
        frame=frame,
        folds=folds,
        feature_cols=feature_cols,
        labels=rr_all,
        horizon_bars=horizon_bars,
        args=args,
    )

    source_idx = scored["source_idx"].to_numpy(np.int64)
    execution_idx = source_idx + args.entry_delay_bars
    valid_execution = execution_idx < len(frame)
    safe_execution_idx = np.where(valid_execution, execution_idx, source_idx)
    execution_times = frame["time"].iloc[safe_execution_idx].to_numpy()
    scored["feature_time"] = scored["time"]
    scored["time"] = execution_times
    scored["outcome_rr"] = rr_all[source_idx]
    scored["close_idx"] = close_idx_all[source_idx]
    scored["start_idx"] = execution_idx
    scored["entry_price"] = open_[safe_execution_idx]
    scored.loc[~valid_execution, "entry_price"] = np.nan
    scored["sl_dist"] = scored["atr"].astype(float) * sl_mult
    is_buy = scored["direction"].astype(int) == 1
    scored["sl_price"] = np.where(
        is_buy,
        scored["entry_price"] - scored["sl_dist"],
        scored["entry_price"] + scored["sl_dist"],
    )
    scored["tp_price"] = np.where(
        is_buy,
        scored["entry_price"] + scored["sl_dist"] * tp_rr,
        scored["entry_price"] - scored["sl_dist"] * tp_rr,
    )

    time_values = frame["time"].astype("int64").to_numpy()
    fold_end_by_id = {
        int(fold["fold"]): int(
            np.searchsorted(
                time_values,
                pd.Timestamp(fold["test_end"], tz="UTC").value,
                side="left",
            )
        )
        for fold in folds
    }
    fold_end_idx = scored["fold"].map(fold_end_by_id).to_numpy(np.int64)
    sim_rr = scored["outcome_rr"].to_numpy(np.float64).copy()
    sim_close_idx = scored["close_idx"].to_numpy(np.int64).copy()
    scored_start_idx = scored["start_idx"].to_numpy(np.int64)
    invalid = (sim_close_idx < 0) | (sim_close_idx >= fold_end_idx) | (scored_start_idx >= fold_end_idx)
    sim_rr[invalid] = 0.0
    sim_close_idx[invalid] = -1

    folds_array = scored["fold"].to_numpy(np.int64)
    proba = scored["probability"].to_numpy(np.float64)
    start_idx = scored_start_idx
    sl_dist_for_sizing = scored["sl_dist"].to_numpy(np.float64)

    rows: list[dict[str, float | int]] = []
    best_payload: dict[str, Any] | None = None
    fold_count = len(folds)
    for min_probability in parse_float_list(args.min_probabilities):
        for top_k in parse_int_list(args.top_ks):
            for risk_pct in parse_float_list(args.risks):
                candidate_mask = None
                if args.mt5_select_tradeable_first:
                    initial_cap = 200.0 * risk_pct
                    one_step_risk = sl_dist_for_sizing * args.mt5_min_lot_risk_per_price
                    candidate_mask = one_step_risk <= initial_cap * 1.001
                selected = build_selected_indices(
                    folds_array=folds_array,
                    proba=proba,
                    start_idx=start_idx,
                    min_probability=min_probability,
                    top_k=top_k,
                    fold_count=fold_count,
                    candidate_mask=candidate_mask,
                    min_signal_gap_bars=args.min_signal_gap_bars,
                )
                if selected.size == 0:
                    continue
                for max_positions in parse_int_list(args.max_positions):
                    if args.mt5_min_lot_sizing:
                        if "simulate_selected_mt5_minlot" not in globals():
                            raise RuntimeError("--mt5-min-lot-sizing requires numba in this script")
                        result = simulate_selected_mt5_minlot(
                            selected=selected,
                            folds=folds_array,
                            start_idx=start_idx,
                            close_idx=sim_close_idx,
                            rr=sim_rr,
                            sl_dist=sl_dist_for_sizing,
                            risk_pct=risk_pct,
                            max_positions=max_positions,
                            max_dd_pct=20.0,
                            fold_count=fold_count,
                            min_lot_risk_per_price=args.mt5_min_lot_risk_per_price,
                        )
                    else:
                        result = simulate_selected(
                            selected=selected,
                            folds=folds_array,
                            start_idx=start_idx,
                            close_idx=sim_close_idx,
                            rr=sim_rr,
                            risk_pct=risk_pct,
                            max_positions=max_positions,
                            max_dd_pct=20.0,
                            fold_count=fold_count,
                        )
                    final_balance, net_pct, max_dd, trades, wins, losses = result
                    summary = summarize_result(final_balance, net_pct, max_dd, trades, wins, losses)
                    for fold_index in range(fold_count):
                        fold_no = fold_index + 1
                        summary[f"fold_{fold_no:02d}_final"] = round(float(final_balance[fold_index]), 2)
                        summary[f"fold_{fold_no:02d}_dd"] = round(float(max_dd[fold_index]), 2)
                        summary[f"fold_{fold_no:02d}_trades"] = int(trades[fold_index])
                    summary.update(
                        {
                            "tp_rr": tp_rr,
                            "sl_mult": sl_mult,
                            "horizon_bars": horizon_bars,
                            "min_probability": min_probability,
                            "top_k_per_fold": top_k,
                            "risk_pct": risk_pct,
                            "max_positions": max_positions,
                            "selected_signals": int(selected.size),
                            "mt5_min_lot_sizing": int(bool(args.mt5_min_lot_sizing)),
                            "mt5_select_tradeable_first": int(bool(args.mt5_select_tradeable_first)),
                            "min_signal_gap_bars": int(args.min_signal_gap_bars),
                        }
                    )
                    rows.append(summary)

                    sort_key = (
                        summary["target_folds"],
                        -summary["loss_folds"],
                        -summary["dd_fail_folds"],
                        summary["median_final_balance"],
                        -summary["worst_dd_pct"],
                    )
                    if best_payload is None or sort_key > best_payload["sort_key"]:
                        best_payload = {
                            "sort_key": sort_key,
                            "summary": summary,
                            "selected": selected.copy(),
                            "fold_result": fold_result_frame(*result),
                        }

    if best_payload is None:
        raise RuntimeError("No selection produced trades")
    search = pd.DataFrame(rows).sort_values(
        ["target_folds", "loss_folds", "dd_fail_folds", "median_final_balance", "worst_dd_pct"],
        ascending=[False, True, True, False, False],
    )
    selected_signals = scored.iloc[best_payload["selected"]].sort_values(["fold", "time"]).copy()
    return search, selected_signals, best_payload["fold_result"], best_payload["summary"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a fresh OOS signal universe against reset-$200 target1200 fold objective."
    )
    parser.add_argument("--features", type=Path, default=ROOT / "outputs/historical_features_2022_2026.csv")
    parser.add_argument("--fold-manifest", type=Path, default=ROOT / "outputs/mt5_wf_r2_5_tradeside/manifest.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/target1200_model_universe")
    parser.add_argument("--broker-gmt", type=int, default=3)
    parser.add_argument("--tp-rrs", default="1.2")
    parser.add_argument("--sl-mults", default="1.0")
    parser.add_argument("--horizons", default="288")
    parser.add_argument("--risks", default="0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.06")
    parser.add_argument("--max-positions", default="1,2,3,4,5,6,8,10,12")
    parser.add_argument("--min-probabilities", default="0.0,0.52,0.55,0.58,0.60,0.63,0.66")
    parser.add_argument("--top-ks", default="100,200,400,800,1200,1600,2400")
    parser.add_argument("--include-broker-hours", default=None)
    parser.add_argument("--exclude-broker-hours", default=None)
    parser.add_argument("--side", choices=["all", "buy", "sell"], default="all")
    parser.add_argument(
        "--direction-mode",
        choices=["trade_side", "reverse_trade_side", "buy", "sell"],
        default="trade_side",
    )
    parser.add_argument("--min-atr-percentile", type=float, default=None)
    parser.add_argument("--max-atr-percentile", type=float, default=None)
    parser.add_argument("--exclude-news-blackout", action="store_true")
    parser.add_argument("--candidate-stride", type=int, default=1)
    parser.add_argument(
        "--entry-delay-bars",
        type=int,
        default=1,
        help="Delay execution after the feature bar. 1 means next M5 bar open.",
    )
    parser.add_argument(
        "--mt5-min-lot-sizing",
        action="store_true",
        help="Score selected signals with MT5 0.01-lot risk granularity and skip trades whose SL exceeds risk cap.",
    )
    parser.add_argument(
        "--mt5-min-lot-risk-per-price",
        type=float,
        default=1.0,
        help="USD risk for a 0.01 XAUUSD lot per 1.0 price unit of SL distance.",
    )
    parser.add_argument(
        "--mt5-select-tradeable-first",
        action="store_true",
        help="When scoring MT5 min-lot sizing, choose top-k only from signals whose SL is tradable from the reset $200 balance at the tested risk.",
    )
    parser.add_argument(
        "--min-signal-gap-bars",
        type=int,
        default=0,
        help="Optional greedy de-clustering gap for selected signals, measured in execution bars.",
    )
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--model-type", choices=["hgb", "extra_trees"], default="hgb")
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.05)
    parser.add_argument("--positive-weight", type=float, default=1.5)
    parser.add_argument("--recency-half-life-days", type=float, default=0.0)
    args = parser.parse_args()

    if args.candidate_stride < 1:
        raise ValueError("--candidate-stride must be >= 1")
    if args.entry_delay_bars < 0:
        raise ValueError("--entry-delay-bars must be >= 0")
    if args.min_signal_gap_bars < 0:
        raise ValueError("--min-signal-gap-bars must be >= 0")

    folds = load_folds(args.fold_manifest)
    include_hours = parse_hours(args.include_broker_hours)
    exclude_hours = parse_hours(args.exclude_broker_hours)
    feature_cols = available_feature_columns(args.features)
    frame = load_feature_frame(
        features_path=args.features,
        feature_cols=feature_cols,
        folds=folds,
        broker_gmt=args.broker_gmt,
        include_hours=include_hours,
        exclude_hours=exclude_hours,
        side=args.side,
        direction_mode=args.direction_mode,
        min_atr_percentile=args.min_atr_percentile,
        max_atr_percentile=args.max_atr_percentile,
        exclude_news_blackout=args.exclude_news_blackout,
        candidate_stride=args.candidate_stride,
    )
    print(f"loaded candidates={len(frame)} features={len(feature_cols)} folds={len(folds)}")

    all_search_parts: list[pd.DataFrame] = []
    best_global: dict[str, Any] | None = None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tp_rr in parse_float_list(args.tp_rrs):
        for sl_mult in parse_float_list(args.sl_mults):
            for horizon_bars in parse_int_list(args.horizons):
                config_key = make_config_key(args, tp_rr, sl_mult, horizon_bars)
                search, selected, fold_result, summary = score_one_label_config(
                    frame=frame,
                    folds=folds,
                    feature_cols=feature_cols,
                    tp_rr=tp_rr,
                    sl_mult=sl_mult,
                    horizon_bars=horizon_bars,
                    args=args,
                )
                search["config_key"] = config_key
                all_search_parts.append(search)

                config_dir = args.out_dir / f"config_{config_key}"
                config_dir.mkdir(parents=True, exist_ok=True)
                search.to_csv(config_dir / "search.csv", index=False)
                fold_result.to_csv(config_dir / "best_fold_results.csv", index=False)
                selected.to_csv(config_dir / "best_selected_signals_debug.csv", index=False)
                (config_dir / "best_summary.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
                print(pd.DataFrame([summary]).to_string(index=False))

                sort_key = (
                    summary["target_folds"],
                    -summary["loss_folds"],
                    -summary["dd_fail_folds"],
                    summary["median_final_balance"],
                    -summary["worst_dd_pct"],
                )
                if best_global is None or sort_key > best_global["sort_key"]:
                    best_global = {
                        "sort_key": sort_key,
                        "summary": summary,
                        "search": search,
                        "selected": selected,
                        "fold_result": fold_result,
                        "config_key": config_key,
                    }

    if best_global is None:
        raise RuntimeError("Search did not produce any candidate")

    all_search = pd.concat(all_search_parts, ignore_index=True).sort_values(
        ["target_folds", "loss_folds", "dd_fail_folds", "median_final_balance", "worst_dd_pct"],
        ascending=[False, True, True, False, False],
    )
    all_search.to_csv(args.out_dir / "target1200_model_search.csv", index=False)
    best = best_global["summary"]
    best_dir = args.out_dir / "best_mt5"
    write_mt5_manifest(best_global["selected"], folds, best_dir, broker_gmt=args.broker_gmt)
    best_global["fold_result"].to_csv(args.out_dir / "best_fold_results.csv", index=False)
    best_global["selected"].to_csv(args.out_dir / "best_selected_signals_debug.csv", index=False)
    (args.out_dir / "best_summary.json").write_text(
        json.dumps(
            {
                "config_key": best_global["config_key"],
                "summary": best,
                "mt5_manifest": str(best_dir / "manifest.json"),
                "gate_pass": bool(
                    best["target_folds"] >= len(folds)
                    and best["loss_folds"] <= 1
                    and best["dd_fail_folds"] == 0
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n=== global best ===")
    print(pd.DataFrame([best]).to_string(index=False))
    print(best_global["fold_result"].to_string(index=False))
    print(f"saved search: {args.out_dir / 'target1200_model_search.csv'}")
    print(f"saved MT5 manifest: {best_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
