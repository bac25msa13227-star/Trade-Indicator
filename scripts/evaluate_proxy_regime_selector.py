from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_proxy_candidate_rolling_selector import load_candidates  # noqa: E402
from scripts.export_mt5_regime_meta_selector import REGIME_COLUMNS, _normalizers  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_folds(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    folds = data.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"No folds in schedule manifest: {path}")
    return folds


def fast_load_feature_regimes(features_path: Path, folds: list[dict[str, Any]], lookback_days: int) -> pd.DataFrame:
    header = pd.read_csv(features_path, nrows=0).columns
    usecols = [c for c in ["time", "close", "trade_side"] + REGIME_COLUMNS if c in header]
    frame = pd.read_csv(features_path, usecols=usecols)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce").dt.tz_localize(None)
    frame = frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    times = frame["time"].to_numpy(dtype="datetime64[ns]")
    numeric_cols = [c for c in frame.columns if c not in {"time", "trade_side"}]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = int(fold["fold"])
        test_start = pd.to_datetime(str(fold["test_start"]), errors="coerce").tz_localize(None)
        if pd.isna(test_start):
            continue
        start = test_start - pd.Timedelta(days=int(lookback_days))
        left = int(np.searchsorted(times, np.datetime64(start), side="left"))
        right = int(np.searchsorted(times, np.datetime64(test_start), side="left"))
        if right <= left:
            continue
        window = frame.iloc[left:right]
        row: dict[str, Any] = {"fold": fold_id, "regime_rows": int(len(window))}
        close = pd.to_numeric(window.get("close"), errors="coerce").dropna()
        if len(close) >= 2 and float(close.iloc[0]) != 0.0:
            ret = float(close.iloc[-1] / close.iloc[0] - 1.0)
            row["regime_close_return"] = ret
            row["regime_abs_close_return"] = abs(ret)
        if "trade_side" in window.columns:
            side = window["trade_side"].astype(str).str.lower()
            row["regime_buy_ratio"] = float((side == "buy").mean())
        for col in REGIME_COLUMNS:
            if col not in window.columns:
                continue
            values = pd.to_numeric(window[col], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"regime_{col}_mean"] = float(values.mean())
            row[f"regime_{col}_std"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).set_index("fold") if rows else pd.DataFrame()


def build_similarity_matrix(
    regimes: pd.DataFrame,
    fold_ids: list[int],
    mean: pd.Series,
    std: pd.Series,
    regime_cols: list[str],
    bandwidth: float,
) -> tuple[dict[int, int], np.ndarray]:
    fold_to_idx = {int(fold_id): idx for idx, fold_id in enumerate(fold_ids)}
    if not regime_cols:
        return fold_to_idx, np.ones((len(fold_ids), len(fold_ids)), dtype=float)
    values = np.zeros((len(fold_ids), len(regime_cols)), dtype=float)
    valid = np.zeros((len(fold_ids), len(regime_cols)), dtype=bool)
    for i, fold_id in enumerate(fold_ids):
        if fold_id not in regimes.index:
            continue
        row = ((regimes.loc[fold_id, regime_cols] - mean[regime_cols]) / std[regime_cols]).replace([np.inf, -np.inf], np.nan)
        arr = row.to_numpy(dtype=float)
        mask = np.isfinite(arr)
        values[i, mask] = arr[mask]
        valid[i, mask] = True
    sims = np.ones((len(fold_ids), len(fold_ids)), dtype=float)
    bw = max(float(bandwidth), 1.0e-6)
    for i in range(len(fold_ids)):
        diff = values - values[i]
        mask = valid & valid[i]
        counts = mask.sum(axis=1)
        dist = np.ones(len(fold_ids), dtype=float)
        usable = counts > 0
        if usable.any():
            sq = np.square(diff) * mask
            dist[usable] = np.sqrt(sq[usable].sum(axis=1) / counts[usable])
        sims[i, :] = np.exp(-0.5 * np.square(dist / bw))
    return fold_to_idx, sims


def outcome_score(row: pd.Series, target: float, deposit: float, max_dd: float) -> float:
    final_balance = float(row.get("final_balance", deposit) or deposit)
    dd = float(row.get("max_dd_pct", -100.0) or -100.0)
    target_hit = final_balance >= target
    dd_pass = dd >= -max_dd
    loss = final_balance < deposit
    capped_final = min(final_balance, target)
    growth = (capped_final - deposit) / max(1.0, target - deposit)
    dd_buffer = max(-1.5, min(1.0, (dd + max_dd) / max_dd))
    value = 2.0 * growth + 0.45 * dd_buffer
    if target_hit:
        value += 1.0
    if dd_pass:
        value += 0.6
    if target_hit and dd_pass and not loss:
        value += 3.0
    if not target_hit:
        value -= 1.2
    if not dd_pass:
        value -= 2.2
    if loss:
        value -= 3.5
    return float(value)


def weighted_candidate_score(
    history: pd.DataFrame,
    fold_id: int,
    fold_to_idx: dict[int, int],
    sims: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float]:
    rows: list[float] = []
    weights: list[float] = []
    strict_values: list[float] = []
    target_values: list[float] = []
    dd_values: list[float] = []
    loss_values: list[float] = []

    for _, row in history.iterrows():
        hist_fold = int(row["fold"])
        if fold_id in fold_to_idx and hist_fold in fold_to_idx:
            sim = float(sims[fold_to_idx[fold_id], fold_to_idx[hist_fold]])
        else:
            sim = 1.0
        age = max(0, fold_id - hist_fold)
        recency = 1.0
        if float(args.recency_half_life_folds) > 0:
            recency = 0.5 ** (age / float(args.recency_half_life_folds))
        weight = max(0.0, sim * recency)
        if weight <= 1.0e-12:
            continue
        final_balance = float(row["final_balance"])
        dd = float(row["max_dd_pct"])
        target_hit = final_balance >= float(args.target_balance)
        dd_pass = dd >= -float(args.max_dd_pct)
        loss = final_balance < float(args.deposit)
        strict = target_hit and dd_pass and not loss
        rows.append(outcome_score(row, float(args.target_balance), float(args.deposit), float(args.max_dd_pct)))
        weights.append(weight)
        strict_values.append(1.0 if strict else 0.0)
        target_values.append(1.0 if target_hit else 0.0)
        dd_values.append(1.0 if dd_pass else 0.0)
        loss_values.append(1.0 if loss else 0.0)

    if not weights:
        return {
            "score": -1.0e18,
            "effective_rows": 0.0,
            "weighted_value": math.nan,
            "weighted_strict": math.nan,
            "weighted_target": math.nan,
            "weighted_dd": math.nan,
            "weighted_loss": math.nan,
        }

    w = np.asarray(weights, dtype=float)
    effective_rows = float((w.sum() ** 2) / max(float(np.square(w).sum()), 1.0e-12))
    weighted_value = float(np.average(np.asarray(rows, dtype=float), weights=w))
    weighted_strict = float(np.average(np.asarray(strict_values, dtype=float), weights=w))
    weighted_target = float(np.average(np.asarray(target_values, dtype=float), weights=w))
    weighted_dd = float(np.average(np.asarray(dd_values, dtype=float), weights=w))
    weighted_loss = float(np.average(np.asarray(loss_values, dtype=float), weights=w))
    recent = history.sort_values("fold").tail(max(1, int(args.recent_folds)))
    recent_strict = float(recent["strict_pass"].mean())
    recent_loss = float(recent["loss_fold"].mean())
    recent_dd = float(recent["dd_pass"].mean())
    recent_target = float(recent["target_hit"].mean())
    score = (
        weighted_value * float(args.value_weight)
        + weighted_strict * float(args.strict_weight)
        + weighted_target * float(args.target_weight)
        + weighted_dd * float(args.dd_weight)
        - weighted_loss * float(args.loss_weight)
        + recent_strict * float(args.recent_strict_weight)
        + recent_target * float(args.recent_target_weight)
        + recent_dd * float(args.recent_dd_weight)
        - recent_loss * float(args.recent_loss_weight)
        + min(effective_rows, 20.0) * float(args.effective_rows_weight)
    )
    return {
        "score": float(score),
        "effective_rows": effective_rows,
        "weighted_value": weighted_value,
        "weighted_strict": weighted_strict,
        "weighted_target": weighted_target,
        "weighted_dd": weighted_dd,
        "weighted_loss": weighted_loss,
        "recent_strict": recent_strict,
        "recent_target": recent_target,
        "recent_dd": recent_dd,
        "recent_loss": recent_loss,
    }


def choose_bootstrap(current: pd.DataFrame) -> pd.Series:
    ranked = current.sort_values(
        ["candidate_target_folds", "candidate_loss_folds", "candidate_dd_fail_folds", "candidate_median_final"],
        ascending=[False, True, True, False],
    )
    return ranked.iloc[0]


def evaluate_once(
    frame: pd.DataFrame,
    folds: list[dict[str, Any]],
    regimes: pd.DataFrame,
    fold_to_idx: dict[int, int],
    sims: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    fold_ids = [int(f["fold"]) for f in folds]
    for fold_id in fold_ids:
        current = frame[frame["fold"] == fold_id].copy()
        if current.empty:
            continue
        start_fold = 1 if int(args.lookback_folds) <= 0 else max(1, fold_id - int(args.lookback_folds))
        best: tuple[float, pd.Series, dict[str, float], int] | None = None
        for label, group in frame.groupby("candidate_label"):
            current_row = current[current["candidate_label"] == label]
            if current_row.empty:
                continue
            history = group[(group["fold"] >= start_fold) & (group["fold"] < fold_id)].copy()
            if len(history) < int(args.min_history_folds):
                continue
            stats = weighted_candidate_score(history, fold_id, fold_to_idx, sims, args)
            if stats["effective_rows"] < float(args.min_effective_rows):
                continue
            score = float(stats["score"])
            if best is None or score > best[0]:
                best = (score, current_row.iloc[0], stats, int(len(history)))
        mode = "regime_prior"
        if best is None:
            row = choose_bootstrap(current)
            stats = {
                "score": math.nan,
                "effective_rows": 0.0,
                "weighted_value": math.nan,
                "weighted_strict": math.nan,
                "weighted_target": math.nan,
                "weighted_dd": math.nan,
                "weighted_loss": math.nan,
            }
            hist_count = 0
            mode = "bootstrap_global_proxy"
        else:
            _, row, stats, hist_count = best
        record = row.to_dict()
        record["selection_mode"] = mode
        record["selection_history_folds"] = hist_count
        for key, value in stats.items():
            record[f"selector_{key}"] = value
        selected.append(record)
    rows = pd.DataFrame(selected)
    strict = (rows["final_balance"] >= float(args.target_balance)) & (rows["max_dd_pct"] >= -float(args.max_dd_pct)) & (rows["final_balance"] >= float(args.deposit))
    summary = {
        "folds": int(len(rows)),
        "target_folds": int((rows["final_balance"] >= float(args.target_balance)).sum()),
        "dd_pass_folds": int((rows["max_dd_pct"] >= -float(args.max_dd_pct)).sum()),
        "loss_folds": int((rows["final_balance"] < float(args.deposit)).sum()),
        "strict_pass_folds": int(strict.sum()),
        "min_final_balance": round(float(rows["final_balance"].min()), 2),
        "median_final_balance": round(float(rows["final_balance"].median()), 2),
        "worst_dd_pct": round(float(rows["max_dd_pct"].min()), 2),
        "bootstrap_folds": int((rows["selection_mode"] == "bootstrap_global_proxy").sum()),
    }
    return rows, summary


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean proxy selector using only prior-fold outcomes and pre-fold feature regimes.")
    parser.add_argument("--candidate-root", action="append", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--min-history-folds", type=int, default=8)
    parser.add_argument("--lookback-folds", type=int, default=40)
    parser.add_argument("--recent-folds", type=int, default=4)
    parser.add_argument("--regime-lookback-days", type=int, default=60)
    parser.add_argument("--regime-bandwidth", type=float, default=2.0)
    parser.add_argument("--recency-half-life-folds", type=float, default=20.0)
    parser.add_argument("--min-effective-rows", type=float, default=3.0)
    parser.add_argument("--value-weight", type=float, default=1000.0)
    parser.add_argument("--strict-weight", type=float, default=2200.0)
    parser.add_argument("--target-weight", type=float, default=800.0)
    parser.add_argument("--dd-weight", type=float, default=700.0)
    parser.add_argument("--loss-weight", type=float, default=1600.0)
    parser.add_argument("--recent-strict-weight", type=float, default=900.0)
    parser.add_argument("--recent-target-weight", type=float, default=300.0)
    parser.add_argument("--recent-dd-weight", type=float, default=300.0)
    parser.add_argument("--recent-loss-weight", type=float, default=1000.0)
    parser.add_argument("--effective-rows-weight", type=float, default=25.0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-lookback-folds", default="20,40,80,0")
    parser.add_argument("--sweep-regime-lookback-days", default="30,60,120,180")
    parser.add_argument("--sweep-regime-bandwidth", default="0.75,1.25,2.0,3.5")
    parser.add_argument("--sweep-min-history-folds", default="5,8,12")
    parser.add_argument("--sweep-recency-half-life-folds", default="10,20,40")
    args = parser.parse_args()

    folds = load_folds(args.fold_manifest)
    frame = load_candidates(args.candidate_root, args.target_balance, args.max_dd_pct, args.deposit)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.sweep:
        regimes = fast_load_feature_regimes(args.features, folds, args.regime_lookback_days)
        mean, std, regime_cols = _normalizers(regimes)
        fold_ids = [int(f["fold"]) for f in folds]
        fold_to_idx, sims = build_similarity_matrix(regimes, fold_ids, mean, std, regime_cols, args.regime_bandwidth)
        selected, summary = evaluate_once(frame, folds, regimes, fold_to_idx, sims, args)
        selected.to_csv(args.out_dir / "regime_proxy_selected_folds.csv", index=False)
        frame.to_csv(args.out_dir / "regime_proxy_candidate_matrix.csv", index=False)
        (args.out_dir / "regime_proxy_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    sweep_rows: list[dict[str, Any]] = []
    best_selected: pd.DataFrame | None = None
    best_summary: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    regime_cache: dict[int, tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]] = {}
    combos = itertools.product(
        parse_int_list(args.sweep_lookback_folds),
        parse_int_list(args.sweep_regime_lookback_days),
        parse_float_list(args.sweep_regime_bandwidth),
        parse_int_list(args.sweep_min_history_folds),
        parse_float_list(args.sweep_recency_half_life_folds),
    )
    for lookback, regime_days, bandwidth, min_hist, half_life in combos:
        if regime_days not in regime_cache:
            regimes = fast_load_feature_regimes(args.features, folds, regime_days)
            mean, std, regime_cols = _normalizers(regimes)
            regime_cache[regime_days] = (regimes, mean, std, regime_cols)
        regimes, mean, std, regime_cols = regime_cache[regime_days]
        run_args = argparse.Namespace(**vars(args))
        run_args.lookback_folds = lookback
        run_args.regime_lookback_days = regime_days
        run_args.regime_bandwidth = bandwidth
        run_args.min_history_folds = min_hist
        run_args.recency_half_life_folds = half_life
        fold_ids = [int(f["fold"]) for f in folds]
        fold_to_idx, sims = build_similarity_matrix(regimes, fold_ids, mean, std, regime_cols, bandwidth)
        selected, summary = evaluate_once(frame, folds, regimes, fold_to_idx, sims, run_args)
        row = {
            **summary,
            "lookback_folds": lookback,
            "regime_lookback_days": regime_days,
            "regime_bandwidth": bandwidth,
            "min_history_folds": min_hist,
            "recency_half_life_folds": half_life,
        }
        sweep_rows.append(row)
        key = (
            float(summary["strict_pass_folds"]),
            float(summary["target_folds"]),
            float(summary["dd_pass_folds"]),
            -float(summary["loss_folds"]),
            float(summary["median_final_balance"]),
            float(summary["min_final_balance"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_selected = selected
            best_summary = row

    sweep = pd.DataFrame(sweep_rows).sort_values(
        ["strict_pass_folds", "target_folds", "dd_pass_folds", "loss_folds", "median_final_balance"],
        ascending=[False, False, False, True, False],
    )
    sweep.to_csv(args.out_dir / "regime_proxy_sweep.csv", index=False)
    assert best_selected is not None and best_summary is not None
    best_selected.to_csv(args.out_dir / "regime_proxy_best_selected_folds.csv", index=False)
    frame.to_csv(args.out_dir / "regime_proxy_candidate_matrix.csv", index=False)
    (args.out_dir / "regime_proxy_best_summary.json").write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    print(json.dumps(best_summary, indent=2))
    print(f"sweep={args.out_dir / 'regime_proxy_sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
