from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mt5_prior_sequence_regime_throttle_sweep import (
    add_realized_r,
    build_training_rows,
    max_drawdown_pct,
    parse_floats,
    state_features,
)


def extra_points_for_trade(row: pd.Series, args: argparse.Namespace) -> float:
    points = float(args.extra_roundtrip_points)
    hour = int(row.get("hour")) if pd.notna(row.get("hour")) else -1
    weekday = int(row.get("weekday")) if pd.notna(row.get("weekday")) else -1
    thin_hours = {int(part.strip()) for part in str(args.thin_hours_utc).split(",") if part.strip()}
    if hour in thin_hours:
        points += float(args.thin_hour_extra_points)
    if weekday == 4 and hour >= int(args.friday_cutoff_hour_utc):
        points += float(args.friday_extra_points)
    return max(0.0, points)


def apply_extra_costs(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if (
        float(args.extra_roundtrip_points) == 0.0
        and float(args.thin_hour_extra_points) == 0.0
        and float(args.friday_extra_points) == 0.0
        and float(args.commission_per_lot_roundtrip) == 0.0
    ):
        return frame
    out = frame.copy()
    costs = []
    for _, row in out.iterrows():
        volume = float(row.get("volume") or 0.0)
        points = extra_points_for_trade(row, args)
        cost = points * float(args.point_size) * float(args.contract_size) * volume
        cost += float(args.commission_per_lot_roundtrip) * volume
        costs.append(cost)
    out["extra_cost"] = costs
    out["profit_raw"] = out["profit"]
    out["profit"] = pd.to_numeric(out["profit"], errors="coerce").fillna(0.0) - out["extra_cost"]
    return out


def session_bucket(hour: float) -> str:
    h = int(hour)
    if 0 <= h <= 6:
        return "asia_thin"
    if 7 <= h <= 12:
        return "london"
    if 13 <= h <= 17:
        return "ny_overlap"
    if 18 <= h <= 21:
        return "ny_late"
    return "rollover"


def enrich_buckets(frame: pd.DataFrame, atr_edges: list[float] | None = None) -> tuple[pd.DataFrame, list[float]]:
    out = frame.copy()
    out["session"] = out["hour"].map(session_bucket)
    out["direction_bucket"] = np.where(pd.to_numeric(out["direction"], errors="coerce") >= 0, "buy", "sell")
    out["prob_bucket"] = pd.cut(
        pd.to_numeric(out["signal_probability"], errors="coerce").fillna(0.5),
        bins=[-np.inf, 0.45, 0.50, 0.55, 0.60, np.inf],
        labels=["p00", "p45", "p50", "p55", "p60"],
    ).astype(str)
    atr = pd.to_numeric(out["signal_atr"], errors="coerce").fillna(0.0)
    if atr_edges is None:
        quantiles = atr[atr > 0].quantile([0.25, 0.5, 0.75]).dropna().tolist()
        atr_edges = sorted(set(float(value) for value in quantiles))
    bins = [-np.inf] + atr_edges + [np.inf]
    labels = [f"atr{i}" for i in range(len(bins) - 1)]
    out["atr_bucket"] = pd.cut(atr, bins=bins, labels=labels, duplicates="drop").astype(str)
    out["loss_bucket"] = pd.cut(
        pd.to_numeric(out["loss_streak"], errors="coerce").fillna(0.0),
        bins=[-np.inf, 0.5, 1.5, 2.5, np.inf],
        labels=["ls0", "ls1", "ls2", "ls3p"],
    ).astype(str)
    out["dd_bucket"] = pd.cut(
        pd.to_numeric(out["pre_dd_pct"], errors="coerce").fillna(0.0),
        bins=[-np.inf, -12.0, -8.0, -4.0, -1.0, np.inf],
        labels=["dd12p", "dd8", "dd4", "dd1", "dd0"],
    ).astype(str)
    out["roll5_bucket"] = pd.cut(
        pd.to_numeric(out["rolling5_r"], errors="coerce").fillna(0.0),
        bins=[-np.inf, -2.0, -1.0, 0.0, 1.0, np.inf],
        labels=["r5m2", "r5m1", "r5neg", "r5pos", "r5hot"],
    ).astype(str)
    return out, atr_edges


KEYS = [
    ("session", "direction_bucket", "prob_bucket", "atr_bucket", "loss_bucket", "dd_bucket", "roll5_bucket"),
    ("session", "direction_bucket", "prob_bucket", "loss_bucket", "dd_bucket"),
    ("session", "direction_bucket", "atr_bucket", "loss_bucket", "dd_bucket"),
    ("session", "direction_bucket", "loss_bucket", "dd_bucket", "roll5_bucket"),
    ("session", "direction_bucket", "loss_bucket", "dd_bucket"),
    ("session", "direction_bucket", "roll5_bucket"),
    ("session", "direction_bucket"),
]


def build_stats(train: pd.DataFrame, min_count: int) -> list[dict[tuple[Any, ...], dict[str, float]]]:
    stats: list[dict[tuple[Any, ...], dict[str, float]]] = []
    for key_cols in KEYS:
        table: dict[tuple[Any, ...], dict[str, float]] = {}
        grouped = train.groupby(list(key_cols), dropna=False)
        for key, group in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            if len(group) < int(min_count):
                continue
            table[key] = {
                "count": float(len(group)),
                "mean_r": float(group["realized_r"].mean()),
                "positive_rate": float(group["label_positive"].mean()),
            }
        stats.append(table)
    return stats


def lookup_stats(row: dict[str, Any], stats: list[dict[tuple[Any, ...], dict[str, float]]]) -> dict[str, float] | None:
    for key_cols, table in zip(KEYS, stats, strict=True):
        key = tuple(row.get(col) for col in key_cols)
        if key in table:
            return table[key]
    return None


def row_with_buckets(row: pd.Series, state: dict[str, float], atr_edges: list[float]) -> dict[str, Any]:
    hour = float(row.get("hour", 0))
    direction = float(row.get("direction", 0))
    prob = float(row.get("signal_probability", 0.5))
    atr = float(row.get("signal_atr", 0.0))
    loss_streak = float(state.get("loss_streak", 0.0))
    pre_dd = float(state.get("pre_dd_pct", 0.0))
    roll5 = float(state.get("rolling5_r", 0.0))

    if prob <= 0.45:
        prob_bucket = "p00"
    elif prob <= 0.50:
        prob_bucket = "p45"
    elif prob <= 0.55:
        prob_bucket = "p50"
    elif prob <= 0.60:
        prob_bucket = "p55"
    else:
        prob_bucket = "p60"

    atr_idx = 0
    for edge in atr_edges:
        if atr > float(edge):
            atr_idx += 1
    atr_bucket = f"atr{atr_idx}"

    if loss_streak <= 0.5:
        loss_bucket = "ls0"
    elif loss_streak <= 1.5:
        loss_bucket = "ls1"
    elif loss_streak <= 2.5:
        loss_bucket = "ls2"
    else:
        loss_bucket = "ls3p"

    if pre_dd <= -12.0:
        dd_bucket = "dd12p"
    elif pre_dd <= -8.0:
        dd_bucket = "dd8"
    elif pre_dd <= -4.0:
        dd_bucket = "dd4"
    elif pre_dd <= -1.0:
        dd_bucket = "dd1"
    else:
        dd_bucket = "dd0"

    if roll5 <= -2.0:
        roll5_bucket = "r5m2"
    elif roll5 <= -1.0:
        roll5_bucket = "r5m1"
    elif roll5 <= 0.0:
        roll5_bucket = "r5neg"
    elif roll5 <= 1.0:
        roll5_bucket = "r5pos"
    else:
        roll5_bucket = "r5hot"

    base = {
        "hour": hour,
        "weekday": float(row.get("weekday", 0)),
        "direction": direction,
        "signal_probability": prob,
        "signal_atr": atr,
        "rr": float(row.get("rr", 0.0)),
        **state,
        "session": session_bucket(hour),
        "direction_bucket": "buy" if direction >= 0 else "sell",
        "prob_bucket": prob_bucket,
        "atr_bucket": atr_bucket,
        "loss_bucket": loss_bucket,
        "dd_bucket": dd_bucket,
        "roll5_bucket": roll5_bucket,
    }
    return base


def simulate_fold(
    group: pd.DataFrame,
    stats: list[dict[tuple[Any, ...], dict[str, float]]],
    atr_edges: list[float],
    *,
    args: argparse.Namespace,
    mean_r_skip: float,
    pos_rate_skip: float,
    reduce_mean_r: float,
    reduce_mult: float,
    pre_dd_hard_skip_pct: float,
    boost_mean_r: float,
    boost_pos_rate: float,
    boost_mult: float,
    boost_pre_dd_floor_pct: float,
    projected_dd_cap_pct: float,
    projected_dd_buffer_pct: float,
) -> dict[str, Any]:
    balance = float(args.deposit)
    peak = balance
    curve = [balance]
    recent_r: list[float] = []
    loss_streak = 0
    win_streak = 0
    used = 0
    skipped = 0
    reduced = 0
    boosted = 0
    dd_capped = 0
    stat_hits = 0
    for trade_index, (_, row) in enumerate(group.sort_values(["close_time", "open_time"]).iterrows(), start=1):
        state = state_features(
            balance=balance,
            peak=peak,
            recent_r=recent_r,
            loss_streak=loss_streak,
            win_streak=win_streak,
            trade_index=trade_index,
        )
        if float(state["pre_dd_pct"]) <= -float(pre_dd_hard_skip_pct):
            skipped += 1
            continue
        enriched = row_with_buckets(row, state, atr_edges)
        stat = lookup_stats(enriched, stats)
        if stat is not None:
            stat_hits += 1
            if stat["mean_r"] < float(mean_r_skip) or stat["positive_rate"] < float(pos_rate_skip):
                skipped += 1
                continue
            mult = float(reduce_mult) if stat["mean_r"] < float(reduce_mean_r) else 1.0
            if (
                stat["mean_r"] >= float(boost_mean_r)
                and stat["positive_rate"] >= float(boost_pos_rate)
                and float(state["pre_dd_pct"]) >= -float(boost_pre_dd_floor_pct)
            ):
                mult = max(mult, float(boost_mult))
        else:
            mult = 1.0
        if mult < 0.999:
            reduced += 1
        if mult > 1.001:
            boosted += 1
        if projected_dd_cap_pct > 0:
            sl_risk = abs(float(row["entry_price"]) - float(row["sl_price"])) * float(row["volume"]) * float(
                args.contract_value_per_lot
            )
            floor_balance = peak * (1.0 - (float(projected_dd_cap_pct) - float(projected_dd_buffer_pct)) / 100.0)
            if sl_risk > 0 and balance - (sl_risk * mult) < floor_balance:
                allowed_loss = max(0.0, balance - floor_balance)
                capped_mult = max(0.0, min(mult, allowed_loss / sl_risk))
                if capped_mult < mult:
                    mult = capped_mult
                    dd_capped += 1
                    if mult <= 0.0:
                        skipped += 1
                        continue
        profit = float(row["profit"]) * mult
        balance += profit
        peak = max(peak, balance)
        curve.append(balance)
        used += 1
        realized_r = float(row["realized_r"]) * mult
        recent_r.append(realized_r)
        if profit < 0:
            loss_streak += 1
            win_streak = 0
        elif profit > 0:
            win_streak += 1
            loss_streak = 0
        if balance >= float(args.target_balance):
            break
    dd = max_drawdown_pct(curve)
    return {
        "fold": int(group["fold"].iloc[0]),
        "final_balance": round(balance, 2),
        "target_hit": bool(balance >= float(args.target_balance)),
        "max_dd_pct": round(dd, 2),
        "dd_pass": bool(dd >= -float(args.max_dd_pct)),
        "loss_fold": bool(balance < float(args.deposit)),
        "trades_used": int(used),
        "trades_skipped": int(skipped),
        "trades_reduced": int(reduced),
        "trades_boosted": int(boosted),
        "trades_dd_capped": int(dd_capped),
        "stat_hits": int(stat_hits),
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    strict = frame["target_hit"] & frame["dd_pass"] & (~frame["loss_fold"])
    return {
        "folds": int(len(frame)),
        "all_pass_folds": int(strict.sum()),
        "target_pass_folds": int(frame["target_hit"].sum()),
        "dd_pass_folds": int(frame["dd_pass"].sum()),
        "loss_folds": int(frame["loss_fold"].sum()),
        "min_final_balance": round(float(frame["final_balance"].min()), 2),
        "median_final_balance": round(float(frame["final_balance"].median()), 2),
        "worst_dd_pct": round(float(frame["max_dd_pct"].min()), 2),
        "total_trades_used": int(frame["trades_used"].sum()),
        "total_trades_skipped": int(frame["trades_skipped"].sum()),
        "total_trades_reduced": int(frame["trades_reduced"].sum()),
        "total_trades_boosted": int(frame["trades_boosted"].sum()),
        "total_trades_dd_capped": int(frame["trades_dd_capped"].sum()),
        "total_stat_hits": int(frame["stat_hits"].sum()),
        "fail_folds": frame.loc[~strict, "fold"].astype(int).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep fast prior-sequence bucket risk throttle.")
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--contract-value-per-lot", type=float, default=100.0)
    parser.add_argument("--extra-roundtrip-points", type=float, default=0.0)
    parser.add_argument("--thin-hours-utc", default="")
    parser.add_argument("--thin-hour-extra-points", type=float, default=0.0)
    parser.add_argument("--friday-cutoff-hour-utc", type=int, default=20)
    parser.add_argument("--friday-extra-points", type=float, default=0.0)
    parser.add_argument("--point-size", type=float, default=0.001)
    parser.add_argument("--contract-size", type=float, default=100.0)
    parser.add_argument("--commission-per-lot-roundtrip", type=float, default=0.0)
    parser.add_argument("--min-counts", default="20,40,80")
    parser.add_argument("--mean-r-skips", default="-0.2,-0.1,0,0.05")
    parser.add_argument("--pos-rate-skips", default="0.35,0.4,0.45")
    parser.add_argument("--reduce-mean-rs", default="0.05,0.1,0.2")
    parser.add_argument("--reduce-mults", default="0.5,0.75,1.0")
    parser.add_argument("--pre-dd-hard-skip-pcts", default="12,14,16,18,20")
    parser.add_argument("--boost-mean-rs", default="999")
    parser.add_argument("--boost-pos-rates", default="0.55")
    parser.add_argument("--boost-mults", default="1.0")
    parser.add_argument("--boost-pre-dd-floor-pcts", default="20")
    parser.add_argument("--projected-dd-cap-pcts", default="0")
    parser.add_argument("--projected-dd-buffer-pcts", default="0")
    args = parser.parse_args()

    feedback = pd.read_csv(args.feedback)
    for col in ["fold", "profit", "hour", "weekday", "direction", "signal_probability", "signal_atr", "rr"]:
        feedback[col] = pd.to_numeric(feedback.get(col), errors="coerce")
    for col in ["open_time", "close_time"]:
        feedback[col] = pd.to_datetime(feedback[col], errors="coerce")
    feedback = feedback.dropna(subset=["fold", "profit"]).copy()
    feedback["fold"] = feedback["fold"].astype(int)
    feedback = apply_extra_costs(feedback, args)
    feedback = add_realized_r(feedback, args.contract_value_per_lot)
    training_rows = build_training_rows(feedback, args.deposit)
    training_rows, atr_edges = enrich_buckets(training_rows)

    folds = sorted(feedback["fold"].unique().tolist())
    records: list[dict[str, Any]] = []
    details: dict[str, list[dict[str, Any]]] = {}
    stats_cache: dict[tuple[int, int], list[dict[tuple[Any, ...], dict[str, float]]]] = {}
    for min_count in [int(value) for value in parse_floats(args.min_counts)]:
        for mean_r_skip in parse_floats(args.mean_r_skips):
            for pos_rate_skip in parse_floats(args.pos_rate_skips):
                for reduce_mean_r in parse_floats(args.reduce_mean_rs):
                    if reduce_mean_r < mean_r_skip:
                        continue
                    for reduce_mult in parse_floats(args.reduce_mults):
                        for pre_dd_hard_skip_pct in parse_floats(args.pre_dd_hard_skip_pcts):
                            for boost_mean_r in parse_floats(args.boost_mean_rs):
                                for boost_pos_rate in parse_floats(args.boost_pos_rates):
                                    for boost_mult in parse_floats(args.boost_mults):
                                        for boost_pre_dd_floor_pct in parse_floats(args.boost_pre_dd_floor_pcts):
                                            for projected_dd_cap_pct in parse_floats(args.projected_dd_cap_pcts):
                                                for projected_dd_buffer_pct in parse_floats(args.projected_dd_buffer_pcts):
                                                    rows: list[dict[str, Any]] = []
                                                    for fold_id in folds:
                                                        cache_key = (fold_id, min_count)
                                                        if cache_key not in stats_cache:
                                                            train = training_rows[training_rows["fold"] < fold_id].copy()
                                                            stats_cache[cache_key] = build_stats(train, min_count)
                                                        rows.append(
                                                            simulate_fold(
                                                                feedback[feedback["fold"] == fold_id].copy(),
                                                                stats_cache[cache_key],
                                                                atr_edges,
                                                                args=args,
                                                                mean_r_skip=mean_r_skip,
                                                                pos_rate_skip=pos_rate_skip,
                                                                reduce_mean_r=reduce_mean_r,
                                                                reduce_mult=reduce_mult,
                                                                pre_dd_hard_skip_pct=pre_dd_hard_skip_pct,
                                                                boost_mean_r=boost_mean_r,
                                                                boost_pos_rate=boost_pos_rate,
                                                                boost_mult=boost_mult,
                                                                boost_pre_dd_floor_pct=boost_pre_dd_floor_pct,
                                                                projected_dd_cap_pct=projected_dd_cap_pct,
                                                                projected_dd_buffer_pct=projected_dd_buffer_pct,
                                                            )
                                                        )
                                                    summary = summarize(rows, args)
                                                    key = (
                                                        f"mc{min_count}_mrs{mean_r_skip:g}_prs{pos_rate_skip:g}_"
                                                        f"rmr{reduce_mean_r:g}_rm{reduce_mult:g}_pdd{pre_dd_hard_skip_pct:g}_"
                                                        f"bmr{boost_mean_r:g}_bpr{boost_pos_rate:g}_bm{boost_mult:g}_"
                                                        f"bf{boost_pre_dd_floor_pct:g}_pdc{projected_dd_cap_pct:g}_"
                                                        f"pdb{projected_dd_buffer_pct:g}"
                                                    )
                                                    summary.update(
                                                        {
                                                            "key": key,
                                                            "min_count": int(min_count),
                                                            "mean_r_skip": float(mean_r_skip),
                                                            "pos_rate_skip": float(pos_rate_skip),
                                                            "reduce_mean_r": float(reduce_mean_r),
                                                            "reduce_mult": float(reduce_mult),
                                                            "pre_dd_hard_skip_pct": float(pre_dd_hard_skip_pct),
                                                            "boost_mean_r": float(boost_mean_r),
                                                            "boost_pos_rate": float(boost_pos_rate),
                                                            "boost_mult": float(boost_mult),
                                                            "boost_pre_dd_floor_pct": float(boost_pre_dd_floor_pct),
                                                            "projected_dd_cap_pct": float(projected_dd_cap_pct),
                                                            "projected_dd_buffer_pct": float(projected_dd_buffer_pct),
                                                        }
                                                    )
                                                    records.append(summary)
                                                    details[key] = rows
    records.sort(
        key=lambda item: (
            item["all_pass_folds"],
            item["target_pass_folds"],
            item["dd_pass_folds"],
            -item["loss_folds"],
            item["worst_dd_pct"],
            item["median_final_balance"],
        ),
        reverse=True,
    )
    report = {
        "best": records[:20],
        "details": {record["key"]: details[record["key"]] for record in records[:10]},
        "keys": [list(key) for key in KEYS],
        "atr_edges": atr_edges,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best": records[:10]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
