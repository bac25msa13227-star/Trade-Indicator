from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DIRTY_KEYS = ("adaptive_per_fold", "research_oracle_fold_selection", "selection_uses_current_fold_metrics")


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except (UnicodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not parse JSON: {path}")


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def label_for(path: Path) -> str:
    return path.name


def max_drawdown_pct(curve: list[float]) -> float:
    peak = curve[0] if curve else 0.0
    worst = 0.0
    for value in curve:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, (value - peak) / peak * 100.0)
    return worst


def load_feedback(candidate_dir: Path, refresh: bool = False) -> pd.DataFrame:
    manifest = candidate_dir / "manifest.json"
    feedback = candidate_dir / "mt5_trade_feedback_detailed.csv"
    if not manifest.exists():
        raise ValueError(f"missing manifest: {manifest}")
    payload = read_json(manifest)
    dirty = [key for key in DIRTY_KEYS if is_true(payload.get(key))]
    if dirty:
        raise ValueError(f"refusing dirty candidate {candidate_dir}: {dirty}")
    if not feedback.exists():
        raise ValueError(f"missing feedback: {feedback}")

    frame = pd.read_csv(feedback)
    if frame.empty:
        return frame
    frame["candidate_label"] = label_for(candidate_dir)
    frame["candidate_dir"] = str(candidate_dir)
    for column in [
        "fold",
        "volume",
        "profit",
        "hour",
        "weekday",
        "signal_probability",
        "signal_atr",
        "signal_match_lag_min",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["open_time", "close_time"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    required = {"fold", "open_time", "close_time", "profit", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{feedback} missing columns: {sorted(missing)}")
    return frame.dropna(subset=["fold", "open_time", "close_time", "profit", "volume"]).copy()


def extra_cost(row: pd.Series, args: argparse.Namespace) -> float:
    volume = float(row["volume"])
    points = float(args.extra_roundtrip_points)
    hour = int(row.get("hour")) if pd.notna(row.get("hour")) else -1
    weekday = int(row.get("weekday")) if pd.notna(row.get("weekday")) else -1
    thin_hours = {int(x) for x in str(args.thin_hours_utc).split(",") if str(x).strip()}
    if hour in thin_hours:
        points += float(args.thin_hour_extra_points)
    if weekday == 4 and hour >= int(args.friday_cutoff_hour_utc):
        points += float(args.friday_extra_points)
    return points * float(args.point_size) * float(args.contract_size) * volume + (
        float(args.commission_per_lot_roundtrip) * volume
    )


def history_scores(stress_folds: pd.DataFrame, fold_id: int, args: argparse.Namespace) -> dict[str, float]:
    hist = stress_folds[stress_folds["fold"] < fold_id].copy()
    if int(args.history_lookback) > 0:
        hist = hist[hist["fold"] >= fold_id - int(args.history_lookback)].copy()
    if hist.empty:
        return {}
    hist["strict"] = hist["target_hit"].astype(bool) & hist["dd_pass"].astype(bool) & ~hist["loss_fold"].astype(bool)
    scores: dict[str, float] = {}
    for label, group in hist.groupby("candidate_label"):
        if len(group) < int(args.min_history_rows):
            continue
        strict_rate = float(group["strict"].mean())
        target_rate = float(group["target_hit"].astype(bool).mean())
        dd_rate = float(group["dd_pass"].astype(bool).mean())
        loss_rate = float(group["loss_fold"].astype(bool).mean())
        median_final = float(pd.to_numeric(group["final_balance"], errors="coerce").median())
        worst_dd = float(pd.to_numeric(group["max_dd_pct"], errors="coerce").min())
        score = (
            strict_rate * float(args.strict_weight)
            + target_rate * float(args.target_weight)
            + dd_rate * float(args.dd_weight)
            - loss_rate * float(args.loss_weight)
            + (median_final / float(args.target_balance)) * float(args.median_final_weight)
            + (worst_dd / 20.0) * float(args.worst_dd_weight)
        )
        scores[str(label)] = float(score)
    return scores


def parse_score_overrides(text: str | None) -> dict[str, float]:
    result: dict[str, float] = {}
    if not text:
        return result
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"score override must be LABEL=SCORE, got {item!r}")
        label, score = item.split("=", 1)
        result[label.strip()] = float(score.strip())
    return result


def row_score(row: pd.Series, prior_score: float, args: argparse.Namespace) -> float:
    probability = float(row.get("signal_probability") or 0.0)
    atr = float(row.get("signal_atr") or 0.0)
    lag = float(row.get("signal_match_lag_min") or 0.0)
    return (
        prior_score
        + probability * float(args.probability_weight)
        + min(atr, float(args.atr_cap)) / max(float(args.atr_cap), 1e-9) * float(args.atr_weight)
        + min(float(row.get("rolling_density") or 0.0), float(args.rolling_density_cap))
        / max(float(args.rolling_density_cap), 1e-9)
        * float(args.rolling_density_weight)
        - max(0.0, lag - float(args.max_free_lag_min)) * float(args.lag_penalty)
    )


def simulate_fold(all_rows: pd.DataFrame, stress_folds: pd.DataFrame, fold_id: int, args: argparse.Namespace) -> dict[str, Any]:
    rows = all_rows[all_rows["fold"].astype(int) == int(fold_id)].copy()
    if rows.empty:
        return {
            "fold": fold_id,
            "trades_available": 0,
            "trades_used": 0,
            "final_balance": float(args.deposit),
            "max_dd_pct": 0.0,
            "target_hit": False,
            "dd_pass": True,
            "loss_fold": False,
            "selected_candidates": "",
        }

    scores = history_scores(stress_folds, fold_id, args)
    overrides = parse_score_overrides(args.candidate_score_overrides)
    if scores:
        allowed_labels = set(scores).union(overrides)
        rows = rows[rows["candidate_label"].isin(allowed_labels)].copy()
    elif args.bootstrap_labels:
        allowed = {x.strip() for x in str(args.bootstrap_labels).split(",") if x.strip()}
        rows = rows[rows["candidate_label"].isin(allowed)].copy()
    if rows.empty:
        return {
            "fold": fold_id,
            "trades_available": 0,
            "trades_used": 0,
            "final_balance": float(args.deposit),
            "max_dd_pct": 0.0,
            "target_hit": False,
            "dd_pass": True,
            "loss_fold": False,
            "selected_candidates": "",
        }

    if float(args.rolling_density_weight) != 0.0 or int(args.min_rolling_density) > 0:
        rows = rows.sort_values(["candidate_label", "open_time"]).copy()
        density_frames: list[pd.DataFrame] = []
        window = pd.Timedelta(hours=float(args.rolling_density_hours))
        for _, group in rows.groupby("candidate_label", sort=False):
            group = group.sort_values("open_time").copy()
            times = group["open_time"].tolist()
            counts: list[int] = []
            left = 0
            for right, ts in enumerate(times):
                while left < right and times[left] < ts - window:
                    left += 1
                counts.append(right - left + 1)
            group["rolling_density"] = counts
            density_frames.append(group)
        rows = pd.concat(density_frames, ignore_index=True)
        if int(args.min_rolling_density) > 0:
            rows = rows[rows["rolling_density"] >= int(args.min_rolling_density)].copy()
    else:
        rows["rolling_density"] = 0

    rows["prior_score"] = rows["candidate_label"].map(scores).fillna(float(args.bootstrap_score))
    if overrides:
        rows["prior_score"] = rows.apply(
            lambda row: float(row["prior_score"]) + float(overrides.get(str(row["candidate_label"]), 0.0)),
            axis=1,
        )
    rows["arb_score"] = rows.apply(lambda row: row_score(row, float(row["prior_score"]), args), axis=1)
    if args.min_row_score is not None:
        rows = rows[rows["arb_score"] >= float(args.min_row_score)].copy()

    rows = rows.sort_values(["open_time", "arb_score", "signal_probability"], ascending=[True, False, False])
    balance = float(args.deposit)
    curve = [balance]
    dd_floor = float(args.deposit) * (1.0 - float(args.max_dd_pct) / 100.0)
    open_until = pd.Timestamp.min
    trades_used = 0
    total_extra_cost = 0.0
    selected: list[str] = []
    hit_target = False
    hit_dd = False
    candidate_pnl: dict[str, float] = {}
    candidate_loss_streak: dict[str, int] = {}
    blacklisted: set[str] = set()
    paper_pnl: dict[str, float] = {}
    paper_wins: dict[str, int] = {}
    paper_trades: dict[str, int] = {}
    paper_loss_streak: dict[str, int] = {}
    pending_paper: list[tuple[pd.Timestamp, str, float]] = []
    fold_start_time = rows["open_time"].min()
    locked_candidates: set[str] | None = None

    for open_time, slot in rows.groupby("open_time", sort=True):
        if pd.isna(open_time) or open_time < open_until:
            continue
        if pending_paper:
            still_pending: list[tuple[pd.Timestamp, str, float]] = []
            for close_time, label, pnl in pending_paper:
                if pd.notna(close_time) and close_time <= open_time:
                    paper_trades[label] = paper_trades.get(label, 0) + 1
                    paper_pnl[label] = paper_pnl.get(label, 0.0) + pnl
                    if pnl > 0:
                        paper_wins[label] = paper_wins.get(label, 0) + 1
                        paper_loss_streak[label] = 0
                    else:
                        paper_loss_streak[label] = paper_loss_streak.get(label, 0) + 1
                else:
                    still_pending.append((close_time, label, pnl))
            pending_paper = still_pending
        for _, paper_row in slot.iterrows():
            paper_label = str(paper_row["candidate_label"])
            paper_close = paper_row["close_time"]
            paper_net = float(paper_row["profit"]) - extra_cost(paper_row, args)
            if pd.notna(paper_close):
                pending_paper.append((paper_close, paper_label, paper_net))
        warmup_done = True
        if float(args.paper_warmup_hours) > 0 and pd.notna(fold_start_time):
            warmup_done = open_time >= fold_start_time + pd.Timedelta(hours=float(args.paper_warmup_hours))
        if int(args.paper_warmup_min_closed) > 0:
            warmup_done = warmup_done and sum(paper_trades.values()) >= int(args.paper_warmup_min_closed)
        if not warmup_done:
            continue
        if int(args.lock_top_paper_candidates) > 0 and locked_candidates is None:
            ranked_labels = sorted(
                paper_trades,
                key=lambda label: (
                    paper_pnl.get(label, 0.0),
                    paper_wins.get(label, 0) / max(paper_trades.get(label, 1), 1),
                    paper_trades.get(label, 0),
                ),
                reverse=True,
            )
            ranked_labels = [
                label
                for label in ranked_labels
                if paper_trades.get(label, 0) >= int(args.paper_min_trades)
                and paper_pnl.get(label, 0.0) >= float(args.paper_lock_min_pnl)
            ]
            if not ranked_labels:
                continue
            locked_candidates = set(ranked_labels[: int(args.lock_top_paper_candidates)])
        eligible = slot[~slot["candidate_label"].astype(str).isin(blacklisted)].copy()
        if locked_candidates is not None:
            eligible = eligible[eligible["candidate_label"].astype(str).isin(locked_candidates)].copy()
        if eligible.empty:
            continue
        if float(args.current_pnl_weight) != 0.0:
            eligible["arb_score"] = eligible.apply(
                lambda row: float(row["arb_score"])
                + candidate_pnl.get(str(row["candidate_label"]), 0.0) / float(args.deposit) * float(args.current_pnl_weight),
                axis=1,
            )
        if (
            float(args.paper_pnl_weight) != 0.0
            or float(args.paper_winrate_weight) != 0.0
            or float(args.paper_loss_streak_penalty) != 0.0
        ):
            def add_paper_score(row: pd.Series) -> float:
                label = str(row["candidate_label"])
                trades = paper_trades.get(label, 0)
                score = float(row["arb_score"])
                if trades >= int(args.paper_min_trades):
                    score += paper_pnl.get(label, 0.0) / float(args.deposit) * float(args.paper_pnl_weight)
                    score += (paper_wins.get(label, 0) / max(trades, 1)) * float(args.paper_winrate_weight)
                    score -= paper_loss_streak.get(label, 0) * float(args.paper_loss_streak_penalty)
                return score

            eligible["arb_score"] = eligible.apply(add_paper_score, axis=1)
        best = eligible.sort_values(["arb_score", "signal_probability"], ascending=[False, False]).iloc[0]
        adjusted_profit = float(best["profit"]) - extra_cost(best, args)
        label = str(best["candidate_label"])
        balance += adjusted_profit
        curve.append(balance)
        trades_used += 1
        total_extra_cost += extra_cost(best, args)
        selected.append(label)
        candidate_pnl[label] = candidate_pnl.get(label, 0.0) + adjusted_profit
        if adjusted_profit < 0:
            candidate_loss_streak[label] = candidate_loss_streak.get(label, 0) + 1
        else:
            candidate_loss_streak[label] = 0
        if float(args.blacklist_candidate_loss_usd) > 0 and candidate_pnl[label] <= -float(args.blacklist_candidate_loss_usd):
            blacklisted.add(label)
        if int(args.blacklist_candidate_loss_streak) > 0 and candidate_loss_streak[label] >= int(args.blacklist_candidate_loss_streak):
            blacklisted.add(label)
        close_time = best["close_time"]
        if pd.notna(close_time):
            open_until = max(open_until, close_time)

        if balance <= dd_floor:
            hit_dd = True
            if args.stop_on_dd:
                break
        if float(args.stop_on_peak_dd_pct) > 0 and max_drawdown_pct(curve) <= -float(args.stop_on_peak_dd_pct):
            break
        if balance >= float(args.target_balance):
            hit_target = True
            if args.stop_on_target:
                break

    worst_dd = max_drawdown_pct(curve)
    return {
        "fold": int(fold_id),
        "trades_available": int(len(rows)),
        "trades_used": int(trades_used),
        "final_balance": round(balance, 2),
        "net_pct": round((balance - float(args.deposit)) / float(args.deposit) * 100.0, 2),
        "max_dd_pct": round(worst_dd, 2),
        "target_hit": bool(hit_target or balance >= float(args.target_balance)),
        "dd_pass": bool(worst_dd >= -float(args.max_dd_pct) and not hit_dd),
        "loss_fold": bool(balance < float(args.deposit)),
        "total_extra_cost": round(total_extra_cost, 2),
        "selected_candidates": ",".join(pd.Series(selected).value_counts().head(5).index.astype(str).tolist()),
    }


def summarize(folds: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    frame = pd.DataFrame(folds)
    strict = frame["target_hit"].astype(bool) & frame["dd_pass"].astype(bool) & ~frame["loss_fold"].astype(bool)
    return {
        "summary": {
            "folds": int(len(frame)),
            "all_pass_folds": int(strict.sum()),
            "target_pass_folds": int(frame["target_hit"].sum()),
            "dd_pass_folds": int(frame["dd_pass"].sum()),
            "loss_folds": int(frame["loss_fold"].sum()),
            "min_final_balance": round(float(frame["final_balance"].min()), 2),
            "median_final_balance": round(float(frame["final_balance"].median()), 2),
            "worst_dd_pct": round(float(frame["max_dd_pct"].min()), 2),
            "total_trades_used": int(frame["trades_used"].sum()),
            "gate_pass": bool(strict.all()),
            "fail_folds": frame.loc[~strict, "fold"].astype(int).tolist(),
        },
        "folds": folds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate clean online candidate arbitration from prior MT5 feedback.")
    parser.add_argument("--candidate-dir", action="append", type=Path, required=True)
    parser.add_argument("--stress-folds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold-report-out", type=Path, default=None)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=30)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--history-lookback", type=int, default=8)
    parser.add_argument("--min-history-rows", type=int, default=3)
    parser.add_argument("--bootstrap-labels", default="")
    parser.add_argument("--bootstrap-score", type=float, default=0.0)
    parser.add_argument("--candidate-score-overrides", default="")
    parser.add_argument("--strict-weight", type=float, default=8.0)
    parser.add_argument("--target-weight", type=float, default=4.0)
    parser.add_argument("--dd-weight", type=float, default=3.0)
    parser.add_argument("--loss-weight", type=float, default=10.0)
    parser.add_argument("--median-final-weight", type=float, default=1.0)
    parser.add_argument("--worst-dd-weight", type=float, default=2.0)
    parser.add_argument("--probability-weight", type=float, default=1.0)
    parser.add_argument("--atr-weight", type=float, default=0.0)
    parser.add_argument("--atr-cap", type=float, default=3.0)
    parser.add_argument("--rolling-density-hours", type=float, default=72.0)
    parser.add_argument("--rolling-density-cap", type=float, default=30.0)
    parser.add_argument("--rolling-density-weight", type=float, default=0.0)
    parser.add_argument("--min-rolling-density", type=int, default=0)
    parser.add_argument("--max-free-lag-min", type=float, default=5.0)
    parser.add_argument("--lag-penalty", type=float, default=0.0)
    parser.add_argument("--min-row-score", type=float, default=None)
    parser.add_argument("--stop-on-peak-dd-pct", type=float, default=0.0)
    parser.add_argument("--current-pnl-weight", type=float, default=0.0)
    parser.add_argument("--blacklist-candidate-loss-usd", type=float, default=0.0)
    parser.add_argument("--blacklist-candidate-loss-streak", type=int, default=0)
    parser.add_argument("--paper-pnl-weight", type=float, default=0.0)
    parser.add_argument("--paper-winrate-weight", type=float, default=0.0)
    parser.add_argument("--paper-loss-streak-penalty", type=float, default=0.0)
    parser.add_argument("--paper-min-trades", type=int, default=5)
    parser.add_argument("--paper-warmup-hours", type=float, default=0.0)
    parser.add_argument("--paper-warmup-min-closed", type=int, default=0)
    parser.add_argument("--lock-top-paper-candidates", type=int, default=0)
    parser.add_argument("--paper-lock-min-pnl", type=float, default=0.0)
    parser.add_argument("--extra-roundtrip-points", type=float, default=0.0)
    parser.add_argument("--thin-hours-utc", default="99")
    parser.add_argument("--thin-hour-extra-points", type=float, default=0.0)
    parser.add_argument("--friday-cutoff-hour-utc", type=int, default=20)
    parser.add_argument("--friday-extra-points", type=float, default=0.0)
    parser.add_argument("--point-size", type=float, default=0.001)
    parser.add_argument("--contract-size", type=float, default=100.0)
    parser.add_argument("--commission-per-lot-roundtrip", type=float, default=0.10)
    parser.add_argument("--no-stop-on-target", dest="stop_on_target", action="store_false")
    parser.add_argument("--no-stop-on-dd", dest="stop_on_dd", action="store_false")
    parser.set_defaults(stop_on_target=True, stop_on_dd=True)
    args = parser.parse_args()

    frames = [load_feedback(path) for path in args.candidate_dir]
    all_rows = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    stress_folds = pd.read_csv(args.stress_folds)
    for column in ["fold", "final_balance", "max_dd_pct"]:
        if column in stress_folds.columns:
            stress_folds[column] = pd.to_numeric(stress_folds[column], errors="coerce")

    folds = [
        simulate_fold(all_rows, stress_folds, fold_id, args)
        for fold_id in range(int(args.fold_start), int(args.fold_end) + 1)
    ]
    report = summarize(folds, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.fold_report_out:
        pd.DataFrame(folds).to_csv(args.fold_report_out, index=False)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
