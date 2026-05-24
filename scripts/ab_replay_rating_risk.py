from __future__ import annotations

import argparse
import json
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_ts(value: object) -> pd.Timestamp:
    return pd.to_datetime(str(value), utc=False, errors="coerce")


def _direction_to_side(value: object) -> str:
    try:
        return "buy" if int(float(value)) > 0 else "sell"
    except Exception:
        text = str(value).strip().lower()
        return "buy" if text in {"buy", "long", "1"} else "sell"


def _rating_multiplier(probability: float | None) -> float:
    if probability is None or pd.isna(probability):
        return 1.0
    prob = float(probability)
    if prob < 0.55:
        return 0.0
    if prob < 0.70:
        return 0.60
    return 1.0


def _load_manifest(path: Path) -> dict[int, Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = Path.cwd()
    result: dict[int, Path] = {}
    for fold in manifest.get("folds", []):
        fold_id = int(fold["fold"])
        csv_path = Path(str(fold["csv"]))
        if not csv_path.is_absolute():
            csv_path = root / csv_path
        result[fold_id] = csv_path
    return result


def _load_signals(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame["signal_time"] = frame["open_time"].map(_parse_ts)
    frame["signal_side"] = frame["direction"].map(_direction_to_side)
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce")
    return frame.sort_values("signal_time").reset_index(drop=True)


def _match_signal(trade: pd.Series, signals: pd.DataFrame, max_lag_minutes: int) -> tuple[float | None, str]:
    if signals.empty:
        return None, "no_signal_file_rows"

    open_time = _parse_ts(trade["open_time"])
    side = str(trade["side"]).strip().lower()
    if pd.isna(open_time):
        return None, "bad_trade_time"

    max_lag = timedelta(minutes=max_lag_minutes)
    subset = signals[
        (signals["signal_side"] == side)
        & (signals["signal_time"] <= open_time)
        & ((open_time - signals["signal_time"]) <= max_lag)
    ]
    if subset.empty:
        return None, "no_signal_match"
    row = subset.iloc[-1]
    return float(row["probability"]), "matched"


def _replay_variant(trades: pd.DataFrame, deposit: float, variant_col: str) -> dict[str, Any]:
    balance = float(deposit)
    peak = float(deposit)
    worst_dd = 0.0
    wins = 0
    losses = 0
    active_trades = 0
    skipped_trades = 0

    for _, row in trades.iterrows():
        multiplier = float(row[variant_col])
        if multiplier <= 0:
            skipped_trades += 1
            continue
        pnl = float(row["pnl"]) * multiplier
        active_trades += 1
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
        balance += pnl
        peak = max(peak, balance)
        dd = (balance - peak) / peak * 100.0 if peak > 0 else 0.0
        worst_dd = min(worst_dd, dd)

    return {
        "final_balance": round(balance, 2),
        "net_pct": round((balance / deposit - 1.0) * 100.0, 2),
        "worst_dd_pct": round(worst_dd, 2),
        "active_trades": int(active_trades),
        "skipped_trades": int(skipped_trades),
        "wins": int(wins),
        "losses": int(losses),
        "wr_pct": round((wins / active_trades * 100.0), 2) if active_trades else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B replay baseline vs rating-risk scaling using MT5 deal replay files.")
    parser.add_argument("--deals-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--deposit", type=float, default=1200.0)
    parser.add_argument("--max-lag-minutes", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold_signal_paths = _load_manifest(args.manifest)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for deal_path in sorted(args.deals_dir.glob("fold_*_deals_replay.csv")):
        match = re.search(r"fold_(\d+)_deals_replay\.csv", deal_path.name)
        if not match:
            continue
        fold = int(match.group(1))
        signal_path = fold_signal_paths.get(fold)
        if signal_path is None or not signal_path.exists():
            continue

        trades = pd.read_csv(deal_path)
        signals = _load_signals(signal_path)
        if trades.empty:
            continue

        probs: list[float | None] = []
        statuses: list[str] = []
        multipliers: list[float] = []
        for _, trade in trades.iterrows():
            probability, status = _match_signal(trade, signals, args.max_lag_minutes)
            probs.append(probability)
            statuses.append(status)
            multipliers.append(_rating_multiplier(probability))

        trades = trades.copy()
        trades["matched_probability"] = probs
        trades["match_status"] = statuses
        trades["baseline_multiplier"] = 1.0
        trades["rating_multiplier"] = multipliers
        trades["rating_scaled_pnl"] = trades["pnl"].astype(float) * trades["rating_multiplier"].astype(float)
        trades.to_csv(args.out_dir / f"fold_{fold:03d}_ab_trades.csv", index=False)

        baseline = _replay_variant(trades, args.deposit, "baseline_multiplier")
        rating = _replay_variant(trades, args.deposit, "rating_multiplier")
        summary = {
            "fold": fold,
            "deals_file": str(deal_path),
            "signals_file": str(signal_path),
            "match_rate_pct": round((pd.Series(statuses).eq("matched").mean() * 100.0), 2),
            "matched_trades": int(pd.Series(statuses).eq("matched").sum()),
            "total_trades": int(len(trades)),
            "baseline_final": baseline["final_balance"],
            "baseline_net_pct": baseline["net_pct"],
            "baseline_worst_dd_pct": baseline["worst_dd_pct"],
            "baseline_active_trades": baseline["active_trades"],
            "baseline_wr_pct": baseline["wr_pct"],
            "rating_final": rating["final_balance"],
            "rating_net_pct": rating["net_pct"],
            "rating_worst_dd_pct": rating["worst_dd_pct"],
            "rating_active_trades": rating["active_trades"],
            "rating_skipped_trades": rating["skipped_trades"],
            "rating_wr_pct": rating["wr_pct"],
            "delta_final": round(rating["final_balance"] - baseline["final_balance"], 2),
            "delta_dd_pct": round(rating["worst_dd_pct"] - baseline["worst_dd_pct"], 2),
        }
        summaries.append(summary)
        rows.extend(trades.to_dict("records"))

    summary_df = pd.DataFrame(summaries).sort_values("fold") if summaries else pd.DataFrame()
    summary_df.to_csv(args.out_dir / "ab_rating_by_fold.csv", index=False)
    if rows:
        pd.DataFrame(rows).to_csv(args.out_dir / "ab_rating_all_trades.csv", index=False)

    report = {
        "folds": int(len(summary_df)),
        "deposit": float(args.deposit),
        "match_rate_median_pct": round(float(summary_df["match_rate_pct"].median()), 2) if not summary_df.empty else None,
        "baseline_median_final": round(float(summary_df["baseline_final"].median()), 2) if not summary_df.empty else None,
        "rating_median_final": round(float(summary_df["rating_final"].median()), 2) if not summary_df.empty else None,
        "baseline_min_final": round(float(summary_df["baseline_final"].min()), 2) if not summary_df.empty else None,
        "rating_min_final": round(float(summary_df["rating_final"].min()), 2) if not summary_df.empty else None,
        "baseline_worst_dd_pct": round(float(summary_df["baseline_worst_dd_pct"].min()), 2) if not summary_df.empty else None,
        "rating_worst_dd_pct": round(float(summary_df["rating_worst_dd_pct"].min()), 2) if not summary_df.empty else None,
        "baseline_loss_folds": int((summary_df["baseline_final"] < args.deposit).sum()) if not summary_df.empty else 0,
        "rating_loss_folds": int((summary_df["rating_final"] < args.deposit).sum()) if not summary_df.empty else 0,
        "rating_better_final_folds": int((summary_df["delta_final"] > 0).sum()) if not summary_df.empty else 0,
        "rating_worse_final_folds": int((summary_df["delta_final"] < 0).sum()) if not summary_df.empty else 0,
        "total_baseline_trades": int(summary_df["baseline_active_trades"].sum()) if not summary_df.empty else 0,
        "total_rating_active_trades": int(summary_df["rating_active_trades"].sum()) if not summary_df.empty else 0,
        "total_rating_skipped_trades": int(summary_df["rating_skipped_trades"].sum()) if not summary_df.empty else 0,
    }
    (args.out_dir / "ab_rating_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
