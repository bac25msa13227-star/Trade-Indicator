from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def load_fold_signals(run_dir: Path, broker_gmt: int) -> pd.DataFrame:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    parts: list[pd.DataFrame] = []
    for fold in manifest["folds"]:
        frame = pd.read_csv(fold["csv"])
        if frame.empty:
            continue
        frame["fold"] = int(fold["fold"])
        frame["test_start"] = fold["test_start"]
        frame["test_end"] = fold["test_end"]
        parts.append(frame)
    if not parts:
        return pd.DataFrame()
    signals = pd.concat(parts, ignore_index=True)
    signals["open_dt_broker"] = pd.to_datetime(
        signals["open_time"], format="%Y.%m.%d %H:%M", errors="coerce"
    )
    signals["time_utc"] = signals["open_dt_broker"].dt.tz_localize("UTC") - pd.Timedelta(hours=broker_gmt)
    signals["end_utc"] = pd.to_datetime(signals["test_end"], utc=True, errors="coerce")
    return signals.dropna(subset=["time_utc", "end_utc"]).sort_values(["fold", "time_utc"]).reset_index(drop=True)


def compute_outcomes(
    signals: pd.DataFrame,
    bars: pd.DataFrame,
    tp_rr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = bars["time"].astype("int64").to_numpy()
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    start_idx = np.searchsorted(times, signals["time_utc"].astype("int64").to_numpy(), side="left")
    end_idx = np.searchsorted(times, signals["end_utc"].astype("int64").to_numpy(), side="left")

    rr = np.zeros(len(signals), dtype=float)
    close_idx = np.full(len(signals), -1, dtype=int)
    for i, row in signals.iterrows():
        si = int(start_idx[i])
        ei = int(end_idx[i])
        if si >= len(times) or ei <= si:
            continue

        direction = int(row.direction)
        entry = float(row.entry_price)
        sl = float(row.sl_price)
        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            continue

        tp = entry + direction * sl_dist * tp_rr
        h = high[si:ei]
        l = low[si:ei]
        if direction == 1:
            hit_sl = l <= sl
            hit_tp = h >= tp
        else:
            hit_sl = h >= sl
            hit_tp = l <= tp

        hit = hit_sl | hit_tp
        if not hit.any():
            continue

        j = int(np.argmax(hit))
        close_idx[i] = si + j
        # Conservative same-bar tie: stop is counted before target.
        rr[i] = -1.0 if hit_sl[j] else tp_rr

    return rr, close_idx, start_idx


def simulate_fold_set(
    signals: pd.DataFrame,
    rr: np.ndarray,
    close_idx: np.ndarray,
    start_idx: np.ndarray,
    risk_pct: float,
    max_positions: int,
    max_dd_pct: float,
    target_balance: float,
    deposit: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    folds = signals["fold"].astype(int).to_numpy()
    for fold_id in sorted(set(int(x) for x in folds)):
        idx = np.where(folds == fold_id)[0]
        balance = float(deposit)
        peak = float(deposit)
        max_dd = 0.0
        open_positions: list[tuple[int, float]] = []
        trades = 0
        wins = 0
        losses = 0

        for i in idx:
            current_idx = int(start_idx[i])
            still_open: list[tuple[int, float]] = []
            for exit_idx, pnl in open_positions:
                if exit_idx <= current_idx:
                    balance += pnl
                    peak = max(peak, balance)
                    max_dd = min(max_dd, (balance - peak) / peak * 100.0 if peak > 0 else -100.0)
                    wins += int(pnl > 0)
                    losses += int(pnl < 0)
                else:
                    still_open.append((exit_idx, pnl))
            open_positions = still_open

            if len(open_positions) >= max_positions:
                continue
            if max_dd <= -max_dd_pct:
                continue
            if target_balance > 0 and balance >= target_balance:
                continue
            if rr[i] == 0.0 or close_idx[i] < 0:
                continue

            pnl = max(0.0, balance) * risk_pct * rr[i]
            open_positions.append((int(close_idx[i]), pnl))
            trades += 1

        for _, pnl in sorted(open_positions, key=lambda item: item[0]):
            balance += pnl
            peak = max(peak, balance)
            max_dd = min(max_dd, (balance - peak) / peak * 100.0 if peak > 0 else -100.0)
            wins += int(pnl > 0)
            losses += int(pnl < 0)

        rows.append(
            {
                "fold": fold_id,
                "final_balance": round(balance, 2),
                "net_pct": round((balance - float(deposit)) / float(deposit) * 100.0, 2),
                "max_dd_pct": round(max_dd, 2),
                "trades": trades,
                "wins": wins,
                "losses": losses,
            }
        )
    return pd.DataFrame(rows)


def summarize(result: pd.DataFrame) -> dict[str, float | int]:
    return {
        "target_folds": int((result["final_balance"] >= 1200.0).sum()),
        "loss_folds": int((result["net_pct"] < 0.0).sum()),
        "dd_fail_folds": int((result["max_dd_pct"] < -20.0).sum()),
        "worst_dd_pct": float(result["max_dd_pct"].min()),
        "worst_net_pct": float(result["net_pct"].min()),
        "median_final_balance": float(result["final_balance"].median()),
        "best_final_balance": float(result["final_balance"].max()),
        "total_trades": int(result["trades"].sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search the reset-$200 fold objective: final>=1200, <=1 negative fold, DD<=20%."
    )
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/mt5_wf_r2_5_tradeside")
    parser.add_argument("--features", type=Path, default=ROOT / "outputs/historical_features_2022_2026.csv")
    parser.add_argument("--tp-rrs", default="0.8,1.0,1.2,1.5,2.0,2.5,3.0,4.0,5.5")
    parser.add_argument("--risks", default="0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.05,0.06,0.08,0.10,0.12,0.15,0.20")
    parser.add_argument("--max-positions", default="1,2,3,4,5,6,7,8,10,12,15,20,30,50")
    parser.add_argument("--broker-gmt", type=int, default=3)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/target1200_signal_objective_search.csv")
    args = parser.parse_args()

    bars = pd.read_csv(args.features, usecols=["time", "high", "low"])
    bars["time"] = pd.to_datetime(bars["time"], utc=True, errors="coerce")
    bars = bars.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    signals = load_fold_signals(args.run_dir, int(args.broker_gmt))
    if signals.empty:
        raise ValueError(f"No signals found in {args.run_dir}")

    rows: list[dict[str, float | int | str]] = []
    for tp_rr in parse_float_list(args.tp_rrs):
        rr, close_idx, start_idx = compute_outcomes(signals, bars, tp_rr)
        resolved = rr != 0.0
        tp_rate = float((rr > 0.0).sum() / max(1, resolved.sum()))
        for risk_pct in parse_float_list(args.risks):
            for max_positions in parse_int_list(args.max_positions):
                result = simulate_fold_set(
                    signals=signals,
                    rr=rr,
                    close_idx=close_idx,
                    start_idx=start_idx,
                    risk_pct=risk_pct,
                    max_positions=max_positions,
                    max_dd_pct=20.0,
                    target_balance=float(args.target_balance),
                    deposit=float(args.deposit),
                )
                summary = summarize(result)
                summary.update(
                    {
                        "run_dir": str(args.run_dir),
                        "tp_rr": tp_rr,
                        "tp_rate": round(tp_rate, 4),
                        "resolved_signals": int(resolved.sum()),
                        "risk_pct": risk_pct,
                        "max_positions": max_positions,
                    }
                )
                rows.append(summary)

    out = pd.DataFrame(rows).sort_values(
        ["target_folds", "loss_folds", "dd_fail_folds", "median_final_balance"],
        ascending=[False, True, True, False],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.head(40).to_string(index=False))
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
