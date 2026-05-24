#!/usr/bin/env python3
"""
decision_parity_test.py

Two validation gates:

1. engine parity:
   Compare the live model scoring path with the raw replay/WF scoring path on
   the same feature rows. This catches feature/model/scaler drift.

2. live WR parity:
   After enough real/demo closed trades, compare the live win rate with the MT5
   WF baseline win rate for the same frozen profile. This catches execution,
   spread, slippage, and broker-behavior drift that static tests cannot see.

Typical commands:
  python scripts/decision_parity_test.py --mode wr
  python scripts/decision_parity_test.py --mode wr --min-trades 30 --tolerance-points 5
  python scripts/decision_parity_test.py --mode engine --config configs/live_acc2.yaml --n-rows 500

Exit codes:
  0 = PASS
  1 = FAIL
  2 = ERROR
  3 = WAIT, not enough live trades yet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from xauusd_ai.config import load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.features.dataset import FEATURE_COLUMNS, prepare_training_dataset
from xauusd_ai.model.trainer import ModelTrainer
from xauusd_ai.strategies.hybrid import HybridStrategy

CONFIDENCE_TOL = 1e-6


def _read_json_url(url: str, timeout: int = 15) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_time_to_epoch(raw: str | None) -> float:
    if not raw:
        return 0.0
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"invalid timestamp: {raw}")
    return float(ts.timestamp())


def _net_pnl(frame: pd.DataFrame) -> pd.Series:
    if "pnl" in frame.columns:
        return pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    total = pd.Series(0.0, index=frame.index)
    for col in ("profit", "swap", "commission"):
        if col in frame.columns:
            total = total + pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return total


def _load_live_closed_trades(
    csv_path: Path,
    bridge_url: str,
    symbol: str,
    magic: int | None,
    since: str | None,
) -> pd.DataFrame:
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
    else:
        since_epoch = _parse_time_to_epoch(since)
        query = {
            "symbol": symbol,
            "since": str(since_epoch),
        }
        if magic is not None:
            query["magic"] = str(magic)
        url = bridge_url.rstrip("/") + "/history/closed?" + urllib.parse.urlencode(query)
        frame = pd.DataFrame(_read_json_url(url))

    if frame.empty:
        return frame

    frame = frame.copy()
    frame["net_pnl"] = _net_pnl(frame)
    frame["is_win"] = frame["net_pnl"] > 0

    time_col = None
    for candidate in ("close_time", "time", "open_time"):
        if candidate in frame.columns:
            time_col = candidate
            break
    if time_col:
        if pd.api.types.is_numeric_dtype(frame[time_col]):
            frame["_sort_time"] = pd.to_datetime(frame[time_col], unit="s", utc=True, errors="coerce")
        else:
            frame["_sort_time"] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
        frame = frame.sort_values("_sort_time")

    if "ticket" in frame.columns:
        frame = frame.drop_duplicates(subset=["ticket"], keep="last")

    return frame.reset_index(drop=True)


def _baseline_wr_from_mt5(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"MT5 WF baseline not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"MT5 WF baseline is empty: {path}")

    trades = pd.to_numeric(df.get("trades", 0), errors="coerce").fillna(0.0)
    tp = pd.to_numeric(df.get("tp_hits", 0), errors="coerce").fillna(0.0)
    sl = pd.to_numeric(df.get("sl_hits", 0), errors="coerce").fillna(0.0)

    resolved = tp + sl
    if resolved.sum() > 0:
        wins = float(tp.sum())
        wr = wins / float(resolved.sum())
        sample = float(resolved.sum())
    elif "win_rate_pct" in df.columns and trades.sum() > 0:
        wr_pct = pd.to_numeric(df["win_rate_pct"], errors="coerce").fillna(0.0)
        wins = float((wr_pct / 100.0 * trades).sum())
        wr = wins / float(trades.sum())
        sample = float(trades.sum())
    else:
        raise ValueError("Cannot compute baseline WR: missing tp/sl or win_rate/trades columns")

    return {
        "wr": wr,
        "sample_trades": sample,
        "folds": float(len(df)),
        "median_final": float(pd.to_numeric(df.get("final_balance", 0), errors="coerce").median()),
        "worst_dd_pct": float(pd.to_numeric(df.get("max_dd_pct", 0), errors="coerce").min()),
    }


def run_wr_parity(args: argparse.Namespace) -> int:
    live = _load_live_closed_trades(
        csv_path=args.live_trades,
        bridge_url=args.bridge_url,
        symbol=args.symbol,
        magic=args.magic,
        since=args.since,
    )
    baseline = _baseline_wr_from_mt5(args.mt5_wf_results)

    n = int(len(live))
    live_wr = float(live["is_win"].mean()) if n else 0.0
    live_pnl = float(live["net_pnl"].sum()) if n else 0.0
    live_avg = float(live["net_pnl"].mean()) if n else 0.0
    delta_points = abs(live_wr - baseline["wr"]) * 100.0

    verdict = "WAIT"
    ok = False
    if n >= args.min_trades:
        ok = delta_points <= args.tolerance_points
        verdict = "PASS" if ok else "FAIL"

    report = {
        "generated_at_epoch": time.time(),
        "mode": "live_wr_parity",
        "verdict": verdict,
        "ok": ok,
        "min_trades": args.min_trades,
        "live": {
            "trades": n,
            "wins": int(live["is_win"].sum()) if n else 0,
            "losses": int(n - live["is_win"].sum()) if n else 0,
            "wr": live_wr,
            "wr_pct": live_wr * 100.0,
            "net_pnl": live_pnl,
            "avg_pnl": live_avg,
            "source_csv": str(args.live_trades),
            "bridge_url": args.bridge_url,
            "symbol": args.symbol,
            "magic": args.magic,
        },
        "baseline": {
            "mt5_wf_results": str(args.mt5_wf_results),
            "wr": baseline["wr"],
            "wr_pct": baseline["wr"] * 100.0,
            "sample_trades": baseline["sample_trades"],
            "folds": baseline["folds"],
            "median_final": baseline["median_final"],
            "worst_dd_pct": baseline["worst_dd_pct"],
        },
        "delta_points": delta_points,
        "tolerance_points": args.tolerance_points,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 78)
    print("  LIVE WR PARITY TEST")
    print("=" * 78)
    print(f"  Live trades       : {n}/{args.min_trades}")
    print(f"  Live WR           : {live_wr * 100:.2f}%")
    print(f"  MT5 WF WR         : {baseline['wr'] * 100:.2f}%")
    print(f"  Delta             : {delta_points:.2f} points  (max {args.tolerance_points:.2f})")
    print(f"  Live net PnL      : ${live_pnl:+.2f}")
    print(f"  Report            : {args.out}")
    print("=" * 78)

    if verdict == "WAIT":
        print("  RESULT: WAIT - not enough live closed trades yet.")
        return 3
    if verdict == "FAIL":
        print("  RESULT: FAIL - live WR diverges from MT5 WF beyond tolerance.")
        return 1
    print("  RESULT: PASS - live WR is within tolerance of MT5 WF.")
    return 0


def run_engine_parity(config_path: Path, n_rows: int) -> int:
    print("=" * 78)
    print("  ENGINE DECISION PARITY TEST")
    print(f"  Config  : {config_path}")
    print(f"  N rows  : {n_rows}")
    print("=" * 78)

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2

    settings = load_settings(config_path)
    model_path = Path(settings.app.model_path)
    scaler_path = Path(settings.app.scaler_path)
    meta_path = Path(settings.app.model_meta_path)
    for p in (model_path, scaler_path, meta_path):
        if not p.exists():
            print(f"Missing artifact: {p}", file=sys.stderr)
            return 2

    trainer_a = ModelTrainer(settings)
    trainer_a.load_artifacts()

    model_b = joblib.load(model_path)
    scaler_b = joblib.load(scaler_path)
    meta_b = json.loads(meta_path.read_text(encoding="utf-8"))
    feat_mask_b = None
    if isinstance(meta_b.get("feature_mask"), list) and meta_b["feature_mask"]:
        feat_mask_b = np.asarray(meta_b["feature_mask"], dtype=bool)
    thr_b = float(
        getattr(settings.strategy, "signal_threshold", None)
        or meta_b.get("decision_threshold")
        or getattr(settings.risk, "min_confidence", 0.70)
    )

    data_service = MarketDataService(settings)
    frames = data_service.fetch_multi_timeframe_data(source="csv_folder", all_bars=True)
    strategy = HybridStrategy(settings)
    full_ds = prepare_training_dataset(settings, frames, strategy)
    window = full_ds.tail(n_rows).copy().reset_index(drop=True)

    proba_a = np.empty(len(window), dtype=float)
    pred_a = np.empty(len(window), dtype=int)
    for i in range(len(window)):
        result = trainer_a.score_live_row(window.iloc[[i]].copy())
        proba_a[i] = result["probability"]
        pred_a[i] = result["prediction"]

    x_b = scaler_b.transform(window[FEATURE_COLUMNS].values)
    if feat_mask_b is not None and len(feat_mask_b) == x_b.shape[1]:
        x_b = x_b[:, feat_mask_b]
    proba_b = model_b.predict_proba(x_b)[:, 1]
    pred_b = (proba_b >= thr_b).astype(int)

    delta = np.abs(proba_a - proba_b)
    n_conf_fail = int((delta > CONFIDENCE_TOL).sum())
    n_pred_fail = int((pred_a != pred_b).sum())
    max_delta = float(delta.max()) if len(delta) else 0.0
    mean_delta = float(delta.mean()) if len(delta) else 0.0

    print(f"  Rows tested       : {len(window)}")
    print(f"  Max conf delta    : {max_delta:.2e}")
    print(f"  Mean conf delta   : {mean_delta:.2e}")
    print(f"  Conf mismatches   : {n_conf_fail}")
    print(f"  Pred mismatches   : {n_pred_fail}")

    fail_reasons: list[str] = []
    if n_conf_fail:
        fail_reasons.append(f"confidence divergence on {n_conf_fail}/{len(window)} rows")
    if n_pred_fail:
        fail_reasons.append(f"prediction mismatch on {n_pred_fail}/{len(window)} rows")
    if trainer_a.feature_columns != list(FEATURE_COLUMNS):
        fail_reasons.append("feature column order mismatch")

    if fail_reasons:
        print("  RESULT: FAIL")
        for reason in fail_reasons:
            print(f"  - {reason}")
        return 1

    print("  RESULT: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["wr", "engine"], default="wr")
    ap.add_argument("--config", type=Path, default=Path("configs/live_acc2.yaml"))
    ap.add_argument("--n-rows", type=int, default=500)

    ap.add_argument("--live-trades", type=Path, default=Path("outputs/live_closed_trades_acc2.csv"))
    ap.add_argument(
        "--mt5-wf-results",
        type=Path,
        default=Path("outputs/mt5_1200_deposit_sweep_20260520/risk_2_0/mt5_wf_results.csv"),
    )
    ap.add_argument("--bridge-url", default="http://localhost:5601")
    ap.add_argument("--symbol", default="XAUUSDm")
    ap.add_argument("--magic", type=int, default=2505201202)
    ap.add_argument("--since", default=None, help="UTC start time for bridge history, e.g. 2026-05-20T00:00:00Z")
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--tolerance-points", type=float, default=5.0)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/acc2_demo_1200_risk2_live_check_20260520/decision_parity_report.json"),
    )
    args = ap.parse_args()

    if args.mode == "engine":
        return run_engine_parity(args.config, args.n_rows)
    return run_wr_parity(args)


if __name__ == "__main__":
    sys.exit(main())
