"""
Full Retrain + Backtest + Walkforward for both ACC1 and ACC2.
- Uses CSV data + self-learning cache from live sessions
- Win = PnL > 0, Loss = PnL < 0, Draw = PnL == 0 (fixed logic)
- Saves final models, runs concurrent backtest and walkforward
Usage:
    python scripts/retrain_full.py acc1   # ACC1 only
    python scripts/retrain_full.py acc2   # ACC2 only
    python scripts/retrain_full.py both   # Both (default)
"""
import os
import sys
import warnings
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
from xauusd_ai.config import load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.dataset import prepare_training_dataset, FEATURE_COLUMNS
from xauusd_ai.model.trainer import ModelTrainer
from xauusd_ai.strategies.hybrid import HybridStrategy
from xauusd_ai.visualization.reports import save_backtest_plots, save_training_plot

BANNER = "=" * 70


def _load_self_learning_frames(log_path: Path, csv_path: Path) -> dict[str, pd.DataFrame] | None:
    """
    Load pre-trained self-learning cached CSV frames (same CSVs used by SelfLearner).
    Returns a dict matching MarketDataService frame format.
    """
    if not csv_path.exists():
        return None
    tf_file_map = {
        "M1":  "XAUUSDm_M1.csv",
        "M5":  "XAUUSDm_M5.csv",
        "M15": "XAUUSDm_M15.csv",
        "M30": "XAUUSDm_M30.csv",
        "H1":  "XAUUSDm_H1.csv",
        "H4":  "XAUUSDm_H4.csv",
        "D1":  "XAUUSDm_D1.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for tf, fname in tf_file_map.items():
        fpath = csv_path / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        # Normalize column names: lowercase + map Volume→tick_volume etc.
        df.columns = [c.lower() for c in df.columns]
        rename_map = {"volume": "tick_volume", "realvolume": "real_volume"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").reset_index(drop=True)
        if "tick_volume_delta" not in df.columns:
            tv = df["tick_volume"] if "tick_volume" in df.columns else pd.Series(0, index=df.index)
            df["tick_volume_delta"] = tv.diff().fillna(0)
        if "volume_imbalance" not in df.columns:
            rng = (df["high"] - df["low"]).replace(0, float("nan"))
            df["volume_imbalance"] = (df["close"] - df["open"]).abs() / rng
            df["volume_imbalance"] = df["volume_imbalance"].fillna(0)
        if "spread_points" not in df.columns:
            df["spread_points"] = df["spread"] if "spread" in df.columns else 0
        frames[tf] = df
        print(f"    Self-learn cache: {tf} → {len(df)} rows  [{str(df['time'].min())[:10]} – {str(df['time'].max())[:10]}]")
    return frames if frames else None


def retrain_account(config_path: Path, label: str) -> dict:
    print(f"\n{BANNER}")
    print(f"  RETRAIN — {label.upper()}")
    print(f"  Config: {config_path}")
    print(BANNER)

    settings = load_settings(config_path)
    csv_folder = Path(settings.market.csv_folder_path)
    strategy = HybridStrategy(settings)
    trainer = ModelTrainer(settings)
    risk_mgr = RiskManager(settings)

    # ── 1. Load full CSV data ────────────────────────────────────────────
    print("\n[1/5] Loading CSV data (all bars)...")
    t0 = time.time()
    svc = MarketDataService(settings)
    frames = svc.fetch_multi_timeframe_data(source="csv_folder", all_bars=True)
    for tf, df in frames.items():
        print(f"    {tf}: {len(df)} bars  [{str(df['time'].min())[:10]} – {str(df['time'].max())[:10]}]")
    print(f"    Done in {time.time()-t0:.1f}s")

    # ── 2. Merge with self-learning cache ────────────────────────────────
    print("\n[2/5] Merging with self-learning session cache...")
    # Keep only the columns that MarketDataService produces so that concat
    # does not introduce extra NaN columns (e.g. 'spread', 'real_volume' from
    # raw CSVs) which would cause dataset.dropna() to drop all rows.
    _REQUIRED_COLS = ["time", "open", "high", "low", "close", "tick_volume",
                      "spread_points", "tick_volume_delta", "volume_imbalance"]
    sl_frames = _load_self_learning_frames(
        Path(settings.app.live_learning_log_path),
        csv_folder,
    )
    if sl_frames:
        for tf in list(frames.keys()):
            if tf in sl_frames and not sl_frames[tf].empty:
                base = frames[tf]
                extra = sl_frames[tf]
                # Align self-learning frame columns to match base frame
                extra_aligned = extra[[c for c in _REQUIRED_COLS if c in extra.columns]]
                combined = pd.concat([base, extra_aligned], ignore_index=True)
                combined = (
                    combined.drop_duplicates("time")
                    .sort_values("time")
                    .reset_index(drop=True)
                )
                before = len(base)
                frames[tf] = combined
                added = len(combined) - before
                print(f"    {tf}: +{added} rows merged (total {len(combined)})")
    else:
        print("    No self-learning cache found — using CSV data only")

    # ── 3. Build dataset + train ─────────────────────────────────────────
    print("\n[3/5] Building feature dataset & training...")
    t0 = time.time()
    dataset = prepare_training_dataset(settings, frames, strategy)
    train_set = dataset[dataset["split"] == "train"]
    test_set  = dataset[dataset["split"] == "test"]
    print(f"    Total rows: {len(dataset)}  Train: {len(train_set)}  Test: {len(test_set)}")
    print(f"    Label dist — Train: {train_set['target'].value_counts().to_dict()}")
    print(f"    Label dist — Test:  {test_set['target'].value_counts().to_dict()}")
    print(f"    Time range test: {str(test_set['time'].min())[:16]} – {str(test_set['time'].max())[:16]}")

    try:
        metrics = trainer.train(dataset)
    except Exception as _train_exc:
        import traceback
        _err_path = Path("outputs/retrain_err.txt")
        _err_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"    [FATAL] Training crashed: {_train_exc}")
        print(f"    Full traceback written to {_err_path}")
        raise
    print(f"    Train done in {time.time()-t0:.1f}s")
    print(f"    Precision: {metrics.get('precision', 0):.4f}  Recall: {metrics.get('recall', 0):.4f}  "
          f"F1: {metrics.get('f1', 0):.4f}  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
    print(f"    Threshold: {metrics.get('threshold', '?')}")

    # Save training plot
    try:
        save_training_plot(metrics, Path(settings.app.training_plot_path))
    except Exception:
        pass

    # ── 4. Backtest ──────────────────────────────────────────────────────
    print("\n[4/5] Running concurrent backtest on test set...")
    t0 = time.time()
    predictions = trainer.predict_dataset(dataset)
    result = simulate_dynamic_concurrent_backtest(predictions, settings, risk_mgr, compound=False)
    r = result.report
    print(f"    Done in {time.time()-t0:.1f}s")
    print(f"\n    {'Metric':<28} {'Value':>12}")
    print(f"    {'-'*42}")
    print(f"    {'Trades':<28} {r['trades']:>12}")
    print(f"    {'Wins (PnL > 0)':<28} {r['wins']:>12}")
    print(f"    {'Losses (PnL < 0)':<28} {r['losses']:>12}")
    print(f"    {'Draws (PnL = 0)':<28} {r.get('draws', 0):>12}")
    print(f"    {'Win Rate (excl. draws)':<28} {r['win_rate']:>12.2%}")
    print(f"    {'Profit Factor':<28} {r['profit_factor']:>12.4f}")
    print(f"    {'Return %':<28} {r['return_pct']:>12.2f}%")
    print(f"    {'Net Profit':<28} ${r['net_profit']:>11.2f}")
    print(f"    {'Max Drawdown':<28} {r['max_drawdown_pct']:>12.2f}%")
    print(f"    {'Avg Win':<28} ${r['avg_win']:>11.2f}")
    print(f"    {'Avg Loss':<28} ${r['avg_loss']:>11.2f}")
    print(f"    {'Sharpe-like':<28} {r['sharpe_like']:>12.4f}")
    print(f"    {'Start Balance':<28} ${r['starting_balance']:>11.2f}")
    print(f"    {'End Balance':<28} ${r['ending_balance']:>11.2f}")

    # Save backtest outputs
    try:
        backtest_report_path = Path(settings.app.backtest_report_path)
        backtest_report_path.write_text(json.dumps({**r, "train_metrics": metrics}, indent=2), encoding="utf-8")
        if not result.trades.empty:
            result.trades.to_csv(settings.app.backtest_trades_path, index=False)
        save_backtest_plots(predictions, result.trades, Path(settings.app.backtest_equity_plot_path), Path(settings.app.backtest_trades_plot_path))
    except Exception as e:
        print(f"    [warn] Saving backtest outputs: {e}")

    # ── 5. Walkforward ── 5 folds (limited) ─────────────────────────────────
    print("\n[5/5] Running walk-forward (max 5 folds)...")
    t0 = time.time()
    import subprocess
    wf_result = subprocess.run(
        [sys.executable, "scripts/walkforward_ict_wyckoff.py", str(config_path), "5"],
        capture_output=False,
        timeout=7200,   # 2-hour cap per account (17 folds × ~5 min each)
    )
    if wf_result.returncode != 0:
        print(f"    [warn] Walk-forward exited with code {wf_result.returncode}")
    else:
        print(f"    Walk-forward done in {time.time()-t0:.1f}s")

    print(f"\n{BANNER}")
    print(f"  {label.upper()} — COMPLETE")
    print(BANNER)
    return r


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

    # ACC1: use dedicated training config with 2022–2026 date-based split and
    #       wider threshold search range (threshold_min=0.55). Saves to same
    #       model paths as live_ict_wyckoff.yaml.
    # ACC2: use live config (ratio split 80/20, threshold_min=0.30).
    configs = {
        "acc1": Path("configs/train_ict_wyckoff_2022_2026.yaml"),
        "acc2": Path("configs/live_acc2.yaml"),
    }

    results = {}
    if mode in ("acc1", "both"):
        results["acc1"] = retrain_account(configs["acc1"], "ACC1 (ICT/Wyckoff — signal_threshold=0.80)")

    if mode in ("acc2", "both"):
        results["acc2"] = retrain_account(configs["acc2"], "ACC2 (Low-threshold — signal_threshold=0.30)")

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for acct, r in results.items():
        print(f"\n  {acct.upper()}:")
        print(f"    Trades: {r['trades']}  Wins: {r['wins']}  Losses: {r['losses']}  Draws: {r.get('draws',0)}")
        print(f"    Win Rate: {r['win_rate']:.2%}  PF: {r['profit_factor']:.4f}  Return: {r['return_pct']:.2f}%")
        print(f"    Max DD: {r['max_drawdown_pct']:.2f}%  Sharpe: {r['sharpe_like']:.4f}")
