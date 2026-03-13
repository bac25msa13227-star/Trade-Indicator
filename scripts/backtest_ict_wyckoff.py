"""
Standalone backtest script — sử dụng model ICT+Wyckoff đã train.
Chạy toàn bộ backtest trên TEST SET 2026 và hiển thị kết quả chi tiết.
"""
import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings
warnings.filterwarnings("ignore")

import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from xauusd_ai.config import load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.dataset import prepare_training_dataset, FEATURE_COLUMNS
from xauusd_ai.model.trainer import ModelTrainer
from xauusd_ai.strategies.hybrid import HybridStrategy
from xauusd_ai.backtesting.engine import simulate_prediction_backtest

CFG = Path("configs/train_ict_wyckoff_2022_2026.yaml")
STARTING_BALANCE = 10_000.0   # USD

print("=" * 70)
print("  ICT + WYCKOFF MODEL — BACKTEST ON 2026")
print("=" * 70)
print()

settings = load_settings(CFG)
strategy  = HybridStrategy(settings)
trainer   = ModelTrainer(settings)
risk_mgr  = RiskManager(settings)

# ── 1. Load data ───────────────────────────────────────────────────────────
print("[1/4] Loading multi-timeframe CSV data...")
t0 = time.time()
svc    = MarketDataService(settings)
frames = svc.fetch_multi_timeframe_data(source="csv_folder", all_bars=True)
print(f"      M15:{len(frames['M15'])}  H4:{len(frames['H4'])}  H1:{len(frames['H1'])}  D1:{len(frames['D1'])}  ({time.time()-t0:.1f}s)")

# ── 2. Build features + labels ────────────────────────────────────────────
print("\n[2/4] Building features & labels...")
t0 = time.time()
dataset   = prepare_training_dataset(settings, frames, strategy)
train_df  = dataset[dataset["split"] == "train"]
test_df   = dataset[dataset["split"] == "test"]
print(f"      TRAIN: {len(train_df):,}  TEST: {len(test_df):,}  ({time.time()-t0:.1f}s)")

# ── 3. Train model (identical to train_ict_wyckoff.py pipeline) ──────────
print("\n[3/4] Training model (HistGBC 500, balanced, threshold-optimised)...")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, f1_score

scaler  = StandardScaler()
X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS])
X_test  = scaler.transform(test_df[FEATURE_COLUMNS])
y_train = train_df["target"].values

# threshold search on 80% of train
val_cut   = int(len(train_df) * 0.80)
opt_model = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.05, max_depth=6,
    min_samples_leaf=20, class_weight="balanced",
    early_stopping=False, random_state=42,
)
t0 = time.time()
opt_model.fit(scaler.transform(train_df.iloc[:val_cut][FEATURE_COLUMNS]),
              train_df.iloc[:val_cut]["target"].values)
val_proba = opt_model.predict_proba(scaler.transform(train_df.iloc[val_cut:][FEATURE_COLUMNS]))[:, 1]
y_val     = train_df.iloc[val_cut:]["target"].values

best_thr, best_f1 = settings.training.threshold_min, 0.0
for thr in np.arange(settings.training.threshold_min,
                     settings.training.threshold_max + settings.training.threshold_step,
                     settings.training.threshold_step):
    preds = (val_proba >= thr).astype(int)
    prec  = precision_score(y_val, preds, zero_division=0)
    f1v   = f1_score(y_val, preds, zero_division=0)
    if prec >= settings.training.min_precision_floor and f1v > best_f1:
        best_f1, best_thr = f1v, float(thr)

# final model on 100% train
final_model = HistGradientBoostingClassifier(
    max_iter=500, learning_rate=0.05, max_depth=6,
    min_samples_leaf=20, class_weight="balanced",
    early_stopping=False, random_state=42,
)
final_model.fit(X_train, y_train)
print(f"      Training done ({time.time()-t0:.1f}s)  |  threshold={best_thr:.2f}")

# attach predictions to dataset
_full_X = scaler.transform(dataset[FEATURE_COLUMNS])
dataset = dataset.copy()
dataset["probability"]  = final_model.predict_proba(_full_X)[:, 1]
dataset["prediction"]   = (dataset["probability"] >= best_thr).astype(int)

# ── 4. Simulate trades on TEST 2026 ───────────────────────────────────────
print("\n[4/4] Simulating trades on test set (2026)...")

# temporarily set threshold on trainer so BacktestEngine picks it up
trainer.model          = final_model
trainer.scaler         = scaler
trainer.decision_threshold = best_thr

result = simulate_prediction_backtest(dataset, settings, risk_mgr)
trades = result.trades
r      = result.report

# ── Display ────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("  BACKTEST RESULTS  —  TEST SET 2026")
print("=" * 70)
print()
print(f"  Period : {test_df['time'].min().date()} -> {test_df['time'].max().date()}")
print(f"  Bars   : {len(test_df):,} M15 bars  ({len(test_df)*15//60//24} trading days)")
print()

# -- Equity
ret_pct = r['return_pct']
emoji_eq = "[+]" if ret_pct >= 0 else "[-]"
print(f"  Equity")
print(f"    Starting balance : ${STARTING_BALANCE:>10,.2f}")
print(f"    Ending balance   : ${r['ending_balance']:>10,.2f}   {emoji_eq} {ret_pct:+.2f}%")
print(f"    Net profit       : ${r['net_profit']:>10,.2f}")
print(f"    Max drawdown     : {r['max_drawdown_pct']:.2f}%")
print()

# -- Trade stats
win_r = r['win_rate']
emoji_wr = "[+]" if win_r >= 0.50 else "[-]"
print(f"  Trade Statistics")
print(f"    Total trades     : {r['trades']}")
print(f"    Filtered (skipped): {r['signals_filtered_out']}")
print(f"    Wins / Losses    : {r['wins']} W  /  {r['losses']} L")
print(f"    Win rate         : {win_r:.1%}   {emoji_wr}")
print(f"    Profit factor    : {r['profit_factor']:.3f}   {'[+]' if r['profit_factor'] >= 1.0 else '[-]'}")
print(f"    Avg win          : ${r['avg_win']:>8,.2f}")
print(f"    Avg loss         : ${r['avg_loss']:>8,.2f}")
print(f"    Best trade       : ${r['best_trade']:>8,.2f}")
print(f"    Worst trade      : ${r['worst_trade']:>8,.2f}")
print(f"    Sharpe-like      : {r['sharpe_like']:.3f}")
print()

if not trades.empty:
    # -- Monthly P&L breakdown
    trades["time_dt"] = pd.to_datetime(trades["time"], utc=True)
    trades["month"]   = trades["time_dt"].dt.to_period("M").astype(str)
    monthly = (
        trades.groupby("month")
        .agg(trades_n=("pnl", "count"), pnl_sum=("pnl", "sum"),
             wins_n=("is_win", "sum"))
        .assign(win_rate=lambda df: df["wins_n"] / df["trades_n"])
        .reset_index()
    )
    print(f"  Monthly Breakdown")
    print(f"    {'Month':<10}  {'Trades':>6}  {'Win%':>6}  {'P&L':>10}")
    print(f"    {'-'*10}  {'-'*6}  {'-'*6}  {'-'*10}")
    for _, row in monthly.iterrows():
        sign = "+" if row["pnl_sum"] >= 0 else ""
        print(f"    {row['month']:<10}  {int(row['trades_n']):>6}  {row['win_rate']:>5.1%}  {sign}{row['pnl_sum']:>9,.2f}")
    print()

    # -- Side breakdown (BUY vs SELL)
    if "side" in trades.columns:
        sides = (
            trades.groupby("side")
            .agg(n=("pnl","count"), pnl=("pnl","sum"), wins=("is_win","sum"))
            .assign(wr=lambda df: df["wins"]/df["n"])
        )
        print(f"  Side Breakdown")
        for side, s in sides.iterrows():
            print(f"    {side.upper():<5}  trades={int(s['n']):>4}  win={s['wr']:.1%}  P&L=${s['pnl']:>+,.2f}")
        print()

    # -- RR distribution
    rr_bins = [-10, -0.5, 0, 0.5, 1.0, 1.5, 2.2, 10]
    rr_labels = ["<-0.5R", "-0.5-0R", "0-0.5R", "0.5-1R", "1-1.5R", "1.5-2.2R", ">2.2R"]
    trades["rr_bucket"] = pd.cut(trades["realized_rr"], bins=rr_bins, labels=rr_labels)
    rr_dist = trades["rr_bucket"].value_counts().sort_index()
    print(f"  R:R Distribution")
    for bucket, cnt in rr_dist.items():
        bar = "#" * min(cnt, 40)
        print(f"    {str(bucket):<12}  {cnt:>4}  {bar}")
    print()

    # -- Save outputs
    out_report = Path("outputs/backtest_report_ict_wyckoff.json")
    out_trades = Path("outputs/backtest_trades_ict_wyckoff.csv")
    out_report.write_text(json.dumps({
        "start": str(test_df["time"].min().date()),
        "end":   str(test_df["time"].max().date()),
        "threshold": best_thr,
        **r,
        "monthly": monthly.to_dict(orient="records"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    trades.drop(columns=["time_dt","month","rr_bucket"], errors="ignore").to_csv(out_trades, index=False)
    print(f"  Saved: {out_report}")
    print(f"  Saved: {out_trades}")

print()
print("=" * 70)
print("Done.")
