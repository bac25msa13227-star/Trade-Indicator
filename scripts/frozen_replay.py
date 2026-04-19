"""
Frozen model replay: run saved pkl model on any date range.
Usage:
  python scripts/frozen_replay.py --acc acc1 --from 2026-04-14 --to 2026-04-17
  python scripts/frozen_replay.py --acc all  --from 2026-04-14 --to 2026-04-17
"""
import sys, os, warnings, argparse
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

import json, pickle
import numpy as np
import pandas as pd
from pathlib import Path

from xauusd_ai.config import load_settings
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest

# ── arg parse ───────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--acc", default="all", help="acc1 | acc2 | all")
p.add_argument("--from", dest="from_date", default="2026-04-14")
p.add_argument("--to",   dest="to_date",   default="2026-04-17")
p.add_argument("--balance", type=float, default=200.0)
p.add_argument("--cache", default=None, help="Path to cached parquet (auto-find v14pp if omitted)")
args = p.parse_args()

CONFIGS = {
    "acc1": "configs/acc1_v14pp_profit.yaml",
    "acc2": "configs/acc1_v14pp_composite.yaml",
    "v3":   "configs/acc1_v14pp_v3.yaml",
}
MODELS = {
    "acc1": ("outputs/acc1_v14pp_model.pkl", "outputs/acc1_v14pp_scaler.pkl", "outputs/acc1_v14pp_model_meta.json"),
    "acc2": ("outputs/acc2_v14pp_model.pkl", "outputs/acc2_v14pp_scaler.pkl", "outputs/acc2_v14pp_model_meta.json"),
    "v3":   ("outputs/acc1_v14pp_model.pkl", "outputs/acc1_v14pp_scaler.pkl", "outputs/acc1_v14pp_model_meta.json"),
}
LABELS = {"acc1": "ACC1_PROFIT", "acc2": "ACC2_COMPOSITE", "v3": "ACC1_V3_DAILY"}

run_accs = list(CONFIGS.keys()) if args.acc == "all" else [args.acc.lower()]
DATE_START = pd.Timestamp(args.from_date, tz="UTC")
DATE_END   = pd.Timestamp(args.to_date + " 23:59:59", tz="UTC")

print("=" * 70)
print(f"  FROZEN MODEL REPLAY  {args.from_date} -> {args.to_date}")
print("=" * 70)

# ── Load feature cache (prefer fresh v14pp parquet) ─────────────────────────
print("\n[1/3] Loading feature dataset...")
cache_dir = Path("outputs/.wf_cache")

if args.cache:
    cache_path = Path(args.cache)
else:
    # Auto-find: pick latest acc1_v14pp_profit parquet
    candidates = sorted(cache_dir.glob("acc1_v14pp_profit_*.parquet"))
    if not candidates:
        print("ERROR: No v14pp feature cache found. Run WF once with --cache first.")
        sys.exit(1)
    cache_path = candidates[-1]
    print(f"      Auto-selected cache: {cache_path.name}")

full_ds = pd.read_parquet(cache_path)
full_ds["time"] = pd.to_datetime(full_ds["time"], utc=True, errors="coerce")
print(f"      Loaded {len(full_ds):,} rows | {full_ds['time'].min().date()} -> {full_ds['time'].max().date()}")

date_ds = full_ds[(full_ds["time"] >= DATE_START) & (full_ds["time"] <= DATE_END)].copy()
print(f"      Window rows: {len(date_ds):,}")
if date_ds.empty:
    print("ERROR: No rows in selected date range.")
    sys.exit(1)

# ── Load M1 ─────────────────────────────────────────────────────────────────
print("[1b] Loading M1...")
m1 = pd.read_csv("src/xauusd_ai/real_data/XAUUSDm_M1.csv", index_col=0, parse_dates=True)
if m1.index.tz is None:
    m1.index = m1.index.tz_localize("UTC")
else:
    m1.index = m1.index.tz_convert("UTC")
m1 = m1.sort_index()
print(f"      M1: {len(m1):,} rows | last bar: {m1.index[-1]}")

# ── Run per account ──────────────────────────────────────────────────────────
print("\n[2/3] Frozen predictions + simulation...")
all_results = {}

for acc in run_accs:
    model_p, scaler_p, meta_p = MODELS[acc]
    meta = json.loads(Path(meta_p).read_text())
    feature_cols = meta["feature_columns"]
    feature_mask = np.array(meta["feature_mask"])
    threshold    = meta["decision_threshold"]
    label = LABELS[acc]

    print(f"\n  {label}  (thr={threshold:.4f})")
    with open(model_p, "rb") as f: model = pickle.load(f)
    with open(scaler_p, "rb") as f: scaler = pickle.load(f)

    X = date_ds[feature_cols].values
    X_sc = scaler.transform(X)
    X_sel = X_sc[:, feature_mask]
    proba = model.predict_proba(X_sel)[:, 1]
    preds = (proba >= threshold).astype(int)
    n_sig = int(preds.sum())
    print(f"      Signals: {n_sig} / {len(preds)} bars  ({n_sig/len(preds)*100:.1f}%)")

    sim_df = date_ds.copy()
    sim_df["split"]       = "test"
    sim_df["prediction"]  = preds
    sim_df["probability"] = proba
    sim_df["trade_side"]  = "buy"

    settings_acc = load_settings(Path(CONFIGS[acc]))
    settings_acc.training.backtest_initial_balance = args.balance
    risk_mgr = RiskManager(settings_acc)

    sim = simulate_dynamic_concurrent_backtest(sim_df, settings_acc, risk_mgr, m1_df=m1)
    r = sim.report
    all_results[label] = {"sim": sim, "report": r, "proba": proba}

    pnl = r["ending_balance"] - r["starting_balance"]
    print(f"      Balance: ${args.balance:.0f} -> ${r['ending_balance']:.2f}  (PnL {pnl:+.2f})")
    print(f"      Trades:  {r.get('n_trades', 0)}  WR: {r.get('win_rate', 0)*100:.1f}%  MaxDD: {r.get('max_dd_pct', 0)*100:.1f}%")

# ── Daily breakdown ──────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print(f"DAILY BREAKDOWN  {args.from_date} -> {args.to_date}:")
print("=" * 75)

for acc, (label, res) in zip(run_accs, all_results.items()):
    trades = res["sim"].trades
    r = res["report"]
    proba_arr = res["proba"]
    sc = load_settings(Path(CONFIGS[acc])).risk.min_confidence

    print(f"\n  {label}  (min_confidence={sc:.2f}):")
    print(f"  {'Ngay':10} | {'Bars':>5} | {'HighConf':>8} | {'MaxProba':>8} | {'Trades':>6} | {'WR':>5} | {'PnL':>9}")
    print(f"  {'-'*10}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*9}")

    dc = date_ds.copy()
    dc["proba"] = proba_arr
    dc["date"] = dc["time"].dt.date.astype(str)
    day_sig = dc.groupby("date").agg(
        bars=("proba", "count"),
        proba_max=("proba", "max"),
        high_conf=("proba", lambda x: (x >= sc).sum()),
    ).reset_index()

    if not trades.empty:
        t = trades.copy()
        t["date"] = pd.to_datetime(t["time"], utc=True, errors="coerce").dt.date.astype(str)
        td = t.groupby("date").agg(
            n=("pnl", "count"), pnl=("pnl", "sum"), w=("pnl", lambda x: (x > 0).sum())
        ).reset_index()
    else:
        td = pd.DataFrame(columns=["date", "n", "pnl", "w"])

    total_trades = 0; total_pnl = 0.0
    for _, row in day_sig.iterrows():
        dr = td[td["date"] == row["date"]]
        if not dr.empty:
            dr = dr.iloc[0]; n = int(dr["n"]); pnl_d = dr["pnl"]
            wr = int(dr["w"]) / n * 100 if n > 0 else 0
            total_trades += n; total_pnl += pnl_d
            mark = "[+]" if pnl_d > 0 else "[-]"
            pnl_str = "$%+.2f" % pnl_d
        else:
            n = 0; wr = 0; mark = "   "; pnl_str = "    -    "
        print("  %s %-10s| %5d | %8d | %8.4f | %6d | %4.0f%% | %s" % (
            mark, row["date"], int(row["bars"]), int(row["high_conf"]),
            row["proba_max"], n, wr, pnl_str))

    print(f"  {'-'*10}---{'-'*5}---{'-'*8}---{'-'*8}---{'-'*6}---{'-'*5}---{'-'*9}")
    verdict = "PROFIT" if total_pnl > 0 else ("LOSS" if total_pnl < 0 else "FLAT")
    print(f"  TOTAL: {total_trades} trades | {verdict} ${total_pnl:+.2f}  |  Balance: $200 -> ${r['ending_balance']:.2f}  MaxDD: {r.get('max_dd_pct',0)*100:.1f}%")

print()
