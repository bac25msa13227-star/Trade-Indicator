"""
Exit Model Impact Analysis — ACC2

Simulates every test-split trade bar-by-bar with the trained exit model,
and compares per-trade outcomes against the baseline (no exit model).

Outcomes classified per trade:
  EARLY_WIN   – exit model fires while trade is in profit  (model says exit, trade was going to TP)
              → subdivided: SMART_EXIT (pulled back after) vs PREMATURE (would have hit TP)
  EARLY_CUT   – exit model fires while trade is in loss   (early stop-loss)
              → subdivided: SMART_CUT (saved loss vs SL) vs BAD_CUT (would have recovered)
  NATURAL_TP  – exit model never fires, trade hits TP naturally
  NATURAL_SL  – exit model never fires, trade hits SL naturally

Run:
    $PY = "...python.exe"
    & $PY scripts/analyze_exit_model_acc2.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import logging

from xauusd_ai.config import load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.features.dataset import prepare_training_dataset
from xauusd_ai.model.exit_model import ExitModel
from xauusd_ai.strategies.hybrid import HybridStrategy
from xauusd_ai.features.exit_dataset import EXIT_EXTRA_FEATURES, EXIT_FEATURE_COLUMNS
from xauusd_ai.features.dataset import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger("analyze_exit_model")


def main() -> None:
    settings = load_settings(Path("configs/live_acc2.yaml"))
    # Point to acc2-specific model artifacts
    settings.execution.exit_model.exit_model_path       = "outputs/exit_model_acc2.pkl"
    settings.execution.exit_model.exit_scaler_path      = "outputs/exit_scaler_acc2.pkl"
    settings.execution.exit_model.exit_model_meta_path  = "outputs/exit_model_meta_acc2.json"

    # Load market data and build dataset
    LOGGER.info("Loading market data …")
    data_service = MarketDataService(settings)
    strategy     = HybridStrategy(settings)
    frames       = data_service.fetch_multi_timeframe_data(
        source=settings.market.training_data_source, all_bars=True
    )
    dataset = prepare_training_dataset(settings, frames, strategy)
    LOGGER.info("Dataset: %d rows | test rows: %d", len(dataset), (dataset["split"] == "test").sum())

    # Load trained exit model
    exit_model = ExitModel(settings)
    if not exit_model.load():
        LOGGER.error("Exit model not found — run train_exit_model.py first")
        sys.exit(1)
    LOGGER.info("Exit model loaded (thr=%.2f)", exit_model.threshold)

    # Parameters
    sl_mult  = float(settings.risk.stop_loss_atr_multiple)
    tp_rr    = float(settings.risk.take_profit_rr)
    max_hold = int(getattr(settings.training, "sltp_label_max_horizon", 32))
    min_rr   = float(settings.execution.exit_model.min_unrealized_rr)
    min_hold = int(settings.execution.exit_model.exit_min_hold_bars)

    # Simulate on TEST split
    ds    = dataset[dataset["split"] == "test"].reset_index(drop=True)
    n     = len(ds)
    entry_idx = ds[ds["target"] == 1].index.tolist()
    LOGGER.info("Test split: %d TP entries to simulate", len(entry_idx))

    closes  = ds["close"].values.astype(float)
    atrs    = ds["atr"].values.astype(float) if "atr" in ds.columns else np.ones(n)
    rsis    = ds["rsi"].values.astype(float) if "rsi" in ds.columns else np.full(n, 50.0)
    sides   = ds["trade_side"].values

    avail_feats  = [c for c in FEATURE_COLUMNS if c in ds.columns]
    feat_matrix  = ds[avail_feats].values.astype(float)

    # Map exit model's feature_columns → column indices in the combined (market + pos_state) vector
    extra_names   = EXIT_EXTRA_FEATURES
    all_names     = avail_feats + extra_names
    name_to_col   = {n_: i for i, n_ in enumerate(all_names)}
    model_col_idx = np.array([name_to_col.get(c, -1) for c in exit_model.feature_columns])
    valid_mask    = model_col_idx >= 0

    LOGGER.info("Pre-building position-bar matrix for batch prediction…")

    # Collect all position-bar rows to predict in one batch per entry
    # per_entry: list of (entry_idx, list of (bar_j, unrealized_rr))
    all_rows_list   = []  # (entry_i, bar_j, rr_at_bar)
    all_feats_list  = []  # combined feature vector

    for i in entry_idx:
        if i + max_hold >= n:
            continue
        entry_price = closes[i]
        atr_i       = max(atrs[i], 1e-6)
        sl_dist     = atr_i * sl_mult
        side_sign   = 1.0 if sides[i] == "buy" else -1.0
        rsi_entry   = float(rsis[i])

        for j in range(i + 1, min(i + max_hold + 1, n)):
            bars_held     = j - i
            current_price = closes[j]
            atr_j         = max(atrs[j], 1e-6)
            unrealized_rr = side_sign * (current_price - entry_price) / sl_dist

            if unrealized_rr <= -1.0 or unrealized_rr >= tp_rr:
                break   # natural exit — no need to predict beyond this

            if bars_held < min_hold or unrealized_rr < min_rr:
                continue   # gate: don't even collect a row

            market_feats = feat_matrix[j].tolist()
            prev_price   = closes[j - 1] if j > 0 else current_price
            prev_rr      = side_sign * (prev_price - entry_price) / sl_dist
            pos_state_vals = [
                min(bars_held, 32) / 32.0,
                float(np.clip(unrealized_rr, -2.0, tp_rr + 0.5)),
                side_sign,
                rsi_entry,
                float(rsis[j]) - rsi_entry,
                float(np.clip(side_sign * (current_price - entry_price) / atr_j, -3.0, 3.0)),
                float(np.clip(unrealized_rr / max(tp_rr, 0.1), -0.5, 1.5)),          # xm_progress_to_tp
                float(np.clip(unrealized_rr - prev_rr, -2.0, 2.0)),                  # xm_rr_momentum
                float(max(0.0, (max_hold - bars_held) / max_hold)),                   # xm_bars_remaining
            ]
            combined = market_feats + pos_state_vals
            all_rows_list.append((i, j, unrealized_rr))
            all_feats_list.append(combined)

    LOGGER.info("Batch predicting on %d candidate exit bars…", len(all_feats_list))

    if not all_feats_list:
        LOGGER.warning("No candidate exit bars collected — check min_rr/min_hold settings")
        preds_arr = np.array([])
    else:
        X_all = np.array(all_feats_list, dtype=float)
        # Select only the columns the model was trained on
        X_model = X_all[:, model_col_idx[valid_mask]]
        X_scaled = exit_model.scaler.transform(X_model)
        preds_arr = exit_model.model.predict_proba(X_scaled)[:, 1]

    LOGGER.info("Batch prediction done. Computing outcomes…")

    # Index predictions by (entry_i, bar_j)
    first_fire: dict[int, tuple[int, float, float]] = {}  # entry_i → (bar_j, rr, prob)
    for row_idx, (i, j, rr) in enumerate(all_rows_list):
        prob = float(preds_arr[row_idx]) if len(preds_arr) > 0 else 0.0
        if prob >= exit_model.threshold:
            if i not in first_fire:   # keep FIRST bar where it fires
                first_fire[i] = (j, rr, prob)

    results = []

    for i in entry_idx:
        if i + max_hold >= n:
            continue
        entry_price = closes[i]
        atr_i       = max(atrs[i], 1e-6)
        sl_dist     = atr_i * sl_mult
        side_sign   = 1.0 if sides[i] == "buy" else -1.0

        # Compute baseline final RR (no exit model)
        final_rr_base = None
        for j in range(i + 1, min(i + max_hold + 1, n)):
            rr = side_sign * (closes[j] - entry_price) / sl_dist
            if rr <= -1.0:
                final_rr_base = -1.0
                break
            if rr >= tp_rr:
                final_rr_base = tp_rr
                break
        if final_rr_base is None:
            final_j = min(i + max_hold, n - 1)
            trail_rr = side_sign * (closes[final_j] - entry_price) / sl_dist
            final_rr_base = float(np.clip(trail_rr, -1.0, tp_rr))

        # Exit model outcome
        exit_bar     = None
        final_rr_exit = final_rr_base
        if i in first_fire:
            exit_bar, fire_rr, fire_prob = first_fire[i]
            final_rr_exit = fire_rr

        # Classify
        if exit_bar is None:
            outcome = "NATURAL_TP" if final_rr_base >= tp_rr - 0.01 else "NATURAL_SL_OR_TIMEOUT"
        elif final_rr_exit > 0 and final_rr_base >= tp_rr - 0.01:
            outcome = "PREMATURE_EXIT"   # left TP on table
        elif final_rr_exit > 0 and final_rr_base < 0:
            outcome = "SMART_EXIT"       # saved from loss
        elif final_rr_exit > 0 and final_rr_base < final_rr_exit - 0.05:
            outcome = "SMART_EXIT"       # locked in more profit than would have gotten
        elif final_rr_exit > 0:
            outcome = "NEUTRAL_EXIT"
        elif final_rr_exit < 0 and final_rr_base < 0:
            if final_rr_exit > final_rr_base + 0.05:
                outcome = "SMART_CUT"
            else:
                outcome = "BAD_CUT"
        else:
            outcome = "NEUTRAL_EXIT"

        results.append({
            "entry_bar":     i,
            "side":          sides[i],
            "exit_bar":      exit_bar,
            "exit_prob":     round(fire_prob, 4) if exit_bar is not None else None,
            "bars_held_exit": (exit_bar - i) if exit_bar else None,
            "final_rr_base": round(final_rr_base, 4),
            "final_rr_exit": round(final_rr_exit, 4),
            "rr_delta":      round(final_rr_exit - final_rr_base, 4),
            "outcome":       outcome,
        })

    df = pd.DataFrame(results)
    if df.empty:
        print("No results to display.")
        return

    total      = len(df)
    outcome_counts = df["outcome"].value_counts()

    # Summary stats
    natural_tp    = int(outcome_counts.get("NATURAL_TP", 0))
    natural_sl    = int(outcome_counts.get("NATURAL_SL", 0))
    smart_exit    = int(outcome_counts.get("SMART_EXIT", 0))
    premature     = int(outcome_counts.get("PREMATURE_EXIT", 0))
    neutral       = int(outcome_counts.get("NEUTRAL_EXIT", 0))
    smart_cut     = int(outcome_counts.get("SMART_CUT", 0))
    bad_cut       = int(outcome_counts.get("BAD_CUT", 0))

    early_exits = smart_exit + premature + neutral

    # R:R totals
    total_rr_base = df["final_rr_base"].sum()
    total_rr_exit = df["final_rr_exit"].sum()
    delta_rr      = total_rr_exit - total_rr_base

    avg_hold_when_fired = df[df["exit_bar"].notna()]["bars_held_exit"].mean()

    print()
    print("=" * 65)
    print("  EXIT MODEL IMPACT ANALYSIS  —  ACC2  (TEST SPLIT)")
    print("=" * 65)
    print(f"  Total TP-entry trades simulated : {total:,}")
    print()
    print("  ── Without exit model (baseline) ──────────────────────────")
    print(f"  Natural TP hits          : {natural_tp:,}  ({natural_tp/total:.1%})")
    print(f"  Natural SL hits          : {natural_sl:,}  ({natural_sl/total:.1%})")  # ← shouldn't happen for target==1
    print(f"  Total R:R                : {total_rr_base:+.1f}R")
    print(f"  Avg R:R per trade        : {total_rr_base/total:+.3f}R")
    print()
    print("  ── With exit model (threshold=%.2f) ────────────────────────" % exit_model.threshold)
    print()
    print("  🟢 SMART_EXIT (exited in profit, would have pulled back)  :")
    print(f"     {smart_exit:,} trades  ({smart_exit/total:.1%})")
    rr = df[df["outcome"]=="SMART_EXIT"]["rr_delta"].mean()
    print(f"     Avg R:R improvement per trade : {rr:+.3f}R")
    print()
    print("  🔴 PREMATURE_EXIT (exited early, left TP profit on table) :")
    print(f"     {premature:,} trades  ({premature/total:.1%})")
    rr = df[df["outcome"]=="PREMATURE_EXIT"]["rr_delta"].mean()
    tp_lost = df[df["outcome"]=="PREMATURE_EXIT"]["rr_delta"].sum()
    print(f"     Avg R:R cost per trade        : {rr:+.3f}R")
    print(f"     Total R:R left on table       : {tp_lost:+.1f}R")
    print()
    print("  ⚪ NEUTRAL_EXIT (minimal ± difference)                    :")
    print(f"     {neutral:,} trades  ({neutral/total:.1%})")
    print()
    print("  ✂️  SMART_CUT (cut loss early, position would have SLed)  :")
    print(f"     {smart_cut:,} trades  ({smart_cut/total:.1%})")
    if smart_cut > 0:
        rr = df[df["outcome"]=="SMART_CUT"]["rr_delta"].mean()
        print(f"     Avg R:R saved per trade       : {rr:+.3f}R")
    print()
    print("  ⚠️  BAD_CUT (cut loss early, trade would have recovered)  :")
    print(f"     {bad_cut:,} trades  ({bad_cut/total:.1%})")
    if bad_cut > 0:
        rr = df[df["outcome"]=="BAD_CUT"]["rr_delta"].mean()
        print(f"     Avg R:R cost per trade        : {rr:+.3f}R")
    print()
    print("  ── Summary ─────────────────────────────────────────────────")
    print(f"  Total R:R baseline        : {total_rr_base:+.2f}R")
    print(f"  Total R:R with exit model : {total_rr_exit:+.2f}R")
    pct_change = (total_rr_exit - total_rr_base) / max(abs(total_rr_base), 0.001) * 100
    print(f"  Net R:R delta             : {delta_rr:+.2f}R  ({pct_change:+.1f}%)")
    print()
    print(f"  Early exit fires on       : {early_exits+smart_cut+bad_cut:,}/{total:,} trades  ({(early_exits+smart_cut+bad_cut)/total:.1%})")
    print(f"  Avg bars held when fired  : {avg_hold_when_fired:.1f} bars (out of max {max_hold})")
    print()

    verdict = ""
    if delta_rr > 0:
        verdict = "✅ EXIT MODEL IS PROFITABLE vs baseline"
    elif delta_rr < -total_rr_base * 0.03:
        verdict = "❌ EXIT MODEL HURTS PERFORMANCE — do NOT enable"
    else:
        verdict = "⚠️  EXIT MODEL IS ROUGHLY NEUTRAL"
    print(" ", verdict)
    print("=" * 65)

    # ── Threshold sweep ──────────────────────────────────────────────────────
    # For trades where the model fired (has exit_prob), simulate different thresholds
    fired_df = df[df["exit_prob"].notna()].copy()
    if len(fired_df) > 0:
        print()
        print("  Threshold sweep (optimal firing point):")
        print("  threshold | fires  |  net_delta_R  | fires_pct")
        print("  " + "-" * 52)
        best_thr, best_delta = 0.52, delta_rr
        for thr in np.arange(0.50, 0.85, 0.03):
            # Trades that would fire at this threshold
            fire_at_thr = fired_df[fired_df["exit_prob"] >= thr]
            # Trades that DON'T fire at this threshold (go to baseline)
            no_fire = df[df["exit_prob"].isna() | (df["exit_prob"] < thr)]
            # Net RR: fired trades use final_rr_exit; no-fire use final_rr_base
            rr_thr = fire_at_thr["final_rr_exit"].sum() + no_fire["final_rr_base"].sum()
            n_fires = len(fire_at_thr)
            net = rr_thr - total_rr_base
            marker = " <-- best" if net > best_delta else ""
            if net > best_delta:
                best_delta = net
                best_thr   = thr
            print(f"    {thr:.2f}    | {n_fires:5d}  | {net:+10.1f}R  | {n_fires/total:.1%}{marker}")
        print()
        print(f"  Best threshold: {best_thr:.2f}  (net delta: {best_delta:+.1f}R)")
        if best_delta > 0:
            print(f"  => Set exit_threshold: {best_thr:.2f} in live_acc2.yaml and retrain")
        else:
            print(f"  => No threshold makes this profitable — exit model not suitable for ACC2")
    # ────────────────────────────────────────────────────────────────────────

    # Save detailed results
    out_path = Path("outputs/exit_model_impact_acc2.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Detailed per-trade results saved to: {out_path}")
    print()


if __name__ == "__main__":
    main()
