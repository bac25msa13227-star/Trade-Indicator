"""
Exit model training dataset builder.

For each entry bar in the training set, simulates the position lifecycle
bar-by-bar and labels each held bar with `should_exit`.

Label definition (outcome-based, v2):
  At bar j (position entered at bar i):
    1. Look ahead across the remaining position budget (j+1 .. i+max_hold):
       will_hit_tp = price reaches tp_rr before SL within max_hold
    2. If will_hit_tp  -->  should_exit = 0  (hold, TP is coming)
    3. Else look ahead exit_label_horizon bars:
       worst_near_rr = min(unrealized_rr in next H bars)
       deterioration = unrealized_rr - worst_near_rr
       should_exit = 1 if deterioration > exit_label_threshold_rr AND unrealized_rr > 0.20

This prevents the model from firing on normal pullbacks while heading to a far TP.

Position-state features (prefix "xm_") added on top of FEATURE_COLUMNS:
  xm_bars_held          – bars since entry, normalised 0→1 (max 32 bars)
  xm_unrealized_rr      – current P/L in R units (negative = in loss)
  xm_pos_side           – +1 = buy, -1 = sell
  xm_rsi_at_entry       – RSI at the entry bar (context)
  xm_rsi_delta          – RSI_now − RSI_at_entry (momentum shift)
  xm_price_vs_entry_atr – side-signed price move from entry / current ATR
  xm_progress_to_tp     – unrealized_rr / tp_rr: how far along to TP (0..1+)
  xm_rr_momentum        – unrealized_rr[j] - unrealized_rr[j-1]: trend of profit
  xm_bars_remaining     – (max_hold - bars_held) / max_hold: time pressure
"""
from __future__ import annotations

import random
import logging

import numpy as np
import pandas as pd

from xauusd_ai.config import Settings
from xauusd_ai.features.dataset import FEATURE_COLUMNS

LOGGER = logging.getLogger(__name__)

EXIT_EXTRA_FEATURES: list[str] = [
    "xm_bars_held",
    "xm_unrealized_rr",
    "xm_pos_side",
    "xm_rsi_at_entry",
    "xm_rsi_delta",
    "xm_price_vs_entry_atr",
    "xm_progress_to_tp",    # unrealized_rr / tp_rr — how far along to TP
    "xm_rr_momentum",      # rr[j] - rr[j-1] — profit trend (positive = growing)
    "xm_bars_remaining",   # (max_hold - bars_held) / max_hold — time pressure
]

EXIT_FEATURE_COLUMNS: list[str] = FEATURE_COLUMNS + EXIT_EXTRA_FEATURES


def build_exit_dataset(
    dataset: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Generate position-lifecycle rows for exit model training.

    Parameters
    ----------
    dataset : pd.DataFrame
        Full training dataset from ``prepare_training_dataset()``.
        Must contain FEATURE_COLUMNS + close, high, low, atr, rsi,
        trade_side, target, split columns.
    settings : Settings
        Project settings (used for SL/TP params and exit_model config).

    Returns
    -------
    pd.DataFrame
        Rows with EXIT_FEATURE_COLUMNS + ``should_exit`` label column.
    """
    exit_cfg = settings.execution.exit_model
    sl_mult = float(settings.risk.stop_loss_atr_multiple)
    tp_rr = float(settings.risk.take_profit_rr)
    max_hold = int(getattr(settings.training, "sltp_label_max_horizon", 32))
    label_horizon = int(exit_cfg.exit_label_horizon)
    label_threshold = float(exit_cfg.exit_label_threshold_rr)
    max_entries = int(exit_cfg.max_entry_samples)

    # Work on training split only, reset to 0-based integer index
    ds = dataset[dataset["split"] == "train"].reset_index(drop=True)
    n = len(ds)

    # Identify entry bars (target == 1 = TP hit in training)
    tp_idx = ds[ds["target"] == 1].index.tolist()
    LOGGER.info("ExitDataset: %d TP entries available", len(tp_idx))

    if exit_cfg.include_loss_entries:
        sl_idx_all = ds[ds["target"] == 0].index.tolist()
        # Sample equal number of loss entries as TP entries (capped)
        n_loss = min(len(tp_idx), len(sl_idx_all))
        sl_idx = random.sample(sl_idx_all, n_loss) if n_loss > 0 else []
        LOGGER.info("ExitDataset: adding %d SL entries for loss-cutting learning", len(sl_idx))
        all_entry_idx = sorted(set(tp_idx + sl_idx))
    else:
        all_entry_idx = tp_idx

    # Sub-sample if too many entries (keep dataset size manageable)
    if len(all_entry_idx) > max_entries:
        random.seed(42)
        all_entry_idx = sorted(random.sample(all_entry_idx, max_entries))

    LOGGER.info(
        "ExitDataset: building from %d entry bars (max_hold=%d label_H=%d)",
        len(all_entry_idx), max_hold, label_horizon,
    )

    # Pre-extract arrays for speed
    closes = ds["close"].values.astype(float)
    atrs = ds["atr"].values.astype(float) if "atr" in ds.columns else np.full(n, 1.0)
    rsis = ds["rsi"].values.astype(float) if "rsi" in ds.columns else np.full(n, 50.0)
    sides = ds["trade_side"].values  # "buy" / "sell"

    # Pre-extract feature matrix for bar-level lookup
    avail_feats = [c for c in FEATURE_COLUMNS if c in ds.columns]
    feat_matrix = ds[avail_feats].values.astype(float)
    feat_names = avail_feats

    exit_rows: list[dict] = []

    for i in all_entry_idx:
        # Must have enough future bars for label lookahead
        if i + max_hold + label_horizon >= n:
            continue

        entry_price = closes[i]
        atr_i = max(atrs[i], 1e-6)
        sl_dist = atr_i * sl_mult
        side_sign = 1.0 if sides[i] == "buy" else -1.0
        rsi_at_entry = float(rsis[i])

        prev_unrealized_rr: float = 0.0  # track previous bar's rr for momentum

        for j in range(i + 1, min(i + max_hold + 1, n - label_horizon)):
            current_price = closes[j]
            atr_j = max(atrs[j], 1e-6)
            bars_held = j - i

            # Unrealized RR at this bar
            unrealized_rr = side_sign * (current_price - entry_price) / sl_dist

            # Natural position exit conditions — stop simulating
            if unrealized_rr <= -1.0:
                prev_unrealized_rr = unrealized_rr
                break   # SL hit
            if unrealized_rr >= tp_rr:
                prev_unrealized_rr = unrealized_rr
                break   # TP hit

            # ── Outcome-based label (v2) ──────────────────────────────────────
            # Step 1: will TP be hit before SL within remaining position budget?
            long_end = min(i + max_hold + 1, n)
            will_hit_tp = any(
                side_sign * (closes[k] - entry_price) / sl_dist >= tp_rr
                for k in range(j + 1, long_end)
            )

            if will_hit_tp:
                # TP is coming — don't exit, just hold
                should_exit = 0
            else:
                # TP won't be hit. Assess short-term deterioration risk.
                look_end = min(j + label_horizon + 1, n)
                future_rrs_short = [
                    side_sign * (closes[k] - entry_price) / sl_dist
                    for k in range(j + 1, look_end)
                ]
                if not future_rrs_short:
                    prev_unrealized_rr = unrealized_rr
                    continue
                worst_near_rr = min(future_rrs_short)
                deterioration = unrealized_rr - worst_near_rr
                # Only suggest exit if currently in meaningful profit
                should_exit = 1 if (deterioration > label_threshold and unrealized_rr > 0.20) else 0

            # Build row: market features at bar j  +  position state
            row: dict = {}
            for fi, fname in enumerate(feat_names):
                row[fname] = float(feat_matrix[j, fi])

            row["xm_bars_held"] = float(min(bars_held, 32)) / 32.0
            row["xm_unrealized_rr"] = float(np.clip(unrealized_rr, -2.0, tp_rr + 0.5))
            row["xm_pos_side"] = side_sign
            row["xm_rsi_at_entry"] = rsi_at_entry
            row["xm_rsi_delta"] = float(rsis[j] - rsi_at_entry)
            row["xm_price_vs_entry_atr"] = float(
                np.clip(side_sign * (current_price - entry_price) / atr_j, -3.0, 3.0)
            )
            row["xm_progress_to_tp"] = float(np.clip(unrealized_rr / max(tp_rr, 0.1), -0.5, 1.5))
            row["xm_rr_momentum"] = float(np.clip(unrealized_rr - prev_unrealized_rr, -2.0, 2.0))
            row["xm_bars_remaining"] = float(max(0.0, (max_hold - bars_held) / max_hold))
            row["should_exit"] = should_exit
            exit_rows.append(row)
            prev_unrealized_rr = unrealized_rr

    exit_df = pd.DataFrame(exit_rows)
    if exit_df.empty:
        LOGGER.warning("ExitDataset: produced 0 rows — check dataset and settings")
        return exit_df

    pos_rate = float(exit_df["should_exit"].mean()) if len(exit_df) > 0 else 0.0
    LOGGER.info(
        "ExitDataset: %d rows built | should_exit rate=%.1f%%",
        len(exit_df), pos_rate * 100,
    )
    return exit_df
