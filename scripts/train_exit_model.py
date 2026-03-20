"""
Train the exit model.

Usage:
    python scripts/train_exit_model.py --config configs/live_ict_wyckoff.yaml
    python scripts/train_exit_model.py --config configs/live_acc2.yaml --output-suffix acc2
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root so imports work when run from workspace root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xauusd_ai.config import load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.features.dataset import prepare_training_dataset
from xauusd_ai.features.exit_dataset import build_exit_dataset
from xauusd_ai.model.exit_model import ExitModel
from xauusd_ai.model.trainer import ModelTrainer
from xauusd_ai.strategies.hybrid import HybridStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("train_exit_model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ExitModel")
    parser.add_argument("--config", required=True, help="Path to YAML settings file")
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix appended to model output filenames (e.g. 'acc2')",
    )
    args = parser.parse_args()

    settings = load_settings(Path(args.config))

    # Apply optional output suffix so two accounts can have separate models
    if args.output_suffix:
        sfx = args.output_suffix.lstrip("_")
        exit_cfg = settings.execution.exit_model
        exit_cfg.exit_model_path  = f"outputs/exit_model_{sfx}.pkl"
        exit_cfg.exit_scaler_path = f"outputs/exit_scaler_{sfx}.pkl"
        exit_cfg.exit_model_meta_path = f"outputs/exit_model_meta_{sfx}.json"
        LOGGER.info("Using suffix '%s' → outputs/exit_model_%s.pkl", sfx, sfx)

    # ── 1. Build entry dataset ────────────────────────────────────────
    LOGGER.info("Loading market data (source: %s) …", settings.market.training_data_source)
    data_service = MarketDataService(settings)
    strategy = HybridStrategy(settings)
    frames = data_service.fetch_multi_timeframe_data(
        source=settings.market.training_data_source, all_bars=True
    )
    entry_dataset = prepare_training_dataset(settings, frames, strategy)
    LOGGER.info("Entry dataset: %d rows", len(entry_dataset))

    # ── 2. Build exit dataset ─────────────────────────────────────────
    LOGGER.info("Building exit dataset …")
    exit_dataset = build_exit_dataset(entry_dataset, settings)
    LOGGER.info("Exit dataset: %d rows", len(exit_dataset))

    if exit_dataset.empty:
        LOGGER.error("Exit dataset is empty — aborting.")
        sys.exit(1)

    should_exit_rate = exit_dataset["should_exit"].mean() * 100
    LOGGER.info("should_exit rate: %.1f%%", should_exit_rate)

    # ── 3. Train ──────────────────────────────────────────────────────
    model = ExitModel(settings)
    metrics = model.train(exit_dataset)

    # ── 4. Save ───────────────────────────────────────────────────────
    model.save()

    # ── 5. Print summary ──────────────────────────────────────────────
    print("\n" + "=" * 54)
    print("  EXIT MODEL TRAINING COMPLETE")
    print("=" * 54)
    print(f"  Train rows       : {metrics['train_rows']:,}")
    print(f"  Val   rows       : {metrics['val_rows']:,}")
    print(f"  Positive rate    : {metrics['positive_rate']:.1%}")
    print(f"  Opt threshold    : {metrics['threshold']:.2f}")
    print(f"  Val precision    : {metrics['val_precision']:.3f}")
    print(f"  Val recall       : {metrics['val_recall']:.3f}")
    print(f"  Val ROC-AUC      : {metrics['val_roc_auc']:.4f}")
    print(f"  Features used    : {len(metrics['feature_columns'])}")
    print("=" * 54)
    print()

    exit_cfg = settings.execution.exit_model
    print("To enable live exit model, set in your YAML config:")
    print()
    print("execution:")
    print("  exit_model:")
    print("    enabled: true")
    print(f"    exit_model_path: \"{exit_cfg.exit_model_path}\"")
    print(f"    exit_scaler_path: \"{exit_cfg.exit_scaler_path}\"")
    print(f"    exit_model_meta_path: \"{exit_cfg.exit_model_meta_path}\"")
    print(f"    exit_threshold: {metrics['threshold']:.2f}")
    print()


if __name__ == "__main__":
    main()
