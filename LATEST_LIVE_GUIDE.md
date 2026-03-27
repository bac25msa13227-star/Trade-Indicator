# Latest Live Guide

Last updated: `2026-03-28`

## Keep these files

Canonical live configs:

- `configs/live_acc2.yaml`
- `configs/live_acc1.yaml`
- `configs/train_acc1.yaml`
- `configs/train_ict_wyckoff_2022_2026.yaml`

Canonical live artifacts:

- `outputs/acc2_live_model.pkl`
- `outputs/acc2_live_model.cal.pkl`
- `outputs/acc2_live_scaler.pkl`
- `outputs/acc2_live_model_meta.json`
- `outputs/acc1_live_model.pkl`
- `outputs/acc1_live_model.cal.pkl`
- `outputs/acc1_live_scaler.pkl`
- `outputs/acc1_live_model_meta.json`

Dashboard/API reports still used at runtime:

- `outputs/backtest_report_acc1.json`
- `outputs/backtest_report_acc2.json`
- `outputs/walkforward_report_acc2.json`

Rollback references kept on purpose:

- `outputs/freeze_manifest_20260327.json`
- `outputs/archive_model2_weekly500_pre_freeze_20260327.pkl`
- `outputs/archive_scaler2_weekly500_pre_freeze_20260327.pkl`
- `outputs/archive_model_meta2_weekly500_pre_freeze_20260327.json`

## Thresholds in force

Runtime threshold is loaded from model meta, not just YAML.

- `outputs/acc2_live_model_meta.json` -> `0.78`
- `outputs/acc1_live_model_meta.json` -> `0.60`

If YAML and model meta disagree, the meta JSON wins during artifact loading.

## Retrain rule

No retrain is required before live startup.

Current policy:

- `retrain_on_startup: false`
- the bot starts from the frozen validated artifacts immediately
- self-learning stays enabled in the background
- only accepted candidates replace the live model
- threshold auto-optimization is disabled in live configs

So self-learning can improve weights, but it should not silently change the validated live thresholds.

## Telegram behavior

Telegram is enabled for both live accounts.

Self-learning notifications include:

- scheduled or loss-driven retrain reason
- accepted / rejected verdict
- ROC AUC delta vs current live model
- precision / recall / F1 snapshot
- short explanation of what improved or failed

## Launch commands

MT5 check:

```bash
PYTHONPATH=src python3 src/xauusd_ai/main.py mt5-check --config configs/live_acc2.yaml
PYTHONPATH=src python3 src/xauusd_ai/main.py mt5-check --config configs/live_acc1.yaml
```

Paper:

```bash
PYTHONPATH=src python3 src/xauusd_ai/main.py paper --config configs/live_acc2.yaml
PYTHONPATH=src python3 src/xauusd_ai/main.py paper --config configs/live_acc1.yaml
```

Live:

```bash
PYTHONPATH=src python3 scripts/live_runner.py live --config configs/live_acc2.yaml
PYTHONPATH=src python3 scripts/live_runner.py live --config configs/live_acc1.yaml
```

## One-line answer for future sessions

Use `configs/live_acc2.yaml` with `outputs/acc2_live_model.*` for ACC2, use `configs/live_acc1.yaml` with `outputs/acc1_live_model.*` for ACC1, do not retrain on startup, and let self-learning hot-reload only accepted candidates.
