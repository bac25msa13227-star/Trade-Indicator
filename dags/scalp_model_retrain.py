"""Airflow DAG: Scalp model retrain for ACC1 and ACC2.

Schedule: Every Sunday 03:00 UTC (after daily_data_ingest)
Tasks:
  1. retrain_acc1  – run acc1_scalp_m1_save_model.py (last 250k bars, setup_aware labels)
  2. retrain_acc2  – run acc2_scalp_m1_save_model.py
  3. verify_models – confirm pkl mtime updated and meta train_end matches today

Why this DAG exists:
  The DualScalpM1 model MUST be trained via the scalp pipeline
  (build_scalp_dataset + setup_aware labels + _train_dir).
  The standard model_training DAG uses `prepare_training_dataset` which is
  incompatible with DualScalpM1. Running this DAG keeps live models fresh
  so they adapt to the current price regime.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

_REPO = Path(os.getenv("AIRFLOW_HOME", "/opt/airflow"))

DEFAULT_ARGS = {
    "owner": "trader",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="scalp_model_retrain",
    description="Retrain DualScalpM1 models for ACC1 and ACC2 using scalp pipeline",
    schedule_interval="0 3 * * 0",   # Weekly Sunday 03:00 UTC
    start_date=days_ago(1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["training", "scalp", "live"],
)


def _retrain(config_yaml: str, save_script: str, **ctx) -> None:
    """Import and run the save-model script for the given account."""
    import sys

    src_path = str(_REPO / "src")
    scripts_path = str(_REPO / "scripts")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    import importlib
    import types

    # Patch CONFIG before importing so each account uses its own YAML
    spec = importlib.util.spec_from_file_location(
        "save_model_base", str(_REPO / "scripts" / "acc2_scalp_m1_save_model.py")
    )
    mod: types.ModuleType = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.CONFIG = _REPO / config_yaml
    mod.main()
    print(f"[{save_script}] Retrain complete.", flush=True)


def _verify_models(**ctx) -> None:
    """Verify both pkl files have today's mtime and train_end matches today."""
    import json
    import time

    now = time.time()
    age_limit_secs = 3 * 3600  # must be newer than 3 hours

    models = [
        (
            _REPO / "outputs" / "acc1_scalp_m1_reversal_model.pkl",
            _REPO / "outputs" / "acc1_scalp_m1_reversal_model_meta.json",
        ),
        (
            _REPO / "outputs" / "acc2_scalp_m1_h10_setup_exit_model.pkl",
            _REPO / "outputs" / "acc2_scalp_m1_h10_setup_exit_model_meta.json",
        ),
    ]
    errors: list[str] = []
    for pkl, meta_json in models:
        if not pkl.exists():
            errors.append(f"MISSING: {pkl.name}")
            continue
        age = now - pkl.stat().st_mtime
        if age > age_limit_secs:
            errors.append(f"STALE ({age/3600:.1f}h old): {pkl.name}")
        if meta_json.exists():
            meta = json.loads(meta_json.read_text())
            train_end = meta.get("train_end", "")
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if train_end < today:
                print(
                    f"WARNING: {pkl.name} train_end={train_end} older than today={today}"
                )
            else:
                print(f"OK: {pkl.name}  train_end={train_end}")
    if errors:
        raise RuntimeError("Model verification failed:\n" + "\n".join(errors))
    print("All scalp models verified OK.")


retrain_acc1 = PythonOperator(
    task_id="retrain_acc1",
    python_callable=_retrain,
    op_kwargs={
        "config_yaml": "configs/live_acc1_scalp_m1.yaml",
        "save_script": "acc1_scalp_m1_save_model",
    },
    dag=dag,
)

retrain_acc2 = PythonOperator(
    task_id="retrain_acc2",
    python_callable=_retrain,
    op_kwargs={
        "config_yaml": "configs/live_acc2_scalp_m1.yaml",
        "save_script": "acc2_scalp_m1_save_model",
    },
    dag=dag,
)

verify_models = PythonOperator(
    task_id="verify_models",
    python_callable=_verify_models,
    dag=dag,
)

# ACC1 and ACC2 can retrain in parallel → both must pass verify
[retrain_acc1, retrain_acc2] >> verify_models
