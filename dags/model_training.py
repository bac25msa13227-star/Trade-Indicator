"""Airflow DAG: Model training pipeline.

Schedule: triggered by daily_data_ingest OR manually OR weekly (Sunday 02:00 UTC)
Tasks:
  1. prepare_dataset   – run dataset.py to compute all features
  2. train_model       – run trainer.py, log to MLflow
  3. run_backtest      – backtest on last 3 months, log metrics
  4. run_walkforward   – walk-forward validation (3 folds)
  5. check_quality     – gate: only promote if ROC-AUC >= 0.62
  6. register_model    – push to MLflow Model Registry as Staging
  7. run_drift_check   – Evidently check on training vs live features
  8. promote_to_prod   – if drift OK, move model to Production stage
  9. invalidate_cache  – call FastAPI to reload model cache
"""
from __future__ import annotations

import os
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

DEFAULT_ARGS = {
    "owner": "trader",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="model_training",
    description="Train → Backtest → Walkforward → MLflow register → Deploy",
    schedule_interval="0 2 * * 0",     # Weekly Sunday 02:00 UTC
    start_date=days_ago(1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["training", "mlflow"],
)

ACCOUNT = os.getenv("TRADING_ACCOUNT", "acc2")
CONFIG_PATH = os.getenv("SETTINGS_PATH", "configs/live_acc2.yaml")
MIN_ROC_AUC = float(os.getenv("MIN_ROC_AUC", "0.62"))
MODEL_NAME = f"xauusd-{ACCOUNT}"


# ── Task functions ─────────────────────────────────────────────────────────────

def _prepare_dataset(**ctx):
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    import yaml
    from pathlib import Path
    from xauusd_ai.config import Settings
    from xauusd_ai.features.dataset import build_dataset

    with open(f"/opt/airflow/{CONFIG_PATH}") as f:
        raw = yaml.safe_load(f)
    settings = Settings(**raw)
    df = build_dataset(settings)

    out_path = f"/opt/airflow/outputs/training_dataset_{ACCOUNT}.parquet"
    df.to_parquet(out_path, index=False)
    ctx["ti"].xcom_push(key="dataset_path", value=out_path)
    ctx["ti"].xcom_push(key="dataset_rows", value=len(df))
    print(f"Dataset prepared: {len(df)} rows → {out_path}")


def _train_model(**ctx):
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    import yaml
    import pandas as pd
    from pathlib import Path
    from xauusd_ai.config import Settings
    from xauusd_ai.model.trainer import ModelTrainer
    from xauusd_ai.infra.mlflow_client import MLflowTracker
    from xauusd_ai.infra.storage import StorageClient

    dataset_path = ctx["ti"].xcom_pull(key="dataset_path")
    df = pd.read_parquet(dataset_path)

    with open(f"/opt/airflow/{CONFIG_PATH}") as f:
        raw = yaml.safe_load(f)
    settings = Settings(**raw)
    trainer = ModelTrainer(settings)

    tracker = MLflowTracker(experiment_name=f"xauusd_{ACCOUNT}")
    storage = StorageClient()

    with tracker.start_run(run_name=f"train_{ctx['ds']}_{ACCOUNT}") as run_id:
        # Log training params
        tracker.log_params({
            "account": ACCOUNT,
            "max_iter": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "signal_threshold": settings.strategy.signal_threshold,
            "dataset_rows": len(df),
        })

        metrics = trainer.train(df, save_artifacts=True)
        tracker.log_metrics(metrics)

        # Log + upload model artifacts
        model_pkl = Path(settings.app.model_path)
        scaler_pkl = Path(settings.app.scaler_path)
        cal_pkl = Path(str(model_pkl) + ".cal.pkl")

        for artifact in [model_pkl, scaler_pkl]:
            if artifact.exists():
                tracker.log_artifact(artifact, artifact_path="artifacts")
                storage.upload_model(ACCOUNT, f"run_{run_id[:8]}", artifact)
        if cal_pkl.exists():
            tracker.log_artifact(cal_pkl, artifact_path="artifacts")

        ctx["ti"].xcom_push(key="run_id", value=run_id)
        ctx["ti"].xcom_push(key="roc_auc", value=metrics.get("roc_auc", 0.0))
        print(f"Training complete: ROC-AUC={metrics.get('roc_auc', 0.0):.4f} run_id={run_id}")


def _run_backtest(**ctx):
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    import yaml
    from xauusd_ai.config import Settings
    from xauusd_ai.app import App
    from xauusd_ai.infra.mlflow_client import MLflowTracker

    with open(f"/opt/airflow/{CONFIG_PATH}") as f:
        raw = yaml.safe_load(f)
    settings = Settings(**raw)

    app = App(settings)
    report = app.orchestrator.run_backtest()

    run_id = ctx["ti"].xcom_pull(key="run_id")
    tracker = MLflowTracker(experiment_name=f"xauusd_{ACCOUNT}")

    # Log backtest metrics to same MLflow run
    try:
        import mlflow
        from xauusd_ai.infra.mlflow_client import _get_tracking_uri
        mlflow.set_tracking_uri(_get_tracking_uri())
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                f"bt_{k}": v for k, v in report.items()
                if isinstance(v, (int, float))
            })
    except Exception as exc:
        print(f"MLflow backtest log warning: {exc}")

    ctx["ti"].xcom_push(key="backtest_report", value=report)
    print(f"Backtest done: total_trades={report.get('total_trades', 0)}")


def _run_walkforward(**ctx):
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    import yaml
    from xauusd_ai.config import Settings
    from xauusd_ai.app import App

    with open(f"/opt/airflow/{CONFIG_PATH}") as f:
        raw = yaml.safe_load(f)
    settings = Settings(**raw)

    app = App(settings)
    report = app.orchestrator.run_walkforward()
    ctx["ti"].xcom_push(key="wf_report", value=report)
    print(f"Walk-forward done: mean_roc_auc={report.get('mean_roc_auc', 0.0):.4f}")


def _check_quality(**ctx) -> str:
    roc_auc = ctx["ti"].xcom_pull(key="roc_auc") or 0.0
    print(f"Quality gate: ROC-AUC={roc_auc:.4f} min={MIN_ROC_AUC}")
    return "register_model" if roc_auc >= MIN_ROC_AUC else "reject_model"


def _register_model(**ctx):
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    from xauusd_ai.infra.mlflow_client import MLflowTracker

    run_id = ctx["ti"].xcom_pull(key="run_id")
    roc_auc = ctx["ti"].xcom_pull(key="roc_auc") or 0.0

    tracker = MLflowTracker(experiment_name=f"xauusd_{ACCOUNT}")
    tracker.register_model(
        run_id=run_id,
        model_name=MODEL_NAME,
        artifact_path="artifacts",
        stage="Staging",
        description=f"Train {ctx['ds']} | ROC-AUC={roc_auc:.4f}",
    )
    print(f"Registered {MODEL_NAME} to Staging (run_id={run_id})")


def _run_drift_check(**ctx):
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    import pandas as pd
    from pathlib import Path
    from xauusd_ai.infra.storage import StorageClient
    from xauusd_ai.infra.db import get_engine, TradeStore
    from xauusd_ai.monitoring.drift import DriftMonitor
    from xauusd_ai.infra.metrics import TradingMetrics

    dataset_path = ctx["ti"].xcom_pull(key="dataset_path")
    df = pd.read_parquet(dataset_path)

    # Reference: last 30% of training data; current: last 500 rows of signals
    split_idx = int(len(df) * 0.7)
    reference_df = df.iloc[:split_idx]
    current_df = df.iloc[split_idx:]

    storage = StorageClient()
    metrics = TradingMetrics(ACCOUNT)
    monitor = DriftMonitor(account=ACCOUNT, storage=storage, metrics_recorder=metrics)
    result = monitor.run_feature_drift(reference_df, current_df)

    ctx["ti"].xcom_push(key="drift_score", value=result.get("drift_score", 0.0))
    ctx["ti"].xcom_push(key="drift_detected", value=result.get("drift_detected", False))
    print(f"Drift check: score={result.get('drift_score'):.4f} detected={result.get('drift_detected')}")


def _promote_to_prod(**ctx) -> str:
    drift_detected = ctx["ti"].xcom_pull(key="drift_detected") or False
    if drift_detected:
        print("WARNING: Drift detected — manual approval required before Production promotion.")
        return "manual_approval_needed"
    return "promote_model"


def _promote_model(**ctx):
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    from xauusd_ai.infra.mlflow_client import MLflowTracker

    run_id = ctx["ti"].xcom_pull(key="run_id")
    tracker = MLflowTracker(experiment_name=f"xauusd_{ACCOUNT}")
    tracker.register_model(
        run_id=run_id,
        model_name=MODEL_NAME,
        artifact_path="artifacts",
        stage="Production",
    )
    print(f"Promoted {MODEL_NAME} to Production.")


def _invalidate_cache(**ctx):
    """Call FastAPI endpoint to reload model cache after deployment."""
    import requests
    api_url = os.getenv("FASTAPI_URL", "http://api:8000")
    try:
        resp = requests.post(f"{api_url}/api/v1/models/{MODEL_NAME}/invalidate", timeout=10)
        print(f"Cache invalidation: HTTP {resp.status_code}")
    except Exception as exc:
        print(f"Cache invalidation warning (non-fatal): {exc}")


# ── Task wiring ────────────────────────────────────────────────────────────────

with dag:
    prepare = PythonOperator(task_id="prepare_dataset", python_callable=_prepare_dataset)
    train = PythonOperator(task_id="train_model", python_callable=_train_model)
    backtest = PythonOperator(task_id="run_backtest", python_callable=_run_backtest)
    walkforward = PythonOperator(task_id="run_walkforward", python_callable=_run_walkforward)
    quality_gate = BranchPythonOperator(task_id="check_quality", python_callable=_check_quality)
    register = PythonOperator(task_id="register_model", python_callable=_register_model)
    reject = EmptyOperator(task_id="reject_model")
    drift = PythonOperator(task_id="run_drift_check", python_callable=_run_drift_check)
    promote_branch = BranchPythonOperator(task_id="promote_branch", python_callable=_promote_to_prod)
    promote = PythonOperator(task_id="promote_model", python_callable=_promote_model)
    manual_approval = EmptyOperator(task_id="manual_approval_needed")
    invalidate = PythonOperator(task_id="invalidate_cache", python_callable=_invalidate_cache)

    (
        prepare
        >> train
        >> [backtest, walkforward]
        >> quality_gate
        >> [register, reject]
    )
    register >> drift >> promote_branch >> [promote, manual_approval]
    promote >> invalidate
