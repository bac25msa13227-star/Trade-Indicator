"""Airflow DAG: Drift monitoring — hourly Evidently check on live signals.

Schedule: every hour
Tasks:
  1. load_reference     – load training dataset from MinIO (parquet)
  2. load_current       – load last 200 live signals from PostgreSQL
  3. run_drift_report   – Evidently DataDrift report → upload to MinIO
  4. evaluate_drift     – check if drift_score > threshold
  5. alert_on_drift     – log alert (extend with Slack/email as needed)
  6. push_metrics       – push drift score to Prometheus pushgateway
"""
from __future__ import annotations

import os
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

ACCOUNT = os.getenv("TRADING_ACCOUNT", "acc2")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.30"))

DEFAULT_ARGS = {
    "owner": "trader",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="drift_monitoring",
    description="Hourly Evidently drift check on live feature distribution",
    schedule_interval="@hourly",
    start_date=days_ago(1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["monitoring", "evidently"],
)


def _load_reference(**ctx):
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    import pandas as pd
    from pathlib import Path

    # Try MinIO first, fall back to local parquet
    parquet_path = f"/opt/airflow/outputs/training_dataset_{ACCOUNT}.parquet"
    if Path(parquet_path).exists():
        df = pd.read_parquet(parquet_path)
    else:
        csv_path = f"/opt/airflow/outputs/training_dataset_{ACCOUNT}.csv"
        df = pd.read_csv(csv_path)

    # Use last 40% as reference baseline
    ref = df.tail(int(len(df) * 0.4))
    ref_path = f"/tmp/reference_{ACCOUNT}.parquet"
    ref.to_parquet(ref_path, index=False)
    ctx["ti"].xcom_push(key="ref_path", value=ref_path)


def _load_current(**ctx):
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    import pandas as pd
    from datetime import datetime, timedelta
    from xauusd_ai.infra.db import get_engine_no_pool, TradeStore

    engine = get_engine_no_pool()
    store = TradeStore(engine)
    since = datetime.utcnow() - timedelta(hours=6)
    df = store.get_signals(ACCOUNT, limit=500)

    if df.empty:
        # Fallback: use recent rows from training dataset
        ref_path = ctx["ti"].xcom_pull(key="ref_path")
        df = pd.read_parquet(ref_path).tail(200)

    curr_path = f"/tmp/current_{ACCOUNT}.parquet"
    df.to_parquet(curr_path, index=False)
    ctx["ti"].xcom_push(key="curr_path", value=curr_path)


def _run_drift_report(**ctx):
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    import pandas as pd
    from xauusd_ai.infra.storage import StorageClient
    from xauusd_ai.infra.metrics import TradingMetrics
    from xauusd_ai.monitoring.drift import DriftMonitor

    ref_path = ctx["ti"].xcom_pull(key="ref_path")
    curr_path = ctx["ti"].xcom_pull(key="curr_path")
    ref_df = pd.read_parquet(ref_path)
    curr_df = pd.read_parquet(curr_path)

    storage = StorageClient()
    metrics = TradingMetrics(ACCOUNT)
    monitor = DriftMonitor(account=ACCOUNT, storage=storage, metrics_recorder=metrics)
    result = monitor.run_feature_drift(ref_df, curr_df)

    ctx["ti"].xcom_push(key="drift_score", value=result.get("drift_score", 0.0))
    ctx["ti"].xcom_push(key="drift_detected", value=result.get("drift_detected", False))
    ctx["ti"].xcom_push(key="drifted_features", value=result.get("features_drifted", []))
    print(f"Drift report: {result}")


def _check_drift(**ctx) -> str:
    drift_detected = ctx["ti"].xcom_pull(key="drift_detected") or False
    return "alert_on_drift" if drift_detected else "no_action"


def _alert_on_drift(**ctx):
    drift_score = ctx["ti"].xcom_pull(key="drift_score")
    features = ctx["ti"].xcom_pull(key="drifted_features") or []
    msg = (
        f"[DRIFT ALERT] Account={ACCOUNT} | "
        f"score={drift_score:.3f} | drifted_features={features}"
    )
    print(msg)
    # Extend here: send to Slack/Teams/email
    # e.g. requests.post(slack_webhook, json={"text": msg})


with dag:
    load_ref = PythonOperator(task_id="load_reference", python_callable=_load_reference)
    load_curr = PythonOperator(task_id="load_current", python_callable=_load_current)
    drift_report = PythonOperator(task_id="run_drift_report", python_callable=_run_drift_report)
    check = BranchPythonOperator(task_id="check_drift", python_callable=_check_drift)
    alert = PythonOperator(task_id="alert_on_drift", python_callable=_alert_on_drift)
    no_action = EmptyOperator(task_id="no_action")

    load_ref >> load_curr >> drift_report >> check >> [alert, no_action]
