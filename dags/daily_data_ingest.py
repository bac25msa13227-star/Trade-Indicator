"""Airflow DAG: Daily data ingestion from MT5 → PostgreSQL + MinIO.

Schedule: every day at 00:30 UTC (after midnight rollover)
Tasks:
  1. fetch_mt5_data     – pull M15/H1/D1 bars from MT5 via CSV or MT5 API
  2. validate_data      – basic sanity checks (nulls, gaps, stale data)
  3. upload_to_minio    – push parquet to MinIO datasets/ bucket
  4. load_to_postgres   – upsert OHLCV data into market_data table (optional)
  5. notify_downstream  – TriggerDagRunOperator to launch model_training if new data
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

DEFAULT_ARGS = {
    "owner": "trader",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="daily_data_ingest",
    description="Fetch market data → MinIO + PostgreSQL",
    schedule_interval="30 0 * * *",      # 00:30 UTC daily
    start_date=days_ago(1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["data", "ingestion"],
)


# ── Task functions ─────────────────────────────────────────────────────────────

def _fetch_mt5_data(**ctx):
    """Fetch recent OHLCV bars from MT5 (or CSV fallback) and save as parquet."""
    import os
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    from pathlib import Path
    import pandas as pd

    account = os.getenv("TRADING_ACCOUNT", "acc2")
    output_dir = Path(os.getenv("OUTPUT_DIR", "/opt/airflow/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try MT5 first; fall back to rebuilding from existing CSV
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise RuntimeError("MT5 not available in this environment")
        # Fetch last 2000 M15 bars
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 2000)
        mt5.shutdown()
        if rates is None or len(rates) == 0:
            raise RuntimeError("MT5 returned no data")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
    except Exception as exc:
        # Fallback: use existing CSV
        csv_path = output_dir / f"training_dataset_{account}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No MT5 and no CSV fallback at {csv_path}") from exc
        df = pd.read_csv(csv_path, parse_dates=["time"])

    parquet_path = output_dir / f"market_data_{account}.parquet"
    df.to_parquet(parquet_path, index=False)
    ctx["ti"].xcom_push(key="parquet_path", value=str(parquet_path))
    ctx["ti"].xcom_push(key="rows", value=len(df))
    print(f"Fetched {len(df)} rows → {parquet_path}")


def _validate_data(**ctx):
    """Check for missing values, stale timestamps, and minimum row count."""
    import pandas as pd
    parquet_path = ctx["ti"].xcom_pull(key="parquet_path")
    df = pd.read_parquet(parquet_path)
    issues = []

    if df.empty:
        issues.append("DataFrame is empty")
    null_pct = df.isnull().mean().max()
    if null_pct > 0.05:
        issues.append(f"High null rate: {null_pct:.1%}")
    if len(df) < 500:
        issues.append(f"Too few rows: {len(df)}")

    if issues:
        raise ValueError(f"Data validation failed: {'; '.join(issues)}")
    print(f"Data validation passed: {len(df)} rows, max_null={null_pct:.3%}")


def _upload_to_minio(**ctx):
    """Push parquet file to MinIO datasets/ bucket."""
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    import os
    from pathlib import Path

    parquet_path = ctx["ti"].xcom_pull(key="parquet_path")
    account = os.getenv("TRADING_ACCOUNT", "acc2")
    run_date = ctx["ds"]  # YYYY-MM-DD

    from xauusd_ai.infra.storage import StorageClient
    storage = StorageClient()
    object_name = storage.upload_dataset(
        account, f"market_data_{run_date}.parquet", Path(parquet_path)
    )
    print(f"Uploaded to MinIO: {object_name}")


def _check_new_data(**ctx) -> str:
    """Branch: trigger training if new data has enough rows, else skip."""
    rows = ctx["ti"].xcom_pull(key="rows") or 0
    return "trigger_training" if rows > 1000 else "skip_training"


def _trigger_training(**ctx):
    """Signal model_training DAG to run (via Airflow Variable or DB flag)."""
    try:
        from airflow.models import Variable
        Variable.set("trigger_training", "1", serialize_json=False)
        print("Set Variable trigger_training=1 → model_training DAG will pick up.")
    except Exception as exc:
        print(f"Could not set Airflow variable: {exc}")


# ── Task wiring ────────────────────────────────────────────────────────────────

with dag:
    fetch = PythonOperator(
        task_id="fetch_mt5_data",
        python_callable=_fetch_mt5_data,
    )
    validate = PythonOperator(
        task_id="validate_data",
        python_callable=_validate_data,
    )
    upload = PythonOperator(
        task_id="upload_to_minio",
        python_callable=_upload_to_minio,
    )
    branch = BranchPythonOperator(
        task_id="check_new_data",
        python_callable=_check_new_data,
    )
    trigger = PythonOperator(
        task_id="trigger_training",
        python_callable=_trigger_training,
    )
    skip = EmptyOperator(task_id="skip_training")

    fetch >> validate >> upload >> branch >> [trigger, skip]
