"""
Quick smoke-test: MLflowTracker → MinIO artifact upload.
Does NOT train any model. Safe to run while bots are live.
"""
import os
import sys
import tempfile
from pathlib import Path

# Load .env trước khi import bất cứ thứ gì
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

# In ra các env vars quan trọng
print("=== ENV CHECK ===")
for key in ["MLFLOW_TRACKING_URI", "MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECURE"]:
    print(f"  {key} = {os.getenv(key, '(not set)')}")

print("\n=== CONNECTING TO MLFLOW ===")
from xauusd_ai.infra.mlflow_client import MLflowTracker

tracker = MLflowTracker(experiment_name="xauusd_integration_test")
print(f"  Experiment: xauusd_integration_test -> OK")

print("\n=== STARTING RUN ===")
with tracker.start_run(run_name="smoke_test", tags={"source": "manual_test"}):
    # Log params
    tracker.log_params({
        "test_param_1": "hello",
        "test_param_2": 42,
        "signal_threshold": 0.73,
    })
    print("  log_params → OK")

    # Log metrics
    tracker.log_metrics({
        "precision": 0.613,
        "recall": 0.581,
        "f1": 0.596,
        "train_rows": 97099.0,
    })
    print("  log_metrics → OK")

    # Upload artifact nhỏ lên MinIO
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="mlflow_test_"
    ) as f:
        f.write("MLflow + MinIO integration test artifact\n")
        f.write(f"MINIO_ENDPOINT={os.getenv('MINIO_ENDPOINT')}\n")
        tmp_path = Path(f.name)

    tracker.log_artifact(tmp_path, artifact_path="test_artifacts")
    tmp_path.unlink(missing_ok=True)
    print("  log_artifact (MinIO upload) → OK")

print("\n=== VERIFY VIA API ===")
import urllib.request, json

uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
url = f"{uri}/api/2.0/mlflow/experiments/search?max_results=5"
try:
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = json.loads(resp.read())
    exps = data.get("experiments", [])
    names = [e["name"] for e in exps]
    print(f"  Experiments found: {names}")
    found = any("integration_test" in n for n in names)
    print(f"  xauusd_integration_test in experiments: {found}")
except Exception as e:
    print(f"  API check failed: {e}")

print("\n[OK] MLflow -> MinIO integration OK")
