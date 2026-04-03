$ErrorActionPreference = "Continue"
$LOG = "C:\Users\Administrator\Documents\Trade-Indicator\outputs\start_all.log"
$PROJ = "C:\Users\Administrator\Documents\Trade-Indicator"
Set-Location $PROJ

function Log($msg) {
    $ts = Get-Date -Format "HH:mm:ss"
    "$ts  $msg" | Tee-Object -FilePath $LOG -Append
}

Log "=== START ALL SERVICES ==="
Log "Working dir: $PROJ"

# ── Step 1: Check postgres + minio ───────────────────────────────────────────
Log "--- [1/6] Checking postgres + minio ---"
$pg = docker inspect xauusd-postgres --format "{{.State.Status}}" 2>&1
$mn = docker inspect xauusd-minio    --format "{{.State.Status}}" 2>&1
Log "postgres: $pg"
Log "minio:    $mn"

if ($pg -ne "running") {
    Log "Starting postgres..."
    docker compose up -d postgres 2>&1 | ForEach-Object { Log $_ }
    Start-Sleep 10
}
if ($mn -ne "running") {
    Log "Starting minio..."
    docker compose up -d minio 2>&1 | ForEach-Object { Log $_ }
    Start-Sleep 10
}
Log "postgres + minio OK"

# ── Step 2: Pull images (retry on failure) ───────────────────────────────────
Log "--- [2/6] Pulling images (may take a few minutes) ---"

$images = @(
    "prom/prometheus:v2.51.2",
    "grafana/grafana:10.4.2",
    "ghcr.io/mlflow/mlflow:v2.13.0",
    "apache/airflow:2.9.1-python3.11"
)

foreach ($img in $images) {
    $localCheck = docker images $img --format "{{.Repository}}:{{.Tag}}" 2>&1
    if ($localCheck -match ($img -replace ":", ":")) {
        Log "Image already local: $img"
        continue
    }
    Log "Pulling: $img ..."
    $tries = 0
    $success = $false
    while ($tries -lt 3 -and -not $success) {
        $tries++
        $out = docker pull $img 2>&1
        if ($LASTEXITCODE -eq 0) {
            Log "  Pulled OK: $img"
            $success = $true
        } else {
            Log "  Attempt $tries failed: $($out | Select-Object -Last 2)"
            Start-Sleep 15
        }
    }
    if (-not $success) { Log "  WARNING: Could not pull $img after 3 tries" }
}

# ── Step 3: Start MLflow ─────────────────────────────────────────────────────
Log "--- [3/6] Starting MLflow ---"
docker compose up -d mlflow 2>&1 | ForEach-Object { Log $_ }
Start-Sleep 10
$mf = docker inspect xauusd-mlflow --format "{{.State.Status}}" 2>&1
Log "mlflow: $mf"

# ── Step 4: Start Prometheus + Grafana ───────────────────────────────────────
Log "--- [4/6] Starting Prometheus + Grafana ---"
docker compose up -d prometheus grafana 2>&1 | ForEach-Object { Log $_ }
Start-Sleep 8
$pr = docker inspect xauusd-prometheus --format "{{.State.Status}}" 2>&1
$gr = docker inspect xauusd-grafana    --format "{{.State.Status}}" 2>&1
Log "prometheus: $pr"
Log "grafana:    $gr"

# ── Step 5: Init Airflow (only if not done) ──────────────────────────────────
Log "--- [5/6] Airflow init + start ---"
$afCheck = docker inspect xauusd-airflow-webserver --format "{{.State.Status}}" 2>&1
if ($afCheck -eq "running") {
    Log "Airflow webserver already running, skipping init"
} else {
    Log "Running airflow-init (may take ~2 min)..."
    docker compose up airflow-init 2>&1 | Select-Object -Last 10 | ForEach-Object { Log $_ }
    Start-Sleep 5
    Log "Starting airflow webserver + scheduler..."
    docker compose up -d airflow-webserver airflow-scheduler 2>&1 | ForEach-Object { Log $_ }
    Start-Sleep 15
    $afWeb = docker inspect xauusd-airflow-webserver --format "{{.State.Status}}" 2>&1
    $afSch = docker inspect xauusd-airflow-scheduler --format "{{.State.Status}}" 2>&1
    Log "airflow-webserver: $afWeb"
    Log "airflow-scheduler:  $afSch"
}

# ── Step 6: Start FastAPI ────────────────────────────────────────────────────
Log "--- [6/6] Starting FastAPI ---"
docker compose up -d api 2>&1 | ForEach-Object { Log $_ }
Start-Sleep 8
$apiStatus = docker inspect xauusd-api --format "{{.State.Status}}" 2>&1
Log "api: $apiStatus"

# ── Final status ─────────────────────────────────────────────────────────────
Log ""
Log "=== FINAL STATUS ==="
docker ps --format "{{.Names}}  {{.Status}}  {{.Ports}}" 2>&1 | ForEach-Object { Log $_ }

Log ""
Log "=== SERVICE URLS ==="
Log "Streamlit Dashboard : http://localhost:8501"
Log "FastAPI Swagger      : http://localhost:8000/docs"
Log "MLflow UI            : http://localhost:5000"
Log "MinIO Console        : http://localhost:9001  (user: minioadmin / minioadmin123)"
Log "Prometheus           : http://localhost:9090"
Log "Grafana              : http://localhost:3000  (user: admin / admin2026)"
Log "Airflow              : http://localhost:8080  (user: airflow / airflow2026)"
Log ""
Log "=== DONE ==="
