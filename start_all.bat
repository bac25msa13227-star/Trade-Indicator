@echo off
setlocal EnableDelayedExpansion
set LOG=outputs\start_all.log
cd /d "%~dp0"

echo. > %LOG%
call :log "=== START ALL SERVICES - %DATE% %TIME% ==="

REM ─── Step 1: postgres + minio ────────────────────────────────────────────
call :log "[1/6] Check postgres + minio..."
docker inspect xauusd-postgres --format "{{.State.Status}}" 2>&1 | findstr "running" >nul
if errorlevel 1 (
    call :log "  Starting postgres..."
    docker compose up -d postgres
) else (
    call :log "  postgres already running"
)
docker inspect xauusd-minio --format "{{.State.Status}}" 2>&1 | findstr "running" >nul
if errorlevel 1 (
    call :log "  Starting minio..."
    docker compose up -d minio
) else (
    call :log "  minio already running"
)
timeout /t 5 /nobreak >nul

REM ─── Step 2: Pull missing images ─────────────────────────────────────────
call :log "[2/6] Pulling images..."

docker image inspect prom/prometheus:v2.51.2 >nul 2>&1
if errorlevel 1 (
    call :log "  Pulling prometheus..."
    docker pull prom/prometheus:v2.51.2 2>&1 | find "Pull complete" && call :log "  prometheus OK"
) else ( call :log "  prometheus: already local" )

docker image inspect grafana/grafana:10.4.2 >nul 2>&1
if errorlevel 1 (
    call :log "  Pulling grafana..."
    docker pull grafana/grafana:10.4.2
) else ( call :log "  grafana: already local" )

docker image inspect ghcr.io/mlflow/mlflow:v2.13.0 >nul 2>&1
if errorlevel 1 (
    call :log "  Pulling mlflow..."
    docker pull ghcr.io/mlflow/mlflow:v2.13.0
) else ( call :log "  mlflow: already local" )

docker image inspect apache/airflow:2.9.1-python3.11 >nul 2>&1
if errorlevel 1 (
    call :log "  Pulling airflow (large image ~1GB, please wait)..."
    docker pull apache/airflow:2.9.1-python3.11
) else ( call :log "  airflow: already local" )

call :log "  Image pulls done."

REM ─── Step 3: MLflow ─────────────────────────────────────────────────────
call :log "[3/6] Starting MLflow..."
docker compose up -d mlflow
timeout /t 10 /nobreak >nul
for /f %%s in ('docker inspect xauusd-mlflow --format "{{.State.Status}}" 2^>^&1') do call :log "  mlflow: %%s"

REM ─── Step 4: Prometheus + Grafana ───────────────────────────────────────
call :log "[4/6] Starting Prometheus + Grafana..."
docker compose up -d prometheus grafana
timeout /t 8 /nobreak >nul
for /f %%s in ('docker inspect xauusd-prometheus --format "{{.State.Status}}" 2^>^&1') do call :log "  prometheus: %%s"
for /f %%s in ('docker inspect xauusd-grafana --format "{{.State.Status}}" 2^>^&1') do call :log "  grafana: %%s"

REM ─── Step 5: Airflow ────────────────────────────────────────────────────
call :log "[5/6] Airflow init + start..."
docker inspect xauusd-airflow-webserver --format "{{.State.Status}}" 2>&1 | findstr "running" >nul
if errorlevel 1 (
    call :log "  Running airflow-init (may take ~2 min)..."
    docker compose up airflow-init
    call :log "  Starting airflow-webserver + scheduler..."
    docker compose up -d airflow-webserver airflow-scheduler
    timeout /t 20 /nobreak >nul
) else (
    call :log "  Airflow already running"
)
for /f %%s in ('docker inspect xauusd-airflow-webserver --format "{{.State.Status}}" 2^>^&1') do call :log "  airflow-webserver: %%s"
for /f %%s in ('docker inspect xauusd-airflow-scheduler --format "{{.State.Status}}" 2^>^&1') do call :log "  airflow-scheduler: %%s"

REM ─── Step 6: FastAPI ────────────────────────────────────────────────────
call :log "[6/6] Starting FastAPI..."
docker compose up -d api
timeout /t 8 /nobreak >nul
for /f %%s in ('docker inspect xauusd-api --format "{{.State.Status}}" 2^>^&1') do call :log "  api: %%s"

REM ─── Final status ───────────────────────────────────────────────────────
call :log ""
call :log "=== FINAL STATUS ==="
docker ps --format "  {{.Names}}  {{.Status}}" 2>&1
docker ps --format "  {{.Names}}  {{.Status}}" 2>&1 >> %LOG%

call :log ""
call :log "=== SERVICE URLS ==="
call :log "  Dashboard   : http://localhost:8000/dashboard"
call :log "  FastAPI     : http://localhost:8000/docs"
call :log "  MLflow      : http://localhost:5000"
call :log "  MinIO UI    : http://localhost:9001  (minioadmin / minioadmin123)"
call :log "  Prometheus  : http://localhost:9090"
call :log "  Grafana     : http://localhost:3000  (admin / admin2026)"
call :log "  Airflow     : http://localhost:8080  (airflow / airflow2026)"
call :log ""
call :log "=== ALL DONE ==="

echo.
echo Done! Check outputs\start_all.log for details.
pause
goto :eof

:log
echo %~1
echo %~1 >> %LOG%
goto :eof
