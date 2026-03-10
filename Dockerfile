FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

ARG INSTALL_DASHBOARD=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-core.txt requirements-dashboard.txt requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-core.txt && \
    pip install --no-cache-dir -e . && \
    if [ "$INSTALL_DASHBOARD" = "true" ]; then pip install --no-cache-dir -r requirements-dashboard.txt; fi

CMD ["python", "-m", "xauusd_ai.main", "train", "--config", "configs/settings.yaml"]
