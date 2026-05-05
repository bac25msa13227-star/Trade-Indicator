"""ChartWF integration router — exposes WF artefacts to the ChartWF microservice.

Mounted by ``api.main`` if the file is present. Read-only.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chartwf", tags=["chartwf"])

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = REPO_ROOT / "outputs"
SIM_RE = re.compile(r"^walkforward_trades_(?P<strategy>.+)_sim_trades\.csv$")
SUM_RE = re.compile(r"^walkforward_trades_(?P<strategy>.+)\.csv$")


@router.get("/strategies")
def chartwf_strategies() -> dict:
    """List all WF strategies discoverable in outputs/."""
    if not OUTPUTS.exists():
        return {"strategies": []}
    found: dict[str, dict] = {}
    for f in sorted(OUTPUTS.glob("walkforward_trades_*.csv")):
        m_sim = SIM_RE.match(f.name)
        m_sum = SUM_RE.match(f.name)
        if m_sim:
            found.setdefault(m_sim.group("strategy"), {"name": m_sim.group("strategy"), "files": {}})["files"]["sim"] = f.name
        elif m_sum:
            name = m_sum.group("strategy")
            found.setdefault(name, {"name": name, "files": {}})["files"]["summary"] = f.name
    return {"strategies": list(found.values())}


@router.get("/trades/{strategy}")
def chartwf_trades(strategy: str, limit: int = Query(5000, ge=1, le=50000)) -> dict:
    """Return raw trades for a given strategy, capped at ``limit`` rows."""
    sim = OUTPUTS / f"walkforward_trades_{strategy}_sim_trades.csv"
    summary = OUTPUTS / f"walkforward_trades_{strategy}.csv"
    path = sim if sim.exists() else summary
    if not path.exists():
        raise HTTPException(404, f"strategy '{strategy}' not found")

    df = pd.read_csv(path).tail(limit)
    return {
        "strategy": strategy,
        "file": path.name,
        "count": int(len(df)),
        "columns": list(df.columns),
        "trades": df.where(df.notna(), None).to_dict(orient="records"),
    }
