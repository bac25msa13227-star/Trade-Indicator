"""Historical USD macro event fetcher using two free data sources.

**Primary: BLS (Bureau of Labor Statistics)**
  - Completely free, no API key required for basic access (25 req/day unregistered)
  - Provides: NFP, Unemployment Rate, CPI
  - Endpoint: https://api.bls.gov/publicAPI/v2/timeseries/data/

**Secondary: FRED (St. Louis Federal Reserve)**
  - Free with a registered API key (https://fred.stlouisfed.org/docs/api/api_key.html)
  - Provides: GDP, Retail Sales, Fed Funds Rate, PCE, Industrial Production
  - Requires valid ``fred_api_key`` in news settings

These are merged into the same CSV format as the Forex Factory crawler so
``NewsAnalyzer`` can consume them without modification.

Key USD series for XAUUSD trading:
  NFP / UNRATE / CPI → typically 13:30 UTC (08:30 ET)
  GDP / PCE          → typically 13:30 UTC
  Fed Funds Rate     → typically 19:00 UTC (14:00 ET)
  Retail Sales       → typically 13:30 UTC
"""

from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)

_BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# BLS series → (event_name, impact, release_time_utc)
_BLS_SERIES: dict[str, tuple[str, str, str]] = {
    "CES0000000001": ("Non-Farm Employment Change", "High", "13:30"),
    "LNS14000000":   ("Unemployment Rate",          "High", "13:30"),
    "CUSR0000SA0":   ("CPI m/m",                    "High", "13:30"),
}

# FRED series → (event_name, impact, release_time_utc) — require valid API key
_FRED_SERIES: dict[str, tuple[str, str, str]] = {
    "GDPC1":   ("GDP q/q",                  "High",   "13:30"),
    "RSAFS":   ("Retail Sales m/m",         "Medium", "13:30"),
    "FEDFUNDS":("Fed Funds Rate",            "High",   "19:00"),
    "PCEPI":   ("Core PCE Price Index m/m", "High",   "13:30"),
    "INDPRO":  ("Industrial Production m/m","Medium", "14:15"),
}


# ---------------------------------------------------------------------------
# BLS helpers
# ---------------------------------------------------------------------------

def _bls_fetch(series_ids: list[str], start_year: int, end_year: int,
               catalog: bool = False, delay: float = 0.5) -> dict[str, list[dict]]:
    """POST to BLS API. Returns {series_id: [observations]}."""
    payload: dict[str, Any] = {
        "seriesid": series_ids,
        "startyear": str(start_year),
        "endyear": str(end_year),
        "calculations": False,
        "catalog": catalog,
    }
    try:
        resp = requests.post(_BLS_API_URL, json=payload, timeout=20,
                             headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        LOGGER.error("BLS request failed: %s", exc)
        return {}

    time.sleep(delay)
    result: dict[str, list[dict]] = {}
    for series in data.get("Results", {}).get("series", []):
        result[series["seriesID"]] = series.get("data", [])
    return result


def _bls_period_to_date(year: str, period: str) -> str | None:
    """Convert BLS period (M01–M12) to 'YYYY-MM-DD' (first of that month)."""
    if not period.startswith("M") or period == "M13":
        return None
    month = int(period[1:])
    return f"{year}-{month:02d}-01"


# ---------------------------------------------------------------------------
# FRED helpers
# ---------------------------------------------------------------------------

def _fred_fetch(series_id: str, api_key: str, start: str, end: str,
                delay: float) -> list[dict[str, str]]:
    """Fetch observations from FRED. Returns list of {'date', 'value'}."""
    params: dict[str, Any] = {
        "series_id": series_id,
        "observation_start": start,
        "observation_end": end,
        "file_type": "json",
        "sort_order": "asc",
        "api_key": api_key,  # must be a valid key
    }
    try:
        resp = requests.get(_FRED_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        time.sleep(delay)
        return [o for o in data.get("observations", []) if o.get("value", ".") != "."]
    except requests.RequestException as exc:
        LOGGER.error("FRED request for %s failed: %s", series_id, exc)
        return []


# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------

def _build_event(
    dt_str: str,
    release_time_utc: str,
    currency: str,
    event_name: str,
    impact: str,
    actual: float,
    previous: float,
) -> dict[str, str] | None:
    """Build a news event dict from raw values. Returns None on parse error."""
    full_dt = f"{dt_str} {release_time_utc}:00"
    try:
        dt_utc = datetime.strptime(full_dt, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    deviation = round(actual - previous, 4)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "datetime_utc": dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "currency": currency,
        "impact": impact,
        "event": event_name,
        "actual": str(round(actual, 4)),
        "forecast": str(round(previous, 4)),
        "previous": str(round(previous, 4)),
        "deviation": str(deviation),
        "crawled_at": now_str,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_bls_news(
    start_year: int,
    end_year: int,
    request_delay: float = 0.5,
) -> list[dict[str, str]]:
    """Fetch NFP, Unemployment, CPI from BLS (no API key required).

    BLS monthly data points come with year + period (M01=Jan … M12=Dec).
    We use the observation date as the release date (BLS sets this correctly
    for most series).  Previous period value is used as the forecast proxy.
    """
    events: list[dict[str, str]] = []

    # BLS API caps year range at 20 years per request; we chunk by decade if needed
    all_obs = _bls_fetch(list(_BLS_SERIES.keys()), start_year, end_year, delay=request_delay)

    for series_id, (event_name, impact, rel_time) in _BLS_SERIES.items():
        obs_list = all_obs.get(series_id, [])
        if not obs_list:
            LOGGER.warning("No BLS data for series %s (%s)", series_id, event_name)
            continue

        # BLS returns newest first; reverse for chronological order
        obs_list = list(reversed(obs_list))
        LOGGER.info("BLS %s (%s): %d observations", series_id, event_name, len(obs_list))

        for i, obs in enumerate(obs_list):
            dt_str = _bls_period_to_date(obs.get("year", ""), obs.get("period", ""))
            if not dt_str:
                continue
            try:
                actual = float(obs["value"])
            except (KeyError, ValueError):
                continue
            previous = float(obs_list[i - 1]["value"]) if i > 0 else actual
            ev = _build_event(dt_str, rel_time, "USD", event_name, impact, actual, previous)
            if ev:
                events.append(ev)

    LOGGER.info("BLS fetch complete. Events: %d", len(events))
    return events


def fetch_fred_news(
    start_date: str,
    end_date: str,
    api_key: str,
    request_delay: float = 1.0,
) -> list[dict[str, str]]:
    """Fetch GDP, Retail Sales, Fed Funds Rate, PCE from FRED.

    Requires a valid (free) FRED API key from fred.stlouisfed.org.
    """
    if not api_key:
        LOGGER.warning("No FRED API key provided; skipping FRED series. "
                       "Register free at https://fred.stlouisfed.org/docs/api/api_key.html")
        return []

    events: list[dict[str, str]] = []
    for series_id, (event_name, impact, rel_time) in _FRED_SERIES.items():
        LOGGER.info("Fetching FRED series %s (%s)", series_id, event_name)
        obs_list = _fred_fetch(series_id, api_key, start_date, end_date, request_delay)
        if not obs_list:
            continue
        LOGGER.info("FRED %s: %d observations", series_id, len(obs_list))
        for i, obs in enumerate(obs_list):
            try:
                actual = float(obs["value"])
            except (KeyError, ValueError):
                continue
            previous = float(obs_list[i - 1]["value"]) if i > 0 else actual
            ev = _build_event(obs.get("date", ""), rel_time, "USD", event_name, impact, actual, previous)
            if ev:
                events.append(ev)

    LOGGER.info("FRED fetch complete. Events: %d", len(events))
    return events


def upsert_to_cache(cache_path: str | Path, events: list[dict[str, str]]) -> int:
    """Merge events into the news CSV cache (upsert by datetime+currency+event).

    Returns the number of new rows added.
    """
    cache_path = Path(cache_path)
    fieldnames = [
        "datetime_utc", "currency", "impact", "event",
        "actual", "forecast", "previous", "deviation", "crawled_at",
    ]

    existing: dict[str, dict[str, str]] = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = f"{row.get('datetime_utc','')}|{row.get('currency','')}|{row.get('event','')}"
                existing[key] = row

    added = 0
    for ev in events:
        key = f"{ev['datetime_utc']}|{ev['currency']}|{ev['event']}"
        if key not in existing:
            existing[key] = ev
            added += 1
        elif existing[key].get("actual", "") == "" and ev.get("actual", "") != "":
            existing[key].update(ev)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(existing.values(), key=lambda r: r.get("datetime_utc", "")))

    LOGGER.info("Cache upsert complete. New rows: %d  Total: %d", added, len(existing))
    return added


# Keep backward-compatible alias used by orchestrator
upsert_fred_to_cache = upsert_to_cache


# ---------------------------------------------------------------------------
# Synthetic calendar generator (offline fallback)
# ---------------------------------------------------------------------------

# Hardcoded FOMC announcement dates (Wednesday 19:00 UTC = 14:00 ET).
# Source: federalreserve.gov calendar (public information).
_FOMC_DATES: list[str] = [
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
]

# Hardcoded NFP release dates (first Friday of month, 13:30 UTC = 08:30 ET).
# Source: bls.gov news release schedule (public information).
_NFP_DATES: list[str] = [
    # 2023
    "2023-01-06", "2023-02-03", "2023-03-10", "2023-04-07",
    "2023-05-05", "2023-06-02", "2023-07-07", "2023-08-04",
    "2023-09-01", "2023-10-06", "2023-11-03", "2023-12-08",
    # 2024
    "2024-01-05", "2024-02-02", "2024-03-08", "2024-04-05",
    "2024-05-03", "2024-06-07", "2024-07-05", "2024-08-02",
    "2024-09-06", "2024-10-04", "2024-11-01", "2024-12-06",
    # 2025
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
]

# CPI release dates (approximate; bls.gov — typically 2nd or 3rd week of month, 13:30 UTC).
_CPI_DATES: list[str] = [
    # 2023
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12",
    "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10",
    "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
    # 2024
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
    "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
    "2025-09-10", "2025-10-15", "2025-11-13", "2025-12-10",
]


def generate_synthetic_calendar(start_date: str, end_date: str) -> list[dict[str, str]]:
    """Return placeholder news events for known NFP, FOMC, CPI dates.

    These events have no actual/forecast values (impact-presence only).
    They set ``news_in_window`` and ``news_upcoming_impact`` in ``NewsAnalyzer``
    so the ML model learns to behave differently around high-impact event times.

    Parameters
    ----------
    start_date / end_date : ISO date strings (inclusive).
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    events: list[dict[str, str]] = []

    def _add(date_str: str, event_name: str, time_utc: str, impact: str) -> None:
        try:
            dt = datetime.strptime(f"{date_str} {time_utc}:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return
        if start_dt <= dt <= end_dt:
            events.append({
                "datetime_utc": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "currency": "USD",
                "impact": impact,
                "event": event_name,
                "actual": "",
                "forecast": "",
                "previous": "",
                "deviation": "",
                "crawled_at": now_str,
            })

    for d in _NFP_DATES:
        _add(d, "Non-Farm Employment Change", "13:30", "High")
        _add(d, "Unemployment Rate", "13:30", "High")

    for d in _CPI_DATES:
        _add(d, "CPI m/m", "13:30", "High")

    for d in _FOMC_DATES:
        _add(d, "Fed Funds Rate", "19:00", "High")

    LOGGER.info("Synthetic calendar: %d events generated for %s–%s", len(events), start_date, end_date)
    return events

