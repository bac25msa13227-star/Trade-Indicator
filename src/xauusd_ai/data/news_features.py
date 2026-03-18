"""News Features cho XAUUSD AI Trading Model
============================================
Tạo 5 features từ lịch tin tức kinh tế, với 3 lớp nguồn dữ liệu:

  Lớp 1: Finnhub Economic Calendar API (FINNHUB_API_KEY trong .env)
          - Có actual + estimate  tính surprise THỰC TẾ
          - Historical data từ 2020 đến nay
          - Free tier: 100 calls/min, không giới hạn lịch sử
          - Cache: outputs/news_cache/YYYY.json (per-year, không fetch lại)
          - Endpoint: GET https://finnhub.io/api/v1/calendar/economic

  Lớp 2: ForexFactory JSON API (không cần key, chỉ có tuần/tháng hiện tại)
          - Dùng khi không có Finnhub key hoặc Finnhub down
          - Merge bổ sung cho dữ liệu hiện tại
          - Endpoint: https://nfs.faireconomy.media/ff_calendar_thisweek.json

  Lớp 3: Rule-based calendar (hoàn toàn offline, deterministic)
          - NFP (thứ 6 đầu tháng), CPI (ngày ~12), FOMC (hardcode), PCE, GDP
          - Luôn dùng nếu cả Finnhub + ForexFactory fail
          - Không có actual/estimate  dùng hướng định sẵn

5 features (total model features: 28  33):
  - news_impact_ahead    : 0=none 1=medium 2=high trong 4h tới
  - news_hours_ahead     : giờ đến tin High-impact tiếp theo (048)
  - news_hours_since     : giờ kể từ tin High-impact gần nhất (048)
  - news_surprise_gold   : +1 bullish gold / -1 bearish / 0 neutral
                           (từ actual vs estimate nếu có Finnhub;
                            từ hướng định sẵn nếu dùng rule-based)
  - news_is_blackout     : 1 = 1h của High-impact news (không trade)

Surprise logic (Finnhub actual vs estimate):
  NFP actual > estimate   USD tăng  Gold DOWN   -1
  CPI actual > estimate   Lạm phát  Gold UP     +1
  FOMC actual > estimate  Lãi cao   Gold DOWN   -1
  PCE/PPI actual > est    Lạm phát  Gold UP     +1
  GDP actual > estimate   USD tăng  Gold DOWN   -1
  Jobless Claims: higher = weak  Gold UP  +1 (inverted)

Setup:
  1. Đăng ký tại https://finnhub.io (free)  lấy API key
  2. Thêm vào .env: FINNHUB_API_KEY=your_key_here
  3. Chạy walk-forward  cache tự động build theo từng năm
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

_FINNHUB_BASE  = "https://finnhub.io/api/v1/calendar/economic"
_FF_WEEK_URL   = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_FF_MONTH_URL  = "https://nfs.faireconomy.media/ff_calendar_thismonth.json"
_CACHE_DIR     = Path("outputs/news_cache")
_HEADERS       = {"User-Agent": "Mozilla/5.0 (compatible; XAUBot/1.0)"}
_REQUEST_TIMEOUT = 20

_FOMC_DATES = [
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]

_EVENT_GOLD_DIRECTION: dict[str, int] = {
    "Non-Farm Payroll": -1, "Nonfarm Payrolls": -1, "NF Payrolls": -1,
    "ADP Employment": -1, "ADP Nonfarm": -1,
    "Jobless Claims": +1, "Initial Claims": +1, "Unemployment Claims": +1,
    "Unemployment Rate": +1,
    "CPI": +1, "Consumer Price Index": +1, "Core CPI": +1,
    "PCE": +1, "Core PCE": +1, "Personal Consumption": +1,
    "PPI": +1, "Producer Price": +1, "Inflation": +1,
    "GDP": -1, "Gross Domestic Product": -1,
    "Retail Sales": -1, "ISM": -1, "PMI": -1, "Purchasing Managers": -1,
    "Durable Goods": -1, "Housing": -1, "Building Permits": -1,
    "New Home Sales": -1, "Consumer Confidence": -1, "Consumer Sentiment": -1,
    "Michigan": -1, "FOMC": -1, "Federal Reserve": -1, "Fed Funds": -1,
    "Interest Rate": -1, "Powell": -1,
    "Trade Balance": +1, "Current Account": +1,
}

_INVERTED_EVENTS = frozenset([
    "jobless claims", "initial claims", "unemployment claims",
    "unemployment rate", "trade balance", "current account",
])


def _direction_from_title(title: str) -> int:
    t = title.lower()
    for kw, d in _EVENT_GOLD_DIRECTION.items():
        if kw.lower() in t:
            return d
    return 0


def _direction_from_surprise(title: str, actual, estimate) -> int:
    try:
        a, e = float(actual), float(estimate)
    except (TypeError, ValueError):
        return _direction_from_title(title)
    if a == e:
        return 0
    base = _direction_from_title(title)
    if base == 0:
        return 0
    is_inv = any(kw in title.lower() for kw in _INVERTED_EVENTS)
    pos = a > e
    return (base if pos else -base) if not is_inv else (base if pos else -base)


def _first_friday(year: int, month: int, hour_utc: int = 13, minute_utc: int = 30) -> datetime:
    d = datetime(year, month, 1, hour_utc, minute_utc, tzinfo=timezone.utc)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def _last_friday(year: int, month: int, hour_utc: int = 13, minute_utc: int = 30) -> datetime:
    nm, ny = (month % 12) + 1, year + (1 if month == 12 else 0)
    last_day = datetime(ny, nm, 1, tzinfo=timezone.utc) - timedelta(days=1)
    fri = last_day - timedelta(days=(last_day.weekday() - 4) % 7)
    return fri.replace(hour=hour_utc, minute=minute_utc, second=0, microsecond=0)


def _weekday_skip(dt: datetime) -> datetime:
    if dt.weekday() == 5:
        return dt + timedelta(days=2)
    if dt.weekday() == 6:
        return dt + timedelta(days=1)
    return dt


#  Layer 1: Finnhub 

# Process-level flag: set True after first 401/403 so we stop retrying all session
_finnhub_disabled: bool = False


def _get_finnhub_key() -> str | None:
    if _finnhub_disabled:
        return None
    key = os.environ.get("FINNHUB_API_KEY", "").strip()
    return key if key else None


def _finnhub_cache_path(year: int) -> Path:
    return _CACHE_DIR / f"finnhub_{year}.json"


def _load_finnhub_cache(year: int) -> list[dict] | None:
    p = _finnhub_cache_path(year)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        cached_at = data.get("cached_at", "")
        if cached_at and year == datetime.now(timezone.utc).year:
            age_h = (datetime.now(timezone.utc) -
                     datetime.fromisoformat(cached_at)).total_seconds() / 3600
            if age_h > 6:
                return None
        return data.get("events", [])
    except Exception:
        return None


def _save_finnhub_cache(year: int, events: list[dict]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _finnhub_cache_path(year).write_text(
        json.dumps({"cached_at": datetime.now(timezone.utc).isoformat(),
                    "year": year, "events": events}, indent=2, default=str),
        encoding="utf-8",
    )


def _fetch_finnhub_year(key: str, year: int) -> list[dict]:
    global _finnhub_disabled
    quarters = [
        (f"{year}-01-01", f"{year}-03-31"),
        (f"{year}-04-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-09-30"),
        (f"{year}-10-01", f"{year}-12-31"),
    ]
    all_events: list[dict] = []
    for from_d, to_d in quarters:
        try:
            r = requests.get(
                _FINNHUB_BASE,
                params={"from": from_d, "to": to_d, "token": key},
                headers=_HEADERS,
                timeout=_REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            for ev in r.json().get("economicCalendar", []):
                country = str(ev.get("country", "") or "").upper()
                impact  = str(ev.get("impact", "") or "").lower()
                if "US" in country and impact in ("high", "medium"):
                    all_events.append(ev)
            LOGGER.info("Finnhub: fetched %s%s", from_d, to_d)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                LOGGER.warning("Finnhub key invalid/quota exceeded (%s) — disabling for this session", status)
                _finnhub_disabled = True
                return all_events  # stop retrying remaining quarters
            LOGGER.warning("Finnhub fetch failed %s%s: %s", from_d, to_d, exc)
    return all_events


def _finnhub_to_df(events: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for ev in events:
        title  = str(ev.get("event", "") or "")
        impact = str(ev.get("impact", "") or "").capitalize()
        if impact not in ("High", "Medium"):
            continue
        raw_time = str(ev.get("time", "") or "")
        try:
            ev_time = (pd.Timestamp(raw_time + " 13:30:00", tz="UTC")
                       if len(raw_time) == 10
                       else pd.Timestamp(raw_time, tz="UTC"))
            if ev_time.tzinfo is None:
                ev_time = ev_time.tz_localize("UTC")
        except Exception:
            continue
        actual, estimate = ev.get("actual"), ev.get("estimate")
        rows.append({
            "datetime_utc":   ev_time,
            "event":          title,
            "impact":         impact,
            "gold_direction": _direction_from_surprise(title, actual, estimate),
            "actual":         actual,
            "estimate":       estimate,
            "prev":           ev.get("prev"),
            "source":         "finnhub",
        })
    if not rows:
        return pd.DataFrame(columns=[
            "datetime_utc", "event", "impact", "gold_direction",
            "actual", "estimate", "prev", "source",
        ])
    df = pd.DataFrame(rows)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df.sort_values("datetime_utc").reset_index(drop=True)


def fetch_finnhub_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch Finnhub economic calendar với per-year disk cache."""
    key = _get_finnhub_key()
    if not key:
        LOGGER.info("Finnhub: FINNHUB_API_KEY not set  skipping")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for year in range(start.year, end.year + 1):
        cached = _load_finnhub_cache(year)
        if cached is not None:
            LOGGER.info("Finnhub: cache hit year=%d (%d events)", year, len(cached))
            df_y = _finnhub_to_df(cached)
        else:
            LOGGER.info("Finnhub: fetching year %d from API...", year)
            raw = _fetch_finnhub_year(key, year)
            if raw:
                _save_finnhub_cache(year, raw)
            df_y = _finnhub_to_df(raw)
        if not df_y.empty:
            frames.append(df_y)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[
        (df["datetime_utc"] >= start) & (df["datetime_utc"] <= end)
    ].sort_values("datetime_utc").reset_index(drop=True)


#  Layer 2: ForexFactory 

def _fetch_forexfactory_live() -> pd.DataFrame:
    rows: list[dict] = []
    for url in [_FF_WEEK_URL, _FF_MONTH_URL]:
        try:
            r = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
            r.raise_for_status()
            for ev in r.json():
                title    = str(ev.get("title", "") or ev.get("name", ""))
                impact   = str(ev.get("impact", "") or "").capitalize()
                currency = str(ev.get("country", "") or ev.get("currency", "") or "").upper()
                if impact not in ("High", "Medium"):
                    continue
                if "USD" not in currency and "US" not in currency:
                    continue
                ev_time = None
                for k in ("date", "datetime", "time"):
                    raw = ev.get(k)
                    if raw:
                        try:
                            ev_time = pd.Timestamp(raw, tz="UTC")
                            break
                        except Exception:
                            pass
                if ev_time is None:
                    continue
                rows.append({
                    "datetime_utc": ev_time, "event": title,
                    "impact": impact, "gold_direction": _direction_from_title(title),
                    "actual": None, "estimate": None, "prev": None,
                    "source": "forexfactory",
                })
            if rows:
                break
        except Exception as exc:
            LOGGER.debug("ForexFactory failed (%s): %s", url, exc)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df.sort_values("datetime_utc").drop_duplicates(
        subset=["datetime_utc", "event"]
    ).reset_index(drop=True)


#  Layer 3: Rule-based 

def _generate_rule_based_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict] = []
    start_py, end_py = start.to_pydatetime(), end.to_pydatetime()
    months: list[tuple[int, int]] = []
    cur = start_py.replace(day=1)
    while cur <= end_py:
        months.append((cur.year, cur.month))
        nm = cur.month % 12 + 1
        ny = cur.year + (1 if cur.month == 12 else 0)
        cur = cur.replace(year=ny, month=nm)

    def _add(dt: datetime, ev: str, gd: int) -> None:
        if start_py <= dt <= end_py:
            rows.append({"datetime_utc": dt, "event": ev, "impact": "High",
                         "gold_direction": gd, "actual": None, "estimate": None,
                         "prev": None, "source": "rule_based"})

    for year, month in months:
        _add(_first_friday(year, month, 13, 30),   "Non-Farm Payroll", -1)
        _add(_weekday_skip(datetime(year, month, 12, 13, 30, tzinfo=timezone.utc)), "CPI", +1)
        _add(_last_friday(year, month, 13, 30),    "PCE", +1)

    for ds in _FOMC_DATES:
        _add(datetime.fromisoformat(ds).replace(hour=19, tzinfo=timezone.utc), "FOMC", -1)

    for year, month in months:
        if month in (1, 4, 7, 10):
            try:
                dt = _weekday_skip(datetime(year, month, 30, 13, 30, tzinfo=timezone.utc))
            except ValueError:
                dt = _weekday_skip(datetime(year, month, 28, 13, 30, tzinfo=timezone.utc))
            _add(dt, "GDP", -1)

    if not rows:
        return pd.DataFrame(columns=[
            "datetime_utc", "event", "impact", "gold_direction",
            "actual", "estimate", "prev", "source",
        ])
    df = pd.DataFrame(rows)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df.sort_values("datetime_utc").reset_index(drop=True)


#  Unified public API 

def build_news_calendar(
    start_date: str | pd.Timestamp,
    end_date:   str | pd.Timestamp,
) -> pd.DataFrame:
    """
    Tạo lịch tin tức unified cho [start_date, end_date].
    Ưu tiên: Finnhub (actual/estimate)  ForexFactory  Rule-based.
    """
    start = pd.Timestamp(start_date, tz="UTC") if not isinstance(start_date, pd.Timestamp) else start_date
    end   = pd.Timestamp(end_date,   tz="UTC") if not isinstance(end_date,   pd.Timestamp) else end_date

    parts: list[pd.DataFrame] = []
    logs:  list[str] = []

    # Layer 1
    fh = fetch_finnhub_calendar(start, end)
    if not fh.empty:
        parts.append(fh)
        logs.append(f"Finnhub({len(fh)})")

    # Layer 2  only when range includes recent/future
    now = pd.Timestamp.now(tz="UTC")
    if end >= now - pd.Timedelta(days=35):
        try:
            ff = _fetch_forexfactory_live()
            if not ff.empty:
                ff_in = ff[(ff["datetime_utc"] >= start) & (ff["datetime_utc"] <= end)]
                if not ff_in.empty:
                    parts.append(ff_in)
                    logs.append(f"ForexFactory({len(ff_in)})")
        except Exception as exc:
            LOGGER.debug("ForexFactory skipped: %s", exc)

    # Layer 3
    rb = _generate_rule_based_calendar(start, end)
    if not rb.empty:
        parts.append(rb)
        logs.append(f"RuleBased({len(rb)})")

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    combined["datetime_utc"] = pd.to_datetime(combined["datetime_utc"], utc=True)
    _prio = {"finnhub": 0, "forexfactory": 1, "rule_based": 2}
    combined["_p"] = combined["source"].map(_prio).fillna(9).astype(int)
    combined = (
        combined
        .sort_values(["datetime_utc", "_p"])
        .drop_duplicates(subset=["datetime_utc", "event"], keep="first")
        .drop(columns="_p")
        .sort_values("datetime_utc")
        .reset_index(drop=True)
    )
    LOGGER.info(
        "build_news_calendar: %d events [%s] %s%s",
        len(combined), " + ".join(logs), start.date(), end.date(),
    )
    return combined


def attach_news_features(
    m15_df: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date:   str | pd.Timestamp | None = None,
    news_calendar: pd.DataFrame | None = None,
    blackout_hours: float = 1.0,
    lookahead_hours: float = 4.0,
) -> pd.DataFrame:
    """
    Thêm 5 news features vào M15 DataFrame (fully vectorized O(n log m)):
      news_impact_ahead, news_hours_ahead, news_hours_since,
      news_surprise_gold, news_is_blackout
    """
    df = m15_df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)

    if news_calendar is None:
        sd = start_date if start_date is not None else df["time"].min() - pd.Timedelta(days=1)
        ed = end_date   if end_date   is not None else df["time"].max() + pd.Timedelta(days=1)
        news_calendar = build_news_calendar(sd, ed)

    _defaults = {"news_impact_ahead": 0, "news_hours_ahead": 48.0,
                 "news_hours_since": 48.0, "news_surprise_gold": 0,
                 "news_is_blackout": 0}
    if news_calendar is None or news_calendar.empty:
        for col, val in _defaults.items():
            df[col] = val
        return df

    nc = news_calendar.copy()
    nc["datetime_utc"] = pd.to_datetime(nc["datetime_utc"], utc=True)
    nc = nc.sort_values("datetime_utc").reset_index(drop=True)
    hi = nc[nc["impact"] == "High"].reset_index(drop=True)

    if hi.empty:
        for col, val in _defaults.items():
            df[col] = val
        return df

    bar_ns  = df["time"].values.astype("int64")
    hi_ns   = hi["datetime_utc"].values.astype("int64")
    hi_gold = hi["gold_direction"].fillna(0).astype(int).values

    NS    = int(3_600_000_000_000)
    ns_la = int(lookahead_hours * NS)
    ns_bo = int(blackout_hours  * NS)
    ns_2h = int(2 * NS)
    ns_48 = int(48 * NS)

    idx_n  = np.searchsorted(hi_ns, bar_ns, side="right")
    idx_p  = idx_n - 1

    valid_n  = idx_n < len(hi_ns)
    clamp_n  = np.minimum(idx_n, len(hi_ns) - 1)
    diff_n   = np.where(valid_n, hi_ns[clamp_n] - bar_ns, ns_48 + NS)
    hours_ahead  = np.clip(diff_n.astype(float) / NS, 0.0, 48.0)
    in_la        = valid_n & (diff_n <= ns_la)
    gold_pre     = np.where(in_la, hi_gold[clamp_n], 0)

    valid_p  = idx_p >= 0
    clamp_p  = np.maximum(idx_p, 0)
    diff_p   = np.where(valid_p, bar_ns - hi_ns[clamp_p], ns_48 + NS)
    hours_since  = np.clip(diff_p.astype(float) / NS, 0.0, 48.0)
    in_post      = valid_p & (diff_p <= ns_2h)
    gold_post    = np.where(in_post, hi_gold[clamp_p], 0)

    df["news_impact_ahead"]  = np.where(in_la, 2, 0).astype(int)
    df["news_hours_ahead"]   = hours_ahead.round(2)
    df["news_hours_since"]   = hours_since.round(2)
    df["news_surprise_gold"] = np.where(gold_post != 0, gold_post, gold_pre).astype(int)
    df["news_is_blackout"]   = (
        (valid_n & (diff_n <= ns_bo)) | (valid_p & (diff_p <= ns_bo))
    ).astype(int)

    src_cnt = nc["source"].value_counts().to_dict() if "source" in nc.columns else {}
    LOGGER.info(
        "attach_news_features: %d bars | hi=%d | blackout=%d | sources=%s",
        len(df), len(hi), int(df["news_is_blackout"].sum()), src_cnt,
    )
    return df
