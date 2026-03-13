"""Forex Factory Economic Calendar Crawler.

Fetches and parses the economic calendar from Forex Factory,
converting all timestamps to UTC and caching results as CSV.

Timezone: ForexFactory displays times in Eastern Time (ET). All datetimes stored as UTC.

Verified HTML structure (March 2026):
- Table:  <table class="calendar__table">
- Rows:   <tr class="calendar__row ...">
- Cells:  <td class="calendar__cell calendar__time">, etc.
- Impact: <span class="icon icon--ff-impact-{red|ora|yel|gra}">
"""
from __future__ import annotations

import csv
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytz
import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)

FF_BASE_URL = "https://www.forexfactory.com/calendar"
FF_TIMEZONE = pytz.timezone("America/New_York")

# Maps FF impact icon CSS suffix to human-readable label
_IMPACT_ICON_MAP = {
    "red": "High",
    "ora": "Medium",
    "yel": "Low",
    "gra": "Bank Holiday",
}

NEWS_COLUMNS = [
    "datetime_utc",
    "currency",
    "impact",
    "event",
    "actual",
    "forecast",
    "previous",
    "deviation",
    "crawled_at",
]

_SUFFIX_MULTIPLIER: dict[str, float] = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}

# Minimal headers – verified to return fully server-rendered HTML
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def parse_numeric(value: str) -> Optional[float]:
    """Parse strings like '245K', '3.2%', '-1.5B', '(50)' to float, or None."""
    if not value or value.strip() in ("", "-", "N/A", "n/a"):
        return None
    value = value.strip().replace(",", "")
    is_negative = value.startswith("(") and value.endswith(")")
    if is_negative:
        value = value[1:-1]
    multiplier = 1.0
    if value.endswith("%"):
        value = value[:-1]
    elif value[-1:].upper() in _SUFFIX_MULTIPLIER:
        multiplier = _SUFFIX_MULTIPLIER[value[-1:].upper()]
        value = value[:-1]
    try:
        result = float(value) * multiplier
        return -result if is_negative else result
    except ValueError:
        return None


class ForexFactoryCrawler:
    """Crawls the ForexFactory calendar and persists results as a CSV cache.

    Example::

        crawler = ForexFactoryCrawler("outputs/news_data.csv")
        crawler.crawl_current_week()
        crawler.crawl_historical("2023-01-01")
    """

    def __init__(
        self,
        cache_path: str = "outputs/news_data.csv",
        request_delay: float = 3.0,
        timeout: int = 30,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.request_delay = request_delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(_DEFAULT_HEADERS)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl_current_week(self) -> list[dict]:
        """Fetch current-week calendar and save to cache. Returns event list."""
        events = self._fetch_and_parse(week_date=None)
        self.save_events(events)
        return events

    def crawl_week(self, week_date: datetime) -> list[dict]:
        """Fetch the week that contains *week_date* and save to cache."""
        events = self._fetch_and_parse(week_date=week_date)
        self.save_events(events)
        return events

    def crawl_historical(
        self,
        start_date: "str | datetime",
        end_date: "Optional[str | datetime]" = None,
    ) -> int:
        """Crawl week-by-week from *start_date* to *end_date* (today if None).

        Uses exponential backoff on HTTP errors to handle Cloudflare rate limits.
        Returns the total number of events saved.
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        # Align current to the Monday of start_date's week
        current = start_date - timedelta(days=start_date.weekday())
        current = current.replace(hour=0, minute=0, second=0, microsecond=0)

        total = 0
        consecutive_errors = 0
        while current <= end_date:
            LOGGER.info("Crawling week of %s", current.strftime("%Y-%m-%d"))
            events = self._fetch_week_with_retry(current)
            if events is None:
                consecutive_errors += 1
                # Exponential backoff: wait up to 120s on repeated failures
                wait = min(self.request_delay * (2 ** consecutive_errors), 120.0)
                LOGGER.warning("Retrying after %.0fs backoff (consecutive errors: %d)", wait, consecutive_errors)
                time.sleep(wait)
                continue
            consecutive_errors = 0
            self.save_events(events)
            total += len(events)
            current += timedelta(weeks=1)
            time.sleep(self.request_delay)

        LOGGER.info("Historical crawl complete. Total events: %d", total)
        return total

    def _fetch_week_with_retry(self, week_date: datetime, max_retries: int = 3):
        """Attempt to fetch a week with retries. Returns None on terminal failure."""
        monday = week_date - timedelta(days=week_date.weekday())
        date_str = monday.strftime("%b%d.%Y").lower()
        url = f"{FF_BASE_URL}?week={date_str}"

        for attempt in range(max_retries):
            try:
                LOGGER.debug("GET %s (attempt %d)", url, attempt + 1)
                # Re-establish session on retries to refresh cookies
                if attempt > 0:
                    time.sleep(self.request_delay * attempt)
                    self.session = requests.Session()
                    self.session.headers.update(_DEFAULT_HEADERS)
                    # Prime session with the base page to get Cloudflare cookies
                    self.session.get(FF_BASE_URL, timeout=self.timeout)
                    time.sleep(2)

                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 403:
                    LOGGER.warning("403 Forbidden for %s (attempt %d) – possible rate limit", url, attempt + 1)
                    if attempt < max_retries - 1:
                        time.sleep(30 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return self._parse_html(resp.text, ref_date=week_date)
            except requests.RequestException as exc:
                LOGGER.error("HTTP error fetching %s (attempt %d): %s", url, attempt + 1, exc)

        LOGGER.error("All %d attempts exhausted for %s. Skipping.", max_retries, url)
        return None  # Signal caller to apply backoff

    def load_as_dataframe(self):
        """Load the cached news CSV as a pandas DataFrame with UTC datetimes."""
        import pandas as pd

        if not self.cache_path.exists():
            return pd.DataFrame(columns=NEWS_COLUMNS)
        try:
            df = pd.read_csv(self.cache_path, dtype=str)
            df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
            return df.dropna(subset=["datetime_utc"]).reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not load news CSV: %s", exc)
            return pd.DataFrame(columns=NEWS_COLUMNS)

    def save_events(self, events: list[dict]) -> None:
        """Upsert *events* into the CSV cache (key = datetime+currency+event)."""
        if not events:
            return
        existing = self._load_existing_as_dict()
        for event in events:
            key = (event["datetime_utc"], event["currency"], event["event"])
            existing[key] = event  # overwrite or insert
        all_sorted = sorted(existing.values(), key=lambda r: r["datetime_utc"])
        with self.cache_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=NEWS_COLUMNS)
            writer.writeheader()
            writer.writerows(all_sorted)
        LOGGER.info("News cache updated: %d total events in %s", len(all_sorted), self.cache_path)

    # ------------------------------------------------------------------
    # Core parsing
    # ------------------------------------------------------------------

    def _fetch_and_parse(self, week_date: Optional[datetime]) -> list[dict]:
        if week_date is None:
            url = FF_BASE_URL
        else:
            # ForexFactory URL format: ?week=jan06.2026 (Monday of the week, lowercase)
            monday = week_date - timedelta(days=week_date.weekday())
            date_str = monday.strftime("%b%d.%Y").lower()
            url = f"{FF_BASE_URL}?week={date_str}"

        try:
            LOGGER.debug("GET %s", url)
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return self._parse_html(resp.text, ref_date=week_date)
        except requests.RequestException as exc:
            LOGGER.error("HTTP error fetching %s: %s", url, exc)
            return []
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Unexpected error fetching %s: %s", url, exc)
            return []

    def _parse_html(self, html: str, ref_date: Optional[datetime] = None) -> list[dict]:
        """Parse ForexFactory calendar HTML into a list of event dicts."""
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", class_="calendar__table")
        if table is None:
            LOGGER.warning("No calendar__table found in HTML.")
            return []

        # Match rows whose class LIST contains the substring "calendar__row"
        rows = table.find_all("tr", class_=lambda c: c and "calendar__row" in c)
        if not rows:
            LOGGER.warning("No calendar__row elements found.")
            return []

        now_et = datetime.now(FF_TIMEZONE)
        current_date: date = ref_date.date() if ref_date is not None else now_et.date()
        current_year = current_date.year
        current_time_et: Optional[datetime] = None
        events: list[dict] = []
        crawled_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        for row in rows:
            # --- Date ---
            date_cell = row.find("td", class_=lambda c: c and "calendar__date" in c)
            if date_cell:
                date_text = date_cell.get_text(separator=" ", strip=True)
                parsed_date = self._parse_date(date_text, current_year)
                if parsed_date is not None:
                    current_date = parsed_date
                    current_year = current_date.year

            # --- Time ---
            time_cell = row.find("td", class_=lambda c: c and "calendar__time" in c)
            if time_cell:
                time_text = time_cell.get_text(strip=True)
                if time_text and time_text.lower() not in ("all day", "tentative", ""):
                    parsed_time = self._parse_time_et(time_text, current_date)
                    if parsed_time is not None:
                        current_time_et = parsed_time

            # --- Currency ---
            currency_cell = row.find("td", class_=lambda c: c and "calendar__currency" in c)
            if not currency_cell:
                continue
            currency = currency_cell.get_text(strip=True)
            if not currency:
                continue

            # --- Impact ---
            impact = self._parse_impact(row)

            # --- Event name ---
            event_cell = row.find("td", class_=lambda c: c and "calendar__event" in c)
            if not event_cell:
                continue
            link = event_cell.find("a")
            event_name = (link or event_cell).get_text(strip=True)
            if not event_name:
                continue

            # --- Data fields ---
            actual = self._cell_text(row, "calendar__actual")
            forecast = self._cell_text(row, "calendar__forecast")
            previous = self._cell_text(row, "calendar__previous")

            # --- UTC timestamp ---
            if current_time_et is not None:
                dt_utc = current_time_et.astimezone(timezone.utc)
            else:
                dt_utc = FF_TIMEZONE.localize(
                    datetime(current_date.year, current_date.month, current_date.day)
                ).astimezone(timezone.utc)

            # --- Deviation ---
            actual_num = parse_numeric(actual)
            forecast_num = parse_numeric(forecast)
            deviation = ""
            if actual_num is not None and forecast_num is not None:
                deviation = str(round(actual_num - forecast_num, 6))

            events.append({
                "datetime_utc": dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "currency": currency,
                "impact": impact,
                "event": event_name,
                "actual": actual,
                "forecast": forecast,
                "previous": previous,
                "deviation": deviation,
                "crawled_at": crawled_at,
            })

        LOGGER.info("Parsed %d events from HTML", len(events))
        return events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_date(self, text: str, year: int) -> Optional[date]:
        """Parse 'Mon Mar 10', 'MonMar 9', 'Mar10', etc. into a date."""
        text = re.sub(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\s*", "", text, flags=re.I).strip()
        if not text:
            return None
        for fmt in ("%b %d", "%B %d", "%b%d", "%B%d"):
            try:
                return datetime.strptime(f"{text} {year}", f"{fmt} %Y").date()
            except ValueError:
                pass
        return None

    def _parse_time_et(self, time_str: str, ref_date: date) -> Optional[datetime]:
        """Parse '8:30am', '10:00pm' into an Eastern-Time-aware datetime."""
        time_str = time_str.strip().lower().replace(" ", "")
        for fmt in ("%I:%M%p", "%I%p"):
            try:
                parsed = datetime.strptime(time_str, fmt)
                naive_dt = datetime(ref_date.year, ref_date.month, ref_date.day, parsed.hour, parsed.minute)
                try:
                    return FF_TIMEZONE.localize(naive_dt, is_dst=None)
                except pytz.exceptions.AmbiguousTimeError:
                    return FF_TIMEZONE.localize(naive_dt, is_dst=False)
                except pytz.exceptions.NonExistentTimeError:
                    return FF_TIMEZONE.localize(naive_dt, is_dst=True)
            except ValueError:
                continue
        return None

    def _parse_impact(self, row) -> str:
        """Extract impact from span's CSS class: icon--ff-impact-{red|ora|yel|gra}."""
        impact_cell = row.find("td", class_=lambda c: c and "calendar__impact" in c)
        if not impact_cell:
            return "Low"
        span = impact_cell.find("span")
        if span:
            classes = " ".join(span.get("class", [])).lower()
            for suffix, label in _IMPACT_ICON_MAP.items():
                if f"impact-{suffix}" in classes:
                    return label
        return "Low"

    def _cell_text(self, row, css_keyword: str) -> str:
        """Get text from the first td whose class list contains *css_keyword*."""
        cell = row.find("td", class_=lambda c: c and css_keyword in c)
        return cell.get_text(strip=True) if cell else ""

    def _load_existing_as_dict(self) -> dict[tuple, dict]:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                return {
                    (row["datetime_utc"], row["currency"], row["event"]): row
                    for row in reader
                    if all(k in row for k in ("datetime_utc", "currency", "event"))
                }
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not load existing cache: %s", exc)
            return {}
