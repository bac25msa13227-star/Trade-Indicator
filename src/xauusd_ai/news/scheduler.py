"""News Event Scheduler.

Runs a background thread that:
1. Periodically refreshes the Forex Factory calendar (every ``check_interval_s`` seconds).
2. Before each high/medium-impact news release (within ``pre_alert_minutes``),
   schedules a targeted "pre-release" crawl at T-``pre_crawl_minutes``.
3. At event time and ``post_crawl_seconds`` seconds after, crawls again to capture
   the *Actual* value and recompute the deviation.

All scheduled wake-ups use ``threading.Event.wait()`` so they do not busy-spin.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable, Optional

import pandas as pd

if TYPE_CHECKING:
    from xauusd_ai.news.crawler import ForexFactoryCrawler

LOGGER = logging.getLogger(__name__)


class NewsScheduler:
    """Background scheduler for Forex Factory news crawls.

    Parameters
    ----------
    crawler:
        A :class:`~xauusd_ai.news.crawler.ForexFactoryCrawler` instance.
    check_interval_s:
        How often (seconds) the scheduler polls for upcoming events. Default: 300 (5 min).
    pre_crawl_minutes:
        How many minutes before an event to perform a pre-release crawl. Default: 15.
    post_crawl_seconds:
        How many seconds after the event time to crawl for the Actual value. Default: 90.
    min_impact:
        Minimum impact level to schedule event-specific crawls.
        Accepted values: ``"Low"``, ``"Medium"``, ``"High"``. Default: ``"Medium"``.
    on_news_update:
        Optional callback called with the list of refreshed events after every crawl.
    """

    _IMPACT_ORDER = {"Bank Holiday": 0, "Low": 1, "Medium": 2, "High": 3}

    def __init__(
        self,
        crawler: ForexFactoryCrawler,
        check_interval_s: int = 300,
        pre_crawl_minutes: int = 15,
        post_crawl_seconds: int = 90,
        min_impact: str = "Medium",
        on_news_update: Optional[Callable[[list[dict]], None]] = None,
    ) -> None:
        self.crawler = crawler
        self.check_interval_s = check_interval_s
        self.pre_crawl_minutes = pre_crawl_minutes
        self.post_crawl_seconds = post_crawl_seconds
        self.min_impact_level = self._IMPACT_ORDER.get(min_impact, 2)
        self.on_news_update = on_news_update

        self._stop_event = threading.Event()
        self._main_thread: Optional[threading.Thread] = None
        # Track event times already scheduled so we don't double-schedule
        self._scheduled_keys: set[tuple] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public control API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the scheduler background thread (non-blocking)."""
        if self._main_thread and self._main_thread.is_alive():
            LOGGER.warning("NewsScheduler is already running.")
            return
        self._stop_event.clear()
        self._main_thread = threading.Thread(
            target=self._main_loop, name="NewsScheduler", daemon=True
        )
        self._main_thread.start()
        LOGGER.info("NewsScheduler started (check_interval=%ds).", self.check_interval_s)

    def stop(self) -> None:
        """Signal the scheduler to stop and wait for the thread to finish."""
        LOGGER.info("Stopping NewsScheduler…")
        self._stop_event.set()
        if self._main_thread:
            self._main_thread.join(timeout=10)
        LOGGER.info("NewsScheduler stopped.")

    def is_running(self) -> bool:
        return bool(self._main_thread and self._main_thread.is_alive())

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _main_loop(self) -> None:
        LOGGER.info("NewsScheduler main loop started.")
        # Initial crawl on startup
        self._do_crawl(label="startup")

        while not self._stop_event.is_set():
            self._schedule_upcoming_events()
            self._stop_event.wait(timeout=self.check_interval_s)

        LOGGER.info("NewsScheduler main loop exiting.")

    def _do_crawl(self, label: str = "") -> list[dict]:
        """Perform a crawl and call the optional callback."""
        try:
            LOGGER.info("Crawling Forex Factory calendar [%s]…", label)
            events = self.crawler.crawl_current_week()
            if self.on_news_update:
                try:
                    self.on_news_update(events)
                except Exception as cb_exc:  # noqa: BLE001
                    LOGGER.error("on_news_update callback raised: %s", cb_exc)
            return events
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Crawl failed [%s]: %s", label, exc)
            return []

    def _schedule_upcoming_events(self) -> None:
        """Look at the cached calendar and schedule precise crawls for near events."""
        news_df = self.crawler.load_as_dataframe()
        if news_df.empty:
            return

        now_utc = datetime.now(timezone.utc)
        lookahead = now_utc + timedelta(minutes=self.pre_crawl_minutes + 5)

        # Filter upcoming high/medium impact events in the look-ahead window
        if "datetime_utc" not in news_df.columns:
            return

        upcoming = news_df[
            (news_df["datetime_utc"] > now_utc)
            & (news_df["datetime_utc"] <= lookahead)
            & (
                news_df["impact"].map(self._IMPACT_ORDER).fillna(0)
                >= self.min_impact_level
            )
        ]

        for _, row in upcoming.iterrows():
            event_time: datetime = row["datetime_utc"]
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)
            key = (str(event_time), str(row.get("currency", "")), str(row.get("event", "")))

            with self._lock:
                if key in self._scheduled_keys:
                    continue
                self._scheduled_keys.add(key)

            LOGGER.info(
                "Scheduling crawls for [%s] %s %s at %s UTC",
                row.get("impact"),
                row.get("currency"),
                row.get("event"),
                event_time.strftime("%Y-%m-%d %H:%M"),
            )

            # Pre-release crawl at T - pre_crawl_minutes
            pre_time = event_time - timedelta(minutes=self.pre_crawl_minutes)
            if pre_time > now_utc:
                self._spawn_timed_crawl(pre_time, f"pre-release:{key[2]}")

            # At-release crawl at T+0
            if event_time > now_utc:
                self._spawn_timed_crawl(event_time, f"at-release:{key[2]}")

            # Post-release crawl at T + post_crawl_seconds (to capture Actual)
            post_time = event_time + timedelta(seconds=self.post_crawl_seconds)
            self._spawn_timed_crawl(post_time, f"post-release:{key[2]}")

    def _spawn_timed_crawl(self, target_time: datetime, label: str) -> None:
        """Spawn a daemon thread that sleeps until *target_time* then crawls."""

        def _run() -> None:
            now = datetime.now(timezone.utc)
            delay_s = (target_time - now).total_seconds()
            if delay_s > 0:
                LOGGER.debug("Timed crawl '%s' sleeping %.1f s", label, delay_s)
                # Use the stop event so we can interrupt if the scheduler stops
                self._stop_event.wait(timeout=max(delay_s, 0))
                if self._stop_event.is_set():
                    return
            self._do_crawl(label=label)

        thread = threading.Thread(target=_run, name=f"NewsCrawl-{label[:40]}", daemon=True)
        thread.start()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_scheduler(settings) -> NewsScheduler:  # noqa: ANN001
    """Create a :class:`NewsScheduler` configured from *settings*.

    Parameters
    ----------
    settings:
        The top-level :class:`~xauusd_ai.config.Settings` object.
    """
    from xauusd_ai.news.crawler import ForexFactoryCrawler

    crawler = ForexFactoryCrawler(
        cache_path=settings.news.cache_path,
        request_delay=settings.news.request_delay_seconds,
    )
    return NewsScheduler(
        crawler=crawler,
        check_interval_s=settings.news.refresh_interval_seconds,
        pre_crawl_minutes=settings.news.pre_crawl_minutes,
        post_crawl_seconds=settings.news.post_crawl_seconds,
        min_impact=settings.news.min_scheduled_impact,
    )
