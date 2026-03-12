"""NewsCrawler — Tải lịch tin tức kinh tế tự động.

Nguồn: ForexFactory JSON API (miễn phí, không cần auth)
  - Tuần này: https://nfs.faireconomy.media/ff_calendar_thisweek.json
  - Tháng này: https://nfs.faireconomy.media/ff_calendar_thismonth.json

Cache: outputs/news_events.json (tự động refresh sau cache_hours giờ)

Chức năng:
  - Block trade TRƯỚC tin (minutes_before) — bảo vệ spread lớn trước tin
  - Block trade SAU tin (minutes_after) — tránh biến động hậu tin
  - Bật trade_before_news=true để SCALP trước tin (rủi ro cao)
  - Bật trade_after_news=true để trade breakout sau tin
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)

_FF_WEEK_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_FF_MONTH_URL = "https://nfs.faireconomy.media/ff_calendar_thismonth.json"
_CACHE_PATH = Path("outputs/news_events.json")
_REQUEST_TIMEOUT = 15
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; XAUBot/1.0)"}


def _fetch_forexfactory() -> list[dict]:
    """Fetch lịch tin từ ForexFactory JSON API."""
    for url in [_FF_WEEK_URL, _FF_MONTH_URL]:
        try:
            r = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                LOGGER.info("NewsCrawler: fetched %d events from %s", len(data), url)
                return data
        except Exception as exc:
            LOGGER.warning("NewsCrawler: fetch from %s failed: %s", url, exc)
    return []


def _parse_event_time(ev: dict) -> datetime | None:
    """Parse thời gian tin tức từ nhiều format khác nhau."""
    for key in ("date", "datetime", "time"):
        raw = ev.get(key)
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(str(raw))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    return None


class NewsCrawler:
    """
    Tải và cache lịch tin tức kinh tế từ ForexFactory.
    Dùng trong live loop để filter trade gần news events.
    """

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._events: list[dict] = []
        news_cfg = getattr(getattr(settings, "integrations", None), "news", None)
        self._cache_hours: int = getattr(news_cfg, "cache_hours", 6)
        self._cache_path = _CACHE_PATH
        self._load_or_fetch()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _load_or_fetch(self) -> None:
        """Load từ cache. Nếu hết hạn hoặc không có → fetch mới."""
        if self._cache_path.exists():
            try:
                stored = json.loads(self._cache_path.read_text(encoding="utf-8"))
                fetched_at = datetime.fromisoformat(stored["fetched_at"])
                age_h = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
                if age_h < stored.get("cache_hours", self._cache_hours):
                    self._events = stored.get("events", [])
                    LOGGER.info(
                        "NewsCrawler: loaded %d events from cache (age=%.1fh)",
                        len(self._events), age_h,
                    )
                    return
            except Exception:
                pass
        self.refresh()

    def refresh(self) -> None:
        """Fetch mới từ ForexFactory và lưu cache."""
        events = _fetch_forexfactory()
        if events:
            self._events = events
            self._save_cache(events)
        else:
            LOGGER.warning("NewsCrawler: fetch failed, using %d cached events", len(self._events))

    def _save_cache(self, events: list[dict]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps({
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "cache_hours": self._cache_hours,
                "events": events,
            }, indent=2, default=str),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_events_in_window(
        self,
        now: datetime | None = None,
        minutes_before: int = 30,
        minutes_after: int = 30,
        currencies: list[str] | None = None,
        high_impact_only: bool = True,
    ) -> list[dict]:
        """
        Lấy danh sách tin trong khoảng:
            [now - minutes_after, now + minutes_before]
        """
        if now is None:
            now = datetime.now(timezone.utc)
        if currencies is None:
            currencies = ["USD"]
        impact_filter = {"High"} if high_impact_only else {"High", "Medium"}

        result = []
        for ev in self._events:
            ev_time = _parse_event_time(ev)
            if ev_time is None:
                continue

            # Lọc theo currency
            currency = (ev.get("country") or ev.get("currency") or "").upper()
            if not any(c.upper() in currency for c in currencies):
                continue

            # Lọc theo impact
            impact = (ev.get("impact") or "").capitalize()
            if impact not in impact_filter:
                continue

            # diff_minutes > 0 = tin chưa xảy ra (tương lai)
            diff_min = (ev_time - now).total_seconds() / 60
            if -minutes_after <= diff_min <= minutes_before:
                result.append({
                    **ev,
                    "ev_time": ev_time.isoformat(),
                    "diff_minutes": round(diff_min, 1),
                })

        # Sắp xếp: gần nhất trước
        result.sort(key=lambda x: abs(x["diff_minutes"]))
        return result

    def is_near_news(
        self,
        now: datetime | None = None,
        minutes_before: int = 30,
        minutes_after: int = 30,
        currencies: list[str] | None = None,
        high_impact_only: bool = True,
    ) -> tuple[bool, str, bool]:
        """
        Kiểm tra có tin quan trọng gần now không.

        Returns:
            (is_near, reason, is_before_news)
            is_before_news=True  → tin chưa xảy ra
            is_before_news=False → tin vừa xảy ra
        """
        events = self.get_events_in_window(
            now=now,
            minutes_before=minutes_before,
            minutes_after=minutes_after,
            currencies=currencies,
            high_impact_only=high_impact_only,
        )
        if not events:
            return False, "clear", False

        ev = events[0]
        title = ev.get("title", "?")
        currency = (ev.get("country") or ev.get("currency") or "")
        diff = ev["diff_minutes"]
        is_before = diff > 0
        direction = f"sau {abs(diff):.0f}p" if is_before else f"cách đây {abs(diff):.0f}p"
        reason = f"Tin {currency} '{title}' {direction}"
        return True, reason, is_before

    def get_next_events(self, limit: int = 5) -> list[dict]:
        """Lấy N tin sắp tới gần nhất (dùng cho dashboard)."""
        now = datetime.now(timezone.utc)
        result = []
        for ev in self._events:
            ev_time = _parse_event_time(ev)
            if ev_time is None:
                continue
            diff_min = (ev_time - now).total_seconds() / 60
            if diff_min > 0:
                result.append({**ev, "ev_time": ev_time.isoformat(), "diff_minutes": round(diff_min, 1)})
        result.sort(key=lambda x: x["diff_minutes"])
        return result[:limit]
