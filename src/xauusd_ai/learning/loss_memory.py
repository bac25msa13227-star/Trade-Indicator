from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class LossMemoryStore:
    """Optional mem0-backed loss-pattern store with JSONL fallback."""

    def __init__(self, account: str | None = None, output_dir: str | Path = "outputs") -> None:
        self.account = (account or os.environ.get("TRADING_ACCOUNT") or "acc1").strip().lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fallback_path = self.output_dir / f"loss_memory_{self.account}.jsonl"
        self.enabled = self._env_bool("MEM0_ENABLED", default=False)
        self._memory = self._init_mem0() if self.enabled else None

    def add_loss(self, event: dict[str, Any]) -> None:
        text = self._event_text(event)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "account": self.account,
            "text": text,
            "event": event,
        }
        if self._memory is not None:
            try:
                self._memory.add(
                    text,
                    user_id=f"trade_indicator_{self.account}",
                    metadata={"account": self.account, "kind": "loss_pattern"},
                )
                payload["mem0_status"] = "added"
            except Exception as exc:
                payload["mem0_status"] = f"fallback:{type(exc).__name__}"
                LOGGER.warning("mem0 add_loss failed; using JSONL fallback: %s", exc)
        else:
            payload["mem0_status"] = "disabled_or_unavailable"
        self._append(payload)

    def search_similar(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if self._memory is not None:
            try:
                result = self._memory.search(query, user_id=f"trade_indicator_{self.account}", limit=limit)
                if isinstance(result, list):
                    return result
                if isinstance(result, dict):
                    return result.get("results", []) or result.get("memories", []) or []
            except Exception as exc:
                LOGGER.warning("mem0 search failed; using JSONL fallback: %s", exc)

        if not self.fallback_path.exists():
            return []
        lines = self.fallback_path.read_text(encoding="utf-8").splitlines()[-200:]
        terms = {term.lower() for term in str(query).split() if len(term) >= 3}
        scored: list[tuple[int, dict[str, Any]]] = []
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(item.get("text", "")).lower()
            score = sum(1 for term in terms if term in text)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def _init_mem0(self) -> object | None:
        try:
            from mem0 import Memory  # type: ignore
        except Exception as exc:
            LOGGER.warning("MEM0_ENABLED=1 but mem0 is not importable: %s", exc)
            return None

        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        collection = os.environ.get("MEM0_COLLECTION", "xauusd_loss_patterns")
        llm_provider = os.environ.get("MEM0_LLM_PROVIDER", "openai")
        embedder_provider = os.environ.get("MEM0_EMBEDDER_PROVIDER", "openai")
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {"url": qdrant_url, "collection_name": collection},
            },
            "embedder": {"provider": embedder_provider, "config": {}},
            "llm": {"provider": llm_provider, "config": {}},
        }
        try:
            return Memory.from_config(config)
        except Exception as exc:
            LOGGER.warning("mem0 init failed; using JSONL fallback: %s", exc)
            return None

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _event_text(event: dict[str, Any]) -> str:
        reasons = " | ".join(str(reason) for reason in event.get("reasons", []))
        features = event.get("features", {})
        return (
            f"LOSS account={event.get('account')} ticket={event.get('ticket')} "
            f"side={event.get('side')} net_pnl={event.get('net_pnl')} "
            f"reasons={reasons} features={json.dumps(features, ensure_ascii=True, default=str)}"
        )

    def _append(self, payload: dict[str, Any]) -> None:
        with self.fallback_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
