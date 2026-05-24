from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def account_from_env(default: str = "acc1") -> str:
    account = os.environ.get("TRADING_ACCOUNT", default).strip().lower()
    return account or default


class DecisionLog:
    """Append-only, per-account trade decision journal with deterministic reflections."""

    def __init__(self, account: str | None = None, output_dir: str | Path = "outputs") -> None:
        self.account = account or account_from_env()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.md_path = self.output_dir / f"decision_log_{self.account}.md"
        self.jsonl_path = self.output_dir / f"decision_log_{self.account}.jsonl"

    def log_entry(self, trade_id: int | str, signal: dict[str, Any], market_data: dict[str, Any] | None = None) -> None:
        market_data = market_data or {}
        event = {
            "type": "decision_entry",
            "timestamp": self._now(),
            "account": self.account,
            "trade_id": trade_id,
            "symbol": signal.get("symbol") or market_data.get("symbol") or "XAUUSD",
            "side": signal.get("side"),
            "confidence": signal.get("confidence", signal.get("probability")),
            "rating": signal.get("rating"),
            "entry_price": signal.get("entry_price"),
            "stop_loss": signal.get("stop_loss"),
            "take_profit": signal.get("take_profit"),
            "reason": signal.get("reason"),
            "market": self._market_context(market_data),
            "status": "pending",
        }
        self._append(event, self._format_entry(event))

    def log_skip(self, reason: str, signal: dict[str, Any] | None = None, market_data: dict[str, Any] | None = None) -> None:
        signal = signal or {}
        market_data = market_data or {}
        event = {
            "type": "decision_skip",
            "timestamp": self._now(),
            "account": self.account,
            "symbol": signal.get("symbol") or market_data.get("symbol") or "XAUUSD",
            "side": signal.get("side"),
            "confidence": signal.get("confidence", signal.get("probability")),
            "rating": signal.get("rating"),
            "reason": reason,
            "market": self._market_context(market_data),
        }
        self._append(event, self._format_skip(event))

    def log_exit(
        self,
        trade_id: int | str,
        exit_price: float,
        exit_reason: str,
        bars_held: int,
        realized_rr: float,
        actual_slippage_pips: float | None = None,
    ) -> None:
        event = {
            "type": "decision_exit",
            "timestamp": self._now(),
            "account": self.account,
            "trade_id": trade_id,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "bars_held": bars_held,
            "realized_rr": realized_rr,
            "actual_slippage_pips": actual_slippage_pips,
            "reflection": self._reflection(exit_reason, realized_rr, bars_held, actual_slippage_pips),
        }
        self._append(event, self._format_exit(event))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _market_context(market_data: dict[str, Any]) -> dict[str, Any]:
        keys = ("atr", "atr_mean", "spread_pips", "volume_ratio", "session", "regime", "volatility_regime")
        return {key: market_data.get(key) for key in keys if market_data.get(key) is not None}

    @staticmethod
    def _reflection(exit_reason: str, realized_rr: float, bars_held: int, slippage: float | None) -> list[str]:
        rr = float(realized_rr)
        direction = "worked" if rr > 0 else "failed"
        exit_label = str(exit_reason or "unknown")
        notes = [
            f"The thesis {direction} with realized_rr={rr:.2f} after {int(bars_held)} bars.",
            f"Exit path was {exit_label}; compare this setup with similar session/regime entries before increasing size.",
        ]
        if rr < 0:
            notes.append("Next similar trade should check whether spread, volatility, or trend alignment deteriorated before entry.")
        else:
            notes.append("This setup can be reused as a positive reference only if live spread and fill lag stayed normal.")
        if slippage is not None:
            notes.append(f"Observed slippage was {float(slippage):.2f} pips and should be compared with the expected cost model.")
        return notes[:4]

    def _append(self, event: dict[str, Any], markdown: str) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=True, default=str) + "\n")
        with self.md_path.open("a", encoding="utf-8") as fh:
            fh.write(markdown.rstrip() + "\n\n")

    @staticmethod
    def _format_entry(event: dict[str, Any]) -> str:
        return (
            f"[{event['timestamp']} | {event['account']} | {event['symbol']} | "
            f"{str(event.get('side') or '').upper()} | confidence:{event.get('confidence')} | "
            f"rating:{event.get('rating') or 'n/a'} | pending]\n"
            "DECISION:\n"
            f"{event.get('reason') or 'No reason supplied.'}\n"
            f"ENTRY: {event.get('entry_price')} | SL: {event.get('stop_loss')} | TP: {event.get('take_profit')}\n"
            f"CONTEXT: {json.dumps(event.get('market') or {}, ensure_ascii=True)}\n"
            "<!-- ENTRY_END -->"
        )

    @staticmethod
    def _format_skip(event: dict[str, Any]) -> str:
        return (
            f"[{event['timestamp']} | {event['account']} | {event['symbol']} | SKIP]\n"
            "DECISION:\n"
            f"Skipped because: {event.get('reason')}\n"
            f"CONTEXT: {json.dumps(event.get('market') or {}, ensure_ascii=True)}\n"
            "<!-- SKIP_END -->"
        )

    @staticmethod
    def _format_exit(event: dict[str, Any]) -> str:
        reflection = "\n".join(f"- {line}" for line in event["reflection"])
        return (
            f"[{event['timestamp']} | {event['account']} | trade_id:{event.get('trade_id')} | closed]\n"
            f"OUTCOME: exit={event.get('exit_reason')} | price={event.get('exit_price')} | "
            f"rr={float(event.get('realized_rr') or 0.0):.2f} | bars={event.get('bars_held')}\n"
            "REFLECTION:\n"
            f"{reflection}\n"
            "<!-- REFLECTION_END -->"
        )
