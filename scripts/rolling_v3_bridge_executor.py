from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("outputs/target1200_live_rolling_v3_current_selected/manifest.json")
DEFAULT_GUARD = Path("outputs/target1200_live_rolling_v3_current_selected/live_canary_guard_report.json")
DEFAULT_CIRCUIT_GUARD = Path("outputs/target1200_live_rolling_v3_current_selected/equity_target_guard_report.json")
DEFAULT_PREFLIGHT = Path("outputs/target1200_live_rolling_v3_current_selected/live_preflight_report.json")
DEFAULT_PAPER_REPORT = Path("outputs/target1200_live_rolling_v3_current_selected/paper_30day_report.json")
DEFAULT_PAPER_EVENT_LOG = Path("outputs/target1200_live_rolling_v3_current_selected/paper_bridge_events.jsonl")
DEFAULT_REALISTIC_GATE = Path("outputs/target1200_live_rolling_v3_current_selected/realistic_gate_report.json")
DEFAULT_STATE = Path("outputs/target1200_live_rolling_v3_current_selected/bridge_live_state.json")
DEFAULT_REPORT = Path("outputs/target1200_live_rolling_v3_current_selected/bridge_live_last_report.json")


@dataclass(frozen=True)
class Signal:
    open_time: datetime
    direction: int
    entry_price: float
    sl_price: float
    tp_price: float
    atr: float
    probability: float
    row_index: int

    @property
    def side(self) -> str:
        return "buy" if self.direction == 1 else "sell"


class BridgeError(RuntimeError):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_signal_time(value: str) -> datetime:
    return datetime.strptime(str(value).strip(), "%Y.%m.%d %H:%M").replace(tzinfo=timezone.utc)


def iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def resolve_path(raw_value: Any, base_path: Path) -> Path:
    raw = Path(str(raw_value))
    if raw.is_absolute():
        return raw
    candidates = [
        Path.cwd() / raw,
        base_path.parent / raw.name,
        base_path.parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / raw).resolve()


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _csv_int_set(value: str) -> set[int]:
    result: set[int] = set()
    for item in str(value or "").split(","):
        item = item.strip()
        if item:
            result.add(int(item))
    return result


def _maybe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _maybe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def bridge_call(
    bridge_url: str,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 15,
) -> Any:
    base = bridge_url.rstrip("/")
    query = ""
    if params:
        query = "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(base + path + query, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise BridgeError(f"{method} {path} HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise BridgeError(f"{method} {path} bridge unreachable: {exc}") from exc

    if isinstance(payload, dict) and payload.get("error"):
        raise BridgeError(f"{method} {path} bridge error: {payload['error']}")
    return payload


def find_current_fold(manifest: dict[str, Any], now: datetime) -> dict[str, Any] | None:
    for fold in manifest.get("folds", []) or []:
        start = datetime.fromisoformat(str(fold["test_start"])).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(str(fold["test_end"])).replace(tzinfo=timezone.utc)
        if start <= now < end:
            return dict(fold)
    return None


def load_signals(path: Path) -> list[Signal]:
    signals: list[Signal] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"open_time", "direction", "entry_price", "sl_price", "tp_price", "atr", "probability"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"signal CSV missing columns: {sorted(missing)}")
        for idx, row in enumerate(reader, start=1):
            signals.append(
                Signal(
                    open_time=parse_signal_time(row["open_time"]),
                    direction=1 if int(float(row["direction"])) >= 0 else -1,
                    entry_price=float(row["entry_price"]),
                    sl_price=float(row["sl_price"]),
                    tp_price=float(row["tp_price"]),
                    atr=float(row["atr"]),
                    probability=float(row["probability"]),
                    row_index=idx,
                )
            )
    signals.sort(key=lambda item: (item.open_time, item.row_index))
    return signals


def latest_m5_bar_times(rates: list[dict[str, Any]]) -> tuple[datetime, datetime]:
    if len(rates) < 2:
        raise ValueError("need at least 2 M5 bars from bridge to identify previous closed bar")
    times = sorted(int(row["time"]) for row in rates)
    return (
        datetime.fromtimestamp(times[-2], tz=timezone.utc),
        datetime.fromtimestamp(times[-1], tz=timezone.utc),
    )


def tick_time_utc(tick: dict[str, Any], fallback: datetime) -> datetime:
    raw = int(tick.get("time") or 0)
    if raw > 0:
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    return fallback


def signal_key(fold_number: int, signal: Signal, magic: int) -> str:
    raw = (
        f"{fold_number}|{signal.open_time.strftime('%Y%m%d%H%M')}|{signal.direction}|"
        f"{signal.entry_price:.5f}|{signal.sl_price:.5f}|{signal.tp_price:.5f}|{magic}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def short_comment(signal: Signal, key: str) -> str:
    side = "B" if signal.direction == 1 else "S"
    return f"rv3_{signal.open_time.strftime('%m%d%H%M')}_{side}_{key[:4]}"


def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        step = 0.01
    return math.floor((value + 1e-12) / step) * step


def price_risk_amount(symbol_info: dict[str, Any], volume: float, entry: float, stop_loss: float) -> float:
    dist = abs(float(entry) - float(stop_loss))
    if volume <= 0 or dist <= 0:
        return 0.0
    tick_value = float(symbol_info.get("trade_tick_value") or 0.0)
    tick_size = float(symbol_info.get("trade_tick_size") or 0.0)
    if tick_value > 0 and tick_size > 0:
        return (dist / tick_size) * tick_value * volume
    contract_size = float(symbol_info.get("trade_contract_size") or 100.0) or 100.0
    return dist * contract_size * volume


def position_risk(symbol_info: dict[str, Any], position: dict[str, Any]) -> float | None:
    sl = float(position.get("sl") or 0.0)
    if sl <= 0:
        return None
    return price_risk_amount(
        symbol_info,
        float(position.get("volume") or 0.0),
        float(position.get("open_price") or 0.0),
        sl,
    )


def calculate_lot(
    symbol_info: dict[str, Any],
    *,
    balance: float,
    risk_pct: float,
    max_risk_pct: float,
    entry: float,
    stop_loss: float,
    canary_mode: bool = False,
) -> tuple[float, float, float, str]:
    effective_risk_pct = min(float(risk_pct), float(max_risk_pct))
    max_order_risk = balance * effective_risk_pct / 100.0
    risk_per_lot = price_risk_amount(symbol_info, 1.0, entry, stop_loss)
    if balance <= 0 or max_order_risk <= 0 or risk_per_lot <= 0:
        return 0.0, 0.0, max_order_risk, "invalid balance/risk/SL distance"

    min_lot = float(symbol_info.get("volume_min") or 0.01)
    max_lot = min(float(symbol_info.get("volume_max") or 5.0), 5.0)
    step = float(symbol_info.get("volume_step") or 0.01)
    min_lot_risk = price_risk_amount(symbol_info, min_lot, entry, stop_loss)
    if canary_mode:
        if min_lot_risk > max_order_risk * 1.001:
            return (
                0.0,
                min_lot_risk,
                max_order_risk,
                f"canary min lot risk {min_lot_risk:.2f} exceeds cap {max_order_risk:.2f}",
            )
        return round(min_lot, 2), min_lot_risk, max_order_risk, "canary_min_lot"

    if min_lot_risk > max_order_risk * 1.001:
        return (
            0.0,
            min_lot_risk,
            max_order_risk,
            f"min lot risk {min_lot_risk:.2f} exceeds cap {max_order_risk:.2f}",
        )

    lot = floor_to_step(max_order_risk / risk_per_lot, step)
    if lot < min_lot:
        lot = min_lot
    lot = min(lot, max_lot)
    current_risk = price_risk_amount(symbol_info, lot, entry, stop_loss)
    if current_risk > max_order_risk * 1.001:
        lot = floor_to_step(max_order_risk / risk_per_lot, step)
        if lot < min_lot:
            return 0.0, current_risk, max_order_risk, "risk cap blocks even minimum lot"
        current_risk = price_risk_amount(symbol_info, lot, entry, stop_loss)
    return round(lot, 2), current_risk, max_order_risk, "ok"


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "created_at": iso(utc_now()),
            "executed_signal_keys": [],
            "orders": [],
            "peak_equity": 0.0,
            "initial_balance": 0.0,
            "target_balance_stop": {"triggered": False},
        }
    return read_json(path)


def update_equity_state(state: dict[str, Any], account: dict[str, Any], initial_balance_hint: float = 0.0) -> None:
    balance = float(account.get("balance") or 0.0)
    equity = float(account.get("equity") or balance)
    if float(state.get("initial_balance") or 0.0) <= 0 and balance > 0:
        state["initial_balance"] = float(initial_balance_hint) if initial_balance_hint > 0 else balance
    if equity > float(state.get("peak_equity") or 0.0):
        state["peak_equity"] = equity


def closed_losses_today(
    bridge_url: str,
    symbol: str,
    magic: int,
    now: datetime,
) -> tuple[float, list[dict[str, Any]]]:
    day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp()
    deals = bridge_call(
        bridge_url,
        "GET",
        "/history/closed",
        params={"symbol": symbol, "since": int(day_start), "magic": magic},
        timeout=20,
    )
    loss = 0.0
    if isinstance(deals, list):
        for deal in deals:
            pnl = float(deal.get("profit") or 0.0) + float(deal.get("swap") or 0.0) + float(deal.get("commission") or 0.0)
            if pnl < 0:
                loss += abs(pnl)
    return loss, deals if isinstance(deals, list) else []


def close_positions(
    bridge_url: str,
    positions: list[dict[str, Any]],
    magic: int,
    deviation: int,
    execute: bool,
) -> list[dict[str, Any]]:
    closed: list[dict[str, Any]] = []
    for position in positions:
        item = {"ticket": position.get("ticket"), "volume": position.get("volume"), "dry_run": not execute}
        if execute:
            response = bridge_call(
                bridge_url,
                "POST",
                f"/positions/{int(position['ticket'])}/close",
                body={"volume": float(position["volume"]), "deviation": deviation, "magic": magic},
                timeout=20,
            )
            item["response"] = response
        closed.append(item)
    return closed


def validate_guard(args: argparse.Namespace, now: datetime) -> dict[str, Any]:
    if not args.guard_report.exists():
        raise RuntimeError(f"guard report missing: {args.guard_report}")
    guard = read_json(args.guard_report)
    blockers = list(guard.get("blockers") or [])
    if not bool(guard.get("live_start_allowed")):
        blockers.append("live_start_allowed is false")
    guard_now_raw = guard.get("now_utc")
    if guard_now_raw:
        guard_now = parse_utc(str(guard_now_raw))
        guard_age_min = (now - guard_now).total_seconds() / 60.0
        if guard_age_min > float(args.max_guard_age_minutes):
            blockers.append(
                f"guard report is stale: {guard_age_min:.1f} min > {args.max_guard_age_minutes:.1f} min"
            )
    if blockers:
        raise RuntimeError("guard blocked bridge live: " + "; ".join(str(x) for x in blockers))
    return guard


def _source_manifest_path(manifest: dict[str, Any], manifest_path: Path) -> Path | None:
    source_validation = manifest.get("source_validation")
    if not isinstance(source_validation, dict):
        return None
    raw = source_validation.get("source_manifest")
    if not raw:
        return None
    return resolve_path(raw, manifest_path)


def _contamination_flags(prefix: str, payload: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if _is_true(payload.get("adaptive_per_fold")):
        blockers.append(f"{prefix} adaptive_per_fold=true")
    if _is_true(payload.get("research_oracle_fold_selection")):
        blockers.append(f"{prefix} research_oracle_fold_selection=true")
    if _is_true(payload.get("selection_uses_current_fold_metrics")):
        blockers.append(f"{prefix} selection_uses_current_fold_metrics=true")
    return blockers


def validate_live_protocol_clean(
    args: argparse.Namespace,
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    current_fold: dict[str, Any],
    guard: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    blockers.extend(_contamination_flags("manifest", manifest))

    if _is_true(current_fold.get("selected_candidate_adaptive_per_fold")):
        blockers.append(f"fold {current_fold.get('fold')} selected source candidate is adaptive_per_fold")
    if _is_true(current_fold.get("selected_candidate_research_oracle_fold_selection")):
        blockers.append(f"fold {current_fold.get('fold')} selected source candidate is research_oracle")
    if _is_true(current_fold.get("selected_candidate_selection_uses_current_fold_metrics")):
        blockers.append(f"fold {current_fold.get('fold')} selected source candidate uses current fold metrics")
    if str(current_fold.get("selection_mode") or "").startswith("bootstrap_declared") and not args.allow_bootstrap_source:
        blockers.append(f"fold {current_fold.get('fold')} uses bootstrap_declared selection")

    source_flags = manifest.get("source_manifest_flags")
    if isinstance(source_flags, dict):
        blockers.extend(_contamination_flags("source_manifest_flags", source_flags))

    source_path = _source_manifest_path(manifest, manifest_path)
    source_summary: dict[str, Any] = {"path": str(source_path) if source_path else None, "checked": False}
    if source_path is not None:
        if source_path.exists():
            source_manifest = read_json(source_path)
            source_summary.update(
                {
                    "checked": True,
                    "adaptive_per_fold": _is_true(source_manifest.get("adaptive_per_fold")),
                    "research_oracle_fold_selection": _is_true(source_manifest.get("research_oracle_fold_selection")),
                    "selection_uses_current_fold_metrics": _is_true(
                        source_manifest.get("selection_uses_current_fold_metrics")
                    ),
                }
            )
            if not args.allow_adaptive_source:
                blockers.extend(_contamination_flags("source_manifest", source_manifest))
        elif args.execute:
            blockers.append(f"source manifest missing: {source_path}")

    preflight_path = args.preflight_report or (manifest_path.parent / "live_preflight_report.json")
    preflight_summary: dict[str, Any] = {"path": str(preflight_path), "exists": preflight_path.exists()}
    if preflight_path.exists():
        preflight = read_json(preflight_path)
        preflight_summary["live_allowed"] = bool(preflight.get("live_allowed"))
        preflight_summary["blockers"] = preflight.get("blockers") or []
        warnings = [str(item) for item in preflight.get("warnings") or []]
        preflight_summary["warnings"] = warnings
        if not bool(preflight.get("live_allowed")):
            blockers.append(f"preflight live_allowed=false: {preflight.get('blockers', [])}")
        if preflight.get("blockers"):
            blockers.append(f"preflight blockers present: {preflight.get('blockers')}")
        dirty_terms = ("adaptive_per_fold", "hindsight", "research_oracle", "current fold metrics")
        dirty_warnings = [item for item in warnings if any(term in item for term in dirty_terms)]
        if dirty_warnings and not args.allow_adaptive_source:
            blockers.append(f"preflight contamination warnings present: {dirty_warnings}")
    elif args.execute:
        blockers.append(f"preflight report missing: {preflight_path}")

    guard_warnings = [str(item) for item in guard.get("warnings") or []]
    dirty_guard_warnings = [
        item
        for item in guard_warnings
        if any(term in item for term in ("adaptive_per_fold", "hindsight", "research_oracle", "current fold metrics"))
    ]
    if dirty_guard_warnings and not args.allow_adaptive_source:
        blockers.append(f"canary guard contamination warnings present: {dirty_guard_warnings}")

    summary = {
        "source_manifest": source_summary,
        "preflight": preflight_summary,
        "allow_adaptive_source": bool(args.allow_adaptive_source),
        "allow_bootstrap_source": bool(args.allow_bootstrap_source),
    }
    if blockers:
        raise RuntimeError("live protocol blocked: " + "; ".join(str(item) for item in blockers))
    return summary


def validate_execute_gate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        return {
            "execute": False,
            "paper_gate_required": bool(args.require_paper_gate),
            "realistic_gate_required": bool(args.require_realistic_gate),
        }

    blockers: list[str] = []
    paper_summary: dict[str, Any] = {"path": str(args.paper_report), "exists": args.paper_report.exists()}
    realistic_summary: dict[str, Any] = {
        "path": str(args.realistic_gate_report),
        "exists": args.realistic_gate_report.exists(),
    }
    if args.require_paper_gate and not args.allow_execute_without_paper:
        if not args.paper_report.exists():
            blockers.append(f"paper report missing: {args.paper_report}")
        else:
            paper = read_json(args.paper_report)
            trades = _maybe_int(paper.get("trades", paper.get("total_trades")), 0)
            days = _maybe_float(paper.get("days", paper.get("paper_days", paper.get("duration_days"))), 0.0)
            win_rate = _maybe_float(paper.get("win_rate_pct", paper.get("win_rate")), 0.0)
            if 0.0 < win_rate <= 1.0:
                win_rate *= 100.0
            paper_summary.update({"trades": trades, "days": days, "win_rate_pct": win_rate})
            if trades < int(args.min_paper_trades):
                blockers.append(f"paper trades {trades} < required {args.min_paper_trades}")
            if days < float(args.min_paper_days):
                blockers.append(f"paper days {days:.2f} < required {args.min_paper_days:.2f}")
            if win_rate < float(args.min_paper_win_rate_pct):
                blockers.append(f"paper win_rate {win_rate:.2f}% < required {args.min_paper_win_rate_pct:.2f}%")

    if args.require_canary_mode and not (args.canary_mode or args.allow_full_risk_live):
        blockers.append("execute requires --canary-mode until full-risk live is explicitly approved")

    if args.require_realistic_gate:
        if not args.realistic_gate_report.exists():
            blockers.append(f"realistic MT5 stress gate report missing: {args.realistic_gate_report}")
        else:
            realistic = read_json(args.realistic_gate_report)
            gate_summary = realistic.get("summary") if isinstance(realistic.get("summary"), dict) else {}
            realistic_summary.update(gate_summary)
            realistic_summary["scenario"] = realistic.get("scenario")
            if not bool(gate_summary.get("realistic_gate_pass")):
                blockers.append(
                    "realistic MT5 stress gate failed: "
                    f"target_pass={gate_summary.get('target_pass_folds')}/{gate_summary.get('folds')}, "
                    f"dd_pass={gate_summary.get('dd_pass_folds')}/{gate_summary.get('folds')}, "
                    f"loss_folds={gate_summary.get('loss_folds')}"
                )

    summary = {
        "execute": True,
        "paper_gate_required": bool(args.require_paper_gate),
        "paper": paper_summary,
        "realistic_gate_required": bool(args.require_realistic_gate),
        "realistic_gate": realistic_summary,
        "canary_mode": bool(args.canary_mode),
        "allow_full_risk_live": bool(args.allow_full_risk_live),
    }
    if blockers:
        raise RuntimeError("execute gate blocked: " + "; ".join(blockers))
    return summary


def session_guard(args: argparse.Namespace, now: datetime) -> dict[str, Any]:
    if not args.session_throttle:
        return {"enabled": False, "risk_multiplier": 1.0, "blocked": False}

    weekday = now.weekday()
    hour = now.hour
    if args.block_friday_close and weekday == 4 and hour >= int(args.friday_close_hour_utc):
        return {
            "enabled": True,
            "blocked": True,
            "reason": f"blocked: avoid Friday close gap risk hour_utc={hour}",
            "risk_multiplier": 0.0,
            "weekday": weekday,
            "hour_utc": hour,
        }

    thin_hours = _csv_int_set(args.asian_thin_hours_utc)
    thin_weekdays = _csv_int_set(args.asian_thin_weekdays)
    asian_thin = weekday in thin_weekdays and hour in thin_hours
    multiplier = float(args.asian_thin_risk_multiplier) if asian_thin else 1.0
    return {
        "enabled": True,
        "blocked": False,
        "asian_thin": asian_thin,
        "risk_multiplier": multiplier,
        "weekday": weekday,
        "hour_utc": hour,
    }


def read_optional_report(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            return read_json(path)
    except Exception:
        return {}
    return {}


def read_online_decision(path: Path) -> dict[str, Any]:
    if not path or str(path) in {".", ""}:
        return {}
    try:
        if path.exists():
            payload = read_json(path)
            return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        return {"read_error": str(exc), "path": str(path)}
    return {}


def build_report(status: str, args: argparse.Namespace, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "execute": bool(args.execute),
        "bridge_url": args.bridge_url,
        "symbol": args.symbol,
        "magic": int(args.magic),
        "generated_at_utc": iso(utc_now()),
    }
    payload.update(extra)
    return payload


def target_stop_record(state: dict[str, Any]) -> dict[str, Any]:
    raw = state.get("target_balance_stop")
    return raw if isinstance(raw, dict) else {}


def target_stop_is_latched(state: dict[str, Any]) -> bool:
    return bool(target_stop_record(state).get("triggered") or state.get("target_balance_stop_triggered"))


def mark_target_stop(
    state: dict[str, Any],
    *,
    account: dict[str, Any],
    target: float,
    reason: str,
    positions: list[dict[str, Any]],
) -> None:
    now = utc_now()
    state["target_balance_stop_triggered"] = True
    state["live_trading_disabled_reason"] = reason
    state["last_stop_reasons"] = [reason]
    state["target_balance_stop"] = {
        "triggered": True,
        "target": float(target),
        "triggered_at_utc": iso(now),
        "reason": reason,
        "account_at_trigger": account,
        "positions_at_trigger": positions,
    }


def maybe_enforce_target_balance_stop(
    args: argparse.Namespace,
    *,
    state: dict[str, Any],
    account: dict[str, Any],
    positions: list[dict[str, Any]],
    current_fold: dict[str, Any],
) -> dict[str, Any] | None:
    target = float(args.target_balance_stop or 0.0)
    if target <= 0:
        return None

    balance = float(account.get("balance") or 0.0)
    equity = float(account.get("equity") or balance)
    latched = target_stop_is_latched(state)
    reached = balance >= target or equity >= target
    if not latched and not reached:
        return None

    if latched:
        record = target_stop_record(state)
        reason = str(record.get("reason") or f"target balance stop already triggered at {target:.2f}")
    else:
        reason = f"target balance stop: balance={balance:.2f} equity={equity:.2f} >= {target:.2f}"
        if args.execute:
            mark_target_stop(state, account=account, target=target, reason=reason, positions=positions)
            write_json_atomic(args.state, state)

    closed = close_positions(args.bridge_url, positions, args.magic, args.deviation, args.execute and args.close_on_stop)
    if args.execute and latched and positions:
        state["last_stop_reasons"] = [reason]
        write_json_atomic(args.state, state)

    return build_report(
        "target_stop",
        args,
        reason=reason,
        current_fold=current_fold,
        account=account,
        positions=positions,
        closed_positions=closed,
        target_balance_stop=target,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    now = parse_utc(args.now) if args.now else utc_now()
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    guard = validate_guard(args, now)
    current_fold = find_current_fold(manifest, now)
    if current_fold is None:
        raise RuntimeError("no manifest fold covers current UTC time")
    live_protocol_summary = validate_live_protocol_clean(
        args,
        manifest=manifest,
        manifest_path=manifest_path,
        current_fold=current_fold,
        guard=guard,
    )
    execute_gate_summary = validate_execute_gate(args)

    signal_path = resolve_path(current_fold.get("csv"), manifest_path)
    signals = load_signals(signal_path)
    account = bridge_call(args.bridge_url, "GET", "/account")
    if not bool(account.get("connected")):
        raise RuntimeError("bridge account is not connected")
    if args.expected_login and int(account.get("login") or 0) != int(args.expected_login):
        raise RuntimeError(f"wrong MT5 account: expected {args.expected_login}, got {account.get('login')}")

    state = load_state(args.state)
    update_equity_state(state, account, float(args.initial_balance or 0.0))
    positions = bridge_call(args.bridge_url, "GET", "/positions", params={"symbol": args.symbol, "magic": args.magic})
    if not isinstance(positions, list):
        positions = []

    balance = float(account.get("balance") or 0.0)
    equity = float(account.get("equity") or balance)
    initial_balance = float(state.get("initial_balance") or args.initial_balance or balance or 0.0)
    if float(args.min_balance or 0.0) > 0 and balance < float(args.min_balance):
        return build_report(
            "blocked",
            args,
            reason=f"absolute min balance stop: balance={balance:.2f} < {args.min_balance:.2f}",
            account=account,
            positions=positions,
            initial_balance=initial_balance,
        )

    circuit_guard = read_optional_report(args.circuit_guard_report)
    circuit_status = str(circuit_guard.get("status") or "")
    if circuit_status in {"target_stop", "circuit_stop", "dd_stop", "min_balance_stop"}:
        return build_report(
            "blocked",
            args,
            reason=f"account circuit guard latched: {circuit_status}: {circuit_guard.get('reason', '')}",
            account=account,
            positions=positions,
            circuit_guard=circuit_guard,
            initial_balance=initial_balance,
        )

    target_stop_report = maybe_enforce_target_balance_stop(
        args,
        state=state,
        account=account,
        positions=positions,
        current_fold=current_fold,
    )
    if target_stop_report is not None:
        return target_stop_report

    market = bridge_call(
        args.bridge_url,
        "GET",
        "/market/state",
        params={"symbol": args.symbol, "stale_seconds": args.stale_seconds},
    )
    if not bool(market.get("is_open")):
        return build_report("blocked", args, reason=f"market closed: {market.get('reason')}", market=market)
    session = session_guard(args, now)
    if session.get("blocked"):
        return build_report(
            "blocked",
            args,
            reason=str(session.get("reason")),
            account=account,
            positions=positions,
            session_guard=session,
            live_protocol=live_protocol_summary,
            execute_gate=execute_gate_summary,
        )

    symbol_info = bridge_call(args.bridge_url, "GET", "/symbol/info", params={"symbol": args.symbol})
    rates = bridge_call(
        args.bridge_url,
        "GET",
        "/rates",
        params={"symbol": args.symbol, "timeframe": "M5", "bars": max(3, int(args.rates_bars))},
    )
    closed_bar, current_bar = latest_m5_bar_times(rates)
    matching = [signal for signal in signals if signal.open_time == closed_bar]

    last_seen_raw = state.get("last_seen_closed_bar_utc")
    last_seen = parse_utc(last_seen_raw) if last_seen_raw else None
    first_run = last_seen is None

    initial_balance = float(state.get("initial_balance") or args.initial_balance or account.get("balance") or 0.0)
    max_dd_kill_pct = float(args.max_dd_kill_pct)
    daily_loss, today_deals = closed_losses_today(args.bridge_url, args.symbol, args.magic, now)

    stop_reasons: list[str] = []
    if max_dd_kill_pct > 0 and initial_balance > 0:
        dd_from_initial = (initial_balance - equity) / initial_balance * 100.0
        if dd_from_initial >= max_dd_kill_pct:
            stop_reasons.append(f"initial DD kill {dd_from_initial:.2f}% >= {max_dd_kill_pct:.2f}%")
    if args.max_peak_dd_kill_pct > 0 and float(state.get("peak_equity") or 0.0) > 0:
        peak = float(state.get("peak_equity") or 0.0)
        peak_dd = (peak - equity) / peak * 100.0
        if peak_dd >= float(args.max_peak_dd_kill_pct):
            stop_reasons.append(f"peak DD kill {peak_dd:.2f}% >= {args.max_peak_dd_kill_pct:.2f}%")
    if args.max_daily_loss_pct > 0 and balance > 0:
        max_daily_loss = balance * float(args.max_daily_loss_pct) / 100.0
        if daily_loss >= max_daily_loss:
            stop_reasons.append(f"daily loss {daily_loss:.2f} >= {max_daily_loss:.2f}")

    if stop_reasons:
        closed = close_positions(args.bridge_url, positions, args.magic, args.deviation, args.execute and args.close_on_stop)
        if args.execute:
            state["last_stop_reasons"] = stop_reasons
            write_json_atomic(args.state, state)
        return build_report(
            "blocked",
            args,
            reason="; ".join(stop_reasons),
            closed_positions=closed,
            account=account,
            positions=positions,
            daily_loss=daily_loss,
            today_closed_deals=len(today_deals),
        )

    if first_run and matching and not args.allow_first_run_trade:
        if args.execute:
            state["last_seen_closed_bar_utc"] = iso(closed_bar)
            state["first_run_skipped_signal_bar_utc"] = iso(closed_bar)
            write_json_atomic(args.state, state)
        return build_report(
            "skipped",
            args,
            reason="first live bridge run skips current closed-bar signal; rerun on next M5 bar",
            closed_bar_utc=iso(closed_bar),
            matching_signals=len(matching),
            account=account,
            current_fold=current_fold,
            guard_now_utc=guard.get("now_utc"),
        )

    if last_seen is not None and closed_bar <= last_seen and not matching:
        return build_report(
            "idle",
            args,
            reason="closed bar already processed",
            closed_bar_utc=iso(closed_bar),
            last_seen_closed_bar_utc=iso(last_seen),
            matching_signals=len(matching),
            account=account,
        )

    if not matching:
        online_decision = read_online_decision(args.online_decision)
        online_target = parse_utc(str(online_decision.get("target_bar_utc") or "")) if online_decision.get("target_bar_utc") else None
        online_reason = str(online_decision.get("reason") or "")
        online_probability = online_decision.get("probability")
        online_threshold = online_decision.get("threshold")
        online_should_trade = bool(online_decision.get("should_trade", False))
        if online_target is not None and online_target < closed_bar:
            return build_report(
                "idle",
                args,
                reason=(
                    "online signal lag: generator evaluated "
                    f"{iso(online_target)} but latest closed M5 bar is {iso(closed_bar)}"
                ),
                closed_bar_utc=iso(closed_bar),
                online_decision_target_bar_utc=iso(online_target),
                online_decision_reason=online_reason,
                online_decision_probability=online_probability,
                online_decision_threshold=online_threshold,
                online_decision_should_trade=online_should_trade,
                signal_file=str(signal_path),
                signal_min_utc=iso(signals[0].open_time) if signals else None,
                signal_max_utc=iso(signals[-1].open_time) if signals else None,
                account=account,
            )
        if online_target is not None and online_target == closed_bar and not online_should_trade:
            reason = online_reason or "online decision did not pass"
            if online_probability is not None and online_threshold is not None:
                reason = (
                    f"{reason}; no order sent because probability "
                    f"{float(online_probability):.4f} < threshold {float(online_threshold):.2f}"
                )
            if args.execute:
                state["last_seen_closed_bar_utc"] = iso(closed_bar)
                write_json_atomic(args.state, state)
            return build_report(
                "idle",
                args,
                reason=reason,
                closed_bar_utc=iso(closed_bar),
                online_decision_target_bar_utc=iso(online_target),
                online_decision_reason=online_reason,
                online_decision_probability=online_probability,
                online_decision_threshold=online_threshold,
                online_decision_should_trade=online_should_trade,
                signal_file=str(signal_path),
                signal_min_utc=iso(signals[0].open_time) if signals else None,
                signal_max_utc=iso(signals[-1].open_time) if signals else None,
                account=account,
            )
        if args.execute:
            state["last_seen_closed_bar_utc"] = iso(closed_bar)
            write_json_atomic(args.state, state)
        return build_report(
            "idle",
            args,
            reason="no signal for latest closed M5 bar",
            closed_bar_utc=iso(closed_bar),
            signal_file=str(signal_path),
            signal_min_utc=iso(signals[0].open_time) if signals else None,
            signal_max_utc=iso(signals[-1].open_time) if signals else None,
            account=account,
        )

    tick = bridge_call(args.bridge_url, "GET", "/tick", params={"symbol": args.symbol})
    point = float(symbol_info.get("point") or symbol_info.get("trade_tick_size") or 0.0)
    bid = float(tick.get("bid") or 0.0)
    ask = float(tick.get("ask") or 0.0)
    spread = ask - bid
    spread_points = spread / point if point > 0 and ask > 0 and bid > 0 else math.inf
    if float(args.max_spread_points or 0.0) > 0 and spread_points > float(args.max_spread_points):
        if args.execute:
            state["last_seen_closed_bar_utc"] = iso(closed_bar)
            state.setdefault("skipped_wide_spread_bars", []).append(
                {
                    "closed_bar_utc": iso(closed_bar),
                    "bid": bid,
                    "ask": ask,
                    "spread_points": round(spread_points, 2),
                    "max_spread_points": float(args.max_spread_points),
                }
            )
            state["skipped_wide_spread_bars"] = state["skipped_wide_spread_bars"][-200:]
            write_json_atomic(args.state, state)
        return build_report(
            "blocked",
            args,
            reason=f"spread {spread_points:.1f} points exceeds cap {float(args.max_spread_points):.1f}",
            closed_bar_utc=iso(closed_bar),
            matching_signals=len(matching),
            tick=tick,
            spread_points=round(spread_points, 2),
            max_spread_points=float(args.max_spread_points),
            account=account,
            current_fold=current_fold,
        )
    entry_due = closed_bar + timedelta(minutes=5)
    entry_tick_time = tick_time_utc(tick, now)
    entry_lag_seconds = (entry_tick_time - entry_due).total_seconds()
    if float(args.max_entry_lag_seconds or 0.0) > 0 and entry_lag_seconds > float(args.max_entry_lag_seconds):
        if args.execute:
            state["last_seen_closed_bar_utc"] = iso(closed_bar)
            state.setdefault("skipped_stale_signal_bars", []).append(
                {
                    "closed_bar_utc": iso(closed_bar),
                    "entry_due_utc": iso(entry_due),
                    "entry_tick_time_utc": iso(entry_tick_time),
                    "entry_lag_seconds": round(entry_lag_seconds, 3),
                    "max_entry_lag_seconds": float(args.max_entry_lag_seconds),
                }
            )
            state["skipped_stale_signal_bars"] = state["skipped_stale_signal_bars"][-200:]
            write_json_atomic(args.state, state)
        return build_report(
            "blocked",
            args,
            reason=(
                f"entry lag {entry_lag_seconds:.1f}s exceeds cap "
                f"{float(args.max_entry_lag_seconds):.1f}s"
            ),
            closed_bar_utc=iso(closed_bar),
            current_bar_utc=iso(current_bar),
            entry_due_utc=iso(entry_due),
            entry_tick_time_utc=iso(entry_tick_time),
            entry_lag_seconds=round(entry_lag_seconds, 3),
            matching_signals=len(matching),
            account=account,
            current_fold=current_fold,
        )

    max_positions = int(current_fold.get("max_positions") or args.max_positions)
    manifest_risk_pct = float(current_fold.get("risk_pct") or args.risk_pct)
    manifest_max_risk_pct = float(current_fold.get("max_risk_pct") or manifest_risk_pct)
    manifest_max_exposure_pct = float(current_fold.get("max_exposure_pct") or args.max_exposure_pct)
    session_multiplier = float(session.get("risk_multiplier", 1.0) or 1.0)
    risk_pct = manifest_risk_pct * session_multiplier
    max_risk_pct = manifest_max_risk_pct * session_multiplier
    max_exposure_pct = manifest_max_exposure_pct * session_multiplier
    if float(args.live_risk_cap_pct or 0.0) > 0:
        risk_pct = min(risk_pct, float(args.live_risk_cap_pct))
        max_risk_pct = min(max_risk_pct, float(args.live_risk_cap_pct))
    if float(args.live_exposure_cap_pct or 0.0) > 0:
        max_exposure_pct = min(max_exposure_pct, float(args.live_exposure_cap_pct))

    unknown_risk = False
    open_risk = 0.0
    for position in positions:
        risk = position_risk(symbol_info, position)
        if risk is None:
            unknown_risk = True
        else:
            open_risk += risk

    if unknown_risk:
        return build_report("blocked", args, reason="open position missing SL; exposure unknown", positions=positions)
    if len(positions) >= max_positions:
        if args.execute:
            state["last_seen_closed_bar_utc"] = iso(closed_bar)
            write_json_atomic(args.state, state)
        return build_report(
            "blocked",
            args,
            reason=f"max positions reached: {len(positions)}/{max_positions}",
            positions=positions,
            closed_bar_utc=iso(closed_bar),
        )

    executed_keys = set(str(item) for item in state.get("executed_signal_keys") or [])
    orders: list[dict[str, Any]] = []
    position_count = len(positions)
    exposure_cap = balance * max_exposure_pct / 100.0 if max_exposure_pct > 0 else math.inf

    for signal in matching:
        if position_count >= max_positions:
            break
        key = signal_key(int(current_fold["fold"]), signal, int(args.magic))
        if key in executed_keys:
            orders.append({"signal_key": key, "status": "skipped", "reason": "signal already executed"})
            continue

        entry = float(tick.get("ask") if signal.side == "buy" else tick.get("bid"))
        ref_sl_dist = abs(signal.entry_price - signal.sl_price)
        intended_rr = abs(signal.tp_price - signal.entry_price) / ref_sl_dist if ref_sl_dist > 0 else 0.0
        if ref_sl_dist <= 0 or intended_rr <= 0:
            orders.append({"signal_key": key, "status": "blocked", "reason": "invalid signal SL/TP distances"})
            continue

        drift_r = signal.direction * (entry - signal.entry_price) / ref_sl_dist
        if drift_r < args.min_entry_drift_r or drift_r > args.max_entry_drift_r:
            orders.append(
                {
                    "signal_key": key,
                    "status": "blocked",
                    "reason": "entry drift outside guard",
                    "drift_r": round(drift_r, 4),
                    "allowed": [args.min_entry_drift_r, args.max_entry_drift_r],
                }
            )
            continue

        actual_sl = entry - ref_sl_dist if signal.side == "buy" else entry + ref_sl_dist
        lot, order_risk, max_order_risk, lot_reason = calculate_lot(
            symbol_info,
            balance=balance,
            risk_pct=risk_pct,
            max_risk_pct=max_risk_pct,
            entry=entry,
            stop_loss=actual_sl,
            canary_mode=bool(args.canary_mode),
        )
        if lot <= 0:
            orders.append({"signal_key": key, "status": "blocked", "reason": lot_reason})
            continue
        if open_risk + order_risk > exposure_cap * 1.001:
            orders.append(
                {
                    "signal_key": key,
                    "status": "blocked",
                    "reason": "exposure cap",
                    "open_risk": round(open_risk, 2),
                    "new_risk": round(order_risk, 2),
                    "cap": round(exposure_cap, 2),
                }
            )
            continue

        order_body = {
            "symbol": args.symbol,
            "side": signal.side,
            "volume": lot,
            "entry_price": signal.entry_price,
            "stop_loss": signal.sl_price,
            "take_profit": signal.tp_price,
            "deviation": int(args.deviation),
            "magic": int(args.magic),
            "comment": short_comment(signal, key),
        }
        order_record: dict[str, Any] = {
            "signal_key": key,
            "status": "dry_run" if not args.execute else "pending",
            "order": order_body,
            "closed_bar_utc": iso(closed_bar),
            "signal_open_time_utc": iso(signal.open_time),
            "probability": signal.probability,
            "entry_tick": entry,
            "drift_r": round(drift_r, 4),
            "lot": lot,
            "order_risk": round(order_risk, 2),
            "max_order_risk": round(max_order_risk, 2),
            "open_risk_before": round(open_risk, 2),
        }
        if args.execute:
            response = bridge_call(args.bridge_url, "POST", "/order", body=order_body, timeout=20)
            order_record["status"] = "sent"
            order_record["response"] = response
            executed_keys.add(key)
            state.setdefault("orders", []).append(
                {
                    "sent_at_utc": iso(utc_now()),
                    "signal_key": key,
                    "fold": int(current_fold["fold"]),
                    "order": order_body,
                    "response": response,
                }
            )
            open_risk += order_risk
            position_count += 1
        orders.append(order_record)

    if args.execute:
        state["executed_signal_keys"] = sorted(executed_keys)[-1000:]
        state["last_seen_closed_bar_utc"] = iso(closed_bar)
        write_json_atomic(args.state, state)

    sent_count = sum(1 for order in orders if order.get("status") == "sent")
    dry_count = sum(1 for order in orders if order.get("status") == "dry_run")
    status = "order_sent" if sent_count else ("signal_ready" if dry_count else "blocked")
    return build_report(
        status,
        args,
        reason="ok" if sent_count or dry_count else "matching signal blocked by live guards",
        closed_bar_utc=iso(closed_bar),
        current_fold=current_fold,
        account=account,
        market=market,
        symbol_info=symbol_info,
        positions_before=positions,
        matching_signals=len(matching),
        orders=orders,
        risk={
            "risk_pct": risk_pct,
            "max_risk_pct": max_risk_pct,
            "max_exposure_pct": max_exposure_pct,
            "manifest_risk_pct": manifest_risk_pct,
            "manifest_max_risk_pct": manifest_max_risk_pct,
            "manifest_max_exposure_pct": manifest_max_exposure_pct,
            "live_risk_cap_pct": float(args.live_risk_cap_pct or 0.0),
            "live_exposure_cap_pct": float(args.live_exposure_cap_pct or 0.0),
            "session_risk_multiplier": session_multiplier,
            "canary_mode": bool(args.canary_mode),
            "open_risk_before": round(open_risk, 2),
            "daily_loss": round(daily_loss, 2),
        },
        session_guard=session,
        live_protocol=live_protocol_summary,
        execute_gate=execute_gate_summary,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute rolling_v3 MT5 signals through the Python/HTTP bridge.")
    parser.add_argument("--bridge-url", default="http://localhost:5601")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--guard-report", type=Path, default=DEFAULT_GUARD)
    parser.add_argument("--circuit-guard-report", type=Path, default=DEFAULT_CIRCUIT_GUARD)
    parser.add_argument("--preflight-report", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--paper-report", type=Path, default=DEFAULT_PAPER_REPORT)
    parser.add_argument("--paper-event-log", type=Path, default=DEFAULT_PAPER_EVENT_LOG)
    parser.add_argument("--realistic-gate-report", type=Path, default=DEFAULT_REALISTIC_GATE)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--online-decision", type=Path, default=Path(""))
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--magic", type=int, default=13030)
    parser.add_argument("--expected-login", type=int, default=0)
    parser.add_argument("--execute", action="store_true", help="Actually POST /order. Default is dry-run.")
    parser.add_argument("--allow-first-run-trade", action="store_true")
    parser.add_argument("--allow-adaptive-source", action="store_true")
    parser.add_argument("--allow-bootstrap-source", action="store_true")
    parser.add_argument("--allow-execute-without-paper", action="store_true")
    parser.add_argument("--allow-full-risk-live", action="store_true")
    parser.add_argument("--no-paper-gate", dest="require_paper_gate", action="store_false")
    parser.add_argument("--no-canary-gate", dest="require_canary_mode", action="store_false")
    parser.add_argument("--no-realistic-gate", dest="require_realistic_gate", action="store_false")
    parser.add_argument("--canary-mode", action="store_true")
    parser.add_argument("--close-on-stop", action="store_true", default=True)
    parser.add_argument("--no-close-on-stop", dest="close_on_stop", action="store_false")
    parser.add_argument("--now", default="")
    parser.add_argument("--rates-bars", type=int, default=5)
    parser.add_argument("--stale-seconds", type=int, default=300)
    parser.add_argument("--max-guard-age-minutes", type=float, default=30.0)
    parser.add_argument("--initial-balance", type=float, default=200.0)
    parser.add_argument("--min-balance", type=float, default=0.0)
    parser.add_argument("--risk-pct", type=float, default=5.0)
    parser.add_argument("--max-risk-pct", type=float, default=5.0)
    parser.add_argument("--max-exposure-pct", type=float, default=5.0)
    parser.add_argument("--live-risk-cap-pct", type=float, default=1.0)
    parser.add_argument("--live-exposure-cap-pct", type=float, default=1.0)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--max-dd-kill-pct", type=float, default=20.0)
    parser.add_argument("--max-peak-dd-kill-pct", type=float, default=0.0)
    parser.add_argument("--max-daily-loss-pct", type=float, default=100.0)
    parser.add_argument("--target-balance-stop", type=float, default=1200.0)
    parser.add_argument("--min-entry-drift-r", type=float, default=-999.0)
    parser.add_argument("--max-entry-drift-r", type=float, default=999.0)
    parser.add_argument("--max-entry-lag-seconds", type=float, default=90.0)
    parser.add_argument("--max-spread-points", type=float, default=120.0)
    parser.add_argument("--min-paper-days", type=float, default=30.0)
    parser.add_argument("--min-paper-trades", type=int, default=100)
    parser.add_argument("--min-paper-win-rate-pct", type=float, default=50.0)
    parser.add_argument("--disable-session-throttle", dest="session_throttle", action="store_false")
    parser.add_argument("--asian-thin-hours-utc", default="2,3,4,5,6")
    parser.add_argument("--asian-thin-weekdays", default="0,1,2,3")
    parser.add_argument("--asian-thin-risk-multiplier", type=float, default=0.3)
    parser.add_argument("--block-friday-close", action="store_true", default=True)
    parser.add_argument("--no-block-friday-close", dest="block_friday_close", action="store_false")
    parser.add_argument("--friday-close-hour-utc", type=int, default=20)
    parser.add_argument("--deviation", type=int, default=30)
    parser.set_defaults(require_paper_gate=True, require_canary_mode=True, require_realistic_gate=True, session_throttle=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        report = run(args)
        if args.out:
            write_json_atomic(args.out, report)
        if not args.execute and args.paper_event_log:
            append_jsonl(args.paper_event_log, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] not in {"blocked"} else 2
    except Exception as exc:
        report = build_report("error", args, error=str(exc))
        if args.out:
            write_json_atomic(args.out, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
