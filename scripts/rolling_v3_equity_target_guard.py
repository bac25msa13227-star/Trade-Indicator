from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STATE = Path("outputs/target1200_live_rolling_v3_current_selected/equity_target_guard_state.json")
DEFAULT_REPORT = Path("outputs/target1200_live_rolling_v3_current_selected/equity_target_guard_report.json")


class BridgeError(RuntimeError):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def bridge_call(
    bridge_url: str,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    timeout: float = 5.0,
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


def load_state(path: Path) -> dict[str, Any]:
    state = read_json(path)
    if not state:
        state = {
            "version": 1,
            "created_at": iso(utc_now()),
            "executed_signal_keys": [],
            "orders": [],
            "peak_equity": 0.0,
            "initial_balance": 0.0,
        }
    return state


def target_stop_record(state: dict[str, Any]) -> dict[str, Any]:
    raw = state.get("target_balance_stop")
    return raw if isinstance(raw, dict) else {}


def target_stop_is_latched(state: dict[str, Any]) -> bool:
    return bool(target_stop_record(state).get("triggered") or state.get("target_balance_stop_triggered"))


def circuit_breaker_record(state: dict[str, Any]) -> dict[str, Any]:
    raw = state.get("circuit_breaker")
    return raw if isinstance(raw, dict) else {}


def circuit_breaker_is_latched(state: dict[str, Any]) -> bool:
    return bool(circuit_breaker_record(state).get("triggered") or state.get("circuit_breaker_triggered"))


def update_equity_state(state: dict[str, Any], account: dict[str, Any], initial_balance_hint: float = 0.0) -> None:
    balance = float(account.get("balance") or 0.0)
    equity = float(account.get("equity") or balance)
    if float(state.get("initial_balance") or 0.0) <= 0 and balance > 0:
        state["initial_balance"] = float(initial_balance_hint) if initial_balance_hint > 0 else balance
    if equity > float(state.get("peak_equity") or 0.0):
        state["peak_equity"] = equity


def mark_target_stop(
    state: dict[str, Any],
    *,
    account: dict[str, Any],
    positions: list[dict[str, Any]],
    target: float,
    reason: str,
) -> None:
    state["target_balance_stop_triggered"] = True
    state["live_trading_disabled_reason"] = reason
    state["last_stop_reasons"] = [reason]
    state["target_balance_stop"] = {
        "triggered": True,
        "target": float(target),
        "triggered_at_utc": iso(utc_now()),
        "reason": reason,
        "account_at_trigger": account,
        "positions_at_trigger": positions,
    }


def mark_circuit_breaker(
    state: dict[str, Any],
    *,
    account: dict[str, Any],
    positions: list[dict[str, Any]],
    status: str,
    reason: str,
) -> None:
    state["circuit_breaker_triggered"] = True
    state["live_trading_disabled_reason"] = reason
    state["last_stop_reasons"] = [reason]
    state["circuit_breaker"] = {
        "triggered": True,
        "status": status,
        "triggered_at_utc": iso(utc_now()),
        "reason": reason,
        "account_at_trigger": account,
        "positions_at_trigger": positions,
        "initial_balance": float(state.get("initial_balance") or 0.0),
        "peak_equity": float(state.get("peak_equity") or 0.0),
    }


def close_positions(args: argparse.Namespace, positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    closed: list[dict[str, Any]] = []
    for position in positions:
        item: dict[str, Any] = {
            "ticket": position.get("ticket"),
            "volume": position.get("volume"),
            "dry_run": not bool(args.execute),
        }
        if args.execute:
            response = bridge_call(
                args.bridge_url,
                "POST",
                f"/positions/{int(position['ticket'])}/close",
                body={
                    "volume": float(position["volume"]),
                    "deviation": int(args.deviation),
                    "magic": int(args.magic),
                },
                timeout=10.0,
            )
            item["response"] = response
        closed.append(item)
    return closed


def build_report(status: str, args: argparse.Namespace, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "execute": bool(args.execute),
        "bridge_url": args.bridge_url,
        "symbol": args.symbol,
        "magic": int(args.magic),
        "initial_balance_config": float(args.initial_balance),
        "min_balance": float(args.min_balance),
        "max_dd_kill_pct": float(args.max_dd_kill_pct),
        "max_peak_dd_kill_pct": float(args.max_peak_dd_kill_pct),
        "target_balance_stop": float(args.target_balance_stop),
        "poll_seconds": float(args.poll_seconds),
        "generated_at_utc": iso(utc_now()),
    }
    payload.update(extra)
    return payload


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    state = load_state(args.state)
    account = bridge_call(args.bridge_url, "GET", "/account", timeout=5.0)
    if not bool(account.get("connected")):
        return build_report("disconnected", args, account=account)

    update_equity_state(state, account, float(args.initial_balance or 0.0))
    positions = bridge_call(
        args.bridge_url,
        "GET",
        "/positions",
        params={"symbol": args.symbol, "magic": args.magic},
        timeout=5.0,
    )
    if not isinstance(positions, list):
        positions = []

    balance = float(account.get("balance") or 0.0)
    equity = float(account.get("equity") or balance)
    initial_balance = float(state.get("initial_balance") or args.initial_balance or balance or 0.0)
    peak_equity = float(state.get("peak_equity") or equity or 0.0)
    target = float(args.target_balance_stop or 0.0)
    latched = target_stop_is_latched(state)
    reached = target > 0 and (balance >= target or equity >= target)

    if latched or reached:
        record = target_stop_record(state)
        reason = str(record.get("reason") or f"target balance stop: balance={balance:.2f} equity={equity:.2f} >= {target:.2f}")
        if reached and not latched and args.execute:
            mark_target_stop(state, account=account, positions=positions, target=target, reason=reason)
            write_json_atomic(args.state, state)
        closed = close_positions(args, positions) if args.close_on_trigger else []
        report = build_report(
            "target_stop",
            args,
            reason=reason,
            account=account,
            positions=positions,
            closed_positions=closed,
        )
        write_json_atomic(args.out, report)
        return report

    circuit_latched = circuit_breaker_is_latched(state)
    circuit_status = "circuit_stop"
    circuit_reason = ""
    dd_floor = initial_balance * (1.0 - float(args.max_dd_kill_pct or 0.0) / 100.0) if initial_balance > 0 else 0.0
    peak_dd_floor = peak_equity * (1.0 - float(args.max_peak_dd_kill_pct or 0.0) / 100.0) if peak_equity > 0 else 0.0
    if circuit_latched:
        record = circuit_breaker_record(state)
        circuit_status = str(record.get("status") or "circuit_stop")
        circuit_reason = str(record.get("reason") or "account circuit breaker already triggered")
    elif float(args.min_balance or 0.0) > 0 and balance < float(args.min_balance):
        circuit_status = "min_balance_stop"
        circuit_reason = f"absolute min balance stop: balance={balance:.2f} < {float(args.min_balance):.2f}"
    elif float(args.max_dd_kill_pct or 0.0) > 0 and initial_balance > 0 and equity <= dd_floor:
        dd_pct = (initial_balance - equity) / initial_balance * 100.0
        circuit_status = "dd_stop"
        circuit_reason = (
            f"initial DD kill: equity={equity:.2f} <= floor={dd_floor:.2f} "
            f"(initial={initial_balance:.2f}, dd={dd_pct:.2f}%, max={float(args.max_dd_kill_pct):.2f}%)"
        )
    elif float(args.max_peak_dd_kill_pct or 0.0) > 0 and peak_equity > 0 and equity <= peak_dd_floor:
        peak_dd_pct = (peak_equity - equity) / peak_equity * 100.0
        circuit_status = "dd_stop"
        circuit_reason = (
            f"peak DD kill: equity={equity:.2f} <= floor={peak_dd_floor:.2f} "
            f"(peak={peak_equity:.2f}, dd={peak_dd_pct:.2f}%, max={float(args.max_peak_dd_kill_pct):.2f}%)"
        )

    if circuit_latched or circuit_reason:
        if circuit_reason and not circuit_latched and args.execute:
            mark_circuit_breaker(
                state,
                account=account,
                positions=positions,
                status=circuit_status,
                reason=circuit_reason,
            )
            write_json_atomic(args.state, state)
        closed = close_positions(args, positions) if args.close_on_trigger else []
        report = build_report(
            circuit_status,
            args,
            reason=circuit_reason,
            account=account,
            positions=positions,
            closed_positions=closed,
            initial_balance=initial_balance,
            peak_equity=peak_equity,
            max_dd_kill_pct=float(args.max_dd_kill_pct),
            max_peak_dd_kill_pct=float(args.max_peak_dd_kill_pct),
            dd_floor=round(dd_floor, 2) if dd_floor else None,
            peak_dd_floor=round(peak_dd_floor, 2) if peak_dd_floor else None,
        )
        write_json_atomic(args.out, report)
        return report

    if args.execute:
        write_json_atomic(args.state, state)
    return build_report(
        "watching",
        args,
        account=account,
        positions=positions,
        initial_balance=initial_balance,
        peak_equity=peak_equity,
        max_dd_kill_pct=float(args.max_dd_kill_pct),
        max_peak_dd_kill_pct=float(args.max_peak_dd_kill_pct),
        min_balance=float(args.min_balance),
        dd_floor=round(dd_floor, 2) if dd_floor else None,
        peak_dd_floor=round(peak_dd_floor, 2) if peak_dd_floor else None,
    )


def run_loop(args: argparse.Namespace) -> int:
    next_heartbeat = 0.0
    last_status = ""
    poll_seconds = max(float(args.poll_seconds), 0.05)
    while True:
        try:
            report = run_once(args)
            now = time.monotonic()
            status = str(report.get("status") or "")
            if args.once or status != last_status or now >= next_heartbeat:
                write_json_atomic(args.out, report)
                print(json.dumps(report, sort_keys=True), flush=True)
                next_heartbeat = now + max(float(args.heartbeat_seconds), poll_seconds)
                last_status = status
            if args.once:
                return 0 if report["status"] in {"watching", "target_stop", "circuit_stop", "dd_stop", "min_balance_stop"} else 2
            if args.exit_after_trigger and report["status"] in {"target_stop", "circuit_stop", "dd_stop", "min_balance_stop"}:
                return 0
        except Exception as exc:
            report = build_report("error", args, error=str(exc))
            write_json_atomic(args.out, report)
            print(json.dumps(report, sort_keys=True), flush=True)
            if args.once:
                return 1
        time.sleep(poll_seconds)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-frequency live account circuit guard for rolling_v3.")
    parser.add_argument("--bridge-url", default="http://localhost:5601")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--magic", type=int, default=13001)
    parser.add_argument("--initial-balance", type=float, default=200.0)
    parser.add_argument("--min-balance", type=float, default=0.0)
    parser.add_argument("--max-dd-kill-pct", type=float, default=20.0)
    parser.add_argument("--max-peak-dd-kill-pct", type=float, default=0.0)
    parser.add_argument("--target-balance-stop", type=float, default=1200.0)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    parser.add_argument("--deviation", type=int, default=30)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--close-on-trigger", action="store_true", default=True)
    parser.add_argument("--no-close-on-trigger", dest="close_on_trigger", action="store_false")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--exit-after-trigger", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if (
        float(args.target_balance_stop or 0.0) <= 0
        and float(args.max_dd_kill_pct or 0.0) <= 0
        and float(args.max_peak_dd_kill_pct or 0.0) <= 0
        and float(args.min_balance or 0.0) <= 0
    ):
        report = build_report("disabled", args, reason="all account circuit guards disabled")
        write_json_atomic(args.out, report)
        print(json.dumps(report, sort_keys=True))
        return 0
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
