"""
Bridge → Dashboard Status Adapter
Translates bridge_live_last_report.json → live_status_acc1.json + paper_trade_signals_acc1.csv
so the existing dashboard shows rolling_v3 bridge state without code changes.
Also writes bridge_signal_log.json — circular buffer of last 200 orders (all statuses
including blocked/skipped) for the "all signals including rejected" dashboard log.

Usage (called automatically by run_rolling_v3_bridge_loop.ps1 after each cycle):
    python scripts/bridge_status_adapter.py
"""
from __future__ import annotations

import csv
import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

REPORT = OUTPUTS / "target1200_live_rolling_v3_current_selected" / "bridge_live_last_report.json"
BRIDGE_STATE = OUTPUTS / "target1200_live_rolling_v3_current_selected" / "bridge_live_state.json"
TARGET_GUARD_REPORT = OUTPUTS / "target1200_live_rolling_v3_current_selected" / "equity_target_guard_report.json"
LIVE_STATUS = OUTPUTS / "live_status_acc1.json"
SIGNALS_CSV = OUTPUTS / "paper_trade_signals_acc1.csv"
SIGNAL_LOG = OUTPUTS / "bridge_signal_log.json"
ONLINE_DECISION = OUTPUTS / "target1200_live_rolling_v3_current_selected" / "online_decision.json"

SIGNAL_LOG_MAX = 200  # circular buffer size

CSV_HEADER = [
    "time", "should_trade", "side", "confidence", "reason",
    "entry_price", "stop_loss", "take_profit", "volume",
    "strategy_score", "volatility_regime",
    "account_balance", "open_positions", "max_positions",
]


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _already_logged(time_str: str) -> bool:
    if not SIGNALS_CSV.exists():
        return False
    try:
        with open(SIGNALS_CSV, encoding="utf-8") as f:
            for row in f:
                if time_str in row:
                    return True
    except Exception:
        pass
    return False


def _load_signal_log() -> list:
    try:
        return json.loads(SIGNAL_LOG.read_text(encoding="utf-8"))
    except Exception:
        return []


def _append_signal_log(new_entries: list) -> None:
    if not new_entries:
        return
    existing = _load_signal_log()
    combined = existing + new_entries
    if len(combined) > SIGNAL_LOG_MAX:
        combined = combined[-SIGNAL_LOG_MAX:]
    tmp = SIGNAL_LOG.with_suffix(".tmp")
    tmp.write_text(json.dumps(combined, indent=None, ensure_ascii=False), encoding="utf-8")
    tmp.replace(SIGNAL_LOG)


def _upsert_signal_log(new_entries: list) -> None:
    """Replace same-bar rows and drop malformed legacy rows."""
    if not new_entries:
        return
    new_keys = {(e.get("bar_time"), e.get("signal_time")) for e in new_entries}
    existing = [
        e
        for e in _load_signal_log()
        if e.get("bar_time") and (e.get("bar_time"), e.get("signal_time")) not in new_keys
    ]
    combined = existing + new_entries
    if len(combined) > SIGNAL_LOG_MAX:
        combined = combined[-SIGNAL_LOG_MAX:]
    tmp = SIGNAL_LOG.with_suffix(".tmp")
    tmp.write_text(json.dumps(combined, indent=None, ensure_ascii=False), encoding="utf-8")
    tmp.replace(SIGNAL_LOG)


def _bridge_get_json(bridge_url: str, path: str, params: dict | None = None) -> dict:
    if not bridge_url:
        return {}
    query = ""
    if params:
        query = "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(bridge_url.rstrip("/") + path + query, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def main() -> int:
    global REPORT, BRIDGE_STATE, TARGET_GUARD_REPORT, LIVE_STATUS, SIGNALS_CSV, SIGNAL_LOG, ONLINE_DECISION

    parser = argparse.ArgumentParser(description="Translate rolling_v3 bridge report into dashboard status files.")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--state", type=Path, default=BRIDGE_STATE)
    parser.add_argument("--target-guard-report", type=Path, default=TARGET_GUARD_REPORT)
    parser.add_argument("--live-status", type=Path, default=LIVE_STATUS)
    parser.add_argument("--signals-csv", type=Path, default=SIGNALS_CSV)
    parser.add_argument("--signal-log", type=Path, default=SIGNAL_LOG)
    parser.add_argument("--online-decision", type=Path, default=ONLINE_DECISION)
    args = parser.parse_args()

    REPORT = args.report
    BRIDGE_STATE = args.state
    TARGET_GUARD_REPORT = args.target_guard_report
    LIVE_STATUS = args.live_status
    SIGNALS_CSV = args.signals_csv
    SIGNAL_LOG = args.signal_log
    ONLINE_DECISION = args.online_decision

    report = _load_json(REPORT)
    state = _load_json(BRIDGE_STATE)
    target_guard = _load_json(TARGET_GUARD_REPORT)
    online_decision = _load_json(ONLINE_DECISION)
    online_base = online_decision.get("base_model") if isinstance(online_decision.get("base_model"), dict) else {}
    online_thresholds = online_decision.get("thresholds") if isinstance(online_decision.get("thresholds"), dict) else {}
    online_probability = float(
        online_decision.get("probability")
        or online_base.get("probability")
        or 0.0
    )
    online_threshold = float(
        online_decision.get("threshold")
        or online_thresholds.get("base_min_probability")
        or online_thresholds.get("feedback_min_probability")
        or 0.5
    )
    online_side = str(online_decision.get("side") or online_base.get("side") or online_base.get("model_side") or "")
    online_entry = float(online_decision.get("entry_price") or online_base.get("entry_price") or 0.0)
    online_sl = float(online_decision.get("sl_price") or online_base.get("sl_price") or 0.0)
    online_tp = float(online_decision.get("tp_price") or online_base.get("tp_price") or 0.0)

    if not report:
        print("bridge_status_adapter: no report found, skipping")
        return 0

    bridge_url = str(report.get("bridge_url") or "")
    account = report.get("account") or {}
    bridge_account = _bridge_get_json(bridge_url, "/account") if bridge_url else {}
    if bridge_account.get("connected") and (not account or not account.get("connected")):
        account = bridge_account
    balance = float(account.get("balance") or 0.0)
    equity = float(account.get("equity") or balance)
    margin = float(account.get("margin") or 0.0)
    status = str(report.get("status") or "idle")
    # Use error field as reason when report has error status
    reason = str(report.get("reason") or report.get("error") or "")
    bar_time = str(report.get("closed_bar_utc") or "")
    generated_at = str(report.get("generated_at_utc") or "")
    signal_max = str(report.get("signal_max_utc") or "")
    orders = report.get("orders") or []
    market_state = _bridge_get_json(bridge_url, "/market/state", {"symbol": "XAUUSD", "stale_seconds": 10})
    tick = _bridge_get_json(bridge_url, "/tick", {"symbol": "XAUUSD"})

    # When bridge account is unavailable, fall back to persisted state only as a last resort.
    if balance == 0.0 and state:
        _state_bal = float(state.get("initial_balance") or state.get("peak_equity") or 0.0)
        if _state_bal > 0.0:
            balance = _state_bal
            equity = _state_bal

    # Infer fields for dashboard
    sent_orders = [o for o in orders if o.get("status") == "sent"]
    should_trade = status in ("order_sent", "signal_ready") or len(sent_orders) > 0

    # Pick best signal info from first sent/dry-run order
    best_order = next((o for o in orders if o.get("status") in ("sent", "dry_run")), None)
    side = "buy"
    confidence = 0.0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    volume = 0.0
    if best_order:
        ob = best_order.get("order") or {}
        side = str(ob.get("side") or "buy")
        confidence = float(best_order.get("probability") or 0.0)
        entry_price = float(ob.get("entry_price") or 0.0)
        stop_loss = float(ob.get("stop_loss") or 0.0)
        take_profit = float(ob.get("take_profit") or 0.0)
        volume = float(best_order.get("lot") or ob.get("volume") or 0.0)
    elif orders:
        # Blocked orders — try to get side from last checked signal
        bo = orders[-1]
        ob2 = bo.get("order") or {}
        side = str(ob2.get("side") or "buy")
    elif online_decision:
        side = online_side or side
        confidence = online_probability
        entry_price = online_entry
        stop_loss = online_sl
        take_profit = online_tp

    display_reason = reason
    if str(online_decision.get("target_bar_utc") or "") == bar_time and not bool(online_decision.get("should_trade", False)):
        display_reason = (
            f"{online_decision.get('reason') or reason}; no order sent because probability "
            f"{online_probability:.4f} < threshold {online_threshold:.2f}"
        )

    open_positions = 1 if margin > 0.0 else 0
    # Count orders in state
    state_orders = state.get("orders") or []
    live_order_count = len([o for o in state_orders if o.get("order", {}).get("status") not in ("closed",)])

    # Build live_status_acc1.json (dashboard-compatible)
    live_status = {
        "ts": generated_at,
        "bar_time": bar_time,
        "account_balance": balance,
        "account_equity": equity,
        "open_positions": open_positions,
        "max_positions": int(report.get("current_fold", {}).get("max_positions") or 1) if report.get("current_fold") else 1,
        "confidence": confidence,
        "should_trade": should_trade,
        "side": side,
        "reason": f"[rolling_v3] {display_reason}",
        "strategy_score": confidence,
        "strategy_raw_score": confidence,
        "strategy_required_min": 0.5,
        "strategy_gate_pass": 1 if should_trade else 0,
        "volatility_regime": 1,
        "market_is_open": bool(account.get("connected", True)),
        "market_state_reason": str(market_state.get("reason") or ("OPEN" if account.get("connected") else "DISCONNECTED")),
        "market_tick_age_sec": float(market_state.get("tick_age_sec") or 0.0),
        "market_tick_is_fresh": bool(market_state.get("tick_is_fresh", False)),
        "market_trade_enabled": bool(market_state.get("trade_enabled", False)),
        "market_symbol": str(market_state.get("symbol") or tick.get("symbol") or "XAUUSD"),
        "market_bid": float(tick.get("bid") or 0.0),
        "market_ask": float(tick.get("ask") or 0.0),
        "market_tick_time_utc": str(market_state.get("tick_time_utc") or ""),
        "signal_threshold": 0.5,
        "signal_max_utc": signal_max,
        "rolling_v3_status": status,
        "rolling_v3_reason": display_reason,
        "rolling_v3_bar": bar_time,
        "rolling_v3_signal_max": signal_max,
        "rolling_v3_orders_today": live_order_count,
        "rolling_v3_online_bar": str(online_decision.get("target_bar_utc") or ""),
        "rolling_v3_online_probability": online_probability,
        "rolling_v3_online_should_trade": bool(online_decision.get("should_trade", False)),
        "rolling_v3_online_reason": str(online_decision.get("reason") or ""),
        "auto_trade_enabled": bool(report.get("execute", False)),
        "bridge_url": bridge_url,
        "magic": int(report.get("magic") or 0),
        "target_balance_stop": float(report.get("target_balance_stop") or target_guard.get("target_balance_stop") or 0.0),
        "target_guard_status": str(target_guard.get("status") or ""),
        "target_guard_reason": str(target_guard.get("reason") or ""),
        "target_guard_poll_seconds": float(target_guard.get("poll_seconds") or 0.0),
        "target_stop_triggered": bool((state.get("target_balance_stop") or {}).get("triggered") or state.get("target_balance_stop_triggered", False)),
    }
    _write_json_atomic(LIVE_STATUS, live_status)

    # Append to paper_trade_signals_acc1.csv for executed/dry-run orders
    orders_to_log = [o for o in orders if o.get("status") in ("sent", "dry_run") and not _already_logged(
        str((o.get("order") or {}).get("entry_price", "")) + str(o.get("signal_open_time_utc") or bar_time)
    )]

    if orders_to_log:
        write_header = not SIGNALS_CSV.exists() or os.path.getsize(SIGNALS_CSV) == 0
        with open(SIGNALS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for o in orders_to_log:
                ob = o.get("order") or {}
                row = {
                    "time": o.get("signal_open_time_utc") or bar_time,
                    "should_trade": True,
                    "side": ob.get("side") or "buy",
                    "confidence": round(float(o.get("probability") or 0.0), 5),
                    "reason": f"rolling_v3 {o.get('status','')}",
                    "entry_price": round(float(ob.get("entry_price") or 0.0), 5),
                    "stop_loss": round(float(ob.get("stop_loss") or 0.0), 5),
                    "take_profit": round(float(ob.get("take_profit") or 0.0), 5),
                    "volume": round(float(o.get("lot") or ob.get("volume") or 0.0), 4),
                    "strategy_score": round(float(o.get("probability") or 0.0), 5),
                    "volatility_regime": 1,
                    "account_balance": balance,
                    "open_positions": open_positions,
                    "max_positions": 1,
                }
                writer.writerow(row)
        print(f"bridge_status_adapter: logged {len(orders_to_log)} order(s) to {SIGNALS_CSV.name}")

    # Append/update ALL decisions (sent/dry_run/blocked/idle) to bridge_signal_log.json.
    # For online no-trade bars, preserve model score/side/plan from online_decision so
    # the dashboard shows why the executor stayed idle instead of a blank row.
    if True:
        new_log_entries: list = []
        if orders:
            for o in orders:
                ob = o.get("order") or {}
                new_log_entries.append({
                    "bar_time": bar_time,
                    "signal_time": o.get("signal_open_time_utc") or bar_time,
                    "status": o.get("status") or "unknown",
                    "reason": o.get("reason") or "",
                    "side": ob.get("side") or "",
                    "entry_price": float(ob.get("entry_price") or 0.0),
                    "stop_loss": float(ob.get("stop_loss") or 0.0),
                    "take_profit": float(ob.get("take_profit") or 0.0),
                    "probability": float(o.get("probability") or 0.0),
                    "lot": float(o.get("lot") or 0.0),
                    "signal_key": o.get("signal_key") or "",
                    "balance": balance,
                })
        else:
            # No orders — idle bar
            new_log_entries.append({
                "bar_time": bar_time,
                "signal_time": str(online_decision.get("target_bar_utc") or bar_time),
                "status": status,
                "reason": (
                    f"{online_decision.get('reason') or reason}; no order sent because probability "
                    f"{online_probability:.4f} < threshold "
                    f"{online_threshold:.2f}"
                )
                if str(online_decision.get("target_bar_utc") or "") == bar_time
                and not bool(online_decision.get("should_trade", False))
                else reason,
                "side": online_side
                if str(online_decision.get("target_bar_utc") or "") == bar_time
                else "",
                "entry_price": online_entry
                if str(online_decision.get("target_bar_utc") or "") == bar_time
                else 0.0,
                "stop_loss": online_sl
                if str(online_decision.get("target_bar_utc") or "") == bar_time
                else 0.0,
                "take_profit": online_tp
                if str(online_decision.get("target_bar_utc") or "") == bar_time
                else 0.0,
                "probability": online_probability
                if str(online_decision.get("target_bar_utc") or "") == bar_time
                else 0.0,
                "lot": 0.0,
                "signal_key": "",
                "balance": balance,
            })
        _upsert_signal_log(new_log_entries)

    print(f"bridge_status_adapter: status={status} balance={balance} bar={bar_time} signal_max={signal_max}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
