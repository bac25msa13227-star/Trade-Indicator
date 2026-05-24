from __future__ import annotations

import argparse
import json
import re
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd


TS_RE = re.compile(r"(?P<ts>\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})")
MARKET_RE = re.compile(
    r"(?P<ts>\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+market "
    r"(?P<side>buy|sell)\s+(?P<volume>[0-9.]+)\s+\S+\s+sl:\s+(?P<sl>[0-9.]+)\s+tp:\s+(?P<tp>[0-9.]+)"
)
ORDER_OPEN_RE = re.compile(
    r"(?P<ts>\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+order performed "
    r"(?P<side>buy|sell)\s+(?P<volume>[0-9.]+)\s+at\s+(?P<price>[0-9.]+)\s+\[#(?P<order>\d+)\s+"
)
FORCED_CLOSE_RE = re.compile(
    r"(?P<ts>\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+market "
    r"(?P<close_side>buy|sell)\s+(?P<volume>[0-9.]+)\s+\S+,\s+close #(?P<open_order>\d+)"
)
CLOSE_RE = re.compile(
    r"(?P<ts>\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"(?P<outcome>take profit|stop loss) triggered #(?P<open_order>\d+)\s+"
    r"(?P<side>buy|sell)\s+(?P<volume>[0-9.]+)\s+\S+\s+(?P<entry>[0-9.]+)\s+"
    r"sl:\s+(?P<sl>[0-9.]+)\s+tp:\s+(?P<tp>[0-9.]+)\s+\[#(?P<close_order>\d+)\s+"
    r"(?P<close_side>buy|sell)\s+(?P<close_volume>[0-9.]+)\s+\S+\s+at\s+(?P<close_price>[0-9.]+)\]"
)
BALANCE_RE = re.compile(
    r"(?P<ts>\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+BALANCE_UPDATE "
    r"balance=(?P<balance>-?[0-9.]+)\s+equity=(?P<equity>-?[0-9.]+)\s+profit=(?P<profit>-?[0-9.]+)"
)


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeError:
            continue
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON file: {path}")


def read_text_lines(path: Path) -> list[str]:
    for encoding in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            return path.read_text(encoding=encoding, errors="strict").splitlines()
        except UnicodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def resolve_path(base_manifest: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, base_manifest.parent / path, base_manifest.resolve().parents[1] / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def as_dt(value: Any) -> pd.Timestamp:
    return pd.to_datetime(str(value).replace(".", "-"), errors="coerce")


def signal_lookup(signal_csv: Path) -> dict[tuple[pd.Timestamp, int], dict[str, Any]]:
    signals = pd.read_csv(signal_csv)
    signals["open_dt"] = pd.to_datetime(signals["open_time"].astype(str).str.replace(".", "-", regex=False), errors="coerce")
    lookup: dict[tuple[pd.Timestamp, int], dict[str, Any]] = {}
    for _, row in signals.iterrows():
        direction = int(float(row.get("direction", 0)))
        key = (row["open_dt"].floor("min"), direction)
        lookup.setdefault(key, row.to_dict())
    return lookup


def parse_log(path: Path) -> list[dict[str, Any]]:
    pending_markets: deque[dict[str, Any]] = deque(maxlen=8)
    pending_forced_closes: deque[dict[str, Any]] = deque(maxlen=8)
    open_trades: dict[int, dict[str, Any]] = {}
    pending_balance_rows: deque[int] = deque()
    rows: list[dict[str, Any]] = []

    for line in read_text_lines(path):
        forced_close = FORCED_CLOSE_RE.search(line)
        if forced_close:
            pending_forced_closes.append(
                {
                    "close_time": as_dt(forced_close.group("ts")),
                    "close_side": forced_close.group("close_side"),
                    "volume": float(forced_close.group("volume")),
                    "open_order": int(forced_close.group("open_order")),
                }
            )
            continue

        market = MARKET_RE.search(line)
        if market:
            pending_markets.append(
                {
                    "open_time": as_dt(market.group("ts")),
                    "side": market.group("side"),
                    "volume": float(market.group("volume")),
                    "sl_price": float(market.group("sl")),
                    "tp_price": float(market.group("tp")),
                }
            )
            continue

        order_open = ORDER_OPEN_RE.search(line)
        if order_open and pending_forced_closes:
            side = order_open.group("side")
            volume = float(order_open.group("volume"))
            match_idx = None
            for idx, pending in enumerate(pending_forced_closes):
                if pending["close_side"] == side and abs(pending["volume"] - volume) < 1e-9:
                    match_idx = idx
                    break
            if match_idx is not None:
                pending = list(pending_forced_closes)[match_idx]
                pending_forced_closes.remove(pending)
                open_order = int(pending["open_order"])
                trade = open_trades.pop(open_order, {})
                rows.append(
                    {
                        **trade,
                        "open_order": open_order,
                        "close_order": int(order_open.group("order")),
                        "close_time": as_dt(order_open.group("ts")),
                        "outcome": "target_stop",
                        "close_price": float(order_open.group("price")),
                    }
                )
                pending_balance_rows.append(len(rows) - 1)
                continue

        if order_open and pending_markets:
            side = order_open.group("side")
            volume = float(order_open.group("volume"))
            open_time = as_dt(order_open.group("ts"))
            match_idx = None
            for idx, pending in enumerate(pending_markets):
                if pending["side"] == side and abs(pending["volume"] - volume) < 1e-9:
                    match_idx = idx
                    break
            if match_idx is not None:
                pending = list(pending_markets)[match_idx]
                pending_markets.remove(pending)
                order = int(order_open.group("order"))
                open_trades[order] = {
                    "open_order": order,
                    "open_time": open_time,
                    "side": side,
                    "direction": 1 if side == "buy" else -1,
                    "volume": volume,
                    "entry_price": float(order_open.group("price")),
                    "sl_price": pending["sl_price"],
                    "tp_price": pending["tp_price"],
                }
            continue

        close = CLOSE_RE.search(line)
        if close:
            open_order = int(close.group("open_order"))
            trade = open_trades.pop(open_order, {})
            row = {
                **trade,
                "open_order": open_order,
                "close_order": int(close.group("close_order")),
                "close_time": as_dt(close.group("ts")),
                "outcome": "tp" if close.group("outcome") == "take profit" else "sl",
                "close_price": float(close.group("close_price")),
            }
            rows.append(row)
            pending_balance_rows.append(len(rows) - 1)
            continue

        balance = BALANCE_RE.search(line)
        if balance and pending_balance_rows:
            row_idx = pending_balance_rows.popleft()
            rows[row_idx]["balance_after"] = float(balance.group("balance"))
            rows[row_idx]["equity_after"] = float(balance.group("equity"))
            rows[row_idx]["profit"] = float(balance.group("profit"))

    return rows


def enrich(rows: list[dict[str, Any]], lookup: dict[tuple[pd.Timestamp, int], dict[str, Any]], fold: int) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        open_time = row.get("open_time")
        direction = int(row.get("direction", 0))
        open_minute = pd.Timestamp(open_time).floor("min")
        signal: dict[str, Any] = {}
        signal_lag_min: int | None = None
        for lag_min in (5, 0, 10):
            candidate = lookup.get((open_minute - pd.Timedelta(minutes=lag_min), direction), {})
            if candidate:
                signal = candidate
                signal_lag_min = lag_min
                break
        entry = float(row.get("entry_price") or 0.0)
        sl = float(row.get("sl_price") or 0.0)
        tp = float(row.get("tp_price") or 0.0)
        sl_dist = abs(entry - sl)
        tp_dist = abs(tp - entry)
        row.update(
            {
                "fold": fold,
                "open_time": pd.Timestamp(open_time).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(open_time) else None,
                "close_time": pd.Timestamp(row.get("close_time")).strftime("%Y-%m-%d %H:%M:%S")
                if pd.notna(row.get("close_time"))
                else None,
                "signal_probability": signal.get("probability"),
                "signal_atr": signal.get("atr"),
                "signal_open_time": signal.get("open_time"),
                "signal_entry_price": signal.get("entry_price"),
                "signal_sl_price": signal.get("sl_price"),
                "signal_tp_price": signal.get("tp_price"),
                "signal_match_lag_min": signal_lag_min,
                "hour": int(pd.Timestamp(open_time).hour) if pd.notna(open_time) else None,
                "weekday": int(pd.Timestamp(open_time).weekday()) if pd.notna(open_time) else None,
                "rr": (tp_dist / sl_dist) if sl_dist > 0 else None,
                "signal_matched": bool(signal),
            }
        )
        enriched.append(row)
    return enriched


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MT5 tester-log position feedback joined to signal rows.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    results_path = args.results or (args.manifest.parent / "mt5_wf_results.csv")
    results = pd.read_csv(results_path)
    fold_to_tester_log = {
        int(row["fold"]): Path(str(row["tester_log"]))
        for _, row in results.iterrows()
        if "tester_log" in row and pd.notna(row["tester_log"])
    }

    all_rows: list[dict[str, Any]] = []
    for fold_meta in manifest.get("folds", []) or []:
        fold = int(fold_meta["fold"])
        tester_log = fold_to_tester_log.get(fold)
        if tester_log is None or not tester_log.exists():
            continue
        csv_path = resolve_path(args.manifest, str(fold_meta["csv"]))
        lookup = signal_lookup(csv_path)
        all_rows.extend(enrich(parse_log(tester_log), lookup, fold))

    frame = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = {
        "rows": int(len(frame)),
        "folds": sorted(int(x) for x in frame["fold"].dropna().unique()) if not frame.empty else [],
        "matched_signals": int(frame["signal_matched"].sum()) if not frame.empty and "signal_matched" in frame else 0,
        "profit_sum": float(frame["profit"].sum()) if not frame.empty and "profit" in frame else 0.0,
    }
    print(json.dumps(summary, indent=2))
    print(f"saved={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
