from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def bridge_get_json(bridge_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
    query = "?" + urllib.parse.urlencode(params or {})
    with urllib.request.urlopen(bridge_url.rstrip("/") + path + query, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=float), q))


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample live MT5 spread distribution from bridge ticks.")
    parser.add_argument("--bridge-url", default="http://localhost:5601")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--max-compatible-p95", type=float, default=120.0)
    parser.add_argument("--stress-multiplier", type=float, default=1.5)
    args = parser.parse_args()

    info = bridge_get_json(args.bridge_url, "/symbol/info", {"symbol": args.symbol})
    point = float(info.get("point") or info.get("trade_tick_size") or 0.0)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    deadline = time.time() + max(int(args.seconds), 1)
    while time.time() < deadline:
        now = datetime.now(timezone.utc)
        try:
            tick = bridge_get_json(args.bridge_url, "/tick", {"symbol": args.symbol})
            bid = float(tick.get("bid") or 0.0)
            ask = float(tick.get("ask") or 0.0)
            spread_price = ask - bid
            spread_points = spread_price / point if point > 0 and bid > 0 and ask > 0 else float("nan")
            rows.append(
                {
                    "sample_utc": now.isoformat(),
                    "symbol": tick.get("symbol") or args.symbol,
                    "bid": bid,
                    "ask": ask,
                    "point": point,
                    "spread_price": spread_price,
                    "spread_points": spread_points,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"sample_utc": now.isoformat(), "symbol": args.symbol, "error": str(exc)})
        time.sleep(max(float(args.interval_seconds), 0.2))

    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["sample_utc", "symbol", "bid", "ask", "point", "spread_price", "spread_points", "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    spreads = [float(row["spread_points"]) for row in rows if "spread_points" in row and np.isfinite(float(row["spread_points"]))]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bridge_url": args.bridge_url,
        "symbol": args.symbol,
        "resolved_symbol": info.get("symbol") or info.get("name"),
        "point": point,
        "samples": len(rows),
        "valid_samples": len(spreads),
        "spread_points": {
            "min": min(spreads) if spreads else None,
            "p50": percentile(spreads, 50),
            "p90": percentile(spreads, 90),
            "p95": percentile(spreads, 95),
            "p99": percentile(spreads, 99),
            "max": max(spreads) if spreads else None,
        },
        "compatible_with_cap": bool(spreads and percentile(spreads, 95) <= float(args.max_compatible_p95)),
        "max_compatible_p95": float(args.max_compatible_p95),
        "recommended_stress_extra_roundtrip_points": (
            round(float(percentile(spreads, 95)) * float(args.stress_multiplier), 2) if spreads else None
        ),
        "recommendation": (
            "broker_or_symbol_ok_for_current_cap"
            if spreads and percentile(spreads, 95) <= float(args.max_compatible_p95)
            else "do_not_raise_cap; switch_to_lower_spread_symbol_or_broker_and_retest"
        ),
        "csv": str(args.out_csv),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
