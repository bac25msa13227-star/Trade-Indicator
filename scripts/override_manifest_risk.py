from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except (UnicodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not parse JSON: {path}")


def parse_overrides(values: list[str]) -> dict[int, float]:
    result: dict[int, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Risk override must be FOLD=RISK_PCT, got {value!r}")
        fold_text, risk_text = value.split("=", 1)
        result[int(fold_text.strip())] = float(risk_text.strip())
    return result


def resolve_csv(manifest_path: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw
    for candidate in [Path.cwd() / raw, manifest_path.parent / raw.name, manifest_path.parent / raw]:
        if candidate.exists():
            return candidate
    return Path.cwd() / raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy or update a manifest and apply risk overrides.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--risk", action="append", default=[])
    parser.add_argument("--risk-pct", type=float, default=None)
    parser.add_argument("--max-risk-pct", type=float, default=None)
    parser.add_argument("--max-exposure-pct", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    args = parser.parse_args()

    overrides = parse_overrides(args.risk)
    source = read_json(args.manifest)
    out_dir = args.out_dir or args.manifest.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    in_place = args.out_dir is None

    folds: list[dict[str, Any]] = []
    total = 0
    for source_fold in source.get("folds", []) or []:
        fold = dict(source_fold)
        fold_id = int(fold["fold"])
        source_csv = resolve_csv(args.manifest, str(fold["csv"]))
        target_csv = out_dir / f"fold_{fold_id:02d}_signals.csv"
        if source_csv.resolve() != target_csv.resolve():
            shutil.copy2(source_csv, target_csv)
        signals = max(0, sum(1 for _ in target_csv.open("r", encoding="utf-8")) - 1)
        fold["csv"] = str(target_csv)
        fold["signals"] = int(signals)
        risk_value = args.risk_pct if args.risk_pct is not None else overrides.get(fold_id)
        if risk_value is not None:
            risk = round(float(risk_value), 4)
            max_positions = int(args.max_positions or fold.get("max_positions", fold.get("max_positions_hint", 1)) or 1)
            fold["risk_pct"] = risk
            fold["max_risk_pct"] = round(float(args.max_risk_pct if args.max_risk_pct is not None else risk), 4)
            if args.max_exposure_pct is not None:
                fold["max_exposure_pct"] = round(float(args.max_exposure_pct), 4)
            else:
                fold["max_exposure_pct"] = round(risk * max_positions, 4)
            fold["max_positions"] = max_positions
            fold["risk_override_safe_live"] = True
        folds.append(fold)
        total += signals

    manifest = dict(source)
    manifest["folds"] = folds
    manifest["total_signals"] = int(total)
    manifest["risk_overrides"] = {str(k): v for k, v in sorted(overrides.items())}
    if args.risk_pct is not None:
        manifest["risk_override_all_folds"] = {
            "risk_pct": float(args.risk_pct),
            "max_risk_pct": float(args.max_risk_pct if args.max_risk_pct is not None else args.risk_pct),
            "max_exposure_pct": float(args.max_exposure_pct if args.max_exposure_pct is not None else args.risk_pct),
            "max_positions": int(args.max_positions or 1),
        }
    manifest["source_manifest"] = str(args.manifest)
    target_manifest = out_dir / "manifest.json"
    if in_place:
        backup = args.manifest.with_suffix(args.manifest.suffix + ".risk_backup")
        if not backup.exists():
            shutil.copy2(args.manifest, backup)
    target_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(target_manifest), "folds": len(folds), "signals": total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
