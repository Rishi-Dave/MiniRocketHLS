#!/usr/bin/env python3
"""Aggregate sweep JSONL outputs and detect saturation points.

Reads one or more directories produced by ``rate_sweep.py`` (each containing
``sweep.jsonl`` + ``sweep_meta.json``) and emits a tidy CSV/JSON summary plus
a per-platform saturation summary.

Saturation point heuristic (paper §4.2):
- The smallest configured rate at which either:
    * achieved/configured drops below ``--throughput-fraction`` (default 0.95)
    * drop_rate exceeds ``--drop-threshold`` (default 0.01)
- If neither condition triggers across the sweep, the platform is reported
  as "unsaturated" — push the sweep further out.

CLI::

    python3 metrics.py --runs runs/smoke_phase1/* --out runs/smoke_phase1/aggregate
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Dict, List, Optional, Tuple


def _read_sweep(run_dir: pathlib.Path) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    sweep_path = run_dir / "sweep.jsonl"
    meta_path = run_dir / "sweep_meta.json"
    if not sweep_path.exists():
        raise FileNotFoundError(f"missing {sweep_path}")
    with open(sweep_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    return rows, meta


def _saturation_point(
    rows: List[dict],
    throughput_fraction: float,
    drop_threshold: float,
) -> Optional[Dict]:
    """Return the first row that meets the saturation criteria, sorted by rate."""
    valid = [r for r in rows if "error" not in r]
    valid.sort(key=lambda r: r["configured_rate_pps"])
    for r in valid:
        cfg_rate = r["configured_rate_pps"]
        ach = r["achieved_throughput_inf_per_s"]
        drop = r.get("drop_rate", 0.0)
        if cfg_rate > 0 and (ach / cfg_rate) < throughput_fraction:
            return {**r, "saturation_reason": "throughput_fraction"}
        if drop >= drop_threshold:
            return {**r, "saturation_reason": "drop_threshold"}
    return None


def _aggregate(
    run_dirs: List[pathlib.Path],
    throughput_fraction: float,
    drop_threshold: float,
) -> Tuple[List[dict], List[dict]]:
    all_rows: List[dict] = []
    summary: List[dict] = []
    for d in run_dirs:
        try:
            rows, meta = _read_sweep(d)
        except FileNotFoundError as e:
            print(f"[metrics] skip {d}: {e}", file=sys.stderr)
            continue
        for r in rows:
            r2 = dict(r)
            r2["run_dir"] = str(d)
            r2["dataset_meta"] = meta.get("dataset", r.get("dataset"))
            r2["adapter_meta"] = meta.get("adapter")
            all_rows.append(r2)
        sat = _saturation_point(rows, throughput_fraction, drop_threshold)
        platform = next((r.get("platform") for r in rows if "platform" in r), "?")
        dataset = next((r.get("dataset") for r in rows if "dataset" in r), meta.get("dataset", "?"))
        valid = [r for r in rows if "error" not in r]
        max_throughput = max((r["achieved_throughput_inf_per_s"] for r in valid), default=0.0)
        summary.append({
            "run_dir": str(d),
            "platform": platform,
            "dataset": dataset,
            "n_points": len(valid),
            "n_errors": sum(1 for r in rows if "error" in r),
            "max_achieved_inf_per_s": max_throughput,
            "saturation_rate_pps": sat["configured_rate_pps"] if sat else None,
            "saturation_throughput_inf_per_s": sat["achieved_throughput_inf_per_s"] if sat else None,
            "saturation_reason": sat["saturation_reason"] if sat else "unsaturated",
            "throughput_fraction_threshold": throughput_fraction,
            "drop_threshold": drop_threshold,
        })
    return all_rows, summary


def _write_csv(path: pathlib.Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen and k != "extra":
                seen.add(k)
                keys.append(k)
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            cells = []
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    cells.append("")
                elif isinstance(v, (dict, list)):
                    cells.append(json.dumps(v, separators=(",", ":")).replace(",", ";"))
                else:
                    s = str(v).replace(",", ";").replace("\n", " ")
                    cells.append(s)
            f.write(",".join(cells) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="one or more rate_sweep.py output directories")
    ap.add_argument("--out", required=True, help="output directory for aggregate.csv + summary.json")
    ap.add_argument("--throughput-fraction", type=float, default=0.95)
    ap.add_argument("--drop-threshold", type=float, default=0.01)
    args = ap.parse_args()

    run_dirs = [pathlib.Path(r).resolve() for r in args.runs]
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = _aggregate(run_dirs, args.throughput_fraction, args.drop_threshold)

    _write_csv(out_dir / "aggregate.csv", rows)
    with open(out_dir / "aggregate.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[metrics] {len(rows)} rows from {len(run_dirs)} runs -> {out_dir}/")
    for s in summary:
        sat = s["saturation_rate_pps"]
        sat_str = f"{sat:.1f} pps" if sat is not None else "unsaturated"
        print(f"  {s['platform']:<28} {s['dataset']:<14} "
              f"max={s['max_achieved_inf_per_s']:>10.1f} inf/s  "
              f"sat@{sat_str}  reason={s['saturation_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
