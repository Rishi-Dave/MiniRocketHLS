#!/usr/bin/env python3
"""Plot throughput-vs-injection-rate saturation curves.

Reads ``aggregate.jsonl`` produced by ``metrics.py`` (or one or more raw
``sweep.jsonl`` files) and produces:
- ``throughput_vs_rate.{png,svg}``: x = configured rate, y = achieved throughput,
  one line per (platform, dataset). Diagonal y=x line annotates ideal pacing.
- ``drop_rate_vs_rate.{png,svg}``: x = configured rate, y = drop rate.
- ``latency_vs_rate.{png,svg}``: x = configured rate, y = p50 / p95 / p99 latency.

CLI::

    python3 plot_saturation.py \\
        --aggregate runs/smoke_phase1/aggregate/aggregate.jsonl \\
        --out runs/smoke_phase1/plots
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None  # type: ignore
    plt = None         # type: ignore


def _read_rows(paths: List[pathlib.Path]) -> List[dict]:
    rows: List[dict] = []
    for p in paths:
        if p.is_dir():
            cand = p / "sweep.jsonl"
            if cand.exists():
                paths_eff = [cand]
            else:
                paths_eff = list(p.rglob("sweep.jsonl"))
        else:
            paths_eff = [p]
        for q in paths_eff:
            with open(q) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
    return rows


def _group(rows: List[dict]) -> Dict[Tuple[str, str], List[dict]]:
    g: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        key = (r.get("platform", "?"), r.get("dataset", "?"))
        g[key].append(r)
    for k in g:
        g[k].sort(key=lambda r: r["configured_rate_pps"])
    return g


def _plot_throughput(groups, out_dir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    all_rates: List[float] = []
    for (platform, dataset), rows in sorted(groups.items()):
        rates = [r["configured_rate_pps"] for r in rows]
        achieved = [r["achieved_throughput_inf_per_s"] for r in rows]
        all_rates.extend(rates)
        label = f"{platform} • {dataset}"
        ax.plot(rates, achieved, marker="o", label=label)
    if all_rates:
        lo = max(min(all_rates), 1e-3)
        hi = max(all_rates)
        ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1, alpha=0.6, label="y = x (ideal pacing)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Configured injection rate (inferences / sec)")
    ax.set_ylabel("Achieved throughput (inferences / sec)")
    ax.set_title("Throughput vs. injection rate (saturation curves)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"throughput_vs_rate.{ext}", dpi=160)
    plt.close(fig)


def _plot_drop_rate(groups, out_dir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for (platform, dataset), rows in sorted(groups.items()):
        rates = [r["configured_rate_pps"] for r in rows]
        drops = [r.get("drop_rate", 0.0) * 100.0 for r in rows]
        ax.plot(rates, drops, marker="s", label=f"{platform} • {dataset}")
    ax.set_xscale("log")
    ax.set_xlabel("Configured injection rate (inferences / sec)")
    ax.set_ylabel("Drop rate (%)")
    ax.set_title("Drop rate vs. injection rate")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"drop_rate_vs_rate.{ext}", dpi=160)
    plt.close(fig)


def _plot_latency(groups, out_dir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for (platform, dataset), rows in sorted(groups.items()):
        rates = [r["configured_rate_pps"] for r in rows]
        p50 = [r.get("latency_us_p50", float("nan")) for r in rows]
        p99 = [r.get("latency_us_p99", float("nan")) for r in rows]
        line, = ax.plot(rates, p50, marker="o", label=f"{platform} • {dataset} p50")
        ax.plot(rates, p99, marker="^", linestyle="--",
                color=line.get_color(), label=f"{platform} • {dataset} p99")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Configured injection rate (inferences / sec)")
    ax.set_ylabel("Latency (µs)")
    ax.set_title("Latency vs. injection rate (p50 solid, p99 dashed)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"latency_vs_rate.{ext}", dpi=160)
    plt.close(fig)


def main() -> int:
    if plt is None:
        print("matplotlib not installed; pip install matplotlib", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", nargs="+",
                    help="one or more aggregate.jsonl or sweep.jsonl files / directories")
    ap.add_argument("--out", required=True, help="output directory for plots")
    args = ap.parse_args()

    if not args.aggregate:
        print("--aggregate required", file=sys.stderr)
        return 2
    paths = [pathlib.Path(p) for p in args.aggregate]
    rows = _read_rows(paths)
    if not rows:
        print("[plot] no rows found", file=sys.stderr)
        return 1
    groups = _group(rows)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_throughput(groups, out_dir)
    _plot_drop_rate(groups, out_dir)
    _plot_latency(groups, out_dir)

    print(f"[plot] {sum(len(v) for v in groups.values())} points, "
          f"{len(groups)} (platform,dataset) curves -> {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
