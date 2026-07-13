#!/usr/bin/env python3
"""Plot the FCCM network-saturation bandwidth chart.

Reproduces the shape of PI Brisk's mock: CPU (red squares) flattens around
the NIC/host packet-rate ceiling (~2^10 pps) while FPGA (green circles)
keeps climbing past the CPU knee.

Inputs are two CSV *schemas* produced by sibling tasks:

FPGA sweep CSV columns (verbatim)::

    variant,dataset,configured_pps,sent,duration_s,eth_in_delta,
    app_in_delta,udp_out_delta,processed_pps,loss_pct,bandwidth_kbps

CPU sweep CSV columns (verbatim)::

    variant,dataset,configured_pps,sent,duration_s,processed_pkts,
    processed_pps,loss_pct,bandwidth_kbps

Both already carry a ``bandwidth_kbps`` column, but this script always
RECOMPUTES bandwidth from ``processed_pps`` using the locked metric below
and warns (does not silently trust the file) if the recomputed value
disagrees with the stored one by more than a small tolerance.

Locked metric::

    bandwidth_kbps = processed_pps * payload_bytes * 8 / 1000
    payload_bytes  = 64   # actual UDP payload sent, same constant for CPU/FPGA

X axis: log2(configured_pps), roughly 6 .. 23.
Y axis: bandwidth_kbps, log scale.
One panel per dataset (small-multiples grid), CPU = red squares,
FPGA = green circles (distinct marker/shade per FPGA variant so multiple
variants can be compared on one panel).

CLI::

    python3 plot_bandwidth.py \\
        --fpga-csv runs/v3_dpr/fpga_sweep.csv runs/hydra_gated/fpga_sweep.csv \\
        --cpu-csv runs/cpu/cpu_sweep.csv \\
        --dataset ItalyPowerDemand --dataset ECG5000 \\
        --out runs/plots
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None  # type: ignore
    plt = None         # type: ignore

# ---------------------------------------------------------------------------
# Locked metric (implement exactly; do not "improve")
# ---------------------------------------------------------------------------
PAYLOAD_BYTES = 64


def bandwidth_kbps(processed_pps: float) -> float:
    """Bandwidth (Kbps) = processed_pps * payload_bytes * 8 / 1000."""
    return processed_pps * PAYLOAD_BYTES * 8 / 1000.0

# Tolerance for warning when a CSV's stored bandwidth_kbps disagrees with
# the recomputed value (relative difference).
_MISMATCH_TOL = 1e-3

# Marker cycle for multiple FPGA variants on the same panel.
_FPGA_MARKERS = ["o", "^", "D", "P", "X", "v", "*", "h"]
# Green shades so distinct variants are still readable on one panel.
_FPGA_COLORS = ["#1a9850", "#66bd63", "#238b45", "#41ab5d", "#006d2c", "#74c476"]

_CPU_COLOR = "#d73027"  # red
_CPU_MARKER = "s"       # square


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
def _read_csv_rows(paths: List[pathlib.Path]) -> List[dict]:
    rows: List[dict] = []
    for p in paths:
        if not p.exists():
            print(f"[plot_bandwidth] WARNING: missing csv {p}, skipping", file=sys.stderr)
            continue
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            n = 0
            for raw in reader:
                rows.append(raw)
                n += 1
            if n == 0:
                print(f"[plot_bandwidth] WARNING: {p} has no data rows", file=sys.stderr)
    return rows


def _to_float(v, default=float("nan")) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _normalize_row(raw: dict, source: str) -> Optional[dict]:
    """Parse a raw CSV row (dict of strings) into typed fields, recomputing
    bandwidth_kbps from processed_pps and warning on mismatch vs. the
    stored value.
    """
    dataset = raw.get("dataset")
    variant = raw.get("variant", "?")
    configured_pps = _to_float(raw.get("configured_pps"))
    processed_pps = _to_float(raw.get("processed_pps"))

    if dataset is None or math.isnan(configured_pps) or math.isnan(processed_pps):
        print(f"[plot_bandwidth] WARNING: skipping malformed row in {source}: {raw}",
              file=sys.stderr)
        return None
    if configured_pps <= 0:
        print(f"[plot_bandwidth] WARNING: skipping non-positive configured_pps in {source}: {raw}",
              file=sys.stderr)
        return None

    recomputed = bandwidth_kbps(processed_pps)
    stored = _to_float(raw.get("bandwidth_kbps"))
    if not math.isnan(stored):
        denom = max(abs(recomputed), 1e-9)
        rel_diff = abs(recomputed - stored) / denom
        if rel_diff > _MISMATCH_TOL:
            print(
                f"[plot_bandwidth] WARNING: {source} row (variant={variant}, "
                f"dataset={dataset}, configured_pps={configured_pps:g}) stored "
                f"bandwidth_kbps={stored:g} disagrees with recomputed "
                f"{recomputed:g} (rel diff {rel_diff:.2%}); using recomputed value",
                file=sys.stderr,
            )

    return {
        "variant": variant,
        "dataset": dataset,
        "configured_pps": configured_pps,
        "processed_pps": processed_pps,
        "log2_configured_pps": math.log2(configured_pps),
        "bandwidth_kbps": recomputed,
        "loss_pct": _to_float(raw.get("loss_pct")),
    }


def _load(paths: List[str], source: str) -> List[dict]:
    p_paths = [pathlib.Path(p) for p in paths] if paths else []
    raw_rows = _read_csv_rows(p_paths)
    out = []
    for raw in raw_rows:
        norm = _normalize_row(raw, source)
        if norm is not None:
            out.append(norm)
    return out


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------
def _group_by_dataset_variant(rows: List[dict]) -> Dict[str, Dict[str, List[dict]]]:
    """dataset -> variant -> sorted rows."""
    g: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g[r["dataset"]][r["variant"]].append(r)
    for dataset, variants in g.items():
        for variant, vrows in variants.items():
            vrows.sort(key=lambda r: r["configured_pps"])
    return g


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _grid_shape(n: int) -> Tuple[int, int]:
    if n <= 0:
        return (1, 1)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    return (nrows, ncols)


def _plot_panel(ax, dataset: str, cpu_variants: Dict[str, List[dict]],
                 fpga_variants: Dict[str, List[dict]]) -> None:
    plotted_anything = False

    for variant, rows in sorted(cpu_variants.items()):
        x = [r["log2_configured_pps"] for r in rows]
        y = [r["bandwidth_kbps"] for r in rows]
        label = f"CPU ({variant})" if variant not in ("?", "cpu") else "CPU"
        ax.plot(x, y, marker=_CPU_MARKER, color=_CPU_COLOR, linestyle="-",
                 label=label, markersize=6, linewidth=1.5)
        plotted_anything = True

    for i, (variant, rows) in enumerate(sorted(fpga_variants.items())):
        x = [r["log2_configured_pps"] for r in rows]
        y = [r["bandwidth_kbps"] for r in rows]
        marker = _FPGA_MARKERS[i % len(_FPGA_MARKERS)]
        color = _FPGA_COLORS[i % len(_FPGA_COLORS)]
        label = f"FPGA ({variant})" if variant not in ("?", "fpga") else "FPGA"
        ax.plot(x, y, marker=marker, color=color, linestyle="-",
                 label=label, markersize=6, linewidth=1.5)
        plotted_anything = True

    ax.set_yscale("log")
    ax.set_xlabel("log2(configured pps)")
    ax.set_ylabel("Bandwidth (Kbps)")
    ax.set_title(dataset)
    ax.grid(True, which="both", alpha=0.3)
    if plotted_anything:
        ax.legend(loc="best", fontsize=7)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                color="grey")


def plot_bandwidth(
    fpga_rows: List[dict],
    cpu_rows: List[dict],
    datasets: Optional[List[str]],
    out_dir: pathlib.Path,
) -> int:
    fpga_g = _group_by_dataset_variant(fpga_rows)
    cpu_g = _group_by_dataset_variant(cpu_rows)

    if datasets:
        dataset_list = list(datasets)
    else:
        dataset_list = sorted(set(fpga_g.keys()) | set(cpu_g.keys()))

    if not dataset_list:
        print("[plot_bandwidth] no datasets found in inputs", file=sys.stderr)
        return 1

    nrows, ncols = _grid_shape(len(dataset_list))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)

    n_points = 0
    for idx, dataset in enumerate(dataset_list):
        ax = axes[idx // ncols][idx % ncols]
        cpu_variants = cpu_g.get(dataset, {})
        fpga_variants = fpga_g.get(dataset, {})
        n_points += sum(len(v) for v in cpu_variants.values())
        n_points += sum(len(v) for v in fpga_variants.values())
        if dataset not in fpga_g and dataset not in cpu_g:
            print(f"[plot_bandwidth] WARNING: no rows for requested dataset '{dataset}'",
                  file=sys.stderr)
        _plot_panel(ax, dataset, cpu_variants, fpga_variants)

    # Hide unused axes in the grid.
    for idx in range(len(dataset_list), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Network saturation: bandwidth vs. injection rate", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"bandwidth_vs_pps.{ext}", dpi=160)
    plt.close(fig)

    print(f"[plot_bandwidth] {n_points} points, {len(dataset_list)} dataset panel(s) -> {out_dir}/")
    return 0


def main() -> int:
    if plt is None:
        print("matplotlib not installed; pip install matplotlib", file=sys.stderr)
        return 2

    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fpga-csv", nargs="+", default=[],
                     help="one or more FPGA sweep CSV files")
    ap.add_argument("--cpu-csv", nargs="+", default=[],
                     help="one or more CPU sweep CSV files")
    ap.add_argument("--dataset", action="append", default=None,
                     help="dataset name to plot (repeatable); default = infer from CSVs")
    ap.add_argument("--out", required=True, help="output directory for plots")
    args = ap.parse_args()

    if not args.fpga_csv and not args.cpu_csv:
        print("[plot_bandwidth] at least one of --fpga-csv / --cpu-csv is required",
              file=sys.stderr)
        return 2

    fpga_rows = _load(args.fpga_csv, "fpga-csv")
    cpu_rows = _load(args.cpu_csv, "cpu-csv")

    if not fpga_rows and not cpu_rows:
        print("[plot_bandwidth] no usable rows parsed from inputs", file=sys.stderr)
        return 1

    out_dir = pathlib.Path(args.out)
    return plot_bandwidth(fpga_rows, cpu_rows, args.dataset, out_dir)


if __name__ == "__main__":
    sys.exit(main())
