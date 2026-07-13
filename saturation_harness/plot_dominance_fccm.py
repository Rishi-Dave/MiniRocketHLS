#!/usr/bin/env python3
"""FCCM camera-ready style figures: CPU vs FPGA bandwidth vs injection rate.

Mimics the group's prior paper figure (bold black frame, boxed top legend,
red CPU squares / green FPGA circles, x = log2 pps, log-scale Kbps y-axis).
Both series start from the same measured point (5k pps, 2560 Kbps, 0% loss).

CPU      : runs/tp_2026-07-06_cpu/cpu_minirocket_3ds.csv            (all 3 datasets)
FPGA low : runs/tp_2026-07-06_csender/fused_csender_sweep.csv       (5k-745k, 0% loss)
FPGA F2F : runs/tp_2026-07-09_f2f_knee/fused_dropfull_knee_plotfmt.csv (InsectSound only,
           137k-34.3M pps -> 2.75M-pps compute plateau)

Outputs per dataset: fccm_dominance_<dataset>.{png,svg}
Combined reference-style multi-panel: fccm_dominance_panels.{png,svg}
bandwidth_kbps = processed_pps * 64 * 8 / 1000 (locked metric, 64B payloads).
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "runs/tp_2026-07-09_f2f_knee"

# Per-variant data sources / output naming. Select with argv[1]
# (default: minirocket).
VARIANTS = {
    "minirocket": {   # v6 (UNROLL=84 @300MHz) — current headline
        "cpu_csv": HERE / "runs/tp_2026-07-06_cpu/cpu_minirocket_3ds.csv",
        "knee_base": "fused_dropfull300u84_knee",
        "out_prefix": "fccm_dominance",
        "title": "MiniRocket (UNROLL=84 @ 300 MHz)",
    },
    "minirocket_v5": {   # v5 facc_t @300MHz, UNROLL=28
        "cpu_csv": HERE / "runs/tp_2026-07-06_cpu/cpu_minirocket_3ds.csv",
        "knee_base": "fused_dropfull300_knee",
        "out_prefix": "fccm_dominance_mrv5",
        "title": "MiniRocket (UNROLL=28 @ 300 MHz)",
    },
    "minirocket250": {   # v4 dropfull @250MHz (pre-facc_t), kept for reference
        "cpu_csv": HERE / "runs/tp_2026-07-06_cpu/cpu_minirocket_3ds.csv",
        "knee_base": "fused_dropfull_knee",
        "out_prefix": "fccm_dominance_mr250",
        "title": "MiniRocket (UNROLL=28 @ 250 MHz)",
    },
    "hydra": {   # u64 batches @200MHz — current hydra headline
        "cpu_csv": HERE / "runs/tp_2026-07-10_cpu_hydra/cpu_hydra_3ds.csv",
        "knee_base": "hydra_dropfull_u64_knee",
        "out_prefix": "fccm_dominance_hydra",
        "title": "HYDRA (UNROLL=64 @ 200 MHz)",
    },
    "hydra_u16": {   # v6 u16 @300MHz, superseded
        "cpu_csv": HERE / "runs/tp_2026-07-10_cpu_hydra/cpu_hydra_3ds.csv",
        "knee_base": "hydra_dropfull_knee",
        "out_prefix": "fccm_dominance_hydra_u16",
        "title": "HYDRA (UNROLL=16 @ 300 MHz)",
    },
}
import sys
VC = VARIANTS[sys.argv[1] if len(sys.argv) > 1 else "minirocket"]
CPU_CSV = VC["cpu_csv"]

DATASETS = ["InsectSound", "MosquitoSound", "FruitFlies"]
START_PPS = 5000   # common measured starting point on both platforms (0% loss)

RED, GREEN, INK = "#d73027", "#1a9850", "#000000"

plt.rcParams.update({
    "font.size": 16, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "font.family": "DejaVu Sans",
})


def load(path, dataset, lo=None, hi=None):
    xs, ys = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["dataset"] != dataset:
                continue
            pps = float(row["configured_pps"])
            if (lo is not None and pps < lo) or (hi is not None and pps >= hi):
                continue
            xs.append(math.log2(pps))
            ys.append(float(row["bandwidth_kbps"]))
    pairs = sorted(set(zip(xs, ys)))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def series(dataset):
    """CPU sweep + pure FPGA-to-FPGA knee sweep (no host sender anywhere)."""
    cx, cy = load(CPU_CSV, dataset, lo=START_PPS)
    fx, fy = [], []
    with open(OUT_DIR / f"{VC['knee_base']}_{dataset}.csv") as f:
        for row in csv.DictReader(f):
            pps = float(row["offered_pps"])
            if pps <= 0:
                continue
            fx.append(math.log2(pps))
            fy.append(float(row["bandwidth_kbps"]))
    pairs = sorted(zip(fx, fy))
    fx, fy = [p[0] for p in pairs], [p[1] for p in pairs]
    return (cx, cy), (fx, fy), True


YTICKS = [4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]
YLABELS = ["4K", "16K", "64K", "256K", "1M", "4M", "16M"]


def draw_panel(ax, dataset, tick_size=16, xmax_shared=None, ymax_shared=None):
    (cx, cy), (fx, fy), saturated = series(dataset)

    for xs, ys, color, marker in ((cx, cy, RED, "s"), (fx, fy, GREEN, "o")):
        ax.plot(xs, ys, color=INK, lw=1.6, zorder=2)
        ax.plot(xs, ys, linestyle="none", marker=marker, markersize=11,
                markerfacecolor=color, markeredgecolor=color, zorder=3)

    xmax = max(fx)
    if xmax_shared is not None:
        xmax = xmax_shared
    if saturated:
        # CPU is loss-bound past its knee: dashed = extrapolated at last measured value
        cpu_plateau, fpga_plateau = cy[-1], max(fy)
        ax.plot([cx[-1], xmax + 0.4], [cpu_plateau, cpu_plateau], color=RED,
                lw=2.4, linestyle=(0, (5, 3)), zorder=2)
        gap_x = xmax - 1.45
        ratio = fpga_plateau / cpu_plateau
        ax.annotate("", xy=(gap_x, fpga_plateau * 0.72),
                    xytext=(gap_x, cpu_plateau * 1.45),
                    arrowprops=dict(arrowstyle="<->", lw=2.6, color=INK))
        ax.text(gap_x + 0.35, math.sqrt(cpu_plateau * fpga_plateau),
                f"{ratio:.0f}×",
                fontsize=tick_size + 3, fontweight="bold", ha="left", va="center")
        ymax_val = max(fy) * 2.6
    else:
        ymax_val = max(fy) * 2.4

    # Shared-scale mode (combined panels figure): every panel gets the same
    # axes so plateaus are visually comparable across datasets.
    if ymax_shared is not None:
        ymax_val = ymax_shared

    ax.set_yscale("log")
    yt = [t for t in YTICKS if t <= ymax_val]
    ax.set_yticks(yt)
    ax.set_yticklabels(YLABELS[: len(yt)], fontsize=tick_size, fontweight="bold")
    ax.set_ylim(1700, ymax_val)
    ax.minorticks_off()

    xhi = int(math.ceil(xmax))
    ax.set_xticks(range(13, xhi + 1, 2))
    ax.set_xticklabels([str(v) for v in range(13, xhi + 1, 2)],
                       fontsize=tick_size, fontweight="bold")
    ax.set_xlim(11.9, xmax + 0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(3.2)
        spine.set_color(INK)
    ax.tick_params(width=3.2, length=7, color=INK)
    ax.grid(False)


def legend_handles():
    return [
        Line2D([], [], linestyle="none", marker="s", markersize=13,
               markerfacecolor=RED, markeredgecolor=RED, label="CPU"),
        Line2D([], [], linestyle="none", marker="o", markersize=13,
               markerfacecolor=GREEN, markeredgecolor=GREEN, label="FPGA"),
    ]


def style_legend(leg):
    leg.get_frame().set_edgecolor(INK)
    leg.get_frame().set_linewidth(3.0)
    for t in leg.get_texts():
        t.set_fontweight("bold")


# --- single-panel figure per dataset ---
for ds in DATASETS:
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)
    draw_panel(ax, ds)
    ax.set_xlabel(r"Transmission Rate (log$_2$ pps)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Bandwidth (Kbps)", fontsize=18, fontweight="bold")
    ax.set_title(f"{VC['title']} \u2014 {ds}", fontsize=16, fontweight="bold", pad=10)
    leg = fig.legend(handles=legend_handles(), loc="upper center", ncol=2,
                     frameon=True, bbox_to_anchor=(0.55, 1.005), fontsize=17,
                     handletextpad=0.4, columnspacing=1.6, borderpad=0.35)
    style_legend(leg)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"{VC['out_prefix']}_{ds}.{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)

# --- combined reference-style panel figure (SHARED SCALE across panels) ---
_all_fx, _all_fy = [], []
for ds in DATASETS:
    (_, _), (fx, fy), _ = series(ds)
    _all_fx.append(max(fx)); _all_fy.append(max(fy))
XMAX_SHARED = max(_all_fx)
YMAX_SHARED = max(_all_fy) * 2.6

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=200)
for i, (ax, ds) in enumerate(zip(axes, DATASETS)):
    draw_panel(ax, ds, tick_size=15, xmax_shared=XMAX_SHARED, ymax_shared=YMAX_SHARED)
    ax.set_title(f"({chr(ord('a') + i)}) {ds}", fontsize=18, fontweight="bold", pad=8)
    ax.set_xlabel(r"Transmission Rate (log$_2$ pps)", fontsize=16, fontweight="bold")
axes[0].set_ylabel("Bandwidth (Kbps)", fontsize=17, fontweight="bold")
fig.suptitle(VC["title"], fontsize=21, fontweight="bold", y=1.10)
leg = fig.legend(handles=legend_handles(), loc="upper center", ncol=2, frameon=True,
                 bbox_to_anchor=(0.5, 1.015), fontsize=17,
                 handletextpad=0.4, columnspacing=1.6, borderpad=0.35)
style_legend(leg)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ("png", "svg"):
    fig.savefig(OUT_DIR / f"{VC['out_prefix']}_panels.{ext}", bbox_inches="tight",
                facecolor="white")
plt.close(fig)

# keep the original filename pointing at the InsectSound figure
import shutil
for ext in ("png", "svg"):
    shutil.copyfile(OUT_DIR / f"{VC['out_prefix']}_InsectSound.{ext}",
                    OUT_DIR / f"{VC['out_prefix']}.{ext}")

print(f"[plot_dominance_fccm] wrote {VC['out_prefix']}_{{{','.join(DATASETS)}}}.{{png,svg}} "
      f"+ fccm_dominance_panels.{{png,svg}} -> {OUT_DIR}")
