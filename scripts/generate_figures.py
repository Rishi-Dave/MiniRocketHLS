#!/usr/bin/env python3
"""
generate_figures.py
Generate publication-quality figures for MiniRocketHLS conference paper/poster.

Output files (results/figures/):
  fig1_throughput_comparison.{pdf,png}   — grouped bar, log-scale, speedup labels
  fig2_latency_cdf.{pdf,png}             — per-sample latency CDF, one subplot/dataset
  fig3_accuracy_comparison.{pdf,png}     — CPU vs FPGA accuracy bar chart + table
  fig4_scaling.{pdf,png}                 — throughput vs CU count, ideal scaling ref
  fig5_latency_breakdown.{pdf,png}       — stacked bar H2D/K1/K2/K3/D2H per dataset
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Color-blind friendly palette — Wong (2011)
# ---------------------------------------------------------------------------
CB = {
    "black":  "#000000",
    "orange": "#E69F00",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "vermil": "#D55E00",
    "pink":   "#CC79A7",
}

# ---------------------------------------------------------------------------
# Global rcParams — poster/paper readable, clean academic style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          13,
    "axes.titlesize":     14,
    "axes.labelsize":     13,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    11,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#cccccc",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.color":         "#dddddd",
    "grid.linewidth":     0.6,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "pdf.fonttype":       42,   # embed fonts as Type 42 (TrueType)
    "ps.fonttype":        42,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_fig(fig, stem):
    """Save figure as both PDF and PNG (300 DPI) to FIGURES_DIR."""
    base = os.path.join(FIGURES_DIR, stem)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=300)
    print(f"  [saved] {base}.pdf  +  .png")
    plt.close(fig)


def load_csv(path, label="CSV"):
    """Load a CSV, returning DataFrame or None with a warning."""
    if not os.path.exists(path):
        warnings.warn(f"  WARNING: {label} not found at {path} — skipping.", stacklevel=2)
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as exc:
        warnings.warn(f"  WARNING: Failed to read {label} ({path}): {exc} — skipping.", stacklevel=2)
        return None


def load_per_sample(algorithm, variant, dataset):
    """Load per-sample CSV: results/{Algorithm}_{variant}_{dataset}_per_sample.csv"""
    fname = f"{algorithm}_{variant}_{dataset}_per_sample.csv"
    return load_csv(os.path.join(RESULTS_DIR, fname), label=fname)


# ---------------------------------------------------------------------------
# Load core data tables (required for most figures)
# ---------------------------------------------------------------------------
df_sum = load_csv(os.path.join(RESULTS_DIR, "latency_summary.csv"),   "latency_summary.csv")
df_bkd = load_csv(os.path.join(RESULTS_DIR, "latency_breakdown.csv"), "latency_breakdown.csv")

# Optional: paper-results master table (may not exist yet)
df_paper = load_csv(os.path.join(RESULTS_DIR, "paper-results", "results_summary.csv"),
                    "paper-results/results_summary.csv")


# ===========================================================================
# Figure 1 — Throughput Comparison (grouped bar, log scale)
# ===========================================================================
def fig1_throughput_comparison():
    print("Generating fig1_throughput_comparison ...")
    if df_sum is None:
        print("  SKIP: latency_summary.csv unavailable.")
        return

    variant_order  = ["CPU_python", "CPU_cpp", "fused_1cu", "fused_2cu", "fused_3cu"]
    variant_labels = {
        "CPU_python": "CPU\n(Python)",
        "CPU_cpp":    "CPU\n(C++ -O3)",
        "fused_1cu":  "FPGA 1-CU",
        "fused_2cu":  "FPGA 2-CU",
        "fused_3cu":  "FPGA 3-CU",
    }
    bar_colors = [CB["black"], CB["orange"], CB["sky"], CB["blue"], CB["vermil"]]

    # Only datasets with CPU baseline (for meaningful speedup comparison)
    datasets = ["InsectSound", "MosquitoSound", "FruitFlies"]
    ds_display = {
        "InsectSound":   "InsectSound\n(L$\\approx$600)",
        "MosquitoSound": "MosquitoSound\n(L=3750)",
        "FruitFlies":    "FruitFlies\n(L$\\approx$5000)",
    }

    # Build lookup: (Variant, Dataset) -> Throughput_InfPerSec
    mr = df_sum[df_sum["Algorithm"] == "MiniRocket"]
    lookup = {(row["Variant"], row["Dataset"]): row["Throughput_InfPerSec"]
              for _, row in mr.iterrows()}

    n_ds  = len(datasets)
    n_var = len(variant_order)
    bw    = 0.17
    gap   = 1.0
    x0    = np.arange(n_ds) * gap
    offsets = np.linspace(-(n_var - 1) / 2, (n_var - 1) / 2, n_var) * bw

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_by_variant = {}
    for vi, variant in enumerate(variant_order):
        vals  = [lookup.get((variant, ds), np.nan) for ds in datasets]
        rects = ax.bar(
            x0 + offsets[vi], vals,
            width=bw * 0.90,
            color=bar_colors[vi],
            label=variant_labels[variant],
            zorder=3,
            edgecolor="white",
            linewidth=0.5,
        )
        bars_by_variant[variant] = (rects, vals)

    # Speedup labels above FPGA bars (vs CPU_python)
    cpu_vals = [lookup.get(("CPU_python", ds), np.nan) for ds in datasets]
    for vi, variant in enumerate(variant_order[1:], start=1):
        rects, fpga_vals = bars_by_variant[variant]
        for rect, fpga_v, cpu_v in zip(rects, fpga_vals, cpu_vals):
            if np.isnan(fpga_v) or np.isnan(cpu_v) or cpu_v == 0:
                continue
            speedup = fpga_v / cpu_v
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() * 1.18,
                f"{speedup:.0f}$\\times$",
                ha="center", va="bottom",
                fontsize=9, color=bar_colors[vi], fontweight="bold",
                rotation=90,
            )

    ax.set_yscale("log")
    ax.set_ylabel("Throughput (inf/s)", labelpad=6)
    ax.set_xticks(x0)
    ax.set_xticklabels([ds_display[d] for d in datasets])
    ax.set_title("Throughput Comparison — MiniRocket on Alveo U280", pad=10)

    # Build clean legend patches
    legend_handles = [
        Patch(facecolor=bar_colors[i], label=variant_labels[v])
        for i, v in enumerate(variant_order)
    ]
    ax.legend(handles=legend_handles, ncol=2, loc="upper left",
              columnspacing=0.8, handlelength=1.2)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}" if x >= 1 else f"{x:.2f}"
    ))
    ax.set_ylim(bottom=8)
    ax.grid(True, which="both", axis="y", linewidth=0.5, color="#dddddd")
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, "fig1_throughput_comparison")


# ===========================================================================
# Figure 2 — Latency CDF (one subplot per dataset, CPU vs FPGA 1-CU)
# ===========================================================================
def fig2_latency_cdf():
    print("Generating fig2_latency_cdf ...")

    datasets = ["GunPoint", "InsectSound", "MosquitoSound", "FruitFlies"]
    ds_titles = {
        "GunPoint":     "GunPoint  (L=150)",
        "InsectSound":  "InsectSound  (L$\\approx$600)",
        "MosquitoSound":"MosquitoSound  (L=3750)",
        "FruitFlies":   "FruitFlies  (L$\\approx$5000)",
    }

    fpga_color = CB["blue"]
    cpp_color  = CB["green"]
    cpu_color  = CB["orange"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
    fig.suptitle("Per-Sample Latency CDF — FPGA 1-CU vs CPU C++ vs CPU Python",
                 fontsize=14, y=1.02)

    any_plotted = False
    for ax, ds in zip(axes, datasets):
        ax.set_title(ds_titles[ds], fontsize=11, pad=6)

        # FPGA 1-CU per-sample data
        df_fpga = load_per_sample("MiniRocket", "fused_1cu", ds)
        if df_fpga is not None and "total_ms" in df_fpga.columns:
            lat_fpga = np.sort(df_fpga["total_ms"].dropna().values)
            cdf_fpga = np.arange(1, len(lat_fpga) + 1) / len(lat_fpga)
            ax.plot(lat_fpga, cdf_fpga, color=fpga_color, lw=2.0,
                    label="FPGA 1-CU")
            any_plotted = True
        else:
            print(f"    WARNING: No FPGA per-sample data for {ds}, skipping subplot.")

        # C++ CPU per-sample data
        df_cpp = load_per_sample("MiniRocket", "CPU_cpp", ds)
        if df_cpp is not None and "total_ms" in df_cpp.columns:
            lat_cpp = np.sort(df_cpp["total_ms"].dropna().values)
            cdf_cpp = np.arange(1, len(lat_cpp) + 1) / len(lat_cpp)
            ax.plot(lat_cpp, cdf_cpp, color=cpp_color, lw=2.0,
                    linestyle="-.", label="CPU (C++)")

        # Python CPU per-sample data (or summary fallback)
        df_cpu = load_per_sample("MiniRocket", "CPU_python", ds)
        if df_cpu is not None and "total_ms" in df_cpu.columns:
            lat_cpu = np.sort(df_cpu["total_ms"].dropna().values)
            cdf_cpu = np.arange(1, len(lat_cpu) + 1) / len(lat_cpu)
            ax.plot(lat_cpu, cdf_cpu, color=cpu_color, lw=2.0,
                    linestyle="--", label="CPU (Python)")
        else:
            # Derive a synthetic vertical line from summary table if available
            if df_sum is not None:
                row = df_sum[(df_sum["Algorithm"] == "MiniRocket") &
                             (df_sum["Variant"]   == "CPU_python") &
                             (df_sum["Dataset"]   == ds)]
                if len(row) > 0 and "Latency_Mean_ms" in row.columns:
                    cpu_mean = float(row["Latency_Mean_ms"].values[0])
                    if cpu_mean > 0:
                        ax.axvline(cpu_mean, color=cpu_color, lw=1.8,
                                   linestyle="--", alpha=0.75,
                                   label=f"CPU mean ({cpu_mean:.1f} ms)")

        ax.set_xscale("log")
        ax.set_xlabel("Latency (ms)", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x:g}"
        ))
        ax.grid(True, which="both", axis="x", linewidth=0.4, color="#e0e0e0")
        ax.grid(True, axis="y", linewidth=0.4, color="#e0e0e0")
        ax.legend(loc="lower right", fontsize=9)

    axes[0].set_ylabel("CDF", labelpad=6)

    if not any_plotted:
        print("  SKIP: No per-sample data found for fig2.")
        plt.close(fig)
        return

    fig.tight_layout()
    save_fig(fig, "fig2_latency_cdf")


# ===========================================================================
# Figure 3 — Accuracy Comparison (CPU vs FPGA, side-by-side bars + table)
# ===========================================================================
def fig3_accuracy_comparison():
    print("Generating fig3_accuracy_comparison ...")
    if df_sum is None:
        print("  SKIP: latency_summary.csv unavailable.")
        return

    datasets = ["GunPoint", "InsectSound", "MosquitoSound", "FruitFlies"]
    ds_short = {
        "GunPoint":     "GunPoint",
        "InsectSound":  "InsectSound",
        "MosquitoSound":"MosquSound",
        "FruitFlies":   "FruitFlies",
    }

    mr = df_sum[df_sum["Algorithm"] == "MiniRocket"]
    cpu_py_acc = {r["Dataset"]: r["Accuracy_Pct"]
                  for _, r in mr[mr["Variant"] == "CPU_python"].iterrows()}
    cpu_cpp_acc = {r["Dataset"]: r["Accuracy_Pct"]
                   for _, r in mr[mr["Variant"] == "CPU_cpp"].iterrows()}
    fpga_acc = {r["Dataset"]: r["Accuracy_Pct"]
                for _, r in mr[mr["Variant"] == "fused_1cu"].iterrows()}

    fig, (ax_bar, ax_tbl) = plt.subplots(1, 2, figsize=(13, 5),
                                          gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle("Classification Accuracy: CPU vs. FPGA (MiniRocket)", fontsize=14)

    # ---- Left panel: grouped bars ----
    x     = np.arange(len(datasets))
    bw    = 0.25
    cpu_py_vals  = [cpu_py_acc.get(ds, np.nan)  for ds in datasets]
    cpu_cpp_vals = [cpu_cpp_acc.get(ds, np.nan) for ds in datasets]
    fpga_vals    = [fpga_acc.get(ds, np.nan) for ds in datasets]

    ax_bar.bar(x - bw, cpu_py_vals,  bw, color=CB["orange"], label="CPU (Python)",
               edgecolor="white", linewidth=0.6, zorder=3)
    ax_bar.bar(x, cpu_cpp_vals, bw, color=CB["green"], label="CPU (C++ -O3)",
               edgecolor="white", linewidth=0.6, zorder=3)
    rects_fpga = ax_bar.bar(x + bw, fpga_vals, bw, color=CB["blue"], label="FPGA 1-CU",
                             edgecolor="white", linewidth=0.6, zorder=3)

    # Annotate exact percentages above bars
    for xi, (pv, cv, fv) in enumerate(zip(cpu_py_vals, cpu_cpp_vals, fpga_vals)):
        if not np.isnan(pv):
            ax_bar.text(xi - bw, pv + 0.4, f"{pv:.1f}%",
                        ha="center", va="bottom", fontsize=8, color=CB["orange"],
                        fontweight="bold")
        if not np.isnan(cv):
            ax_bar.text(xi, cv + 0.4, f"{cv:.1f}%",
                        ha="center", va="bottom", fontsize=8, color=CB["green"],
                        fontweight="bold")
        if not np.isnan(fv):
            ax_bar.text(xi + bw, fv + 0.4, f"{fv:.1f}%",
                        ha="center", va="bottom", fontsize=8, color=CB["blue"],
                        fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([ds_short[d] for d in datasets])
    ax_bar.set_ylabel("Accuracy (%)")
    ax_bar.set_ylim(0, 115)
    ax_bar.set_title("Accuracy per Dataset")
    ax_bar.legend(loc="lower right")
    ax_bar.grid(True, axis="y", linewidth=0.5, color="#dddddd")
    ax_bar.set_axisbelow(True)

    # ---- Right panel: accuracy table ----
    ax_tbl.axis("off")
    rows = []
    for ds in datasets:
        cpu_py_v  = cpu_py_acc.get(ds, None)
        cpu_cpp_v = cpu_cpp_acc.get(ds, None)
        fpga_v    = fpga_acc.get(ds, None)
        py_str   = f"{cpu_py_v:.2f}%" if cpu_py_v  is not None else "N/A"
        cpp_str  = f"{cpu_cpp_v:.2f}%" if cpu_cpp_v is not None else "N/A"
        fpga_str = f"{fpga_v:.2f}%" if fpga_v is not None else "N/A"
        if cpu_cpp_v is not None and fpga_v is not None:
            delta = fpga_v - cpu_cpp_v
            match_str = f"{delta:+.2f}%"
        else:
            match_str = "\u2014"
        rows.append([ds, py_str, cpp_str, fpga_str, match_str])

    col_labels = ["Dataset", "Python", "C++", "FPGA", "\u0394 (FPGA\u2212C++)"]
    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1.1, 2.0)

    # Style header row
    for ci in range(len(col_labels)):
        cell = tbl[(0, ci)]
        cell.set_facecolor(CB["blue"])
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading + highlight delta column
    for ri in range(1, len(rows) + 1):
        bg = "#F0F4FA" if ri % 2 == 0 else "white"
        for ci in range(len(col_labels)):
            cell = tbl[(ri, ci)]
            cell.set_facecolor(bg)
            if ci == 4:
                val_str = rows[ri - 1][3]
                if val_str not in ("—", "N/A"):
                    try:
                        val = float(val_str.replace("%", "").replace("+", ""))
                        if abs(val) < 0.05:
                            cell.set_text_props(color=CB["green"], fontweight="bold")
                        elif val < 0:
                            cell.set_text_props(color=CB["vermil"], fontweight="bold")
                        else:
                            cell.set_text_props(color=CB["orange"], fontweight="bold")
                    except ValueError:
                        pass

    ax_tbl.set_title("Accuracy Table", pad=8)

    fig.tight_layout()
    save_fig(fig, "fig3_accuracy_comparison")


# ===========================================================================
# Figure 4 — CU Scaling (throughput vs CU count, ideal reference)
# ===========================================================================
def fig4_scaling():
    print("Generating fig4_scaling ...")
    if df_sum is None:
        print("  SKIP: latency_summary.csv unavailable.")
        return

    cu_variants = {1: "fused_1cu", 2: "fused_2cu", 3: "fused_3cu"}
    datasets    = ["GunPoint", "InsectSound", "MosquitoSound", "FruitFlies"]
    ds_labels   = {
        "GunPoint":     "GunPoint (L=150)",
        "InsectSound":  "InsectSound (L$\\approx$600)",
        "MosquitoSound":"MosquitoSound (L=3750)",
        "FruitFlies":   "FruitFlies (L$\\approx$5000)",
    }
    ds_markers = ["o", "s", "^", "D"]
    ds_colors  = [CB["orange"], CB["sky"], CB["green"], CB["vermil"]]

    mr = df_sum[df_sum["Algorithm"] == "MiniRocket"]
    lookup = {(r["Variant"], r["Dataset"]): r["Throughput_InfPerSec"]
              for _, r in mr.iterrows()}

    cu_x = np.array([1, 2, 3])

    fig, (ax_abs, ax_eff) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Compute Unit Scaling — MiniRocket Fused Kernel (Alveo U280)",
                 fontsize=14)

    # ---- Left: absolute throughput ----
    for ds, color, marker in zip(datasets, ds_colors, ds_markers):
        ys = [lookup.get((cu_variants[n], ds), np.nan) for n in cu_x]
        ax_abs.plot(cu_x, ys, color=color, marker=marker, lw=2.0,
                    markersize=7, label=ds_labels[ds])

    # Ideal linear scaling anchored to 1-CU GunPoint (representative)
    ref_1cu = lookup.get(("fused_1cu", "GunPoint"), None)
    if ref_1cu is not None:
        ideal = ref_1cu * cu_x
        ax_abs.plot(cu_x, ideal, "k--", lw=1.4, alpha=0.5, label="Ideal (linear)")

    ax_abs.set_xlabel("Number of Compute Units")
    ax_abs.set_ylabel("Throughput (inf/s)")
    ax_abs.set_xticks([1, 2, 3])
    ax_abs.set_title("Absolute Throughput")
    ax_abs.legend(fontsize=10, loc="upper left")
    ax_abs.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_abs.grid(True, axis="y", linewidth=0.5, color="#dddddd")
    ax_abs.set_axisbelow(True)

    # ---- Right: scaling efficiency (normalised to 1-CU) ----
    for ds, color, marker in zip(datasets, ds_colors, ds_markers):
        ys_abs = [lookup.get((cu_variants[n], ds), np.nan) for n in cu_x]
        base   = ys_abs[0] if (len(ys_abs) > 0 and not np.isnan(ys_abs[0])) else None
        if base is None or base == 0:
            continue
        eff = [y / (base * n) if not np.isnan(y) else np.nan
               for n, y in zip(cu_x, ys_abs)]
        ax_eff.plot(cu_x, eff, color=color, marker=marker, lw=2.0,
                    markersize=7, label=ds_labels[ds])

    ax_eff.axhline(1.0, color="k", linestyle="--", lw=1.4, alpha=0.5, label="Ideal (100%)")
    ax_eff.set_xlabel("Number of Compute Units")
    ax_eff.set_ylabel("Scaling Efficiency")
    ax_eff.set_xticks([1, 2, 3])
    ax_eff.set_ylim(0, 1.3)
    ax_eff.set_title("Scaling Efficiency (vs. ideal linear)")
    ax_eff.legend(fontsize=10, loc="upper right")
    ax_eff.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_eff.grid(True, axis="y", linewidth=0.5, color="#dddddd")
    ax_eff.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, "fig4_scaling")


# ===========================================================================
# Figure 5 — Latency Breakdown (stacked bar, fused 1-CU)
# ===========================================================================
def fig5_latency_breakdown():
    print("Generating fig5_latency_breakdown ...")
    if df_bkd is None:
        print("  SKIP: latency_breakdown.csv unavailable.")
        return

    datasets   = ["GunPoint", "InsectSound", "MosquitoSound", "FruitFlies"]
    ds_labels  = {
        "GunPoint":     "GunPoint\n(L=150)",
        "InsectSound":  "InsectSound\n(L$\\approx$600)",
        "MosquitoSound":"MosquitoSound\n(L=3750)",
        "FruitFlies":   "FruitFlies\n(L$\\approx$5000)",
    }

    components  = ["H2D", "K1_FeatureExt", "K2_Scaler", "K3_Classifier", "D2H"]
    comp_labels = {
        "H2D":           "H2D transfer",
        "K1_FeatureExt": "K1: Feature Extraction",
        "K2_Scaler":     "K2: Scaler",
        "K3_Classifier": "K3: Classifier",
        "D2H":           "D2H transfer",
    }
    comp_colors = [CB["sky"], CB["vermil"], CB["orange"], CB["green"], CB["pink"]]

    df_fused = df_bkd[(df_bkd["Algorithm"] == "MiniRocket") &
                      (df_bkd["Variant"]   == "fused_1cu")]

    if df_fused.empty:
        print("  SKIP: No fused_1cu breakdown rows found.")
        return

    x       = np.arange(len(datasets))
    bw      = 0.52
    bottoms = np.zeros(len(datasets))

    fig, ax = plt.subplots(figsize=(8, 5))

    for comp, color in zip(components, comp_colors):
        vals = []
        for ds in datasets:
            row = df_fused[(df_fused["Dataset"]   == ds) &
                           (df_fused["Component"] == comp)]
            vals.append(float(row["Mean_ms"].values[0]) if len(row) > 0 else 0.0)
        vals = np.array(vals)
        ax.bar(x, vals, bw, bottom=bottoms,
               color=color, label=comp_labels[comp],
               edgecolor="white", linewidth=0.6, zorder=3)
        # Annotate segment if tall enough
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 0.04:
                ax.text(xi, b + v / 2, f"{v:.2f}",
                        ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="bold")
        bottoms += vals

    # Annotate total latency on top of each bar
    for xi, total in enumerate(bottoms):
        ax.text(xi, total + 0.012, f"{total:.2f} ms",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Mean latency per sample (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([ds_labels[d] for d in datasets])
    ax.set_title("Latency Breakdown — MiniRocket Fused Kernel (FPGA 1-CU)", pad=10)
    ax.legend(loc="upper left", ncol=1, fontsize=11)
    ax.grid(True, axis="y", linewidth=0.5, color="#dddddd")
    ax.set_axisbelow(True)
    ax.set_ylim(top=bottoms.max() * 1.18)

    fig.tight_layout()
    save_fig(fig, "fig5_latency_breakdown")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print(f"Results dir  : {RESULTS_DIR}")
    print(f"Figures dir  : {FIGURES_DIR}")
    print()

    try:
        fig1_throughput_comparison()
    except Exception as exc:
        print(f"  ERROR in fig1: {exc}")

    try:
        fig2_latency_cdf()
    except Exception as exc:
        print(f"  ERROR in fig2: {exc}")

    try:
        fig3_accuracy_comparison()
    except Exception as exc:
        print(f"  ERROR in fig3: {exc}")

    try:
        fig4_scaling()
    except Exception as exc:
        print(f"  ERROR in fig4: {exc}")

    try:
        fig5_latency_breakdown()
    except Exception as exc:
        print(f"  ERROR in fig5: {exc}")

    print()
    print("Done.")
    print(f"Output : {FIGURES_DIR}/")
