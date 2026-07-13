#!/usr/bin/env python3
"""
generate_cu_scaling_figure.py
Generate CU scaling plot and LaTeX table for MiniRocket fused kernel.

Shows throughput vs CU count (1/2/3) with ideal scaling reference line.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR   = os.path.join(RESULTS_DIR, "tables")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

CB = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "vermil": "#D55E00",
    "sky":    "#56B4E9",
    "pink":   "#CC79A7",
    "black":  "#000000",
}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        13,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.color":       "#dddddd",
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})


def main():
    master = pd.read_csv(os.path.join(RESULTS_DIR, "results_master.csv"))
    mr = master[master["Algorithm"] == "MiniRocket"].copy()

    datasets = ["GunPoint", "InsectSound", "MosquitoSound", "FruitFlies"]
    cu_configs = [
        ("FPGA U280 (fused 1CU)", 1, "1CU (300 MHz)"),
        ("FPGA U280 (fused 2CU)", 2, "2CU (300 MHz)"),
        ("FPGA U280 (fused 3CU)", 3, "3CU (269 MHz)"),
    ]

    colors = [CB["blue"], CB["orange"], CB["green"], CB["vermil"]]
    markers = ["o", "s", "^", "D"]

    # Build data: dataset -> [tp_1cu, tp_2cu, tp_3cu]
    data = {}
    for ds in datasets:
        tps = []
        for plat, ncu, label in cu_configs:
            row = mr[(mr["Platform"] == plat) & (mr["Dataset"] == ds)]
            if not row.empty:
                tps.append(row.iloc[0]["Throughput_InfPerSec"])
            else:
                tps.append(None)
        data[ds] = tps

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    cus = [1, 2, 3]

    for i, ds in enumerate(datasets):
        tps = data[ds]
        valid = [(c, t) for c, t in zip(cus, tps) if t is not None]
        if not valid:
            continue
        xs, ys = zip(*valid)
        ax.plot(xs, ys, marker=markers[i], color=colors[i], linewidth=2,
                markersize=8, label=ds, zorder=3)

    # Ideal scaling reference (from 1CU GunPoint)
    ref_tp = data["GunPoint"][0]
    if ref_tp:
        ideal = [ref_tp * c for c in cus]
        ax.plot(cus, ideal, "--", color="#999999", linewidth=1.5, alpha=0.7,
                label="Ideal scaling", zorder=1)

    ax.set_xlabel("Number of Compute Units")
    ax.set_ylabel("Throughput (inferences/sec)")
    ax.set_xticks(cus)
    ax.set_xticklabels(["1 CU\n(300 MHz)", "2 CU\n(300 MHz)", "3 CU\n(269 MHz)"])
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("MiniRocket CU Scaling (Fused CONV+PPV)")

    base = os.path.join(FIGURES_DIR, "cu_scaling")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=300)
    print(f"[saved] {base}.pdf + .png")
    plt.close(fig)

    # --- Console + LaTeX table ---
    print("\n=== CU Scaling Results ===\n")
    print(f"{'Dataset':<16} {'1CU':>10} {'2CU':>10} {'3CU':>10} {'2/1 Scale':>10} {'3/1 Scale':>10}")
    print("-" * 70)

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{MiniRocket CU scaling with fused CONV+PPV kernel. "
                 r"2-CU runs at 300 MHz (timing met), 3-CU at 269 MHz (timing failed, auto-throttled). "
                 r"Scaling efficiency ranges from 75--88\% for 2 CU and 59--94\% for 3 CU.}")
    latex.append(r"\label{tab:cu_scaling}")
    latex.append(r"\begin{tabular}{l r r r r r}")
    latex.append(r"\toprule")
    latex.append(r"Dataset & 1 CU & 2 CU & 3 CU & 2/1 Scaling & 3/1 Scaling \\")
    latex.append(r"\midrule")

    for ds in datasets:
        tps = data[ds]
        t1 = tps[0] if tps[0] else 0
        t2 = tps[1] if tps[1] else 0
        t3 = tps[2] if tps[2] else 0
        s21 = t2 / t1 if t1 > 0 else 0
        s31 = t3 / t1 if t1 > 0 else 0
        print(f"{ds:<16} {t1:>10,.1f} {t2:>10,.1f} {t3:>10,.1f} {s21:>9.2f}x {s31:>9.2f}x")
        latex.append(f"{ds} & {t1:,.1f} & {t2:,.1f} & {t3:,.1f} & {s21:.2f}$\\times$ & {s31:.2f}$\\times$ \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    tex_path = os.path.join(TABLES_DIR, "cu_scaling.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(latex) + "\n")
    print(f"\n[saved] {tex_path}")

    # Scaling efficiency analysis
    print("\n=== Scaling Efficiency ===")
    for ds in datasets:
        tps = data[ds]
        t1 = tps[0] if tps[0] else 0
        t2 = tps[1] if tps[1] else 0
        t3 = tps[2] if tps[2] else 0
        if t1 > 0:
            eff_2 = (t2 / (2 * t1)) * 100
            # 3CU runs at 269/300 = 0.897x frequency, adjust ideal
            freq_ratio = 269.0 / 300.0
            eff_3 = (t3 / (3 * t1 * freq_ratio)) * 100
            print(f"  {ds:16s}: 2CU eff={eff_2:.0f}%, 3CU eff (freq-adjusted)={eff_3:.0f}%")


if __name__ == "__main__":
    main()
