#!/usr/bin/env python3
"""
generate_energy_figure.py
Generate energy efficiency bar chart for the revised paper.

Output: results/figures/fig6_energy_efficiency.{pdf,png}
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CB = {
    "orange": "#E69F00", "sky": "#56B4E9", "blue": "#0072B2",
    "vermil": "#D55E00", "green": "#009E73", "black": "#000000",
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#dddddd",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Power data
FPGA_W = 24.4
CPU_W = 100.0

# Throughput data from results_master.csv (hw-validated)
DATA = {
    "InsectSound": {
        "CPU C++": 288.9,
        "FPGA 1-CU": 1966.8,
        "FPGA 2-CU": 3191.1,
        "FPGA 3-CU": 5012.2,
    },
    "MosquitoSound": {
        "CPU C++": 48.0,
        "FPGA 1-CU": 937.0,
        "FPGA 2-CU": 1496.9,
        "FPGA 3-CU": 2819.6,
    },
    "FruitFlies": {
        "CPU C++": 258.2,
        "FPGA 1-CU": 742.3,
        "FPGA 2-CU": 1121.4,
        "FPGA 3-CU": 2190.7,
    },
}

POWER = {"CPU C++": CPU_W, "FPGA 1-CU": FPGA_W, "FPGA 2-CU": FPGA_W, "FPGA 3-CU": FPGA_W}

datasets = list(DATA.keys())
platforms = ["CPU C++", "FPGA 1-CU", "FPGA 2-CU", "FPGA 3-CU"]
colors = [CB["black"], CB["sky"], CB["blue"], CB["vermil"]]

fig, ax = plt.subplots(figsize=(8, 5))

n_ds = len(datasets)
n_pl = len(platforms)
bw = 0.18
x0 = np.arange(n_ds) * 1.0
offsets = np.linspace(-(n_pl - 1) / 2, (n_pl - 1) / 2, n_pl) * bw

for pi, platform in enumerate(platforms):
    effs = []
    for ds in datasets:
        tp = DATA[ds][platform]
        pwr = POWER[platform]
        effs.append(tp / pwr)  # inf/J

    rects = ax.bar(x0 + offsets[pi], effs, width=bw * 0.90,
                   color=colors[pi], zorder=3, edgecolor="white", linewidth=0.5)

    # Add efficiency ratio labels above FPGA bars
    for i, (rect, eff) in enumerate(zip(rects, effs)):
        cpu_eff = DATA[datasets[i]]["CPU C++"] / POWER["CPU C++"]
        ratio = eff / cpu_eff
        if ratio > 1:
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() * 1.05,
                    f"{ratio:.0f}x",
                    ha="center", va="bottom", fontsize=9,
                    color=colors[pi], fontweight="bold")

ax.set_yscale("log")
ax.set_ylabel("Energy Efficiency (inf/J)", labelpad=6)
ax.set_xticks(x0)
ax.set_xticklabels([f"{ds}\n(L={l})" for ds, l in
                     zip(datasets, [600, 3750, 5000])])
ax.set_title("Energy Efficiency: FPGA (24.4W) vs CPU (100W TDP)", pad=10)

legend_handles = [Patch(facecolor=c, label=p) for c, p in zip(colors, platforms)]
ax.legend(handles=legend_handles, ncol=2, loc="upper left")
ax.set_axisbelow(True)

fig.tight_layout()
base = os.path.join(FIGURES_DIR, "fig6_energy_efficiency")
fig.savefig(base + ".pdf")
fig.savefig(base + ".png", dpi=300)
print(f"Saved: {base}.pdf + .png")
plt.close(fig)
