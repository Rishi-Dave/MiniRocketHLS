#!/usr/bin/env python3
"""
generate_fused_comparison.py
Generate LaTeX table comparing separate CONV+PPV (v16_fixed) vs fused kernel.

Data sources:
  - v16_fixed throughput: from MEMORY.md validated hw results
  - fused throughput: from results_master.csv
  - resource data: from resource_utilization.csv
"""

import os
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
TABLES_DIR   = os.path.join(RESULTS_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

# v16_fixed throughput data (from hw-validated runs, MEMORY.md)
V16_FIXED = {
    "GunPoint":      {"throughput": 507.5,  "accuracy": 98.33},
    "InsectSound":   {"throughput": 295.9,  "accuracy": 74.12},
    "MosquitoSound": {"throughput": 88.1,   "accuracy": 87.88},
    "FruitFlies":    {"throughput": 67.9,   "accuracy": 95.82},
}

# Resource utilization from resource_utilization.csv (hls_csynth)
RESOURCES = {
    "v16_fixed": {"LUT": 48812, "FF": 88492, "BRAM": 576, "DSP": 724},
    "fused":     {"LUT": 57090, "FF": 94283, "BRAM": 185, "DSP": 0},  # 0 DSP!
}


def main():
    # Load fused throughput from results_master
    master = pd.read_csv(os.path.join(RESULTS_DIR, "results_master.csv"))
    fused = master[(master["Algorithm"] == "MiniRocket") &
                   (master["Platform"] == "FPGA U280 (fused 1CU)")]

    fused_tp = {}
    for _, row in fused.iterrows():
        fused_tp[row["Dataset"]] = row["Throughput_InfPerSec"]

    datasets = ["GunPoint", "InsectSound", "MosquitoSound", "FruitFlies"]

    # --- Console output ---
    print("\n=== Fused CONV+PPV vs Separate (v16_fixed) — 1CU Comparison ===\n")
    print(f"{'Dataset':<16} {'v16_fixed':>12} {'Fused':>12} {'Speedup':>10}")
    print("-" * 55)
    for ds in datasets:
        v16 = V16_FIXED[ds]["throughput"]
        ftp = fused_tp.get(ds, 0)
        speedup = ftp / v16 if v16 > 0 else 0
        print(f"{ds:<16} {v16:>12.1f} {ftp:>12.1f} {speedup:>9.1f}x")

    print(f"\n{'Resource':<16} {'v16_fixed':>12} {'Fused':>12} {'Change':>12}")
    print("-" * 55)
    for res in ["LUT", "FF", "BRAM", "DSP"]:
        v = RESOURCES["v16_fixed"][res]
        f = RESOURCES["fused"][res]
        if v > 0:
            pct = ((f - v) / v) * 100
            print(f"{res:<16} {v:>12,} {f:>12,} {pct:>+11.0f}%")
        else:
            print(f"{res:<16} {v:>12,} {f:>12,} {'N/A':>12}")

    # --- LaTeX table ---
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Impact of fused CONV+PPV kernel on MiniRocket (1 CU, 300 MHz). "
                 r"Fusion eliminates the intermediate \texttt{convolutions[16][8192]} array, "
                 r"achieving 0 DSP by exploiting ternary weights, and uses dynamic loop bounds "
                 r"that scale with time series length.}")
    latex.append(r"\label{tab:fused}")
    latex.append(r"\begin{tabular}{l r r r}")
    latex.append(r"\toprule")
    latex.append(r"& Separate & Fused & Improvement \\")
    latex.append(r"\midrule")
    latex.append(r"\multicolumn{4}{l}{\textit{Throughput (inferences/sec)}} \\")

    for ds in datasets:
        v16 = V16_FIXED[ds]["throughput"]
        ftp = fused_tp.get(ds, 0)
        speedup = ftp / v16 if v16 > 0 else 0
        latex.append(f"\\quad {ds} & {v16:,.1f} & {ftp:,.1f} & {speedup:.1f}$\\times$ \\\\")

    latex.append(r"\addlinespace")
    latex.append(r"\multicolumn{4}{l}{\textit{Resource Utilization (HLS synthesis)}} \\")

    res_labels = {"LUT": "LUTs", "FF": "FFs", "BRAM": "BRAM (18K)", "DSP": "DSP48E2"}
    for res in ["LUT", "FF", "BRAM", "DSP"]:
        v = RESOURCES["v16_fixed"][res]
        f = RESOURCES["fused"][res]
        if v > 0:
            pct = ((f - v) / v) * 100
            sign = "+" if pct > 0 else ""
            latex.append(f"\\quad {res_labels[res]} & {v:,} & {f:,} & {sign}{pct:.0f}\\% \\\\")
        else:
            latex.append(f"\\quad {res_labels[res]} & {v:,} & {f:,} & -- \\\\")

    latex.append(r"\addlinespace")
    latex.append(r"\quad Intermediate array & $16 \times 8192$ floats & Eliminated & $-$100\% \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    tex_path = os.path.join(TABLES_DIR, "fused_comparison.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(latex) + "\n")
    print(f"\n[saved] {tex_path}")

    # Key insight for the paper
    print("\n=== Key Insight ===")
    gp_speedup = fused_tp.get("GunPoint", 0) / V16_FIXED["GunPoint"]["throughput"]
    ms_speedup = fused_tp.get("MosquitoSound", 0) / V16_FIXED["MosquitoSound"]["throughput"]
    print(f"GunPoint (TS=150):      {gp_speedup:.1f}x speedup")
    print(f"MosquitoSound (TS=3750): {ms_speedup:.1f}x speedup")
    print(f"Longer time series benefit MORE from dynamic loop bounds")
    print(f"(GunPoint has 150 samples vs MosquitoSound's 3750 → {ms_speedup/gp_speedup:.1f}x more benefit)")


if __name__ == "__main__":
    main()
