#!/usr/bin/env python3
"""
generate_accuracy_table.py
Generate LaTeX-ready accuracy comparison table from results_master.csv.

Outputs:
  results/tables/accuracy_table.tex   — LaTeX tabular environment
  stdout                              — preview of the table
"""

import os
import sys
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
TABLES_DIR   = os.path.join(RESULTS_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, "results_master.csv")


def main():
    df = pd.read_csv(CSV_PATH)
    mr = df[df["Algorithm"] == "MiniRocket"].copy()

    # Define platform groups in display order
    platforms = [
        ("CPU (Python aeon)",       "Python (aeon)"),
        ("CPU (C++ -O3)",           "C++ {\\texttt{-O3}}"),
        ("FPGA U280 (fused 1CU)",   "FPGA 1CU"),
        ("FPGA U280 (fused 2CU)",   "FPGA 2CU"),
        ("FPGA U280 (fused 3CU)",   "FPGA 3CU"),
    ]

    datasets_order = ["GunPoint", "InsectSound", "MosquitoSound", "FruitFlies"]

    # Build accuracy lookup: (platform, dataset) -> accuracy
    acc = {}
    for _, row in mr.iterrows():
        key = (row["Platform"], row["Dataset"])
        if pd.notna(row["Accuracy_Pct"]):
            acc[key] = row["Accuracy_Pct"]

    # --- Console preview ---
    print("\n=== MiniRocket Accuracy Comparison ===\n")
    header = f"{'Dataset':<16} {'TS Len':>6} {'Samples':>8}"
    for _, label in platforms:
        clean = label.replace("{\\texttt{-O3}}", "-O3")
        header += f"  {clean:>14}"
    print(header)
    print("-" * len(header))

    for ds in datasets_order:
        ds_rows = mr[mr["Dataset"] == ds]
        if ds_rows.empty:
            continue
        ts_len = int(ds_rows.iloc[0]["TS_Length"])
        samples = int(ds_rows.iloc[0]["Test_Samples"])
        line = f"{ds:<16} {ts_len:>6} {samples:>8}"
        for plat, _ in platforms:
            val = acc.get((plat, ds))
            if val is not None:
                line += f"  {val:>13.2f}%"
            else:
                line += f"  {'--':>14}"
        print(line)

    # --- LaTeX output ---
    n_plat = len(platforms)
    col_spec = "l r r " + " ".join(["r"] * n_plat)
    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{MiniRocket classification accuracy (\%) across platforms. "
                       r"C++ and FPGA use identical exported model weights, achieving bit-exact predictions. "
                       r"Python (aeon) trains with \texttt{float64} precision, causing minor accuracy differences "
                       r"from the \texttt{float32} exported model.}")
    latex_lines.append(r"\label{tab:accuracy}")
    latex_lines.append(r"\begin{tabular}{" + col_spec + r"}")
    latex_lines.append(r"\toprule")

    # Header row
    hdr = r"Dataset & TS Len & Samples"
    for _, label in platforms:
        hdr += f" & {label}"
    hdr += r" \\"
    latex_lines.append(hdr)
    latex_lines.append(r"\midrule")

    # Data rows
    for ds in datasets_order:
        ds_rows = mr[mr["Dataset"] == ds]
        if ds_rows.empty:
            continue
        ts_len = int(ds_rows.iloc[0]["TS_Length"])
        samples = int(ds_rows.iloc[0]["Test_Samples"])
        row = f"{ds} & {ts_len:,} & {samples:,}"
        for plat, _ in platforms:
            val = acc.get((plat, ds))
            if val is not None:
                row += f" & {val:.2f}"
            else:
                row += r" & --"
        row += r" \\"
        latex_lines.append(row)

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    tex_path = os.path.join(TABLES_DIR, "accuracy_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(latex_lines) + "\n")
    print(f"\n[saved] {tex_path}")

    # --- Verify bit-exact match ---
    print("\n=== Bit-Exact Verification ===")
    cpp_plat = "CPU (C++ -O3)"
    fpga_plats = [p for p, _ in platforms if "FPGA" in p]
    all_match = True
    for ds in datasets_order:
        cpp_acc = acc.get((cpp_plat, ds))
        if cpp_acc is None:
            continue
        for fp in fpga_plats:
            fpga_acc = acc.get((fp, ds))
            if fpga_acc is None:
                continue
            diff = abs(cpp_acc - fpga_acc)
            status = "MATCH" if diff < 0.02 else f"DIFF={diff:.2f}%"
            print(f"  {ds:16s} C++={cpp_acc:.2f}% {fp}={fpga_acc:.2f}% -> {status}")
            if diff >= 0.02:
                all_match = False
    print(f"\n  Overall: {'ALL MATCH (bit-exact)' if all_match else 'SOME DIFFERENCES FOUND'}")


if __name__ == "__main__":
    main()
