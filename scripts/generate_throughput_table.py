#!/usr/bin/env python3
"""
generate_throughput_table.py
Generate LaTeX-ready throughput comparison table with speedup vs C++ baseline.

Outputs:
  results/tables/throughput_table.tex   — LaTeX tabular environment
  stdout                                — preview of the table
"""

import os
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

    platforms = [
        ("CPU (Python aeon)",       "Python",   False),
        ("CPU (C++ -O3)",           "C++ -O3",  True),   # baseline for speedup
        ("FPGA U280 (fused 1CU)",   "FPGA 1CU", False),
        ("FPGA U280 (fused 2CU)",   "FPGA 2CU", False),
        ("FPGA U280 (fused 3CU)",   "FPGA 3CU", False),
    ]

    datasets_order = ["GunPoint", "InsectSound", "MosquitoSound", "FruitFlies"]

    # Build lookup: (platform, dataset) -> (throughput, latency_mean)
    data = {}
    for _, row in mr.iterrows():
        key = (row["Platform"], row["Dataset"])
        tp = row["Throughput_InfPerSec"] if pd.notna(row["Throughput_InfPerSec"]) else None
        lat = row["Latency_Mean_ms"] if pd.notna(row["Latency_Mean_ms"]) else None
        data[key] = (tp, lat)

    # Get C++ baseline throughput per dataset
    cpp_tp = {}
    for ds in datasets_order:
        val = data.get(("CPU (C++ -O3)", ds))
        if val and val[0]:
            cpp_tp[ds] = val[0]

    # --- Console preview ---
    print("\n=== MiniRocket Throughput Comparison (inf/s) ===\n")
    print(f"{'Dataset':<16}", end="")
    for _, label, _ in platforms:
        print(f"  {label:>12}", end="")
    print(f"  {'Speedup':>10}")
    print("-" * 90)

    for ds in datasets_order:
        print(f"{ds:<16}", end="")
        best_fpga_tp = None
        for plat, label, _ in platforms:
            val = data.get((plat, ds))
            tp = val[0] if val else None
            if tp:
                print(f"  {tp:>12,.1f}", end="")
                if "FPGA" in plat:
                    if best_fpga_tp is None or tp > best_fpga_tp:
                        best_fpga_tp = tp
            else:
                print(f"  {'--':>12}", end="")
        # Best speedup vs C++
        if best_fpga_tp and ds in cpp_tp:
            speedup = best_fpga_tp / cpp_tp[ds]
            print(f"  {speedup:>9.1f}x")
        else:
            print(f"  {'--':>10}")

    # --- LaTeX output ---
    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{MiniRocket throughput (inferences/sec) and speedup over C++ baseline. "
                       r"C++ compiled with \texttt{-O3 -march=native}. "
                       r"Longer time series benefit disproportionately from the fused CONV+PPV kernel.}")
    latex_lines.append(r"\label{tab:throughput}")
    latex_lines.append(r"\resizebox{\columnwidth}{!}{")
    latex_lines.append(r"\begin{tabular}{l r r r r r r r}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Dataset & TS Len & Python & C++ \texttt{-O3} & FPGA 1CU & FPGA 2CU & FPGA 3CU & Speedup \\")
    latex_lines.append(r"\midrule")

    for ds in datasets_order:
        ds_rows = mr[mr["Dataset"] == ds]
        if ds_rows.empty:
            continue
        ts_len = int(ds_rows.iloc[0]["TS_Length"])
        row = f"{ds} & {ts_len:,}"

        for plat, _, _ in platforms:
            val = data.get((plat, ds))
            tp = val[0] if val else None
            if tp:
                row += f" & {tp:,.1f}"
            else:
                row += r" & --"

        # Best FPGA speedup vs C++
        best_fpga = None
        for plat, _, _ in platforms:
            if "FPGA" not in plat:
                continue
            val = data.get((plat, ds))
            tp = val[0] if val else None
            if tp and (best_fpga is None or tp > best_fpga):
                best_fpga = tp

        if best_fpga and ds in cpp_tp:
            speedup = best_fpga / cpp_tp[ds]
            row += f" & {speedup:.1f}$\\times$"
        else:
            row += r" & --"

        row += r" \\"
        latex_lines.append(row)

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(r"\end{table}")

    tex_path = os.path.join(TABLES_DIR, "throughput_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(latex_lines) + "\n")
    print(f"\n[saved] {tex_path}")

    # --- Latency table ---
    print("\n=== MiniRocket Per-Sample Latency (ms) ===\n")
    print(f"{'Dataset':<16} {'Platform':<25} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    print("-" * 80)
    for ds in datasets_order:
        for plat, label, _ in platforms:
            val = data.get((plat, ds))
            if not val or val[1] is None:
                continue
            row_data = mr[(mr["Platform"] == plat) & (mr["Dataset"] == ds)].iloc[0]
            mean = row_data["Latency_Mean_ms"]
            p50 = row_data["Latency_P50_ms"] if pd.notna(row_data["Latency_P50_ms"]) else None
            p95 = row_data["Latency_P95_ms"] if pd.notna(row_data["Latency_P95_ms"]) else None
            p99 = row_data["Latency_P99_ms"] if pd.notna(row_data["Latency_P99_ms"]) else None
            line = f"{ds:<16} {label:<25} {mean:>8.3f}"
            line += f" {p50:>8.3f}" if p50 else f" {'--':>8}"
            line += f" {p95:>8.3f}" if p95 else f" {'--':>8}"
            line += f" {p99:>8.3f}" if p99 else f" {'--':>8}"
            print(line)

    # --- Latency LaTeX ---
    lat_lines = []
    lat_lines.append(r"\begin{table}[t]")
    lat_lines.append(r"\centering")
    lat_lines.append(r"\caption{MiniRocket per-sample inference latency (ms).}")
    lat_lines.append(r"\label{tab:latency}")
    lat_lines.append(r"\begin{tabular}{l l r r r r}")
    lat_lines.append(r"\toprule")
    lat_lines.append(r"Dataset & Platform & Mean & P50 & P95 & P99 \\")
    lat_lines.append(r"\midrule")

    prev_ds = None
    for ds in datasets_order:
        for plat, label, _ in platforms:
            rows = mr[(mr["Platform"] == plat) & (mr["Dataset"] == ds)]
            if rows.empty:
                continue
            row_data = rows.iloc[0]
            if pd.isna(row_data["Latency_Mean_ms"]):
                continue

            ds_label = ds if ds != prev_ds else ""
            prev_ds = ds

            mean = row_data["Latency_Mean_ms"]
            p50 = f"{row_data['Latency_P50_ms']:.3f}" if pd.notna(row_data["Latency_P50_ms"]) else "--"
            p95 = f"{row_data['Latency_P95_ms']:.3f}" if pd.notna(row_data["Latency_P95_ms"]) else "--"
            p99 = f"{row_data['Latency_P99_ms']:.3f}" if pd.notna(row_data["Latency_P99_ms"]) else "--"

            lat_lines.append(f"{ds_label} & {label} & {mean:.3f} & {p50} & {p95} & {p99} \\\\")

        if ds != datasets_order[-1]:
            lat_lines.append(r"\addlinespace")

    lat_lines.append(r"\bottomrule")
    lat_lines.append(r"\end{tabular}")
    lat_lines.append(r"\end{table}")

    lat_path = os.path.join(TABLES_DIR, "latency_table.tex")
    with open(lat_path, "w") as f:
        f.write("\n".join(lat_lines) + "\n")
    print(f"\n[saved] {lat_path}")


if __name__ == "__main__":
    main()
