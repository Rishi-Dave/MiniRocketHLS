#!/usr/bin/env python3
"""Parse FPGA inference log files to extract per-sample latency and produce CSV summaries.

Handles three log formats:
  Format A (MiniRocket modular 1CU): per-sample TIMING RESULTS blocks with H2D/K1/K2/K3/D2H
  Format B (MiniRocket multi-CU): summary-only (wall time, throughput, avg latency)
  Format C (HYDRA): per-sample lines "Sample N/M: predicted=X, actual=Y, time=Z ms"
"""

import re
import os
import csv
import glob
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Log file registry: (path_pattern, algorithm, variant, dataset)
LOG_REGISTRY = [
    # MiniRocket modular fused 1CU
    ("minirocket_modular/fused_gunpoint_test.log", "MiniRocket", "fused_1cu", "GunPoint"),
    ("minirocket_modular/fused_insectsound_test.log", "MiniRocket", "fused_1cu", "InsectSound"),
    ("minirocket_modular/fused_mosquitosound_test.log", "MiniRocket", "fused_1cu", "MosquitoSound"),
    ("minirocket_modular/fused_fruitflies_test.log", "MiniRocket", "fused_1cu", "FruitFlies"),
    # MiniRocket modular fused 2CU
    ("minirocket_modular/fused_2cu_gunpoint_test.log", "MiniRocket", "fused_2cu", "GunPoint"),
    ("minirocket_modular/fused_2cu_insectsound_test.log", "MiniRocket", "fused_2cu", "InsectSound"),
    ("minirocket_modular/fused_2cu_mosquitosound_test.log", "MiniRocket", "fused_2cu", "MosquitoSound"),
    ("minirocket_modular/fused_2cu_fruitflies_test.log", "MiniRocket", "fused_2cu", "FruitFlies"),
    # MiniRocket modular fused 3CU
    ("minirocket_modular/fused_3cu_gunpoint_test.log", "MiniRocket", "fused_3cu", "GunPoint"),
    ("minirocket_modular/fused_3cu_insectsound_test.log", "MiniRocket", "fused_3cu", "InsectSound"),
    ("minirocket_modular/fused_3cu_mosquitosound_test.log", "MiniRocket", "fused_3cu", "MosquitoSound"),
    ("minirocket_modular/fused_3cu_fruitflies_test.log", "MiniRocket", "fused_3cu", "FruitFlies"),
    # HYDRA
    ("hydra_optimized/hydra_hw_test.log", "HYDRA", "v1_float", "GunPoint"),
    ("hydra_optimized/fused_insectsound_test.log", "HYDRA", "v2_apfixed", "InsectSound"),
    # v16_fixed (pre-fused baseline)
    ("minirocket_modular/v16_fixed_hw_test.log", "MiniRocket", "v16_fixed_1cu", "GunPoint"),
]

# CPU logs
CPU_LOG_REGISTRY = [
    ("cpu/minirocket_cpu_insectsound.log", "MiniRocket", "CPU_python", "InsectSound"),
    ("cpu/minirocket_cpu_fruitflies.log", "MiniRocket", "CPU_python", "FruitFlies"),
    ("cpu/minirocket_cpu_mosquitosound.log", "MiniRocket", "CPU_python", "MosquitoSound"),
]


@dataclass
class SampleTiming:
    sample_id: int
    h2d_ms: float = 0.0
    kernel_ms: float = 0.0
    k1_ms: float = 0.0
    k2_ms: float = 0.0
    k3_ms: float = 0.0
    d2h_ms: float = 0.0
    total_ms: float = 0.0
    predicted: int = -1
    actual: int = -1
    correct: bool = False


@dataclass
class LogResults:
    algorithm: str
    variant: str
    dataset: str
    log_file: str
    samples: list = field(default_factory=list)
    # Summary fields (for multi-CU logs without per-sample data)
    total_samples: int = 0
    correct_predictions: int = 0
    accuracy_pct: float = 0.0
    throughput: float = 0.0
    wall_time_ms: float = 0.0
    avg_latency_ms: float = 0.0
    has_per_sample: bool = False


def parse_modular_1cu_log(filepath: str) -> list[SampleTiming]:
    """Parse Format A: MiniRocket modular 1CU logs with per-sample TIMING RESULTS blocks."""
    samples = []
    with open(filepath, "r") as f:
        content = f.read()

    # Find all per-sample timing blocks (skip the final summary block which has Throughput)
    pattern = (
        r"STARTING pipeline for input\[(\d+)\].*?"
        r"Predicted class: (\d+).*?"
        r"Expected class: (\d+).*?"
        r"Prediction (correct|INCORRECT).*?"
        r"={5,}\s*TIMING RESULTS\s*={5,}\s*"
        r"H2D transfer:\s+([\d.]+)\s*ms\s*"
        r"Kernel pipeline:\s+([\d.]+)\s*ms\s*"
        r"K1 \(FeatureExt\):\s+([\d.]+)\s*ms\s*"
        r"K2 \(Scaler\):\s+([\d.]+)\s*ms\s*"
        r"K3 \(Classifier\):\s+([\d.]+)\s*ms\s*"
        r"D2H transfer:\s+([\d.]+)\s*ms\s*"
        r"Total latency:\s+([\d.]+)\s*ms"
    )
    for m in re.finditer(pattern, content, re.DOTALL):
        s = SampleTiming(
            sample_id=int(m.group(1)),
            predicted=int(m.group(2)),
            actual=int(m.group(3)),
            correct=(m.group(4) == "correct"),
            h2d_ms=float(m.group(5)),
            kernel_ms=float(m.group(6)),
            k1_ms=float(m.group(7)),
            k2_ms=float(m.group(8)),
            k3_ms=float(m.group(9)),
            d2h_ms=float(m.group(10)),
            total_ms=float(m.group(11)),
        )
        samples.append(s)
    return samples


def parse_modular_multiCU_log(filepath: str) -> dict:
    """Parse Format B: Multi-CU logs with only summary stats."""
    with open(filepath, "r") as f:
        content = f.read()

    result = {}

    m = re.search(r"Total samples:\s+(\d+)", content)
    if m:
        result["total_samples"] = int(m.group(1))

    m = re.search(r"Correct predictions:\s+(\d+)\s*/\s*(\d+)", content)
    if m:
        result["correct"] = int(m.group(1))
        result["total"] = int(m.group(2))

    m = re.search(r"Overall accuracy:\s+([\d.]+)", content)
    if m:
        result["accuracy_pct"] = float(m.group(1))

    m = re.search(r"Total wall time:\s+([\d.]+)\s*ms", content)
    if m:
        result["wall_time_ms"] = float(m.group(1))

    m = re.search(r"Throughput:\s+([\d.]+)\s*inferences/sec", content)
    if m:
        result["throughput"] = float(m.group(1))

    m = re.search(r"Avg latency/sample:\s+([\d.]+)\s*ms", content)
    if m:
        result["avg_latency_ms"] = float(m.group(1))

    return result


def parse_hydra_log(filepath: str) -> list[SampleTiming]:
    """Parse Format C: HYDRA logs with 'Sample N/M: predicted=X, actual=Y, time=Z ms'."""
    samples = []
    with open(filepath, "r") as f:
        for line in f:
            m = re.match(
                r"Sample\s+(\d+)/(\d+):\s+predicted=(\d+),\s+actual=(\d+),\s+time=([\d.]+)\s*ms\s*\[(.*?)\]",
                line.strip(),
            )
            if m:
                s = SampleTiming(
                    sample_id=int(m.group(1)),
                    predicted=int(m.group(3)),
                    actual=int(m.group(4)),
                    total_ms=float(m.group(5)),
                    correct=(m.group(6).strip() == "\u2713"),
                )
                # For HYDRA, total_ms = kernel time (no H2D/D2H breakdown in log)
                s.kernel_ms = s.total_ms
                samples.append(s)
    return samples


def parse_cpu_log(filepath: str) -> dict:
    """Parse CPU benchmark log files."""
    with open(filepath, "r") as f:
        content = f.read()

    result = {}

    # Accuracy: format is "Accuracy: 0.788400 (19710/25000)" or "Accuracy: 0.788400"
    m = re.search(r"Accuracy:\s+([\d.]+)(?:\s*\((\d+)/(\d+)\))?", content)
    if m:
        acc_val = float(m.group(1))
        # If value < 1, it's a fraction; if > 1, it's a percentage
        result["accuracy_pct"] = acc_val * 100 if acc_val <= 1.0 else acc_val
        if m.group(2) and m.group(3):
            result["correct"] = int(m.group(2))
            result["total_samples"] = int(m.group(3))

    # Throughput: "Throughput: 519.9 inferences/sec"
    m = re.search(r"Throughput:\s+([\d.]+)\s*inferences/sec", content)
    if m:
        result["throughput"] = float(m.group(1))

    m = re.search(r"Latency per sample:\s+([\d.]+)\s*ms", content)
    if m:
        result["avg_latency_ms"] = float(m.group(1))

    m = re.search(r"Test time:\s+([\d.]+)\s*seconds", content)
    if m:
        result["total_time_s"] = float(m.group(1))
        # Derive throughput/latency from total time if not already set
        if "total_samples" in result and "throughput" not in result:
            result["throughput"] = result["total_samples"] / result["total_time_s"]
        if "total_samples" in result and "avg_latency_ms" not in result:
            result["avg_latency_ms"] = result["total_time_s"] * 1000 / result["total_samples"]

    return result


def compute_percentiles(values: list[float]) -> dict:
    """Compute standard percentile statistics."""
    if not values:
        return {}
    arr = np.array(values)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def detect_log_format(filepath: str) -> str:
    """Detect which format a log file uses."""
    with open(filepath, "r") as f:
        content = f.read(10000)  # Read first 10KB

    if "STARTING pipeline for input" in content:
        return "modular_1cu"
    elif "CU pipelines:" in content or "Avg latency/sample:" in content:
        return "multi_cu"
    elif re.search(r"Sample\s+\d+/\d+:.*time=", content):
        return "hydra"
    else:
        return "unknown"


def process_log(entry: tuple) -> LogResults:
    """Process a single log file entry."""
    relpath, algorithm, variant, dataset = entry
    filepath = str(PROJECT_ROOT / relpath)

    if not os.path.exists(filepath):
        print(f"  SKIP: {relpath} (file not found)")
        return None

    fmt = detect_log_format(filepath)
    result = LogResults(
        algorithm=algorithm,
        variant=variant,
        dataset=dataset,
        log_file=relpath,
    )

    if fmt == "modular_1cu":
        samples = parse_modular_1cu_log(filepath)
        result.samples = samples
        result.has_per_sample = True
        # Use the FINAL RESULTS block for authoritative accuracy/throughput
        with open(filepath) as f:
            content = f.read()
        m = re.search(r"Total correct predictions:\s+(\d+)\s*/\s*(\d+)", content)
        if m:
            result.correct_predictions = int(m.group(1))
            result.total_samples = int(m.group(2))
            result.accuracy_pct = 100.0 * result.correct_predictions / result.total_samples
        else:
            result.total_samples = len(samples)
            result.correct_predictions = sum(1 for s in samples if s.correct)
            result.accuracy_pct = 100.0 * result.correct_predictions / result.total_samples if samples else 0
        if samples:
            result.avg_latency_ms = np.mean([s.total_ms for s in samples])
            result.throughput = 1000.0 / result.avg_latency_ms if result.avg_latency_ms > 0 else 0
        m = re.search(r"Throughput:\s+([\d.]+)\s*inferences/sec", content)
        if m:
            result.throughput = float(m.group(1))

    elif fmt == "multi_cu":
        summary = parse_modular_multiCU_log(filepath)
        result.total_samples = summary.get("total", summary.get("total_samples", 0))
        result.correct_predictions = summary.get("correct", 0)
        result.accuracy_pct = summary.get("accuracy_pct", 0)
        result.throughput = summary.get("throughput", 0)
        result.wall_time_ms = summary.get("wall_time_ms", 0)
        result.avg_latency_ms = summary.get("avg_latency_ms", 0)

    elif fmt == "hydra":
        samples = parse_hydra_log(filepath)
        result.samples = samples
        result.has_per_sample = True
        result.total_samples = len(samples)
        result.correct_predictions = sum(1 for s in samples if s.correct)
        if samples:
            result.accuracy_pct = 100.0 * result.correct_predictions / result.total_samples
            result.avg_latency_ms = np.mean([s.total_ms for s in samples])
            result.throughput = 1000.0 / result.avg_latency_ms if result.avg_latency_ms > 0 else 0

    return result


def write_per_sample_csv(result: LogResults, output_dir: Path):
    """Write per-sample timing to CSV."""
    if not result.samples:
        return

    fname = f"{result.algorithm}_{result.variant}_{result.dataset}_per_sample.csv"
    outpath = output_dir / fname

    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "h2d_ms", "kernel_ms", "k1_ms", "k2_ms", "k3_ms",
            "d2h_ms", "total_ms", "predicted", "actual", "correct"
        ])
        for s in result.samples:
            writer.writerow([
                s.sample_id, f"{s.h2d_ms:.3f}", f"{s.kernel_ms:.3f}",
                f"{s.k1_ms:.3f}", f"{s.k2_ms:.3f}", f"{s.k3_ms:.3f}",
                f"{s.d2h_ms:.3f}", f"{s.total_ms:.3f}",
                s.predicted, s.actual, int(s.correct)
            ])
    print(f"  Written: {outpath.name} ({len(result.samples)} samples)")


def write_summary_csv(all_results: list[LogResults], output_dir: Path):
    """Write consolidated summary CSV with latency percentiles."""
    outpath = output_dir / "latency_summary.csv"

    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Algorithm", "Variant", "Dataset", "Num_Samples",
            "Correct", "Accuracy_Pct",
            "Throughput_InfPerSec",
            "Latency_Mean_ms", "Latency_P50_ms", "Latency_P95_ms", "Latency_P99_ms",
            "Latency_Min_ms", "Latency_Max_ms", "Latency_Std_ms",
            "H2D_Mean_ms", "Kernel_Mean_ms", "K1_Mean_ms", "K2_Mean_ms", "K3_Mean_ms", "D2H_Mean_ms",
            "Has_PerSample", "Source_Log"
        ])

        for r in all_results:
            if r is None:
                continue

            if r.has_per_sample and r.samples:
                total_times = [s.total_ms for s in r.samples]
                stats = compute_percentiles(total_times)

                h2d_mean = np.mean([s.h2d_ms for s in r.samples]) if r.samples[0].h2d_ms > 0 else 0
                kern_mean = np.mean([s.kernel_ms for s in r.samples])
                k1_mean = np.mean([s.k1_ms for s in r.samples]) if r.samples[0].k1_ms > 0 else 0
                k2_mean = np.mean([s.k2_ms for s in r.samples]) if r.samples[0].k2_ms > 0 else 0
                k3_mean = np.mean([s.k3_ms for s in r.samples]) if r.samples[0].k3_ms > 0 else 0
                d2h_mean = np.mean([s.d2h_ms for s in r.samples]) if r.samples[0].d2h_ms > 0 else 0

                writer.writerow([
                    r.algorithm, r.variant, r.dataset, r.total_samples,
                    r.correct_predictions, f"{r.accuracy_pct:.2f}",
                    f"{r.throughput:.1f}",
                    f"{stats['mean']:.3f}", f"{stats['p50']:.3f}",
                    f"{stats['p95']:.3f}", f"{stats['p99']:.3f}",
                    f"{stats['min']:.3f}", f"{stats['max']:.3f}",
                    f"{stats['std']:.3f}",
                    f"{h2d_mean:.3f}", f"{kern_mean:.3f}",
                    f"{k1_mean:.3f}", f"{k2_mean:.3f}", f"{k3_mean:.3f}",
                    f"{d2h_mean:.3f}",
                    "yes", r.log_file
                ])
            else:
                writer.writerow([
                    r.algorithm, r.variant, r.dataset, r.total_samples,
                    r.correct_predictions, f"{r.accuracy_pct:.2f}",
                    f"{r.throughput:.1f}",
                    f"{r.avg_latency_ms:.3f}", "", "", "",
                    "", "", "",
                    "", "", "", "", "", "",
                    "no", r.log_file
                ])

    print(f"\nSummary written: {outpath}")


def write_latency_breakdown_csv(all_results: list[LogResults], output_dir: Path):
    """Write detailed latency breakdown CSV for 1CU results (for paper tables)."""
    outpath = output_dir / "latency_breakdown.csv"

    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Algorithm", "Variant", "Dataset",
            "Component", "Mean_ms", "P50_ms", "P95_ms", "P99_ms", "Min_ms", "Max_ms"
        ])

        for r in all_results:
            if r is None or not r.has_per_sample or not r.samples:
                continue
            # Only write breakdown for logs that have H2D/K1/K2/K3/D2H
            if r.samples[0].h2d_ms == 0 and r.samples[0].k1_ms == 0:
                continue

            components = [
                ("H2D", [s.h2d_ms for s in r.samples]),
                ("K1_FeatureExt", [s.k1_ms for s in r.samples]),
                ("K2_Scaler", [s.k2_ms for s in r.samples]),
                ("K3_Classifier", [s.k3_ms for s in r.samples]),
                ("D2H", [s.d2h_ms for s in r.samples]),
                ("Total", [s.total_ms for s in r.samples]),
            ]

            for comp_name, values in components:
                stats = compute_percentiles(values)
                writer.writerow([
                    r.algorithm, r.variant, r.dataset, comp_name,
                    f"{stats['mean']:.3f}", f"{stats['p50']:.3f}",
                    f"{stats['p95']:.3f}", f"{stats['p99']:.3f}",
                    f"{stats['min']:.3f}", f"{stats['max']:.3f}",
                ])

    print(f"Breakdown written: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Parse FPGA inference logs to latency CSVs")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR),
                        help="Output directory for CSVs")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FPGA Log Parser - Latency Extraction")
    print("=" * 60)

    all_results = []

    # Process FPGA logs
    print("\n--- FPGA Logs ---")
    for entry in LOG_REGISTRY:
        relpath = entry[0]
        print(f"\nProcessing: {relpath}")
        result = process_log(entry)
        if result:
            all_results.append(result)
            if result.has_per_sample:
                write_per_sample_csv(result, output_dir)
                print(f"  Samples: {result.total_samples}, "
                      f"Accuracy: {result.accuracy_pct:.2f}%, "
                      f"Throughput: {result.throughput:.1f} inf/s")
            else:
                print(f"  Summary-only: {result.total_samples} samples, "
                      f"Accuracy: {result.accuracy_pct:.2f}%, "
                      f"Throughput: {result.throughput:.1f} inf/s, "
                      f"Avg latency: {result.avg_latency_ms:.3f} ms")

    # Process CPU logs
    print("\n--- CPU Logs ---")
    for entry in CPU_LOG_REGISTRY:
        relpath, algorithm, variant, dataset = entry
        filepath = str(PROJECT_ROOT / relpath)
        if not os.path.exists(filepath):
            print(f"  SKIP: {relpath}")
            continue
        print(f"\nProcessing: {relpath}")
        cpu_data = parse_cpu_log(filepath)
        r = LogResults(
            algorithm=algorithm, variant=variant, dataset=dataset,
            log_file=relpath,
            total_samples=cpu_data.get("total_samples", 0),
            correct_predictions=cpu_data.get("correct", 0),
            accuracy_pct=cpu_data.get("accuracy_pct", 0),
            throughput=cpu_data.get("throughput", 0),
            avg_latency_ms=cpu_data.get("avg_latency_ms", 0),
        )
        all_results.append(r)
        print(f"  Accuracy: {r.accuracy_pct:.2f}%, Throughput: {r.throughput:.1f} inf/s")

    # Write outputs
    print("\n" + "=" * 60)
    write_summary_csv(all_results, output_dir)
    write_latency_breakdown_csv(all_results, output_dir)

    # Print quick summary table
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<12} {'Variant':<16} {'Dataset':<16} {'Samples':>8} {'Acc%':>7} {'Tput':>10} {'Lat_P50':>8} {'Lat_P99':>8}")
    print("-" * 95)
    for r in all_results:
        if r is None:
            continue
        p50 = p99 = ""
        if r.has_per_sample and r.samples:
            times = [s.total_ms for s in r.samples]
            p50 = f"{np.percentile(times, 50):.3f}"
            p99 = f"{np.percentile(times, 99):.3f}"
        print(f"{r.algorithm:<12} {r.variant:<16} {r.dataset:<16} {r.total_samples:>8} "
              f"{r.accuracy_pct:>6.2f}% {r.throughput:>9.1f} {p50:>8} {p99:>8}")


if __name__ == "__main__":
    main()
