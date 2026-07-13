#!/bin/bash
# Scalability Sweep: Measure throughput vs time series length on FPGA
# Uses the fused 1-CU bitstream with varying time_series_length.
#
# The fused kernel has MAX_TIME_SERIES_LENGTH=8192 and accepts
# time_series_length as a runtime parameter via AXI-lite.
#
# This script generates synthetic test data at each length,
# then runs inference and records throughput.
#
# Usage: bash scripts/run_scalability_sweep.sh
# Output: results/scalability_sweep.csv

set -euo pipefail
cd "$(dirname "$0")/.."

# Ensure XRT is sourced
if [ -z "${XILINX_XRT:-}" ]; then
    echo "Sourcing XRT environment..."
    source /opt/xilinx/xrt/setup.sh
fi

MODULAR_DIR="minirocket_modular"
HOST="$MODULAR_DIR/modular_host"
XCLBIN="$MODULAR_DIR/build_dir.hw.fused/krnl.xclbin"
RESULTS_DIR="results"

# Verify files exist
for f in "$HOST" "$XCLBIN"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing file: $f"
        exit 1
    fi
done

mkdir -p "$RESULTS_DIR"

echo "=== MiniRocket Scalability Sweep ==="
echo "Using: $XCLBIN (fused 1-CU, 300 MHz)"
echo ""

# Generate test data at each length using Python
python3 - <<'PYTHON_SCRIPT'
import json
import numpy as np
import os

# Use a real model as the base (just need the weights/structure)
# Try to load the InsectSound model which was used for benchmarks
model_base = None
for candidate in [
    "minirocket_modular/insectsound_minirocket_model.json",
    "minirocket_modular/mosquitosound_minirocket_model.json",
    "minirocket_modular/minirocket_model.json",
    "minirocket_modular/gunpoint_minirocket_model.json",
]:
    if os.path.exists(candidate):
        model_base = candidate
        break

if model_base is None:
    # Search for any model JSON
    import glob
    candidates = glob.glob("minirocket_modular/*_model.json")
    if candidates:
        model_base = candidates[0]
    else:
        print("ERROR: No model JSON found in minirocket_modular/")
        exit(1)

print(f"Using base model: {model_base}")

with open(model_base) as f:
    model = json.load(f)

lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
num_samples = 100  # enough for timing, not so many it takes forever

os.makedirs("results/scalability_data", exist_ok=True)

for length in lengths:
    # Generate synthetic test data at this length
    np.random.seed(42)
    test_data = {
        "num_samples": num_samples,
        "time_series_length": length,
        "num_classes": model.get("num_classes", 2),
        "time_series": np.random.randn(num_samples, length).tolist(),
        "labels": [0] * num_samples
    }

    test_file = f"results/scalability_data/test_len{length}.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    print(f"  Generated: {test_file} ({num_samples} x {length})")

    # Also create a modified model with the correct time_series_length
    model_copy = dict(model)
    model_copy["time_series_length"] = length
    model_file = f"results/scalability_data/model_len{length}.json"
    with open(model_file, 'w') as f:
        json.dump(model_copy, f)

print("Done generating test data.")
PYTHON_SCRIPT

echo ""
echo "Running FPGA benchmarks at each length..."
echo ""

# CSV output
CSV="$RESULTS_DIR/scalability_sweep.csv"
echo "time_series_length,num_samples,total_time_s,throughput_inf_per_s,mean_latency_ms" > "$CSV"

for length in 64 128 256 512 1024 2048 4096 8192; do
    model_file="results/scalability_data/model_len${length}.json"
    test_file="results/scalability_data/test_len${length}.json"
    logfile="results/scalability_data/run_len${length}.log"

    echo "--- Length: $length ---"
    "$HOST" "$XCLBIN" "$model_file" "$test_file" 2>&1 | tee "$logfile"

    # Parse results from log
    throughput=$(grep -oP 'Throughput:\s+\K[\d.]+' "$logfile" 2>/dev/null || echo "N/A")
    latency=$(grep -oP 'Average.*latency:\s+\K[\d.]+' "$logfile" 2>/dev/null || echo "N/A")
    total=$(grep -oP 'Total.*time:\s+\K[\d.]+' "$logfile" 2>/dev/null || echo "N/A")

    echo "${length},100,${total},${throughput},${latency}" >> "$CSV"
    echo ""
done

echo "============================================"
echo "  Scalability sweep complete!"
echo "  Results: $CSV"
echo "============================================"
cat "$CSV"
