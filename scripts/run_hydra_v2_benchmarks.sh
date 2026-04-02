#!/bin/bash
# Run HYDRA v2_fixed (ap_fixed, UNROLL=16) benchmarks on all UCR datasets
# The v2_fixed kernel uses float at the HBM interface, ap_fixed<32,16> internally
# So the existing float host code is compatible.
#
# Usage: bash scripts/run_hydra_v2_benchmarks.sh [quick|full]
# Default: quick (1000 samples per dataset)

set -euo pipefail
cd "$(dirname "$0")/.."

# Ensure XRT is sourced
if [ -z "${XILINX_XRT:-}" ]; then
    echo "Sourcing XRT environment..."
    source /opt/xilinx/xrt/setup.sh
fi

MODE="${1:-quick}"
HYDRA_DIR="hydra_optimized"
HOST="$HYDRA_DIR/host/hydra_host"
XCLBIN="$HYDRA_DIR/build_dir.hw.v2_fixed/krnl.xclbin"
MODELS_DIR="$HYDRA_DIR/models"
RESULTS_DIR="$HYDRA_DIR/results/v2_fixed"

# Verify files exist
for f in "$HOST" "$XCLBIN"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing file: $f"
        exit 1
    fi
done

mkdir -p "$RESULTS_DIR"

# Check FPGA is ready
echo "=== Checking FPGA status ==="
xbutil examine 2>&1 | head -5
echo ""

DATASETS=("insectsound" "mosquitosound" "fruitflies")

for dataset in "${DATASETS[@]}"; do
    model="$MODELS_DIR/hydra_${dataset}_model.json"

    if [ "$MODE" = "full" ]; then
        test_data="$MODELS_DIR/hydra_${dataset}_test.json"
        suffix="full"
    else
        test_data="$MODELS_DIR/hydra_${dataset}_test_1000.json"
        suffix="1000"
    fi

    if [ ! -f "$model" ]; then
        echo "WARNING: Model not found: $model — skipping $dataset"
        continue
    fi
    if [ ! -f "$test_data" ]; then
        echo "WARNING: Test data not found: $test_data — skipping $dataset"
        continue
    fi

    logfile="$RESULTS_DIR/hydra_v2fixed_${dataset}_${suffix}.log"

    echo "============================================"
    echo "  Running HYDRA v2_fixed on $dataset ($suffix)"
    echo "  Model: $model"
    echo "  Test:  $test_data"
    echo "  Log:   $logfile"
    echo "============================================"

    "$HOST" "$XCLBIN" "$model" "$test_data" 2>&1 | tee "$logfile"

    echo ""
    echo "=== $dataset complete ==="
    echo ""
done

echo ""
echo "============================================"
echo "  All benchmarks complete!"
echo "  Results in: $RESULTS_DIR/"
echo "============================================"

# Parse results into CSV
echo ""
echo "=== Summary ==="
echo "Dataset,Samples,Accuracy(%),Throughput(inf/s),Mean_Latency(ms)" > "$RESULTS_DIR/hydra_v2fixed_summary.csv"

for dataset in "${DATASETS[@]}"; do
    logfile="$RESULTS_DIR/hydra_v2fixed_${dataset}_${MODE == "full" && echo "full" || echo "1000"}.log"
    if [ -f "$logfile" ]; then
        # Extract key metrics from log
        accuracy=$(grep -oP 'Accuracy:\s+\K[\d.]+' "$logfile" 2>/dev/null || echo "N/A")
        throughput=$(grep -oP 'Throughput:\s+\K[\d.]+' "$logfile" 2>/dev/null || echo "N/A")
        latency=$(grep -oP 'Average.*latency:\s+\K[\d.]+' "$logfile" 2>/dev/null || echo "N/A")
        samples=$(grep -oP 'Samples:\s+\K\d+' "$logfile" 2>/dev/null || echo "N/A")
        echo "$dataset,$samples,$accuracy,$throughput,$latency" >> "$RESULTS_DIR/hydra_v2fixed_summary.csv"
        echo "  $dataset: accuracy=$accuracy%, throughput=$throughput inf/s, latency=$latency ms"
    fi
done

echo ""
echo "CSV saved to: $RESULTS_DIR/hydra_v2fixed_summary.csv"
