#!/bin/bash
# MiniRocket FPGA vs CPU Benchmark Suite
# Usage: ./run_benchmarks.sh [--fpga-only] [--cpu-only] [--samples N]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default parameters
NUM_SAMPLES=100
RUN_FPGA=true
RUN_CPU=true
XCLBIN="build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/krnl.xclbin"
MODEL="minirocket_model.json"
RESULTS_DIR="benchmark_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fpga-only) RUN_CPU=false; shift ;;
        --cpu-only) RUN_FPGA=false; shift ;;
        --samples) NUM_SAMPLES="$2"; shift 2 ;;
        --xclbin) XCLBIN="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Create results directory
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "MiniRocket FPGA vs CPU Benchmark Suite"
echo "=============================================="
echo "Samples: $NUM_SAMPLES"
echo "Model: $MODEL"
echo "Results dir: $RESULTS_DIR"
echo "=============================================="

# Check if model file exists
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model file not found: $MODEL"
    exit 1
fi

# Build benchmark host if needed
if [ "$RUN_FPGA" = true ]; then
    if [ ! -f "benchmark_host" ]; then
        echo ""
        echo "Building benchmark host application..."
        g++ -o benchmark_host benchmark_host.cpp \
            -I/opt/xilinx/xrt/include \
            -I/home/Xilinx/Vivado/2023.2/include \
            -Wall -O2 -std=c++1y \
            -L/opt/xilinx/xrt/lib -lOpenCL -pthread -lrt -lstdc++
        echo "Benchmark host built successfully."
    fi
fi

# Run CPU benchmark
if [ "$RUN_CPU" = true ]; then
    echo ""
    echo "=============================================="
    echo "Running CPU Benchmark..."
    echo "=============================================="

    python3 benchmark_cpu.py "$MODEL" --synthetic "$NUM_SAMPLES" --warmup 5 \
        | tee "$RESULTS_DIR/cpu_results_${TIMESTAMP}.txt"

    # Also save CSV format
    python3 benchmark_cpu.py "$MODEL" --synthetic "$NUM_SAMPLES" --warmup 5 --csv \
        > "$RESULTS_DIR/cpu_results_${TIMESTAMP}.csv"

    echo "CPU results saved to $RESULTS_DIR/cpu_results_${TIMESTAMP}.txt"
fi

# Run FPGA benchmark
if [ "$RUN_FPGA" = true ]; then
    echo ""
    echo "=============================================="
    echo "Running FPGA Benchmark..."
    echo "=============================================="

    if [ ! -f "$XCLBIN" ]; then
        echo "WARNING: XCLBIN file not found: $XCLBIN"
        echo "Skipping FPGA benchmark. Build with: make build TARGET=hw"
    else
        ./benchmark_host "$XCLBIN" "$MODEL" --synthetic "$NUM_SAMPLES" --warmup 5 \
            | tee "$RESULTS_DIR/fpga_results_${TIMESTAMP}.txt"

        # Also save CSV format
        ./benchmark_host "$XCLBIN" "$MODEL" --synthetic "$NUM_SAMPLES" --warmup 5 --csv \
            > "$RESULTS_DIR/fpga_results_${TIMESTAMP}.csv"

        echo "FPGA results saved to $RESULTS_DIR/fpga_results_${TIMESTAMP}.txt"
    fi
fi

# Generate comparison summary
echo ""
echo "=============================================="
echo "Benchmark Complete"
echo "=============================================="
echo "Results saved in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"*"${TIMESTAMP}"* 2>/dev/null || echo "No results files generated."

echo ""
echo "To generate comparison plots, run:"
echo "  python3 generate_plots.py $RESULTS_DIR"
