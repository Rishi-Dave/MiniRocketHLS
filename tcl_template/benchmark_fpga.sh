#!/bin/bash
# Simple FPGA benchmark script using the original working host
# Runs multiple iterations and measures timing

XCLBIN="${1:-build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/krnl.xclbin}"
MODEL="${2:-minirocket_model.json}"
NUM_ITERATIONS="${3:-100}"

echo "=============================================="
echo "MiniRocket FPGA Benchmark"
echo "=============================================="
echo "XCLBIN: $XCLBIN"
echo "Model: $MODEL"
echo "Iterations: $NUM_ITERATIONS"
echo "=============================================="

if [ ! -f "$XCLBIN" ]; then
    echo "ERROR: XCLBIN file not found: $XCLBIN"
    exit 1
fi

if [ ! -f "host" ]; then
    echo "Building host application..."
    make host
fi

# Warmup run
echo ""
echo "Warmup run..."
./host "$XCLBIN" "$MODEL" > /dev/null 2>&1

# Timed runs
echo ""
echo "Running $NUM_ITERATIONS timed iterations..."

# Use bash's time builtin with a loop
start_time=$(date +%s.%N)

for ((i=1; i<=NUM_ITERATIONS; i++)); do
    ./host "$XCLBIN" "$MODEL" > /dev/null 2>&1
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Completed $i iterations..."
    fi
done

end_time=$(date +%s.%N)

# Calculate timing
total_time=$(echo "$end_time - $start_time" | bc)
avg_time=$(echo "scale=4; $total_time / $NUM_ITERATIONS * 1000" | bc)
throughput=$(echo "scale=2; $NUM_ITERATIONS / $total_time" | bc)

echo ""
echo "=============================================="
echo "FPGA TIMING RESULTS"
echo "=============================================="
echo "Iterations:        $NUM_ITERATIONS"
echo "Total time:        ${total_time} seconds"
echo "Avg per inference: ${avg_time} ms"
echo "Throughput:        ${throughput} inferences/sec"
echo "=============================================="

# Save results to CSV
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p benchmark_results
echo "iterations,total_sec,avg_ms,throughput" > "benchmark_results/fpga_simple_${TIMESTAMP}.csv"
echo "$NUM_ITERATIONS,$total_time,$avg_time,$throughput" >> "benchmark_results/fpga_simple_${TIMESTAMP}.csv"
echo "Results saved to benchmark_results/fpga_simple_${TIMESTAMP}.csv"
