#!/bin/bash
# Run power benchmarks for all HYDRA models
# Generates power measurements for each dataset

cd /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized

XCLBIN="build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/krnl.xclbin"
RESULTS_DIR="power_results"
mkdir -p $RESULTS_DIR

echo "========================================="
echo "HYDRA Power Benchmarking Suite"
echo "========================================="
echo ""

# Array of models and test files
declare -A models
models["InsectSound"]="models/hydra_insectsound_model.json models/hydra_insectsound_test_1000.json"
models["FruitFlies"]="models/hydra_fruitflies_model.json models/hydra_fruitflies_test_1000.json"
models["MosquitoSound"]="models/hydra_mosquitosound_model.json models/hydra_mosquitosound_test_1000.json"

# Run benchmarks for each model
for dataset in "${!models[@]}"; do
    echo "========================================="
    echo "Dataset: $dataset"
    echo "========================================="

    # Parse model and test files
    read -r model_file test_file <<< "${models[$dataset]}"

    # Check if files exist
    if [ ! -f "$model_file" ]; then
        echo "⚠️  Model file not found: $model_file"
        echo "Skipping $dataset..."
        echo ""
        continue
    fi

    if [ ! -f "$test_file" ]; then
        echo "⚠️  Test file not found: $test_file"
        echo "Skipping $dataset..."
        echo ""
        continue
    fi

    echo "Running inference with power profiling..."
    echo "  Model: $model_file"
    echo "  Test:  $test_file"

    # Run FPGA inference
    ./host/hydra_host $XCLBIN $model_file $test_file \
        2>&1 | tee "$RESULTS_DIR/${dataset}_fpga_output.log"

    # Check if power CSV was generated
    if [ -f "power_profile_xilinx_u280_gen3x16_xdma_base_1-0.csv" ]; then
        echo ""
        echo "✓ Power data collected"

        # Analyze power and save results
        python3 power.py > "$RESULTS_DIR/${dataset}_power_analysis.txt" 2>&1

        # Move CSV to results directory
        mv power_profile_xilinx_u280_gen3x16_xdma_base_1-0.csv \
           "$RESULTS_DIR/${dataset}_power_profile.csv"

        echo "✓ Results saved to $RESULTS_DIR/${dataset}_*"
    else
        echo "⚠️  Power profile CSV not generated"
    fi

    echo ""
    echo "========================================="
    echo ""
    sleep 2
done

echo ""
echo "========================================="
echo "Benchmark Complete!"
echo "========================================="
echo ""
echo "Results directory: $RESULTS_DIR/"
echo ""
echo "Summary of power measurements:"
echo "------------------------------"

for dataset in "${!models[@]}"; do
    if [ -f "$RESULTS_DIR/${dataset}_power_analysis.txt" ]; then
        echo ""
        echo "=== $dataset ==="
        grep -A 6 "TOTAL BOARD POWER" "$RESULTS_DIR/${dataset}_power_analysis.txt"
        grep "Throughput:" "$RESULTS_DIR/${dataset}_fpga_output.log" || \
            grep "inferences/sec" "$RESULTS_DIR/${dataset}_fpga_output.log" | tail -1
    fi
done

echo ""
echo "========================================="
echo "Full results available in: $RESULTS_DIR/"
echo "========================================="
