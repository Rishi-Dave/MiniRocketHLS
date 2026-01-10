#!/bin/bash
# Run full HYDRA FPGA test on all 25,000 samples
# Est time: ~130 seconds (2 minutes)

cd /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized

echo "========================================="
echo "HYDRA FPGA Full Test - InsectSound"
echo "========================================="
echo "Test samples: 25,000"
echo "Expected time: ~2 minutes"
echo "Output: /tmp/hydra_fpga_full_test.log"
echo "========================================="
echo ""

./host/hydra_host \
    build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/krnl.xclbin \
    models/hydra_insectsound_model.json \
    models/hydra_insectsound_test.json \
    2>&1 | tee /tmp/hydra_fpga_full_test.log

echo ""
echo "========================================="
echo "Test complete! Results saved to:"
echo "/tmp/hydra_fpga_full_test.log"
echo "========================================="
