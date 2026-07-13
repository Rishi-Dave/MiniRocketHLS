#!/bin/bash
# Build feature_extraction_v16 for hw target
set -e
cd /home/rdave009/minirocket-hls/MiniRocketHLS/minirocket_modular
source /opt/xilinx/xrt/setup.sh
source /tools/Xilinx/Vitis/2023.2/settings64.sh

mkdir -p build/iprepo _x_v16_hw/feature_extraction

echo "=== Starting feature_extraction_v16 hw compile at $(date) ==="
echo "Working dir: $(pwd)"

v++ -t hw \
    --platform xilinx_u280_gen3x16_xdma_1_202211_1 \
    --save-temps \
    -I./include \
    -I./feature_extraction/src \
    -c -k feature_extraction \
    --temp_dir _x_v16_hw/feature_extraction \
    -o build/iprepo/feature_extraction_v16_hw.xo \
    feature_extraction/src/feature_extraction_v16.cpp \
    2>&1 | tee _x_v16_hw/feature_extraction_v16_hw_build.log

echo "=== Build finished at $(date) with EXIT_CODE=$? ==="
