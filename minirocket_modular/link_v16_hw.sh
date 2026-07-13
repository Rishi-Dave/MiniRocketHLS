#!/bin/bash
# Link feature_extraction_v16_hw.xo + scaler.xo + classifier.xo for hw target
set -e
cd /home/rdave009/minirocket-hls/MiniRocketHLS/minirocket_modular

PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
BUILD_DIR=build_dir_v16.hw.${PLATFORM}
TEMP_DIR=_x_v16_hw_link

mkdir -p ${BUILD_DIR} ${TEMP_DIR}

echo "=== Starting hw link at $(date) ==="
echo "Working dir: $(pwd)"
echo "FE .xo: build/iprepo/feature_extraction_v16_hw.xo"
echo "Scaler .xo: build/iprepo/scaler.xo"
echo "Classifier .xo: build/iprepo/classifier.xo"
echo "Config: config.cfg"
echo "Output: ${BUILD_DIR}/krnl.link.xclbin"

v++ -t hw \
    --platform ${PLATFORM} \
    --save-temps \
    -I./include \
    --config config.cfg \
    -l \
    --temp_dir ${TEMP_DIR} \
    -o ${BUILD_DIR}/krnl.link.xclbin \
    build/iprepo/feature_extraction_v16_hw.xo \
    build/iprepo/scaler.xo \
    build/iprepo/classifier.xo \
    2>&1 | tee ${TEMP_DIR}/link_v16_hw.log

echo ""
echo "=== Link finished at $(date) with EXIT_CODE=$? ==="

# Package the xclbin
echo "=== Packaging xclbin ==="
v++ -p ${BUILD_DIR}/krnl.link.xclbin \
    -t hw \
    --platform ${PLATFORM} \
    --package.out_dir ${BUILD_DIR}/package \
    -o ${BUILD_DIR}/krnl.xclbin \
    2>&1 | tee -a ${TEMP_DIR}/link_v16_hw.log

echo ""
echo "=== Package finished at $(date) with EXIT_CODE=$? ==="
ls -lh ${BUILD_DIR}/krnl.xclbin 2>/dev/null || echo "WARNING: krnl.xclbin not found"
