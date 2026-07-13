#!/bin/bash
# DP-Reuse + bFP hw link, retry 2 (A+ fix: len=192 conv map + 250 MHz kernel clock)
cd /home/rdave009/minirocket-hls/MiniRocketHLS/fpga-network
echo "START_DPR_LINK2 $(date)"
make build TARGET=hw \
  PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1 \
  MINIROCKET_KRNL=minirocket_inference_dpr \
  BUILD_DIR=./build_dir.hw.dpr \
  VPP_LDFLAGS="--kernel_frequency 250" 2>&1 | tee build_dpr2.log
echo "DPR_LINK_DONE exit=${PIPESTATUS[0]} $(date)" | tee -a build_dpr2.log
