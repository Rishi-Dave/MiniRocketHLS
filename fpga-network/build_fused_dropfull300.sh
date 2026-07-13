#!/bin/bash
# build_fused_dropfull300.sh — link the v5 fused kernel (facc_t classifier,
# FEATURE_LOOP II=1) at 300 MHz. Link-only: assumes the v5 .xo is already
# installed at build/iprepo/minirocket_fused_inference.xo (csim 5/5 +
# csynth-all-II=1 gated first). Identical recipe to build_fused_dropfull.sh
# except KFREQ (default 300; KFREQ=250 for the timing-fallback rebuild),
# temp_dir and output dir.
set -uo pipefail
export PATH=/home/Xilinx/Vitis_HLS/2023.2/bin:/home/Xilinx/Vitis/2023.2/bin:$PATH
NET=/home/rdave009/minirocket-hls/MiniRocketHLS/fpga-network
KFREQ=${KFREQ:-300}
TAG=${TAG:-fused_dropfull${KFREQ}}
LOG=$NET/build_${TAG}.log
exec > >(tee -a "$LOG") 2>&1
echo "############ START $(date) @ ${KFREQ}MHz ############"
cd "$NET" || exit 9
v++ \
  --vivado.prop "run.__KERNEL__.{STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS}={-directive sdx_optimization_effort_high}" \
  --advanced.misc "report=type report_timing_summary name impl_report_timing_summary_route_design_summary steps {route_design} runs {impl_1} options {-max_paths 10}" \
  --advanced.misc "report=type report_timing_summary name impl_report_timing_summary_post_route_phys_opt_design_summary steps {post_route_phys_opt_design} runs {impl_1} options {-max_paths 10}" \
  -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 --save-temps -I./src --config minirocket_fused/config.cfg -l \
  --kernel_frequency $KFREQ \
  --user_ip_repo_paths $NET/NetLayers/100G-fpga-network-stack-core/synthesis_results_HBM \
  --user_ip_repo_paths ./iprepo \
  --temp_dir _x.$TAG \
  -o./build_dir.hw.$TAG/krnl.xclbin \
  build/iprepo/minirocket_fused_inference.xo \
  NetLayers/_x.xilinx_u280_gen3x16_xdma_1_202211_1/networklayer.xo \
  Ethernet/_x.xilinx_u280_gen3x16_xdma_1_202211_1/cmac_0.xo \
  build/iprepo/pktDropper.xo
RC=$?
if [ $RC -eq 0 ] && [ -f "$NET/build_dir.hw.$TAG/krnl.xclbin" ]; then
  echo "############ BUILD OK $(date) :: $(ls -la $NET/build_dir.hw.$TAG/krnl.xclbin) ############"
else
  echo "############ LINK FAILED rc=$RC $(date) ############"
fi
