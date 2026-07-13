#!/usr/bin/env bash
# Build a network-wired xclbin for hydra_axis_inference.
#
# Pipeline:
#   1. vitis_hls   ->  hydra_axis_inference.xo
#   2. v++ --link  with cmac_0.xo + networklayer.xo (from fpga-network)
#                  and config_axis.cfg
#
# Usage:
#   ./build_axis_xclbin.sh <hw|hw_emu> [PLATFORM]
# Defaults: PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
#
# Output: build_dir.<TARGET>/hydra_axis_krnl.xclbin (and hydra_axis_inference.xo)
#
# Designed to be the ONE command run in tmux on OCT. Per CLAUDE.md Rule #7,
# launch a build-monitor subagent immediately after starting this in tmux.

set -euo pipefail

TARGET="${1:?usage: $0 <hw|hw_emu> [PLATFORM]}"
PLATFORM="${2:-xilinx_u280_gen3x16_xdma_1_202211_1}"

case "$TARGET" in
    hw|hw_emu) ;;
    *) echo "TARGET must be hw or hw_emu" >&2; exit 2 ;;
esac

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
FPGA_NET_DIR="$(cd "$THIS_DIR/../../fpga-network" && pwd)"
BUILD_DIR="$THIS_DIR/build_dir.${TARGET}"

XO_LOCAL="$THIS_DIR/hydra_axis_inference.xo"
XO_CMAC="$FPGA_NET_DIR/Ethernet/_x.${PLATFORM}/cmac_0.xo"
XO_NETLAYER="$FPGA_NET_DIR/NetLayers/_x.${PLATFORM}/networklayer.xo"
XO_PKTDROP="$FPGA_NET_DIR/build/iprepo/pktDropper.xo"

echo "[build] target=$TARGET platform=$PLATFORM"
echo "[build] this dir   : $THIS_DIR"
echo "[build] fpga-net   : $FPGA_NET_DIR"
echo "[build] build dir  : $BUILD_DIR"

# Sanity: prebuilt CMAC + NetLayer + pktDropper XOs must exist in fpga-network.
for xo in "$XO_CMAC" "$XO_NETLAYER" "$XO_PKTDROP"; do
    if [ ! -f "$xo" ]; then
        echo "[build] MISSING prebuilt XO: $xo" >&2
        echo "[build] Run the fpga-network Makefile target that builds CMAC + NetLayer first." >&2
        exit 3
    fi
done

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Step 1: vitis_hls export to .xo (re-runs csynth_design + export_design).
if [ ! -f "$XO_LOCAL" ]; then
    echo "[build] step 1: vitis_hls ip export ($XO_LOCAL)"
    ( cd "$THIS_DIR" && make ip )
fi
[ -f "$XO_LOCAL" ] || { echo "[build] vitis_hls did not produce $XO_LOCAL" >&2; exit 4; }

# Step 2: substitute __FPGA_NETWORK_DIR__ in config_axis.cfg so post_sys_link.tcl resolves.
sed "s|__FPGA_NETWORK_DIR__|$FPGA_NET_DIR|g" "$THIS_DIR/config_axis.cfg" > config_axis.resolved.cfg

# Step 3: v++ link. Mirrors the PROVEN fused_dropfull recipe
# (fpga-network/build_dropfull.log): sdx_optimization_effort_high synth
# directive, route/physopt timing reports, NetLayer HBM user IP repo.
# Frequency 300 (hydra v5 axis closed timing at 300 with this same
# CMAC+NetLayer chain); fall back to 250 if timing fails.
KFREQ="${KFREQ:-300}"
echo "[build] step 2: v++ --link @ ${KFREQ}MHz"
v++ -l \
    --vivado.prop "run.__KERNEL__.{STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS}={-directive sdx_optimization_effort_high}" \
    --advanced.misc "report=type report_timing_summary name impl_report_timing_summary_route_design_summary steps {route_design} runs {impl_1} options {-max_paths 10}" \
    --advanced.misc "report=type report_timing_summary name impl_report_timing_summary_post_route_phys_opt_design_summary steps {post_route_phys_opt_design} runs {impl_1} options {-max_paths 10}" \
    -t "$TARGET" \
    --platform "$PLATFORM" \
    --config config_axis.resolved.cfg \
    -o krnl.xclbin \
    "$XO_LOCAL" "$XO_CMAC" "$XO_NETLAYER" "$XO_PKTDROP" \
    --save-temps \
    --kernel_frequency "$KFREQ" \
    --user_ip_repo_paths "$FPGA_NET_DIR/NetLayers/100G-fpga-network-stack-core/synthesis_results_HBM" \
    --user_ip_repo_paths "$FPGA_NET_DIR/build/iprepo" \
    2>&1 | tee v++.log

echo "[build] DONE -> $BUILD_DIR/krnl.xclbin"
