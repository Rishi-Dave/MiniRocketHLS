#!/usr/bin/env bash
# Build a network-wired xclbin for multirocket_axis_inference.
#
# Pipeline:
#   1. vitis_hls   ->  multirocket_axis_inference.xo
#   2. v++ --link  with cmac_0.xo + networklayer.xo (from fpga-network)
#                  and config_axis.cfg
#
# Usage:
#   ./build_axis_xclbin.sh <hw|hw_emu> [PLATFORM]
# Defaults: PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
#
# Output: build_dir.<TARGET>/hydra_axis_krnl.xclbin (and multirocket_axis_inference.xo)
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

XO_LOCAL="$THIS_DIR/multirocket_axis_inference.xo"
XO_CMAC="$FPGA_NET_DIR/_x.${PLATFORM}/cmac_0.xo"
XO_NETLAYER="$FPGA_NET_DIR/_x.${PLATFORM}/networklayer.xo"

echo "[build] target=$TARGET platform=$PLATFORM"
echo "[build] this dir   : $THIS_DIR"
echo "[build] fpga-net   : $FPGA_NET_DIR"
echo "[build] build dir  : $BUILD_DIR"

# Sanity: prebuilt CMAC + NetLayer XOs must exist in fpga-network.
for xo in "$XO_CMAC" "$XO_NETLAYER"; do
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

# Step 3: v++ link.
echo "[build] step 2: v++ --link"
v++ -l \
    -t "$TARGET" \
    --platform "$PLATFORM" \
    --config config_axis.resolved.cfg \
    -o krnl.xclbin \
    "$XO_LOCAL" "$XO_CMAC" "$XO_NETLAYER" \
    --save-temps \
    --kernel_frequency 300 \
    2>&1 | tee v++.log

echo "[build] DONE -> $BUILD_DIR/krnl.xclbin"
