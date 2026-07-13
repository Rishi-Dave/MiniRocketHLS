#!/usr/bin/env bash
# TP sweep that also captures FPGA-side udp_in counter deltas via SSH.
# Usage: bash run_sweep.sh <out_dir> [fpga_ssh_alias]
#
# Env overrides:
#   XCLBIN    path to xclbin on the FPGA host (default: HYDRA v2_fixed build_dir.hw)
#   BDF       PCIe BDF (default: 0000:3b:00.1 -- pc158)
#   FPGA_IP/FPGA_PORT/LISTEN_PORT -- forwarded to throughput.py
set -euo pipefail

OUT="${1:?usage: $0 <out_dir> [fpga_host]}"
FPGA_HOST="${2:-localhost}"

# Default xclbin = the v5 hydra-axis build used in the May 2026 saturation runs.
XCLBIN="${XCLBIN:-/users/rdave009/MiniRocketHLS/hydra_optimized/hydra_axis/build_dir.hw/krnl.xclbin}"
BDF="${BDF:-0000:3b:00.1}"
READ_CNT_BIN="${READ_CNT_BIN:-/users/rdave009/MiniRocketHLS/saturation_harness/read_netlayer_counters}"

mkdir -p "$OUT"

read_cnt() {
    if [[ "$FPGA_HOST" == "localhost" ]]; then
        "$READ_CNT_BIN" "$XCLBIN" "$BDF" 2>/dev/null \
            | awk '/udp_in_packets/ {print $NF; exit}'
    else
        ssh "$FPGA_HOST" \
            "source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1; \
             $READ_CNT_BIN $XCLBIN $BDF" 2>/dev/null \
            | awk '/udp_in_packets/ {print $NF; exit}'
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BEFORE="$(read_cnt || echo 0)"
echo "fpga_udp_in_before=$BEFORE" | tee "$OUT/fpga_counters.txt"

# Drop the first arg (out_dir) so the rest get forwarded to throughput.py.
shift || true
shift || true
python3 "$SCRIPT_DIR/throughput.py" --out "$OUT" "$@"

AFTER="$(read_cnt || echo 0)"
echo "fpga_udp_in_after=$AFTER"  | tee -a "$OUT/fpga_counters.txt"
echo "fpga_udp_in_delta=$((AFTER - BEFORE))" | tee -a "$OUT/fpga_counters.txt"
