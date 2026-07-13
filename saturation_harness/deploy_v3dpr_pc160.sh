#!/bin/bash
# Deploy v3-dpr (MR DP-Reuse+bFP) on pc160 FPGA; SocketTable -> pc161 sender (.31).
# Loader LAST (after netlayer+slots) to avoid device-context contention.
set -e
BDF=0000:3b:00.1
XCLBIN=/users/rdave009/krnl.xclbin.v3-dpr
MODEL=/users/rdave009/dpr_minirocket_model_len150_fixed.json
NET=/users/rdave009
FPGA_MAC=02:00:00:00:0a:0a
PC161_MAC=f8:f2:1e:d3:26:b0
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

echo "[1] kill loader"; tmux kill-session -t loader 2>/dev/null || true; sleep 1
echo "[2] reset user partition"; sudo /opt/xilinx/xrt/bin/xbutil reset --device $BDF --type user --force | tail -2; sleep 2
echo "[3] program v3-dpr"; sudo /opt/xilinx/xrt/bin/xbutil program --device $BDF --user $XCLBIN | tail -2
echo "[4] setup_netlayer -> pc161 (.31, $PC161_MAC)"; $NET/setup_netlayer $XCLBIN $BDF $FPGA_MAC 192.168.40.10 62177 192.168.40.31 $PC161_MAC 62178 | tail -3
echo "[5+6] gateway -> .31 + fill 16 slots (single batched tweak2, opens device once)"
ARGS="1C C0A8281F"
for s in $(seq 0 15); do
  ARGS="$ARGS $(printf '%X' $((0x820 + s*8))) C0A8281F"   # theirIP=.31
  ARGS="$ARGS $(printf '%X' $((0x8A0 + s*8))) F2E2"        # theirPort=62178
  ARGS="$ARGS $(printf '%X' $((0x920 + s*8))) F2E1"        # myPort=62177
  ARGS="$ARGS $(printf '%X' $((0x9A0 + s*8))) 1"           # valid=1
done
$NET/tweak2 $XCLBIN $BDF $ARGS | tail -2
echo "[7] start loader LAST (len-150 model -> HBM, per-packet spin)"
tmux new-session -d -s loader "source /opt/xilinx/xrt/setup.sh; cd $NET && ./load_minirocket_hbm $XCLBIN $BDF $MODEL 2>&1 | tee /tmp/loader_dpr.log"
sleep 10
echo "[8] verify"
echo "--- loader log ---"; tail -8 /tmp/loader_dpr.log 2>/dev/null
echo "--- counters (baseline) ---"
$NET/read_netlayer_counters_ext $XCLBIN $BDF 2>/dev/null | grep -E "eth_in_packets|udp_in_packets|app_in_packets|udp_out_packets" || true
echo "[done] v3-dpr on pc160, SocketTable->pc161. Thermal:"
/opt/xilinx/xrt/bin/xbutil examine --report thermal --device $BDF 2>/dev/null | grep -iE "FPGA" | head -2
