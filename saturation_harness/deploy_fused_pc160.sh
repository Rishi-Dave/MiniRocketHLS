#!/bin/bash
# Deploy the fused MiniRocket network kernel (minirocket_fused_inference, 250 MHz,
# BUILD=1 free-run) on pc160. Parameterized by model JSON ($1) so the same script
# serves FruitFlies / MosquitoSound / InsectSound. Fully pc160-local (no ssh out).
# Loader LAST. Run via: ssh pc160 'bash ~/deploy_fused_pc160.sh <model.json>'
set -e
BDF=0000:3b:00.1
XCLBIN=/users/rdave009/krnl.xclbin.fused
MODEL=${1:-/users/rdave009/InsectSound_minirocket_model.json}
LOADER=/users/rdave009/load_minirocket_hbm_fused   # defaults to fused CU name
NET=/users/rdave009
FPGA_MAC=02:00:00:00:0a:0a
PC161_MAC=f8:f2:1e:d3:26:b0
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

echo "[model] $MODEL"
echo "[1] kill loader"; tmux kill-session -t loader 2>/dev/null || true; sleep 1
echo "[2] reset user partition"; sudo /opt/xilinx/xrt/bin/xbutil reset --device $BDF --type user --force | tail -2; sleep 2
echo "[3] program fused xclbin"; sudo /opt/xilinx/xrt/bin/xbutil program --device $BDF --user $XCLBIN | tail -2
echo "[4] setup_netlayer -> pc161 (.31)"; $NET/setup_netlayer $XCLBIN $BDF $FPGA_MAC 192.168.40.10 62177 192.168.40.31 $PC161_MAC 62178 | tail -3
echo "[5+6] gateway(.31) + fill 16 slots (batched tweak2)"
ARGS="1C C0A8281F"
for s in $(seq 0 15); do
  ARGS="$ARGS $(printf '%X' $((0x820 + s*8))) C0A8281F"
  ARGS="$ARGS $(printf '%X' $((0x8A0 + s*8))) F2E2"
  ARGS="$ARGS $(printf '%X' $((0x920 + s*8))) F2E1"
  ARGS="$ARGS $(printf '%X' $((0x9A0 + s*8))) 1"
done
$NET/tweak2 $XCLBIN $BDF $ARGS | tail -2
echo "[7] start loader LAST (model -> HBM; BUILD=1 free-run: one start, wait blocks)"
tmux new-session -d -s loader "source /opt/xilinx/xrt/setup.sh; cd $NET && $LOADER $XCLBIN $BDF $MODEL 2>&1 | tee /tmp/loader_fused.log"
sleep 12
echo "[8] loader log"; tail -15 /tmp/loader_fused.log 2>/dev/null
echo "[9] baseline counters"; $NET/read_netlayer_counters_ext $XCLBIN $BDF 2>/dev/null | grep -E "eth_in_packets|udp_in_packets|app_in_packets|udp_out_packets" || true
echo "[done] fused on pc160, SocketTable->pc161"
