#!/bin/bash
# dpr_sweep_orchestrate.sh  —  RUNS ON WOLVERINE (the build/controller host).
#
# Network-saturation rate sweep against the ALREADY-DEPLOYED v3-dpr FPGA
# kernel on pc160 (BDF 0000:3b:00.1), sender = throughput.py ON pc161.
#
# WHY WOLVERINE-DRIVEN: pc160 cannot ssh to pc161 (Permission denied
# (publickey)) so the original "run on pc160, ssh to pc161" design silently
# produced all-zero rows (throughput.py never ran -> Host key verification
# failed). Wolverine authenticates to BOTH nodes, so it conducts:
#   * throughput.py           -> ssh pc161   (the actual UDP sender, .31 NIC)
#   * NetLayer counter reads   -> ssh pc160   (ground truth, source setup.sh)
#   * post-wedge recovery      -> ssh pc160 'bash ~/deploy_v3dpr_pc160.sh'
#                                 (fully pc160-local; no cross-node steps)
#
# Ground truth = FPGA NetLayer counter deltas (eth_in/app_in/udp_out), NOT
# sender ACKs (recv_acks reads 0 due to the slot-14 routing quirk; validated
# 2026-07-04: at 500pps app_in delta == sent == 1001, udp_out == 2*app_in).
#
# pc161's static ARP (192.168.40.10 -> 02:00:00:00:0a:0a) is PERMANENT and
# survives FPGA resets, so no ARP re-add is needed; a defensive replace is
# issued anyway (tolerated to fail).
#
# *** PERFORMS REPEATED sudo xbutil reset/program ON SHARED CLOUDLAB HARDWARE
# *** VIA pc160, PLUS A UDP FLOOD FROM pc161. Authorized in-session 2026-07-04
# *** (plan approval + allowedPrompts). CLAUDE.md Rule 2.

set -uo pipefail

# ---- nodes / ssh -----------------------------------------------------------
PC160=rdave009@pc160.cloudlab.umass.edu
PC161=rdave009@pc161.cloudlab.umass.edu
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

# ---- fixed rig parameters (verified live 2026-07-04) -----------------------
BDF=0000:3b:00.1
XCLBIN=/users/rdave009/krnl.xclbin.v3-dpr
FPGA_MAC=02:00:00:00:0a:0a
FPGA_IP=192.168.40.10
FPGA_PORT=62177
LISTEN_PORT=62178
PC161_IFACE=enp135s0f0

# ---- sweep parameters ------------------------------------------------------
RATES=(500 1000 2000 5000 10000 25000 50000 100000 250000 500000 1000000 2000000 5000000)
DURATION_S=3
WEDGE_RECOVERY_THRESHOLD=2000   # recover after any step >= this rate

VARIANT=v3-dpr
DATASET=len150-binary
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="$SCRIPT_DIR/runs/tp_2026-07-04"
CSV="$OUTDIR/dpr_sweep.csv"
LOGDIR="$OUTDIR/logs"

mkdir -p "$OUTDIR" "$LOGDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/orchestrate.log"; }

if [ ! -f "$CSV" ]; then
  echo "variant,dataset,configured_pps,sent,duration_s,eth_in_delta,app_in_delta,udp_out_delta,processed_pps,loss_pct,bandwidth_kbps" > "$CSV"
fi

# Prints "eth_in app_in udp_out" (0 if a name is missing). Reads on pc160;
# MUST source setup.sh in the remote shell or the tool dies on libxrt.
get_counters() {
  local raw eth app udp
  raw=$($SSH "$PC160" "source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1; ~/read_netlayer_counters_ext $XCLBIN $BDF 2>/dev/null")
  eth=$(echo "$raw" | awk '$1=="eth_in_packets"{print $2}')
  app=$(echo "$raw" | awk '$1=="app_in_packets"{print $2}')
  udp=$(echo "$raw" | awk '$1=="udp_out_packets"{print $2}')
  echo "${eth:-0} ${app:-0} ${udp:-0}"
}

# Full recovery = the pc160-local deploy script (reset->program->setup_netlayer
# ->batched tweak2 gateway+16 slots->loader). All local to pc160.
recover() {
  log "RECOVERY: ssh pc160 'bash ~/deploy_v3dpr_pc160.sh'"
  $SSH "$PC160" 'bash ~/deploy_v3dpr_pc160.sh' >>"$LOGDIR/recovery.log" 2>&1
  local rc=$?
  # Defensive ARP replace on pc161 (permanent already; tolerate failure).
  $SSH "$PC161" "sudo ip neigh replace $FPGA_IP lladdr $FPGA_MAC dev $PC161_IFACE nud permanent" >>"$LOGDIR/recovery.log" 2>&1 || true
  if [ $rc -ne 0 ]; then
    log "  WARNING: recovery deploy exited $rc (see $LOGDIR/recovery.log)"
  else
    log "  recovery ok"
  fi
}

run_rate_step() {
  local rate=$1
  log "=== rate step: ${rate} pps ==="

  read -r eth_b app_b udp_b <<< "$(get_counters)"
  log "  counters BEFORE: eth_in=$eth_b app_in=$app_b udp_out=$udp_b"

  $SSH "$PC161" "python3 ~/throughput.py --fpga-ip $FPGA_IP --fpga-port $FPGA_PORT --listen-port $LISTEN_PORT --rate-min $rate --rate-max $rate --rate-steps 1 --duration-s $DURATION_S --abort-loss-pct 200 --out /tmp/tp_${rate}" > "$LOGDIR/tp_${rate}.log" 2>&1
  local tp_rc=$?
  [ $tp_rc -ne 0 ] && log "  WARNING: throughput.py exited $tp_rc (see $LOGDIR/tp_${rate}.log)"

  local sent
  sent=$($SSH "$PC161" "tail -n1 /tmp/tp_${rate}/sweep.csv 2>/dev/null | cut -d, -f2")
  [[ "$sent" =~ ^[0-9]+$ ]] || sent=0

  read -r eth_a app_a udp_a <<< "$(get_counters)"
  log "  counters AFTER:  eth_in=$eth_a app_in=$app_a udp_out=$udp_a"

  local eth_d=$((eth_a - eth_b)) app_d=$((app_a - app_b)) udp_d=$((udp_a - udp_b))
  local pps loss bw
  pps=$(awk -v a="$app_d" -v d="$DURATION_S" 'BEGIN{printf "%.3f", a/d}')
  loss=$(awk -v a="$app_d" -v s="$sent" 'BEGIN{if(s<1)s=1; printf "%.3f", 100*(1-a/s)}')
  bw=$(awk -v p="$pps" 'BEGIN{printf "%.3f", p*64*8/1000}')

  echo "${VARIANT},${DATASET},${rate},${sent},${DURATION_S},${eth_d},${app_d},${udp_d},${pps},${loss},${bw}" >> "$CSV"
  log "  row: sent=$sent eth_d=$eth_d app_d=$app_d udp_d=$udp_d pps=$pps loss=$loss% bw=${bw}kbps"

  if [ "$rate" -ge "$WEDGE_RECOVERY_THRESHOLD" ]; then
    recover
  else
    log "  (rate < ${WEDGE_RECOVERY_THRESHOLD}, no recovery)"
  fi
}

log "=== dpr_sweep_orchestrate.sh (Wolverine-driven) starting; rates: ${RATES[*]} ==="
for r in "${RATES[@]}"; do
  run_rate_step "$r"
done
log "=== sweep complete. CSV: $CSV ==="
cat "$CSV" | tee -a "$LOGDIR/orchestrate.log"
