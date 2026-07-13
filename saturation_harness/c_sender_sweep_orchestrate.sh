#!/bin/bash
# c_sender_sweep_orchestrate.sh — RUNS ON WOLVERINE.
# Full 3-dataset sweep of the fused v4 FPGA driven by the C AF_PACKET sender
# (hi_rate_sender on pc161), to extend the FPGA curve past Python's ~312k
# ceiling toward the C sender's ~703k limit. Ground truth = pc160 app_in delta.
# configured_pps recorded = ACHIEVED send rate (sent/duration) so each point
# sits at its true offered rate. CRITICAL: --src-port 62178 (socket-table match).
set -uo pipefail
PC160=rdave009@pc160.cloudlab.umass.edu
PC161=rdave009@pc161.cloudlab.umass.edu
SSH="ssh -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new"
BDF=0000:3b:00.1; XCLBIN=/users/rdave009/krnl.xclbin.fused
DATASETS=(
  "InsectSound:/users/rdave009/InsectSound_minirocket_model.json"
  "FruitFlies:/users/rdave009/FruitFlies_minirocket_model.json"
  "MosquitoSound:/users/rdave009/MosquitoSound_minirocket_model.json"
)
# paced targets; --rate 1000000 achieves ~703k (C sender per-sendto ceiling)
RATES=(5000 10000 25000 50000 100000 250000 500000 1000000)
DUR=5
OUTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runs/tp_2026-07-06_csender"
CSV="$OUTDIR/fused_csender_sweep.csv"; LOGDIR="$OUTDIR/logs"
mkdir -p "$OUTDIR" "$LOGDIR"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGDIR/orchestrate.log"; }
[ -f "$CSV" ] || echo "variant,dataset,configured_pps,sent,duration_s,eth_in_delta,app_in_delta,udp_out_delta,processed_pps,loss_pct,bandwidth_kbps" > "$CSV"
get_app(){ $SSH "$PC160" "source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1; ~/read_netlayer_counters_ext $XCLBIN $BDF 2>/dev/null" | awk '$1=="eth_in_packets"{e=$2} $1=="app_in_packets"{a=$2} $1=="udp_out_packets"{u=$2} END{print e" "a" "u}'; }
for ds in "${DATASETS[@]}"; do
  NAME="${ds%%:*}"; MODEL="${ds#*:}"
  log "=== $NAME ==="
  $SSH "$PC160" "bash ~/deploy_fused_pc160.sh $MODEL" >>"$LOGDIR/deploy.log" 2>&1; log "  deployed"
  for rate in "${RATES[@]}"; do
    read -r eb ab ub <<< "$(get_app)"
    OUT=$($SSH "$PC161" "sudo ~/hi_rate_sender --iface enp135s0f0 --src-port 62178 --rate $rate --duration $DUR 2>&1 | tail -1")
    sent=$(echo "$OUT" | grep -oP 'sent=\K[0-9]+'); [[ "$sent" =~ ^[0-9]+$ ]] || sent=0
    achieved=$(echo "$OUT" | grep -oP 'achieved=\K[0-9]+'); [[ "$achieved" =~ ^[0-9]+$ ]] || achieved=$rate
    read -r ea aa ua <<< "$(get_app)"
    ethd=$((ea-eb)); appd=$((aa-ab)); udpd=$((ua-ub))
    pps=$(awk -v a="$appd" -v d="$DUR" 'BEGIN{printf "%.3f",a/d}')
    loss=$(awk -v a="$appd" -v s="$sent" 'BEGIN{if(s<1)s=1;printf "%.3f",100*(1-a/s)}')
    bw=$(awk -v p="$pps" 'BEGIN{printf "%.3f",p*64*8/1000}')
    echo "fused,${NAME},${achieved},${sent},${DUR},${ethd},${appd},${udpd},${pps},${loss},${bw}" >> "$CSV"
    log "  $NAME target=$rate achieved=$achieved sent=$sent app_d=$appd pps=$pps loss=$loss% bw=${bw}kbps"
  done
done
log "=== C-SENDER SWEEP COMPLETE: $CSV ==="; cat "$CSV" | tee -a "$LOGDIR/orchestrate.log"
