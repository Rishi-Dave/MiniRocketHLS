#!/bin/bash
# fused_v4_sweep_orchestrate.sh — RUNS ON WOLVERINE.
# Full 3-dataset sweep of the v4 free-run fused kernel (0% loss to Python
# sender ceiling ~250-311k, no wedge). NO per-rate recovery (v4 is stable;
# recovery would needlessly reset the working kernel). One deploy per dataset.
set -uo pipefail
PC160=rdave009@pc160.cloudlab.umass.edu
PC161=rdave009@pc161.cloudlab.umass.edu
SSH="ssh -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new"
BDF=0000:3b:00.1; XCLBIN=/users/rdave009/krnl.xclbin.fused
FPGA_IP=192.168.40.10; FPGA_PORT=62177; LISTEN_PORT=62178
DATASETS=(
  "InsectSound:/users/rdave009/InsectSound_minirocket_model.json:600"
  "FruitFlies:/users/rdave009/FruitFlies_minirocket_model.json:5000"
  "MosquitoSound:/users/rdave009/MosquitoSound_minirocket_model.json:3750"
)
RATES=(500 1000 2000 5000 10000 25000 50000 100000 250000 500000)
DURATION_S=3
OUTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runs/tp_2026-07-06_fused_v4"
CSV="$OUTDIR/fused_v4_sweep.csv"; LOGDIR="$OUTDIR/logs"
mkdir -p "$OUTDIR" "$LOGDIR"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGDIR/orchestrate.log"; }
[ -f "$CSV" ] || echo "variant,dataset,configured_pps,sent,duration_s,eth_in_delta,app_in_delta,udp_out_delta,processed_pps,loss_pct,bandwidth_kbps" > "$CSV"
get_counters(){
  local raw eth app udp
  raw=$($SSH "$PC160" "source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1; ~/read_netlayer_counters_ext $XCLBIN $BDF 2>/dev/null")
  eth=$(echo "$raw"|awk '$1=="eth_in_packets"{print $2}')
  app=$(echo "$raw"|awk '$1=="app_in_packets"{print $2}')
  udp=$(echo "$raw"|awk '$1=="udp_out_packets"{print $2}')
  echo "${eth:-0} ${app:-0} ${udp:-0}"
}
for ds in "${DATASETS[@]}"; do
  NAME="${ds%%:*}"; rest="${ds#*:}"; MODEL="${rest%%:*}"; TSLEN="${rest##*:}"
  log "=== DATASET $NAME (ts_len=$TSLEN) ==="
  $SSH "$PC160" "bash ~/deploy_fused_pc160.sh $MODEL" >>"$LOGDIR/deploy.log" 2>&1; log "  deployed"
  for rate in "${RATES[@]}"; do
    read -r eb ab ub <<< "$(get_counters)"
    $SSH "$PC161" "python3 ~/throughput.py --fpga-ip $FPGA_IP --fpga-port $FPGA_PORT --listen-port $LISTEN_PORT --rate-min $rate --rate-max $rate --rate-steps 1 --duration-s $DURATION_S --abort-loss-pct 200 --out /tmp/tp_v4_${NAME}_${rate}" >"$LOGDIR/tp_${NAME}_${rate}.log" 2>&1
    sent=$($SSH "$PC161" "tail -n1 /tmp/tp_v4_${NAME}_${rate}/sweep.csv 2>/dev/null|cut -d, -f2"); [[ "$sent" =~ ^[0-9]+$ ]] || sent=0
    read -r ea aa ua <<< "$(get_counters)"
    ethd=$((ea-eb)); appd=$((aa-ab)); udpd=$((ua-ub))
    pps=$(awk -v a="$appd" -v d="$DURATION_S" 'BEGIN{printf "%.3f",a/d}')
    loss=$(awk -v a="$appd" -v s="$sent" 'BEGIN{if(s<1)s=1;printf "%.3f",100*(1-a/s)}')
    bw=$(awk -v p="$pps" 'BEGIN{printf "%.3f",p*64*8/1000}')
    echo "fused,${NAME},${rate},${sent},${DURATION_S},${ethd},${appd},${udpd},${pps},${loss},${bw}" >> "$CSV"
    log "  $NAME @${rate}: sent=$sent app_d=$appd udp_d=$udpd pps=$pps loss=$loss% bw=${bw}kbps"
  done
done
log "=== v4 SWEEP COMPLETE: $CSV ==="; cat "$CSV" | tee -a "$LOGDIR/orchestrate.log"
