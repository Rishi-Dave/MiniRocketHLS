#!/bin/bash
# fused_sweep_orchestrate.sh  —  RUNS ON WOLVERINE (conductor).
# Headline network-saturation sweep of the fused MiniRocket kernel (v3:
# single-shot + packed-emit, HW-validated 2026-07-06) on pc160, sender pc161.
# Loops over 3 datasets (reload model per dataset). Ground truth = FPGA
# NetLayer counter deltas (app_in = ingested/inference-input; udp_out =
# packed prediction packets = app_in/ts_len). Same Wolverine-driven pattern
# as dpr_sweep_orchestrate.sh (pc160 can't ssh pc161).
set -uo pipefail

PC160=rdave009@pc160.cloudlab.umass.edu
PC161=rdave009@pc161.cloudlab.umass.edu
SSH="ssh -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new"
BDF=0000:3b:00.1
XCLBIN=/users/rdave009/krnl.xclbin.fused
FPGA_IP=192.168.40.10; FPGA_PORT=62177; LISTEN_PORT=62178

# datasets: name:model_json:ts_len
DATASETS=(
  "InsectSound:/users/rdave009/InsectSound_minirocket_model.json:600"
  "FruitFlies:/users/rdave009/FruitFlies_minirocket_model.json:5000"
  "MosquitoSound:/users/rdave009/MosquitoSound_minirocket_model.json:3750"
)
# rate points match the CPU baseline sweep (apples-to-apples)
RATES=(500 1000 2000 5000 10000 25000 50000 100000 250000 500000)
DURATION_S=3
RECOVERY_THRESHOLD=25000   # reset+reprogram+reload after any step >= this

OUTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runs/tp_2026-07-06_fused"
CSV="$OUTDIR/fused_sweep.csv"
LOGDIR="$OUTDIR/logs"
mkdir -p "$OUTDIR" "$LOGDIR"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGDIR/orchestrate.log"; }
[ -f "$CSV" ] || echo "variant,dataset,configured_pps,sent,duration_s,eth_in_delta,app_in_delta,udp_out_delta,processed_pps,loss_pct,bandwidth_kbps" > "$CSV"

get_counters(){  # -> "eth app udp"
  local raw eth app udp
  raw=$($SSH "$PC160" "source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1; ~/read_netlayer_counters_ext $XCLBIN $BDF 2>/dev/null")
  eth=$(echo "$raw"|awk '$1=="eth_in_packets"{print $2}')
  app=$(echo "$raw"|awk '$1=="app_in_packets"{print $2}')
  udp=$(echo "$raw"|awk '$1=="udp_out_packets"{print $2}')
  echo "${eth:-0} ${app:-0} ${udp:-0}"
}
deploy(){ log "  DEPLOY/RECOVER fused (model=$1)"; $SSH "$PC160" "bash ~/deploy_fused_pc160.sh $1" >>"$LOGDIR/deploy.log" 2>&1; }

for ds in "${DATASETS[@]}"; do
  NAME="${ds%%:*}"; rest="${ds#*:}"; MODEL="${rest%%:*}"; TSLEN="${rest##*:}"
  log "=== DATASET $NAME (model=$MODEL ts_len=$TSLEN) ==="
  deploy "$MODEL"
  for rate in "${RATES[@]}"; do
    log "  --- $NAME @ ${rate} pps ---"
    read -r eb ab ub <<< "$(get_counters)"
    $SSH "$PC161" "python3 ~/throughput.py --fpga-ip $FPGA_IP --fpga-port $FPGA_PORT --listen-port $LISTEN_PORT --rate-min $rate --rate-max $rate --rate-steps 1 --duration-s $DURATION_S --abort-loss-pct 200 --out /tmp/tp_fz_${NAME}_${rate}" >"$LOGDIR/tp_${NAME}_${rate}.log" 2>&1
    local_sent=$($SSH "$PC161" "tail -n1 /tmp/tp_fz_${NAME}_${rate}/sweep.csv 2>/dev/null | cut -d, -f2")
    [[ "$local_sent" =~ ^[0-9]+$ ]] || local_sent=0
    read -r ea aa ua <<< "$(get_counters)"
    ethd=$((ea-eb)); appd=$((aa-ab)); udpd=$((ua-ub))
    pps=$(awk -v a="$appd" -v d="$DURATION_S" 'BEGIN{printf "%.3f",a/d}')
    loss=$(awk -v a="$appd" -v s="$local_sent" 'BEGIN{if(s<1)s=1;printf "%.3f",100*(1-a/s)}')
    bw=$(awk -v p="$pps" 'BEGIN{printf "%.3f",p*64*8/1000}')
    echo "fused,${NAME},${rate},${local_sent},${DURATION_S},${ethd},${appd},${udpd},${pps},${loss},${bw}" >> "$CSV"
    log "    sent=$local_sent eth_d=$ethd app_d=$appd udp_d=$udpd pps=$pps loss=$loss% bw=${bw}kbps"
    if [ "$rate" -ge "$RECOVERY_THRESHOLD" ]; then deploy "$MODEL"; fi
  done
done
log "=== SWEEP COMPLETE. CSV: $CSV ==="
cat "$CSV" | tee -a "$LOGDIR/orchestrate.log"
