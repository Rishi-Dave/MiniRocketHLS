#!/bin/bash
# cpu_sweep_3ds_orchestrate.sh  —  RUNS ON WOLVERINE (the build/controller host).
#
# CPU-baseline ("red curve") network-saturation rate sweep for MiniRocket on
# the THREE target datasets (InsectSound, FruitFlies, MosquitoSound), run
# over the REAL pc160<->pc161 100G data NIC (192.168.40.30/.31,
# enp135s0f0) -- pure software over UDP, NO xbutil, NO FPGA
# programming/reset. Adapted from cpu_sweep_orchestrate.sh (single-dataset
# len150-binary variant) to loop sequentially over the 3 fused-FPGA-sweep
# datasets so the CPU curve is apples-to-apples with the fused FPGA MiniRocket
# sweep.
#
# WHY WOLVERINE-DRIVEN: pc160 cannot ssh to pc161 (Permission denied
# publickey), so Wolverine (which authenticates to both) conducts:
#   * responder (minirocket_udp_responder.py) -> ssh pc160, backgrounded+
#       detached, bound to 192.168.40.30:51000 (distinct from the FPGA's
#       62177/62178 ports; the FPGA on pc160 is never touched)
#   * sender (throughput.py)                  -> ssh pc161 (the actual UDP
#       sender, .31 NIC)
#   * ground truth reads                      -> ssh pc160 (responder's own
#       StatsReporter JSON file -- packets_received counter)
#
# Ground truth = responder's own packets_received counter (NOT throughput.py
# recv_acks -- the responder only acks once per L-sample window, not once
# per packet, so throughput.py's built-in loss metric is meaningless here;
# --abort-loss-pct 200 disables its 2-consecutive-high-loss sweep abort).
#
# One responder at a time on pc160, port 51000 reused sequentially across
# datasets (stopped/restarted with the next dataset's model between legs).

set -uo pipefail

# ---- nodes / ssh -----------------------------------------------------------
PC160=rdave009@pc160.cloudlab.umass.edu
PC161=rdave009@pc161.cloudlab.umass.edu
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

# ---- fixed rig parameters ---------------------------------------------------
FPGA_DATA_IP=192.168.40.30       # pc160 data-NIC IP (responder binds here)
REMOTE_DIR=/users/rdave009/cpu_baseline
PORT=51000
SCRIPT=minirocket_udp_responder.py

RATES=(500 1000 2000 5000 10000 25000 50000 100000 250000 500000)
DURATION_S=3

DATASETS=(InsectSound FruitFlies MosquitoSound)
declare -A CFG_MODEL=(
  [InsectSound]="models/InsectSound_minirocket_model.json"
  [FruitFlies]="models/FruitFlies_minirocket_model.json"
  [MosquitoSound]="models/MosquitoSound_minirocket_model.json"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="$SCRIPT_DIR/runs/tp_2026-07-06_cpu"
LOGDIR="$OUTDIR/logs"
mkdir -p "$OUTDIR" "$LOGDIR"
CSV="$OUTDIR/cpu_minirocket_3ds.csv"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/cpu_3ds_orchestrate.log"; }

start_responder() {
  local dataset=$1
  local model=${CFG_MODEL[$dataset]}
  local stats="/tmp/cpu-minirocket-${dataset}_stats.json"
  log "starting minirocket responder on pc160 ${FPGA_DATA_IP}:${PORT} dataset=$dataset model=$model"
  $SSH "$PC160" "rm -f $stats"
  timeout 10 $SSH "$PC160" "cd $REMOTE_DIR && setsid nohup python3 cpu_servers/$SCRIPT \
      --listen-ip $FPGA_DATA_IP --listen-port $PORT \
      --model-json $model \
      --variant cpu-minirocket --dataset '$dataset' \
      --stats-file $stats \
      > /tmp/minirocket_${dataset}_responder.log 2>&1 < /dev/null & echo \$!" > "$LOGDIR/minirocket_${dataset}_responder.pid" 2>&1 || true
  sleep 1
  local pat="cpu_servers/[${SCRIPT:0:1}]${SCRIPT:1}.*--listen-port $PORT"
  local check
  check=$($SSH "$PC160" "pgrep -f '$pat' | head -1")
  if [[ -z "$check" ]]; then
    log "  ERROR: minirocket/$dataset responder did not start; log:"
    $SSH "$PC160" "cat /tmp/minirocket_${dataset}_responder.log" | tee -a "$LOGDIR/cpu_3ds_orchestrate.log"
    return 1
  fi
  log "  responder pid(s) on pc160: $check"
  echo "$check" > "$LOGDIR/minirocket_${dataset}_responder.pid"
  return 0
}

stop_responder() {
  local pat="cpu_servers/[${SCRIPT:0:1}]${SCRIPT:1}.*--listen-port $PORT"
  log "stopping minirocket responder on pc160"
  timeout 10 $SSH "$PC160" "pkill -f '$pat'" || true
  sleep 0.3
}

read_processed() {
  local stats=$1
  timeout 10 $SSH "$PC160" "python3 -c \"
import json
try:
    d = json.load(open('$stats'))
    print(d.get('packets_received', 0))
except Exception:
    print(0)
\""
}

run_rate_step() {
  local dataset=$1 rate=$2 step_no=$3
  local stats="/tmp/cpu-minirocket-${dataset}_stats.json"
  local send_listen=$((51300 + step_no))
  local tp_out="/tmp/tp_cpu_minirocket_${dataset}_${rate}"

  log "=== [$dataset] rate step: ${rate} pps ==="

  local before after sent
  before=$(read_processed "$stats")
  log "  packets_received BEFORE: $before"

  timeout $((DURATION_S + 20)) $SSH "$PC161" "python3 ~/throughput.py --fpga-ip $FPGA_DATA_IP --fpga-port $PORT \
      --listen-port $send_listen --rate-min $rate --rate-max $rate --rate-steps 1 \
      --duration-s $DURATION_S --abort-loss-pct 200 --out $tp_out" \
      > "$LOGDIR/minirocket_${dataset}_tp_${rate}.log" 2>&1
  local tp_rc=$?
  [ $tp_rc -ne 0 ] && log "  WARNING: throughput.py exited $tp_rc (see $LOGDIR/minirocket_${dataset}_tp_${rate}.log)"

  # Grace period: let the responder drain in-flight packets and flush its
  # periodic StatsReporter snapshot (default interval 0.2s) before sampling
  # the "after" counter. No further packets are sent during this window.
  sleep 0.5

  after=$(read_processed "$stats")
  sent=$(timeout 10 $SSH "$PC161" "tail -n1 ${tp_out}/sweep.csv 2>/dev/null | cut -d, -f2")
  [[ "$sent" =~ ^[0-9]+$ ]] || sent=0

  local processed=$((after - before))
  [[ $processed -lt 0 ]] && processed=0

  python3 - "cpu-minirocket" "$dataset" "$rate" "$sent" "$DURATION_S" "$processed" "$CSV" <<'PYEOF'
import sys
variant, dataset, configured_pps, sent, duration_s, processed, out_csv = sys.argv[1:8]
configured_pps = float(configured_pps)
sent = int(sent)
duration_s = float(duration_s)
processed = int(processed)

processed_pps = processed / duration_s if duration_s > 0 else 0.0
bandwidth_kbps = processed_pps * 64 * 8 / 1000.0
loss_pct = 100.0 * (1.0 - processed / max(sent, 1))

with open(out_csv, "a") as f:
    f.write(f"{variant},{dataset},{configured_pps},{sent},{duration_s},{processed},{processed_pps},{loss_pct},{bandwidth_kbps}\n")
PYEOF

  log "  row: sent=$sent before=$before after=$after processed=$processed processed_pps=$(awk -v p=$processed -v d=$DURATION_S 'BEGIN{printf "%.1f", p/d}')"
}

run_dataset_sweep() {
  local dataset=$1

  start_responder "$dataset" || { log "aborting $dataset sweep -- responder failed to start"; return 1; }

  local step_no=0
  for r in "${RATES[@]}"; do
    step_no=$((step_no + 1))
    run_rate_step "$dataset" "$r" "$step_no"
  done

  stop_responder
  log "=== $dataset sweep complete ==="
}

echo "variant,dataset,configured_pps,sent,duration_s,processed_pkts,processed_pps,loss_pct,bandwidth_kbps" > "$CSV"

log "=== cpu_sweep_3ds_orchestrate.sh (Wolverine-driven) starting; datasets: ${DATASETS[*]}; rates: ${RATES[*]} ==="
log "*** FPGA on pc160/pc161 left completely untouched -- no xbutil, no program, no reset. ***"

for ds in "${DATASETS[@]}"; do
  run_dataset_sweep "$ds"
done

log "=== all CPU MiniRocket 3-dataset sweeps complete. CSV: $CSV ==="
cat "$CSV" | tee -a "$LOGDIR/cpu_3ds_orchestrate.log"
