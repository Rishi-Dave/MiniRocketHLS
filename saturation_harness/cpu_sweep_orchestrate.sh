#!/bin/bash
# cpu_sweep_orchestrate.sh  —  RUNS ON WOLVERINE (the build/controller host).
#
# CPU-baseline ("red curve") network-saturation rate sweep, run over the REAL
# pc160<->pc161 100G data NIC (192.168.40.30/.31, enp135s0f0) -- pure
# software over UDP, NO xbutil, NO FPGA programming/reset. The already-
# deployed v3-dpr FPGA kernel + its loader tmux session on pc160 are left
# completely untouched for the entire duration of this script.
#
# WHY WOLVERINE-DRIVEN (same reasoning as dpr_sweep_orchestrate.sh): pc160
# cannot ssh to pc161 (Permission denied publickey), so Wolverine (which
# authenticates to both) conducts:
#   * responder (minirocket_udp_responder.py / hydra_udp_responder.py)
#       -> ssh pc160, backgrounded+detached, bound to 192.168.40.30:<port>
#   * sender (throughput.py)   -> ssh pc161 (the actual UDP sender, .31 NIC)
#   * ground truth reads       -> ssh pc160 (responder's own StatsReporter
#                                 JSON file -- packets_received counter,
#                                 same spirit as the FPGA's app_in counter)
#
# Ground truth = responder's own packets_received counter (NOT throughput.py
# recv_acks -- the responders only ack once per L-sample window, not once
# per packet, so throughput.py's built-in loss metric is meaningless here;
# --abort-loss-pct 200 disables its 2-consecutive-high-loss sweep abort).
#
# Two variants, run sequentially (one responder at a time on pc160):
#   cpu-minirocket : dataset len150-binary, L=150,  port 51000
#   cpu-hydra      : dataset InsectSound,   L=600,  port 51001
# (51000/51001 chosen to stay well clear of the FPGA's 62177/62178.)

set -uo pipefail

# ---- nodes / ssh -----------------------------------------------------------
PC160=rdave009@pc160.cloudlab.umass.edu
PC161=rdave009@pc161.cloudlab.umass.edu
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

# ---- fixed rig parameters ---------------------------------------------------
FPGA_DATA_IP=192.168.40.30       # pc160 data-NIC IP (responder binds here)
REMOTE_DIR=/users/rdave009/cpu_baseline

RATES=(500 1000 2000 5000 10000 25000 50000 100000 250000 500000)
DURATION_S=3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="$SCRIPT_DIR/runs/tp_2026-07-04"
LOGDIR="$OUTDIR/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/cpu_orchestrate.log"; }

# ---- per-variant config ------------------------------------------------------
# name:script:model:port:variant_label:dataset_label:sender_listen_base
declare -A CFG_SCRIPT=( [minirocket]="minirocket_udp_responder.py" [hydra]="hydra_udp_responder.py" )
declare -A CFG_MODEL=(  [minirocket]="models/minirocket_model.json" [hydra]="models/hydra_insectsound_model.json" )
declare -A CFG_PORT=(   [minirocket]=51000 [hydra]=51001 )
declare -A CFG_VARIANT=([minirocket]="cpu-minirocket" [hydra]="cpu-hydra" )
declare -A CFG_DATASET=([minirocket]="len150-binary" [hydra]="InsectSound" )
declare -A CFG_SENDBASE=([minirocket]=51100 [hydra]=51200 )
declare -A CFG_STATS=(  [minirocket]="/tmp/cpu-minirocket_stats.json" [hydra]="/tmp/cpu-hydra_stats.json" )

start_responder() {
  local which=$1
  local script=${CFG_SCRIPT[$which]} model=${CFG_MODEL[$which]} port=${CFG_PORT[$which]}
  local variant=${CFG_VARIANT[$which]} dataset=${CFG_DATASET[$which]} stats=${CFG_STATS[$which]}
  log "starting $which responder on pc160 ${FPGA_DATA_IP}:${port} (model=$model)"
  # Wipe any stale stats file so "before" reads start from a clean 0.
  $SSH "$PC160" "rm -f $stats"
  # NOTE: this ssh invocation reliably starts the detached (setsid+nohup)
  # remote process and its stdout ("started") comes back fine, but the local
  # ssh client has been observed to sometimes not exit promptly afterward
  # (channel is freed per -v trace, but the process lingers) -- a sandbox/
  # pty quirk, not a remote-side problem. Wrap in `timeout` + `|| true` and
  # always verify success via a separate pgrep call below rather than
  # trusting this call's own return.
  timeout 10 $SSH "$PC160" "cd $REMOTE_DIR && setsid nohup python3 cpu_servers/$script \
      --listen-ip $FPGA_DATA_IP --listen-port $port \
      --model-json $model \
      --variant $variant --dataset '$dataset' \
      --stats-file $stats \
      > /tmp/${which}_responder.log 2>&1 < /dev/null & echo \$!" > "$LOGDIR/${which}_responder.pid" 2>&1 || true
  sleep 1
  # Confirm it's alive and listening. NOTE: bracket-trick the first letter of
  # $script ("[m]inirocket...") so this pgrep pattern can never self-match
  # the invoking pgrep/pkill process's own command line (which literally
  # contains the plain search string as a substring) -- a plain match here
  # would (for pkill) send SIGTERM to the ssh session's own remote shell and
  # kill the whole ssh channel with exit-signal instead of exit-status.
  local pat="cpu_servers/[${script:0:1}]${script:1}.*--listen-port $port"
  local check
  check=$($SSH "$PC160" "pgrep -f '$pat' | head -1")
  if [[ -z "$check" ]]; then
    log "  ERROR: $which responder did not start; log:"
    $SSH "$PC160" "cat /tmp/${which}_responder.log" | tee -a "$LOGDIR/cpu_orchestrate.log"
    return 1
  fi
  log "  responder pid(s) on pc160: $check"
  echo "$check" > "$LOGDIR/${which}_responder.pid"
  return 0
}

stop_responder() {
  local which=$1
  local script=${CFG_SCRIPT[$which]} port=${CFG_PORT[$which]}
  local pat="cpu_servers/[${script:0:1}]${script:1}.*--listen-port $port"
  log "stopping $which responder on pc160"
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
  local which=$1 rate=$2 step_no=$3
  local port=${CFG_PORT[$which]} variant=${CFG_VARIANT[$which]} dataset=${CFG_DATASET[$which]}
  local stats=${CFG_STATS[$which]} sendbase=${CFG_SENDBASE[$which]}
  local send_listen=$((sendbase + step_no))
  local csv=$4
  local tp_out="/tmp/tp_cpu_${which}_${rate}"

  log "=== [$which] rate step: ${rate} pps ==="

  local before after sent
  before=$(read_processed "$stats")
  log "  packets_received BEFORE: $before"

  timeout $((DURATION_S + 20)) $SSH "$PC161" "python3 ~/throughput.py --fpga-ip $FPGA_DATA_IP --fpga-port $port \
      --listen-port $send_listen --rate-min $rate --rate-max $rate --rate-steps 1 \
      --duration-s $DURATION_S --abort-loss-pct 200 --out $tp_out" \
      > "$LOGDIR/${which}_tp_${rate}.log" 2>&1
  local tp_rc=$?
  [ $tp_rc -ne 0 ] && log "  WARNING: throughput.py exited $tp_rc (see $LOGDIR/${which}_tp_${rate}.log)"

  # Grace period: let the responder drain in-flight packets and flush its
  # periodic StatsReporter snapshot (default interval 0.2s) before sampling
  # the "after" counter. No further packets are sent during this window.
  sleep 0.5

  after=$(read_processed "$stats")
  sent=$(timeout 10 $SSH "$PC161" "tail -n1 ${tp_out}/sweep.csv 2>/dev/null | cut -d, -f2")
  [[ "$sent" =~ ^[0-9]+$ ]] || sent=0

  local processed=$((after - before))
  [[ $processed -lt 0 ]] && processed=0

  python3 - "$variant" "$dataset" "$rate" "$sent" "$DURATION_S" "$processed" "$csv" <<'PYEOF'
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

run_variant_sweep() {
  local which=$1
  local csv="$OUTDIR/cpu_${which}_sweep.csv"
  echo "variant,dataset,configured_pps,sent,duration_s,processed_pkts,processed_pps,loss_pct,bandwidth_kbps" > "$csv"

  start_responder "$which" || { log "aborting $which sweep -- responder failed to start"; return 1; }

  local step_no=0
  for r in "${RATES[@]}"; do
    step_no=$((step_no + 1))
    run_rate_step "$which" "$r" "$step_no" "$csv"
  done

  stop_responder "$which"
  log "=== $which sweep complete. CSV: $csv ==="
  cat "$csv" | tee -a "$LOGDIR/cpu_orchestrate.log"
}

log "=== cpu_sweep_orchestrate.sh (Wolverine-driven) starting; rates: ${RATES[*]} ==="
log "*** FPGA on pc160 is left untouched -- no xbutil, no program, no reset. ***"

run_variant_sweep minirocket
run_variant_sweep hydra

log "=== all CPU baseline sweeps complete ==="
