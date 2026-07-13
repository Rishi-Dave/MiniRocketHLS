#!/usr/bin/env bash
# cpu_sweep_runner.sh -- drive a CPU streaming-inference UDP responder
# (minirocket_udp_responder.py / hydra_udp_responder.py) with the SAME
# throughput.py sweep driver used against the FPGA, across a fixed list of
# target rates, and emit a CSV with the shared saturation-chart schema:
#
#   variant,dataset,configured_pps,sent,duration_s,processed_pkts,processed_pps,loss_pct,bandwidth_kbps
#
# where:
#   processed_pps   = processed_pkts / duration_s
#   bandwidth_kbps  = processed_pps * 64 * 8 / 1000
#   loss_pct        = 100 * (1 - processed_pkts / max(sent, 1))
#
# "processed_pkts" is read from the RESPONDER's own live stats file (ground
# truth on the CPU side -- same spirit as the FPGA's app_in packet counter),
# NOT from throughput.py's "recv_acks" column. throughput.py's own loss
# metric is irrelevant here (our responder only acks once per L samples,
# not once per packet) which is exactly why every invocation below passes
# --abort-loss-pct 200 -- effectively disables throughput.py's "2
# consecutive high-loss steps -> abort sweep" behavior, since loss_pct is
# bounded at 100 and can never exceed 200.
#
# IMPORTANT (2026-07-04 scope): this script is prepared and smoke-tested
# over local loopback ONLY. Do NOT point --target-ip at pc161 / the FPGA
# host while pc161's throughput.py is busy driving the FPGA sweep -- both
# would contend for the same NIC. Leave the live cross-host run for the
# orchestrator to sequence.
#
# Usage:
#   ./cpu_sweep_runner.sh \
#       --variant cpu-minirocket --dataset dpr_len150 \
#       --target-ip 127.0.0.1 --target-port 15000 \
#       --stats-file /tmp/cpu-minirocket_stats.json \
#       --out-csv /path/to/cpu_minirocket_sweep.csv
#
# The responder must already be running and listening on --target-ip:
# --target-port with a matching --stats-file before this script is invoked.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THROUGHPUT_PY_DEFAULT="/home/rdave009/minirocket-hls/MiniRocketHLS/fpga-network/network_experiments/TP/throughput.py"

VARIANT=""
DATASET=""
TARGET_IP="127.0.0.1"
TARGET_PORT=""
LISTEN_PORT_BASE=40000
STATS_FILE=""
OUT_CSV=""
DURATION_S="2.0"
ABORT_LOSS_PCT="200"
THROUGHPUT_PY="${THROUGHPUT_PY_DEFAULT}"
RUN_ROOT="${SCRIPT_DIR}/../runs/cpu_baseline_$(date +%Y%m%d_%H%M%S)"
RATES=(500 1000 2000 5000 10000 25000 50000 100000 250000 500000 1000000)
SETTLE_S="0.5"

usage() {
    grep '^#' "$0" | sed -e 's/^#//' -e 's/^ //'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --target-ip) TARGET_IP="$2"; shift 2 ;;
        --target-port) TARGET_PORT="$2"; shift 2 ;;
        --listen-port-base) LISTEN_PORT_BASE="$2"; shift 2 ;;
        --stats-file) STATS_FILE="$2"; shift 2 ;;
        --out-csv) OUT_CSV="$2"; shift 2 ;;
        --duration-s) DURATION_S="$2"; shift 2 ;;
        --abort-loss-pct) ABORT_LOSS_PCT="$2"; shift 2 ;;
        --throughput-py) THROUGHPUT_PY="$2"; shift 2 ;;
        --run-root) RUN_ROOT="$2"; shift 2 ;;
        --rates) IFS=',' read -r -a RATES <<< "$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

if [[ -z "$VARIANT" || -z "$TARGET_PORT" || -z "$STATS_FILE" || -z "$OUT_CSV" ]]; then
    echo "ERROR: --variant, --target-port, --stats-file, and --out-csv are required." >&2
    usage
fi

if [[ ! -f "$THROUGHPUT_PY" ]]; then
    echo "ERROR: throughput.py not found at $THROUGHPUT_PY" >&2
    exit 1
fi

mkdir -p "$RUN_ROOT"
mkdir -p "$(dirname "$OUT_CSV")"

echo "variant,dataset,configured_pps,sent,duration_s,processed_pkts,processed_pps,loss_pct,bandwidth_kbps" > "$OUT_CSV"

read_processed() {
    # Print the responder's cumulative packets_received counter, or 0 if the
    # stats file doesn't exist yet.
    python3 - "$STATS_FILE" <<'PYEOF'
import json, sys
path = sys.argv[1]
try:
    with open(path) as f:
        d = json.load(f)
    print(d.get("packets_received", 0))
except (FileNotFoundError, json.JSONDecodeError):
    print(0)
PYEOF
}

read_sent() {
    # Print the "sent" column from throughput.py's sweep.csv (single row,
    # since each invocation below uses --rate-steps 1).
    python3 - "$1" <<'PYEOF'
import csv, sys
path = sys.argv[1]
with open(path) as f:
    rows = list(csv.DictReader(f))
print(int(float(rows[0]["sent"])) if rows else 0)
PYEOF
}

step_no=0
for rate in "${RATES[@]}"; do
    step_no=$((step_no + 1))
    listen_port=$((LISTEN_PORT_BASE + step_no))
    step_out="${RUN_ROOT}/rate_${rate}"

    before=$(read_processed)

    python3 "$THROUGHPUT_PY" \
        --fpga-ip "$TARGET_IP" \
        --fpga-port "$TARGET_PORT" \
        --listen-port "$listen_port" \
        --rate-min "$rate" \
        --rate-max "$rate" \
        --rate-steps 1 \
        --duration-s "$DURATION_S" \
        --abort-loss-pct "$ABORT_LOSS_PCT" \
        --out "$step_out" \
        1>"${step_out}.stdout.log" 2>"${step_out}.stderr.log" || {
            echo "WARNING: throughput.py failed at rate=$rate, see ${step_out}.stderr.log" >&2
            continue
        }

    # Grace period: let the responder drain any in-flight packets and flush
    # its periodic stats snapshot (StatsReporter interval default 0.2s)
    # before we sample the "after" counter. No further packets are sent
    # during this window.
    sleep "$SETTLE_S"

    after=$(read_processed)
    sent=$(read_sent "${step_out}/sweep.csv")
    processed=$((after - before))
    if [[ $processed -lt 0 ]]; then
        processed=0
    fi

    python3 - "$VARIANT" "$DATASET" "$rate" "$sent" "$DURATION_S" "$processed" "$OUT_CSV" <<'PYEOF'
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

    echo "[cpu_sweep_runner] rate=${rate} sent=${sent} processed=${processed} " \
         "processed_pps=$(python3 -c "print(f'{${processed}/${DURATION_S}:.1f}')")" >&2
done

echo "Wrote $OUT_CSV" >&2
