#!/bin/bash
# f2f_knee_sweep.sh — RUNS ON WOLVERINE. Post drop-on-full-ingest rebuild.
# FPGA-to-FPGA saturation-knee sweep: pc161 U280 generates in-fabric at a range
# of rates; pc160 fused-dropfull DUT processes. With drop-on-full ingest the DUT
# NEVER wedges, so we sweep continuously (no per-point redeploy).
#
# METRIC CHANGE vs the old app_in sweep: once ingest always drains NetLayer,
# app_in tracks the OFFERED rate (no loss visible there). The graceful-saturation
# ground truth is the PREDICTION rate: udp_out_delta (one prediction packet per
# completed L-sample window) x L = samples actually processed.
#   processed_pps = udp_out_delta * L / dur
#   offered_pps   = app_in_delta / dur          (== pc161 eth_out, cross-checked)
#   loss_pct      = 100 * (1 - processed/offered)
#   bandwidth_kbps= processed_pps * 64 * 8 / 1000     (locked metric)
set -uo pipefail
PC160=rdave009@pc160.cloudlab.umass.edu
PC161=rdave009@pc161.cloudlab.umass.edu
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30"
GEN=/users/rdave009/krnl.xclbin.gen
BDF=0000:3b:00.1
DATASET=${DATASET:-InsectSound}
VARIANT=${VARIANT:-minirocket}                     # minirocket | hydra
case "$DATASET" in                                 # time_series_length per model JSON
  InsectSound)   L=600  ;;
  MosquitoSound) L=3750 ;;
  FruitFlies)    L=5000 ;;
  *) echo "unknown DATASET=$DATASET"; exit 1 ;;
esac
case "$VARIANT" in
  minirocket)
    FUSED=/users/rdave009/krnl.xclbin.fused_dropfull
    MODEL=/users/rdave009/${DATASET}_minirocket_model.json
    DEPLOY=deploy_dropfull_f2f.sh
    CSVBASE=fused_dropfull_knee ;;
  hydra)
    FUSED=/users/rdave009/krnl.xclbin.hydra_dropfull
    MODEL=/users/rdave009/hydra_$(echo "$DATASET" | tr 'A-Z' 'a-z')_model.json
    DEPLOY=deploy_hydra_f2f.sh
    CSVBASE=hydra_dropfull_knee ;;
  minirocket300)   # v5: facc_t classifier (II=1) @ 300 MHz
    FUSED=/users/rdave009/krnl.xclbin.fused_dropfull300
    MODEL=/users/rdave009/${DATASET}_minirocket_model.json
    DEPLOY=deploy_dropfull300_f2f.sh
    CSVBASE=fused_dropfull300_knee ;;
  hydra64)         # hydra v7: UNROLL=64 batches @ 200 MHz
    FUSED=/users/rdave009/krnl.xclbin.hydra_dropfull_u64
    MODEL=/users/rdave009/hydra_$(echo "$DATASET" | tr 'A-Z' 'a-z')_model.json
    DEPLOY=deploy_hydra_u64_f2f.sh
    CSVBASE=hydra_dropfull_u64_knee ;;
  minirocket84)    # v6: UNROLL=84 (one sweep per dilation) @ 300 MHz
    FUSED=/users/rdave009/krnl.xclbin.fused_dropfull300u84
    MODEL=/users/rdave009/${DATASET}_minirocket_model.json
    DEPLOY=deploy_dropfull300u84_f2f.sh
    CSVBASE=fused_dropfull300u84_knee ;;
  *) echo "unknown VARIANT=$VARIANT"; exit 1 ;;
esac
OUT=/home/rdave009/minirocket-hls/MiniRocketHLS/saturation_harness/runs/tp_2026-07-09_f2f_knee
CSV=$OUT/${CSVBASE}_${DATASET}.csv
mkdir -p "$OUT"
[ -s "$CSV" ] && mv "$CSV" "$CSV.old.$(date +%H%M%S)"   # keep prior attempts, never clobber
echo "rate_divider,offered_pps,processed_pps,loss_pct,bandwidth_kbps,udp_out_delta,app_in_delta" > "$CSV"

# pc161 TX socket re-assert args (corrected 8-byte-stride offsets): dest .10:62177, myPort 62178
TXARGS=""; for i in $(seq 0 15); do
  ti=$(printf '%X' $((0x810+i*8))); tp=$(printf '%X' $((0x890+i*8)))
  mp=$(printf '%X' $((0x910+i*8))); vv=$(printf '%X' $((0x990+i*8)))
  TXARGS="$TXARGS $ti C0A8280A $tp 0000F2E1 $mp 0000F2E2 $vv 00000001"; done

# SKIP_DEPLOY=1 when pc160 is already deployed with a verified-live link: the deploy
# reprograms the FPGA (xbutil --force program) and would tear down the trained CMAC state.
if [ "${SKIP_DEPLOY:-0}" != "1" ]; then
  echo "[setup] deploy $VARIANT DUT on pc160 ($DEPLOY: --force program + corrected socket)"
  $SSH "$PC160" "bash ~/$DEPLOY $MODEL" 2>/dev/null | grep -E "^\[" || true
else
  echo "[setup] SKIP_DEPLOY=1 — using already-deployed pc160 DUT"
fi

# sweep from the common CPU-comparison start (5k pps) through the ceiling well past it.
# emit rate ~ 66.5M/div: 13300->5k 6650->10k 2660->25k 1330->50k 665->100k 500->137k ...
# div=2 (~22.8M pps, the generator's actual max) tested SAFE 2026-07-11 on the
# dropfull300u84 build (drop-on-full + pktDropper absorbed a 4s dwell, no wedge).
# STILL FORBIDDEN: div<=1 (historical RX wedge) and div=0 (wedges the GENERATOR
# permanently — needs force-reprogram).
for DIV in 13300 6650 2660 1330 665 500 200 100 65 45 30 22 15 10 6 3 2; do
  # low rates: longer dwell so udp_out (predictions, 1 per L samples) has enough counts
  if [ "$DIV" -ge 665 ]; then DUR=12; else DUR=4; fi
  TRY=0
  while :; do
    $SSH "$PC161" "source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1; ~/start_gen $GEN $BDF $DIV >/dev/null 2>&1" 2>/dev/null
    # keep switch MAC entry for pc160 fresh (arp_discovery @0x1010), then measure
    R=$($SSH "$PC160" "source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
      ~/tweak2 $FUSED $BDF 1010 C0A8280B >/dev/null 2>&1
      rd(){ ~/read_netlayer_counters_ext $FUSED $BDF 2>/dev/null | awk '\$1==\"app_in_packets\"{a=\$2}\$1==\"udp_out_packets\"{u=\$2}END{print a,u}'; }
      read -r a0 u0 <<< \"\$(rd)\"; sleep $DUR; read -r a1 u1 <<< \"\$(rd)\"
      echo \$((a1-a0)) \$((u1-u0))" 2>/dev/null | grep -viE "Auto|XILINX|PATH")
    read -r APPD UDPD <<< "$R"
    # zero-guard: dead point -> re-assert pc161 socket AFTER start_gen, retry once
    if [ "${APPD:-0}" -eq 0 ] && [ "$TRY" -eq 0 ]; then
      TRY=1
      echo "[warn] div=$DIV read zero — re-asserting pc161 socket and retrying"
      $SSH "$PC161" "source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
        ~/setup_netlayer $GEN $BDF 02:00:00:00:0b:0b 192.168.40.11 62178 192.168.40.10 02:00:00:00:0a:0a 62177 >/dev/null 2>&1
        ~/tweak2 $GEN $BDF $TXARGS >/dev/null 2>&1" 2>/dev/null
      continue
    fi
    break
  done
  read -r OFF PROC LOSS BW <<< "$(awk -v a="${APPD:-0}" -v u="${UDPD:-0}" -v L="$L" -v d="$DUR" \
    'BEGIN{off=a/d; proc=u*L/d; loss=(off>0)?100*(1-proc/off):0; bw=proc*64*8/1000; printf "%.0f %.0f %.2f %.1f",off,proc,loss,bw}')"
  echo "$DIV,$OFF,$PROC,$LOSS,$BW,${UDPD:-0},${APPD:-0}" >> "$CSV"
  echo "[$(date '+%H:%M:%S')] div=$DIV offered=$OFF processed=$PROC loss=$LOSS% bw=${BW}kbps (udp_out_d=$UDPD)"
done
# quiesce generator: leave link up but stop the blast (~0 pps at div=1e9)
$SSH "$PC161" "source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1; ~/start_gen $GEN $BDF 1000000000 >/dev/null 2>&1" 2>/dev/null
echo "[done] generator quiesced (div=1000000000)"
echo "=== KNEE SWEEP DONE: $CSV ==="; column -t -s, "$CSV"
