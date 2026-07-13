# OCT parallel-build runbook

**Window:** 2-day OCT reservation (1 build server + 2 FPGAs).

**Goal of this doc:** specify exactly which xclbin builds to fan out in
parallel on OCT, in priority order, with copy-pasteable commands and the
mandatory monitoring subagent (CLAUDE.md Rule #7).

All sources are csim-clean as of 2026-04-28:
- `hydra_optimized/hydra_axis/` — csim PASS, OCT-ready (7 m_axi ports)
- `multirocket_optimized/multirocket_axis/` — csim PASS, OCT-ready (10 m_axi ports)
- `multirocket_optimized/` — m_axi sources clean post Jan-6 algo fix; xclbin stale (predates the fix)
- `fpga-network/` — MiniRocket-axis xclbin already built (Phase 2 drop-module integration BLOCKED on Prith)

## Build queue (priority order)

### Build 1 — MultiRocket m_axi rebuild (Phase 4)
**Why first:** the existing xclbin predates the Jan-6 algorithm rewrite (now 4 pooling ops + first-order diff + 5,376 features). All other MultiRocket work (Phase 5 axis variant) gates on confirming this one produces correct accuracy on InsectSound / MosquitoSound / FruitFlies.

**Time:** ~4–8 h. **Resources:** uses CMAC/NetLayer? **No** — pure PCIe/HBM compute kernel.

```bash
# On OCT build node, from MiniRocketHLS/multirocket_optimized
tmux new -d -s mr-mAxi-hw 'make TARGET=hw PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1 |& tee build_dir.hw.post_jan6.log'

# Within 30s of starting tmux, IMMEDIATELY launch the monitoring subagent
# per CLAUDE.md Rule #7. From the Claude Code session:
#
#   Task(subagent_type="hls-monitor", model="haiku", run_in_background=true,
#        prompt="Monitor tmux session 'mr-mAxi-hw' every 2 min for 15 min;
#                report phase, errors. Exit after 15 min or on failure.")
```

**Validation after build:**
```bash
./multirocket_host build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/krnl.xclbin \
    models/multirocket84_insectsound_model.json \
    models/insectsound_test_data.json
# Expect accuracy within 1% of CPU baseline; if not, debug ap_fixed/float
# overflow per MEMORY.md note before kicking off Phase 5.
```

### Build 2 — HYDRA axis-stream xclbin (Phase 3)
**Why parallel-friendly:** independent IP synth + v++ link from Build 1.

**Time:** ~30 min HLS export + ~3–5 h v++ link. **Uses:** prebuilt CMAC + NetLayer XOs from `fpga-network/_x.xilinx_u280_gen3x16_xdma_1_202211_1/`.

```bash
# On OCT build node, from MiniRocketHLS/hydra_optimized/hydra_axis
tmux new -d -s hydra-axis-hw './build_axis_xclbin.sh hw |& tee hw_build.log'

# Monitor:
#   Task(subagent_type="hls-monitor", model="haiku", run_in_background=true,
#        prompt="Monitor tmux session 'hydra-axis-hw' every 2 min for 15 min...")
```

**Output:** `hydra_optimized/hydra_axis/build_dir.hw/krnl.xclbin`

### Build 3 — MultiRocket axis-stream xclbin (Phase 5)
**Why third:** csim PASS but unverified vs. m_axi (Build 1 must finish + accuracy parity confirmed before relying on this for paper data). Build can still START in parallel — if Build 1 surfaces a bug, kill this and rerun.

**Time:** ~1 h HLS export + ~5–8 h v++ link (10 m_axi ports → wider crossbar).

```bash
# On OCT build node, from MiniRocketHLS/multirocket_optimized/multirocket_axis
tmux new -d -s mr-axis-hw './build_axis_xclbin.sh hw |& tee hw_build.log'

# Monitor (same pattern).
```

**Output:** `multirocket_optimized/multirocket_axis/build_dir.hw/krnl.xclbin`

### Build 4 — fpga-network MiniRocket + drop module (Phase 2) — BLOCKED on Prith
Cannot start until Prith ships his packet-drop kernel + integration notes.
Once received, the build path is the existing `fpga-network/Makefile`
plus the inserted `stream_connect` lines in `fpga-network/config.cfg`.

## Parallelism advice

If OCT lets you allocate the build server + 2 FPGA nodes simultaneously:
- **All three builds can run on the same build server** (they don't need
  the FPGA — `make TARGET=hw` and `v++` are CPU+RAM bound).
- The 2 FPGA nodes are for **post-build validation runs** (Phase 6
  saturation experiments). Don't waste the FPGA reservation on builds.
- A single 32-core build node typically handles 2–3 concurrent v++ jobs
  before swap thrashing — watch `free -h` and `uptime`.

## After all 3 builds finish

1. Validate each xclbin via the saturation harness:
   ```bash
   cd MiniRocketHLS/saturation_harness
   python3 rate_sweep.py --adapter minirocket-axis  --dataset InsectSound \
       --fpga-ip <FPGA_IP> --fpga-port <PORT> --listen-port <RX_PORT>     \
       --rate-min 100 --rate-max 50000 --rate-steps 12 --duration-s 5    \
       --out runs/oct_smoke/minirocket_InsectSound
   # Same for hydra-axis (after wiring its adapter) and multirocket-axis.
   ```
2. Update `MEMORY.md` variant-status table with new xclbin paths + accuracies.
3. Once Prith's drop module integrates, run the full saturation sweep
   (Phase 6) — the lead figure for the paper.

## CLAUDE.md compliance reminder

For ANY of the above hw builds:
- **Always tmux** the build (Rule #3).
- **Always launch the monitoring subagent within 30s** (Rule #7).
- **Always confirm with the user before triggering** (Rule #2) — even though OCT has the headroom, the lab convention is explicit confirmation before each `TARGET=hw`.
