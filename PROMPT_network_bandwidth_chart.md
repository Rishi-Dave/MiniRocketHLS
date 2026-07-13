# Session Prompt — MiniRocket Network Bandwidth Chart (FCCM resubmission)

> **SUPERSEDED — work completed and superseded by the F2F knee sweeps; see
> `docs/NETWORK_RESULTS.md`.** Kept for historical record of the original
> Phase-0 orchestration prompt only.

> Paste this whole file as the first message of a new Claude Code session in
> `/home/rdave009/minirocket-hls`. Run the orchestrator on **Claude Fable**.

---

## 0. How you operate (READ FIRST)

You are the **orchestrator**. You do **not** do task work yourself — **every actual
task (reading, editing, building, deploying, sweeping, plotting, analysis) is done
by a subagent** you dispatch with the `Agent` tool. Your job is to plan, decompose,
dispatch, verify subagent output, and integrate. Keep conclusions, not file dumps.

Before doing anything else:
1. Invoke the **`superpowers:using-superpowers`** skill (how to find/use skills).
2. Invoke the **`superpowers:subagent-driven-development`** skill (executing plans
   via subagents) — and use **`superpowers:dispatching-parallel-agents`** whenever
   you have 2+ independent tasks to run concurrently.
3. Also load project skills as needed: `/hls-review` before pragma edits, `/build`
   + the `hls-monitor` subagent for hw builds (Rule 7), `/safe-deploy` before any
   `xbutil program`, `hls-timing-closure` on negative WNS, `hls-pipeline-debug` on II>1.

Project rules that still bind you (from `CLAUDE.md`):
- **Rule 0 subagent-first** — every audit/multi-file read → subagent (you're already
  doing this by design).
- **Rule 2** — NEVER `xbutil program` / hw run without explicit user ack. State
  variant, freq, CUs, HBM ports first.
- **Rule 7** — builds >10 min → tmux + `hls-monitor` (haiku) subagent, 15-min window.
- Never delete `.xclbin`/`.xo`. Build on Wolverine (Vitis 2023.2), deploy on the FPGA node.
- Update `memory/` (`debugging.md`, `hls-patterns.md`, etc.) within 10 min of any fix >5 min.

Read these memory files before planning (dispatch a subagent to digest them):
`memory/MEMORY.md`, `memory/oct_cluster_state_2026_05_19.md` (bottom: the 2026-07-04
pc160/161 section — CURRENT hardware), `memory/dpr_source_recovery_2026_06_04.md`,
`memory/benchmarks.md`, `memory/saturation_pivot_2026_04_28.md`,
`PaperTexFiles/04_experimental_setup.tex` + `03_architecture.tex`.

---

## 1. The mission (what we're actually building)

We are NOT chasing raw speedup anymore (that data exists and stays in the paper as
support). The **decisive win** for the FCCM resubmission is a **network-saturation
bandwidth chart**. PI Philip Brisk, verbatim:

> "We'll use both. My take on the FCCM reviews is that we need more decisive wins…
> but there is also going to be pushes for GPU results, where we are less likely to
> win. So the new results with faster HBM loaded builds help. But **the network
> experiments is where we can most definitively beat either a CPU or a GPU because
> the CPU/GPU need to go through a NIC and we don't.**"

**The argument:** a CPU/GPU running streaming inference must take every packet
NIC → DMA → kernel → userspace → inference → response. Per-packet host overhead
caps small-packet throughput (it *saturates*). The FPGA ingests at the CMAC (100G
line rate) and computes **inline in-fabric — no NIC, no host** — so it keeps scaling.

## 2. The target chart

Reproduce the shape of `bandwidth-experiment.png` (Brisk's mock), per dataset:
- **X axis:** Transmission Rate, `log2(pps)`, ~6 → 17.
- **Y axis:** Bandwidth (Kbps), **log scale**.
- **CPU series (red squares):** tracks the FPGA at low rates, then **flattens /
  saturates around ~2^10 pps** — the NIC/host ceiling.
- **FPGA series (green circles):** keeps **climbing past the CPU knee** (up to line
  rate / its compute ceiling).
- The **gap between the flattened CPU line and the still-climbing FPGA line is the
  result.** Two example datasets in the mock: (a) NLP, (b) C22.
- "Bandwidth" = achieved/processed throughput = (packets successfully processed per
  sec) × (payload bytes/packet) × 8 / 1000. **Ground truth = FPGA-side NetLayer
  counters (`app_in`/`eth_in` deltas), NOT the sender's ACK count** (ACKs read 0 due
  to a slot-14 routing quirk — see §5). Lock the exact metric definition early and
  make CPU + FPGA apples-to-apples.

---

## 3. Current state (as of 2026-07-04, verified)

**Hardware (this is the working setup — details in `oct_cluster_state_2026_05_19.md`
bottom section):**
- **pc160 = FPGA host.** ssh `rdave009@pc160.cloudlab.umass.edu`. BDF `0000:3b:00.1`.
  Data NIC `enp135s0f0 @ 192.168.40.30`. FPGA CMAC IP `192.168.40.10`, MAC
  `02:00:00:00:0a:0a`. Card shelled/HEALTHY.
- **pc161 = sender.** ssh `rdave009@pc161.cloudlab.umass.edu`. `enp135s0f0 @
  192.168.40.31`, MAC `f8:f2:1e:d3:26:b0`. Has `throughput.py` + static ARP to FPGA.
- pc158/pc168 DEAD. pc170/pc171 stuck GOLDEN (`903f`, unflashable in-band — needs OCT
  admin / `config-fpga boot <RSA cloudlab.pem>` interactively). Do not sink time there.
- Build host = **Wolverine** (this machine, `/home/rdave009/minirocket-hls`, Vitis 2023.2).

**Already deployed + working on pc160:** `v3-dpr` (MR DP-Reuse + bFP, 250 MHz, len-192
cap) serving inference over the network — cross-node smoke confirmed
CMAC→NetLayer→kernel→responses (`app_in`/`udp_out` increment). Tooling ported to
pc160 `~/`: `krnl.xclbin.v3-dpr`, `dpr_minirocket_model_len150_fixed.json`, and
compiled `setup_netlayer`, `read_netlayer_counters_ext`, `load_minirocket_hbm`,
`tweak2` (batch). Deploy script `~/deploy_v3dpr_pc160.sh`; source copies in
`MiniRocketHLS/saturation_harness/`.

---

## 4. The two workstreams

### A. Full saturation sweep on pc160/161 (bank a clean data point)
- Sweep rates 500,1k,2k,5k,10k,25k,50k,100k,250k,500k,1M,2M,5M pps from pc161 via
  `throughput.py` (`--abort-loss-pct 200` to disable the bogus ACK auto-abort).
- Ground truth = `read_netlayer_counters_ext` deltas on pc160 (`eth_in`,`app_in`,
  `udp_out`) per rate. Loss = 1 − app_in_delta/sent.
- Expect a cliff ~2k and CMAC **wedge ~5k** (eth_in→0). After any rate ≥2k, RECOVER:
  `xbutil reset --user` + reprogram v3-dpr (mandatory).
- Persist CSV to pc160 `~/network_experiments/TP/results/tp_<date>/` + update memory.
- Orchestrator note: adapt `saturation_harness/dpr_sweep_orchestrate.sh` for
  pc160/pc161; **counter reads MUST `source /opt/xilinx/xrt/setup.sh`** (else
  `libxrt_coreutil.so.2` load error). Run the deploy/sweep **detached in tmux**.

### B. The chart-enabling work (the substantive path)
1. **CPU baseline (red curve) — FPGA-independent, do this in parallel with everything.**
   Build a CPU MiniRocket-inference-over-UDP responder (run on pc161 or a spare host),
   drive it with the same `throughput.py` sweep, measure bandwidth-vs-rate. Find where
   the CPU flattens (NIC/host ceiling). This is required regardless of FPGA state.
   (Note: memory has a raw CPU↔CPU forwarding baseline ~292k pps, but that's *not*
   inference-in-the-loop — the red curve needs the actual inference responder.)
2. **#2 Kill the CMAC wedge (FPGA must flatten gracefully, never collapse to 0).**
   The `pktDropper` is supposed to drop excess so CMAC RX never wedges, yet v3-dpr
   still wedged at 5k. Investigate pktDropper integration / NetLayer RX FIFO depth /
   backpressure; fix so the FPGA curve plateaus at its compute ceiling instead of
   crashing. Likely a kernel/config.cfg change → ~3h v++ rebuild on Wolverine.
3. **Loader start-once / true inline** — `load_minirocket_hbm` uses a host `start/wait`
   per-packet spin (caps ~5-10k for non-BUILD=1 kernels). `BUILD=1` kernels already
   free-run inline (start fires once, wait blocks). Replace the spin with a clean
   start-once so there's zero ambiguity that the FPGA path is host-free. **Every
   variant we chart MUST be `BUILD=1`** — audit each xclbin.
4. **Fast variant for a dramatic gap** — DP-Reuse is slow (~1.3k inf/s → barely clears
   the CPU). Use a FAST `BUILD=1` inline variant (HYDRA fixed-point ≈6.3k, or fused
   ≈3.8k inf/s) as the headline FPGA curve; keep naïve/bFP/DP-Reuse as ablation points.
   HYDRA source: `hydra_optimized/hydra_axis/`; fused: recover per
   `dpr_source_recovery` memory. Confirm timing closes (watch the CMAC `txoutclk`
   system clock — congestion there failed v3-dpr's first build; fixed via smaller
   conv map + 250 MHz kernel — see `dpr_source_recovery` memory).
5. **Bandwidth metric + plotting scaffolding** — lock the Kbps formula + packet
   payload size, build the plot (matplotlib) matching Brisk's mock (log-log, CPU
   red squares flattening, FPGA green circles climbing), per dataset.

**Suggested order:** dispatch **B1 (CPU baseline)** and **A (sweep)** in parallel
immediately (independent). Then **B3 (loader) + B2 (wedge fix)** feed a rebuild;
then **B4 (fast variant)** headline curve; **B5 (plot)** last. Gate every `xbutil
program` / hw build on user ack (Rule 2/7).

---

## 5. Critical gotchas (save subagents from re-discovering these)

- **Counters are ground truth, not ACKs.** FPGA responses misroute (slot-14 HW quirk
  → dest `0.0.0.16`), so `throughput.py` ACK count reads 0 / "100% loss" — ignore it;
  use `read_netlayer_counters_ext` deltas (or AF_PACKET raw RX if you need host-side).
- **Bug-B is a HOST-side fix**, not the kernel sideband: `fill_all_slots` (all 16
  SocketTable slots valid) makes TX work. The kernel `out_pkt.dest/user/id/strb`
  writes are constant-folded to no-ops. Slots 14-15 mangle (HW) — ~6 readback
  mismatches on high slots are EXPECTED/harmless.
- **XRT tools need `source /opt/xilinx/xrt/setup.sh`** or die on `libxrt_coreutil.so.2`.
- **CMAC recovery after wedge:** `xbutil reset --user` + reprogram. `setup_netlayer`
  sets gateway to the network addr (.0) → ARP storm; MUST overwrite reg `0x1C` with
  the peer IP (`tweak2 <xclbin> <bdf> 1C C0A8281F` for pc161/.31).
- **`v3-dpr` MAX_TIME_SERIES_LENGTH=192** → model series length must be ≤192. Binary
  sklearn models store ONE coef vector → pad `classifier_coef` to `[num_classes][nf]`
  or the loader throws "classifier_coef size mismatch".
- **networklayer.xo / cmac_0.xo must be same-Vitis-version** as the link (2023.2) —
  `[v++ 17-70]` forward-incompat otherwise. Build the whole flow on Wolverine.
- Deploy scripts: run **detached in tmux** (ssh timeout kills a foreground deploy).
  `tweak2` is BATCH now (one device-open, all writes) — don't regress to per-call.

---

## 6. First actions for the orchestrator

1. Load the two superpowers skills (§0) + confirm they're present (if a skill is
   missing, STOP and tell the user to install the `superpowers` plugin).
2. Dispatch a subagent to digest the memory files (§0) and return a tight state map.
3. Verify hardware live: subagent checks pc160 (`xbutil examine` device 0000:3b:00.1,
   loader tmux) + pc161 reachable, both `source setup.sh` first.
4. Present a short plan (which subagents, parallel vs sequential, what needs user ack)
   and confirm the metric definition + which dataset(s) for the headline chart, THEN
   dispatch A + B1 in parallel. Ask the user before any `xbutil program` or v++ build.

Deliverable: the bandwidth-vs-rate chart(s) with CPU (flattening) vs FPGA (climbing),
plus the CSVs/scripts to regenerate them, ready for the FCCM resubmission.
