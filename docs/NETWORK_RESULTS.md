# FPGA-to-FPGA Network-Saturation Results

**The decisive FCCM result:** an Alveo U280 ingests UDP inference traffic directly
at the 100G CMAC — no NIC, no host, no PCIe round-trip on either the packet-generation
or the inference side — and out-throughputs a CPU inference server by two to three
orders of magnitude. This document is the single current synthesis of that result.
It supersedes the chronological session log
(`saturation_harness/FCCM_network_result_SUMMARY.md`) and several stale
handoff/planning docs (see "Reproduction pointers" below for pointers, kept in
place but marked superseded).

## TL;DR

| Variant | Dataset | Samples/s (plateau) | Gbps | Limiting factor | vs CPU | Source CSV |
|---|---|---|---|---|---|---|
| MiniRocket fused v6 (UNROLL=84, 300 MHz, WNS 0.000) | InsectSound (L=600) | 8.78M | ~4.50 | **classifier-bound** (genuine compute limit) | **239×** | `fused_dropfull300u84_knee_InsectSound.csv` |
| MiniRocket fused v6 | MosquitoSound (L=3750) | 21.7M (@4.6% loss) | ~11.1 | **generator-limited** (not FPGA ceiling) | **172×** | `fused_dropfull300u84_knee_MosquitoSound.csv` |
| MiniRocket fused v6 | FruitFlies (L=5000) | 22.0M (@3.6% loss) | ~11.2 | **generator-limited** (not FPGA ceiling) | **161×** | `fused_dropfull300u84_knee_FruitFlies.csv` |
| HYDRA hydra_axis u64 (UNROLL=64, 200 MHz — 300/250 failed timing) | InsectSound | 5.95M | ~3.05 | compute-bound | **~50×** (sources vary 49–56×; recompute before publication) | `hydra_dropfull_u64_knee_InsectSound.csv` |
| HYDRA hydra_axis u64 | MosquitoSound | 12.4M | ~6.36 | compute-bound | **104×** | `hydra_dropfull_u64_knee_MosquitoSound.csv` |
| HYDRA hydra_axis u64 | FruitFlies | 13.4M | ~6.85 | compute-bound | **116×** | `hydra_dropfull_u64_knee_FruitFlies.csv` |

All FPGA CSVs live in `saturation_harness/runs/tp_2026-07-09_f2f_knee/`. "vs CPU"
uses the CPU's own peak *overloaded* throughput at 500k configured pps (a
generous baseline for the CPU — see "CPU baselines" below); the gap against the
CPU's clean, loss-free rate (~5k pps) is far larger still.

**Why the numbers aren't all "the FPGA's ceiling":** InsectSound's short
600-sample windows make the ridge classifier the bottleneck — a genuine
fixed-hardware compute limit. MosquitoSound/FruitFlies have longer windows that
amortize per-window overhead, so the DUT keeps up with everything the in-fabric
packet generator can offer (measured generator ceiling ≈22.8M pps); those two
numbers are a floor, not the FPGA's true ceiling.

## Why this matters (FCCM pitch)

Every other network-attached inference comparison in this space (including
prior work this project is answering) has the CPU or GPU pull packets off a
NIC, up through the kernel network stack, into user space, then back down
again to respond. This project's FPGA path never leaves the fabric: a second
U280 generates the UDP inference stream **in-fabric** (no host packet
generator), it crosses the wire at 100G, and the DUT's CMAC → NetLayer →
compute pipeline classifies it **in-fabric**, with no CPU or NIC anywhere in
the loop on either side.

The PI's framing: *"the network experiments is where we can most definitively
beat either a CPU or a GPU because the CPU/GPU need to go through a NIC and we
don't."* The measurements here **refine, not just confirm**, that framing: the
CPU baselines' knees are set by inference *compute* cost (5k pps for MiniRocket,
lower still for HYDRA under a 1% loss bar), which is well below the raw
NIC/host packet-handling ceiling (~292k pps measured separately, see
`network_bandwidth_2026_07_04.md`). So the win documented here is **compute
in-fabric beating compute-on-CPU**, not merely "skipping a NIC" — the NIC was
never actually the CPU's bottleneck in these experiments.

## Experimental setup

- **Topology:** cross-node CloudLab/OCT (`octfpga-PG0`), two Alveo U280s on one
  switched LAN (VLAN 2501, 100G-capable). **pc161** = generator FPGA (in-fabric
  UDP packet generator, free-running). **pc160** = inference FPGA (fused
  MiniRocket / HYDRA DUT). Earlier phases (superseded by the F2F result) used a
  host-driven sender from pc158/pc168.
- **Stack (both nodes):** 100G CMAC → NetLayer UDP stack → `pktDropper`
  (drop-on-full ingest, the fix that converts a hard NetLayer wedge into a
  graceful loss curve — see caveats) → per-variant inference kernel.
- **Generator:** `fpga-network/udp_generator/`, hw-built, free-run, emits at a
  rate controlled by `rate_divider` (nominal formula ≈66.5M/`rate_divider`;
  true measured max emit ≈22.8M pps — see caveats).
- **Loaders:** `load_minirocket_hbm` / `load_hydra_hbm` push per-dataset model
  weights (scaler + ridge/HYDRA coefficients) into the DUT's HBM banks before
  each sweep; HYDRA's loader additionally dilation-sorts kernels and permutes
  features to match the batched compute core.
- **Metrics:** FPGA-side NetLayer counters, `app_in`/`eth_in` deltas across a
  fixed dwell window (**not** `cmac_status`, which had a buggy register map
  before 2026-07-08 and read all-zero on working links). `bandwidth_kbps =
  processed_pps × 64 × 8 / 1000` (locked 64B-payload metric, one payload byte
  per time-series sample).
- **Figures:** `fccm_dominance_*.png/svg` in
  `saturation_harness/runs/tp_2026-07-09_f2f_knee/` (regenerated 2026-07-13
  from the v6 / u64 CSVs), produced by `saturation_harness/plot_dominance_fccm.py`.

## Results

### MiniRocket fused v6 (UNROLL=84, multiplier-free, 300 MHz, WNS 0.000)

0%-loss knee and overload plateau per dataset (full sweeps in the CSVs):

| Dataset | L | 0%-loss through | Plateau (samples/s) | Loss at plateau | Bandwidth | Note |
|---|---|---|---|---|---|---|
| InsectSound | 600 | 6.17M pps | 8.78M | 9.44% | ~4.50 Gbps | classifier-bound (10-class × 840-feature ridge is the next unroll target) |
| MosquitoSound | 3750 | 16.95M pps | 21.7M | 4.6% | ~11.1 Gbps | generator-limited (0% loss through 17M offered) |
| FruitFlies | 5000 | 17.37M pps | 22.0M | 3.6% | ~11.2 Gbps | generator-limited (0% loss through 17M offered) |

Source: `fused_dropfull300u84_knee_{InsectSound,MosquitoSound,FruitFlies}.csv`.
InsectSound's classifier bottleneck is real hardware saturation (verified in
the HLS schedule report); the MS/FF numbers are capped by the in-fabric
generator's own emission ceiling (~22.8M pps), not by the DUT — the true
MiniRocket compute ceiling on long series is unmeasured (above ~22M samples/s).

### HYDRA hydra_axis u64 (UNROLL=64, 200 MHz)

300 MHz and 250 MHz builds both **failed routed timing** (OOC @300 MHz: WNS
−0.644 ns; full link @250 MHz: WNS −0.706 ns, 39k failing endpoints, never
deployed) — wide `ap_fixed` MAC arrays route poorly at this width. The 200 MHz
relink passed (WNS +0.016 ns, 0 failing endpoints).

| Dataset | L | 0%-loss through | Plateau (samples/s) | Bandwidth | vs CPU-HYDRA@500k |
|---|---|---|---|---|---|
| InsectSound | 600 | 4.19M pps | 5.95M | ~3.05 Gbps | ~50× (**see caveat below**) |
| MosquitoSound | 3750 | 9.57M pps | 12.4M | ~6.36 Gbps | 104× |
| FruitFlies | 5000 | 9.59M pps | 13.4M | ~6.85 Gbps | 116× |

Source: `hydra_dropfull_u64_knee_{InsectSound,MosquitoSound,FruitFlies}.csv`.
MiniRocket's multiplier-free 84-wide core still leads HYDRA on all three
datasets, and closes 300 MHz where HYDRA's wide MAC arrays cannot — algorithm
structure, not just tuning, sets the width-vs-clock frontier here.

**Multiplier discrepancy (flagged per instructions, not papered over):**
recomputing InsectSound's gap directly from the CSVs
(`5,951,250 / 109,598.33` samples/s) gives **54.3×**, not the 49× or 56×
figures that appear at different points in the chronological log. All three
numbers are in the same ballpark and none of them changes the qualitative
result (>1 order of magnitude), but the exact multiplier should be
recalculated from source before this goes into the paper. State it as
**"~50× (sources vary 49–56×)"** until resolved.

### CPU baselines

**CPU-MiniRocket** (Python UDP responder, `runs/tp_2026-07-06_cpu/cpu_minirocket_3ds.csv`):
loss-free (0% loss) at 5,000 configured pps on InsectSound and MosquitoSound;
FruitFlies shows 0.98% loss at the same rate (still effectively at the knee).
Beyond that, `processed_pps` keeps climbing under overload — but with
escalating loss: at 500k configured pps, processed throughput reaches
36.8k/126.2k/136.0k pps (IS/MS/FF) at **59–88% loss**. That inflated overload
number is what the TL;DR's "vs CPU" column divides into, which is the more
conservative (CPU-favorable) comparison; using the honest loss-free ~5k figure
instead would make every FPGA multiplier roughly an order of magnitude larger.
A distinct, older ~14k pps CPU figure from an earlier session used a
different, shorter (len-150) synthetic model — **do not conflate it with these
3-dataset, real-dataset-length results.**

**CPU-HYDRA** (`runs/tp_2026-07-10_cpu_hydra/cpu_hydra_3ds.csv`): no clean flat
knee — loss is already present at 5k pps on Mosquito/FruitFlies (1.7%/2.1%)
and grows steadily with rate (e.g. 58.5–70.3% loss at 500k configured pps,
processed 104k–120k pps). Report processed-pps-with-loss% at the cited rate
rather than claiming a loss-free ceiling.

### Progression: v4 → v5 → v6 (MiniRocket), u16 → u64 (HYDRA)

| Build | Clock | Key change | InsectSound 0-loss / plateau | Status |
|---|---|---|---|---|
| v4 dropfull (250 MHz) | 250 MHz | drop-on-full ingest fix (first knee-shaped, non-wedging sweep) | 2.16M / ~2.69M samples/s | superseded — `fused_dropfull_knee_InsectSound.csv` |
| v5 (300 MHz, `facc_t` classifier) | 300 MHz | ridge-classifier accumulator moved to `ap_fixed<48,24>`, FEATURE_LOOP II 4→1 | 4.30M / ~5.67M samples/s | superseded — `fused_dropfull300_knee_InsectSound.csv` |
| **v6 (300 MHz, UNROLL=84)** | 300 MHz | UNROLL 28→84 (multiplier-free, ±1/0/2 weights → adder/LUT area, not DSPs) | 6.17M / **8.78M samples/s** | **current headline** — `fused_dropfull300u84_knee_InsectSound.csv` |

| Build | Clock | Key change | InsectSound 0-loss / plateau | Status |
|---|---|---|---|---|
| hydra_axis u16 (v6, `v2_fixed` port) | 300 MHz | first network-attached HYDRA, ap_fixed<32,16>, UNROLL=16, dilation-sorted | 4.24M / ~5.39M samples/s | superseded — `hydra_dropfull_knee_InsectSound.csv` |
| **hydra_axis u64** | 200 MHz (300/250 failed timing) | batch width 16→64 (widening costs DSPs only, shared 9-tap window) | 4.19M / **5.95M samples/s** | **current headline** — `hydra_dropfull_u64_knee_InsectSound.csv` |

An earlier intermediate result, the host-C-sender-limited MiniRocket v4 (before
the F2F generator existed): 0% loss to ~745k pps ≈ 148–149× vs CPU — sender
limited, not compute-limited; retained only as a sanity checkpoint, not a
headline number.

**DP-Reuse (`v3-dpr`), honest negative result:** a len-150-model DP-Reuse
variant plateaus at ~694 pps and **loses to the CPU baseline** (~14k pps on
that model). Its cliff lands at the same point as the naive (non-DP-Reuse)
variants tested against it — evidence the cliff is a CMAC RX FIFO/backpressure
limit, not a compute limit, so DP-Reuse's area savings bought nothing on this
metric. This is a paper-grade negative finding, not a bug to hide. (Source:
memory `network_bandwidth_2026_07_04.md`, no dedicated CSV — this predates the
per-dataset knee-sweep CSV format.)

## Honest caveats & known limitations

- **Samples/s ceiling is NOT series-length-independent.** Longer windows
  amortize per-window overhead; MosquitoSound/FruitFlies plateaus run ~2.7×
  higher samples/s than InsectSound's on the same hardware. Do not quote a
  single "FPGA ceiling" number across datasets without this caveat.
- **Knee vs. wedge.** Pre-drop-on-full builds *hard-wedged* the NetLayer RX
  path above their compute ceiling (a sticky failure requiring reprogram, not
  a measurable data point). The drop-on-full ingest fix
  (`minirocket_fused.cpp:370`, guarded non-blocking stream write) converts
  that cliff into a graceful, F2F-measurable loss curve. Always cite the
  loss% curve, not a single "it wedged here" cliff.
- **Pre-2026-07-05 network numbers are invalid.** `fpga-network/common/build.hpp`'s
  `BUILD=1` free-run gate was never actually wired into any build (`-DBUILD=1`
  was in no build script), so early kernels ran host-loop-limited (bounded by
  XRT CU-autorestart re-triggering, not true free-run). Any throughput number
  from before that date should be treated as a host-dispatch-overhead
  measurement, not a compute ceiling.
- **DP-Reuse negative result stands.** ~694 pps, loses to CPU (~14k pps on the
  same len-150 model), cliffs at the same point as naive variants — the
  bottleneck is the CMAC RX FIFO, not compute; DP-Reuse's resource savings did
  not translate to a throughput win in this harness.
- **HYDRA u64 needed 200 MHz.** Wide `ap_fixed` MAC arrays (UNROLL=64 batches)
  route poorly on this shell at 300/250 MHz (routed WNS −0.644/−0.706 ns);
  200 MHz was the first passing build (WNS +0.016 ns). MiniRocket's
  multiplier-free 84-wide core closes 300 MHz on the same shell — algorithm
  structure, not effort, sets the achievable width-vs-clock point.
- **The in-fabric generator is the current measurement ceiling for
  MosquitoSound/FruitFlies**, not the DUT. Measured generator max emit ≈22.8M
  pps (not the naive 66.5M/`rate_divider` formula, which breaks down at small
  divider values). `rate_divider` ≤1 and `=0` are known to wedge the generator
  itself; `div=2` was validated safe for sustained 4s dwells and is the
  highest-rate point in the standard sweep.
- **HYDRA InsectSound multiplier is unresolved** — see the flagged 49×/54×/56×
  discrepancy above; recompute from the CSVs before publication.
- **CPU "vs CPU" multipliers in the TL;DR use the CPU's peak overloaded
  throughput (500k configured pps, 59–88% loss)**, not its clean loss-free
  rate (~5k pps). This is the more conservative, CPU-favorable comparison —
  real, but it understates the gap relative to a fair "both loss-free" metric.

## Reproduction pointers

- Rate sweeps: `saturation_harness/f2f_knee_sweep.sh` (`VARIANT=minirocket84`
  → v6 headline CSVs; `VARIANT=hydra64` → HYDRA u64 headline CSVs;
  `VARIANT=minirocket300`/`minirocket250`/`hydra` reproduce the superseded
  generations). `DATASET=<InsectSound|MosquitoSound|FruitFlies>` selects the
  dataset; `SKIP_DEPLOY=1` reuses an already-loaded xclbin.
- CPU baselines: `saturation_harness/cpu_sweep_3ds_orchestrate.sh`
  (MiniRocket) and `saturation_harness/cpu_hydra_3ds_orchestrate.sh` (HYDRA).
- Figures: `saturation_harness/plot_dominance_fccm.py <variant>` where
  `<variant>` ∈ `{minirocket, minirocket_v5, minirocket250, hydra, hydra_u16}`
  (default `minirocket` = v6 headline). Outputs
  `fccm_dominance_<prefix>_<dataset>.{png,svg}` and a combined `*_panels.*` into
  `saturation_harness/runs/tp_2026-07-09_f2f_knee/`.
- Deploy scripts (pc160): `~/deploy_dropfull300u84_f2f.sh` (MiniRocket v6),
  `~/deploy_hydra_u64_f2f.sh` (HYDRA u64); generator control on pc161 via
  `saturation_harness/start_gen`.
- Link recovery recipe (force-reprogram + GT reset ordering, socket-table
  offsets, generator wedge conditions): the "DEFINITIVE F2F LINK RECOVERY
  RECIPE" section of `saturation_harness/FCCM_network_result_SUMMARY.md` and
  memory file `network_bandwidth_2026_07_04.md`.
- `saturation_harness/runbook.md` is **Phase-1 vintage** (April 2026,
  pre-F2F) — useful for harness layout/history, not for current numbers or
  procedure; this document is the current state.

## Document provenance

- Written: 2026-07-13.
- Sources (numbers pulled only from these, each cited inline above):
  - `saturation_harness/FCCM_network_result_SUMMARY.md` (2026-07-12 chronological log)
  - `saturation_harness/runs/tp_2026-07-09_f2f_knee/fused_dropfull300u84_knee_{InsectSound,MosquitoSound,FruitFlies}.csv` (MiniRocket v6 headline)
  - `saturation_harness/runs/tp_2026-07-09_f2f_knee/hydra_dropfull_u64_knee_{InsectSound,MosquitoSound,FruitFlies}.csv` (HYDRA u64 headline)
  - `saturation_harness/runs/tp_2026-07-09_f2f_knee/fused_dropfull_knee_InsectSound.csv` and `fused_dropfull300_knee_InsectSound.csv` (v4/v5 progression)
  - `saturation_harness/runs/tp_2026-07-06_cpu/cpu_minirocket_3ds.csv` (CPU-MiniRocket baseline)
  - `saturation_harness/runs/tp_2026-07-10_cpu_hydra/cpu_hydra_3ds.csv` (CPU-HYDRA baseline)
  - memory `network_bandwidth_2026_07_04.md` (DP-Reuse negative result, BUILD=1 gap, F2F recovery recipe context)
  - `saturation_harness/plot_dominance_fccm.py` (variant/CSV mapping, verified against the above)
- Verification method: every headline number above was independently
  recomputed from the raw CSV rows (not just copied from the prose log) —
  offered/processed pps, loss%, bandwidth_kbps, and vs-CPU ratios were
  recalculated and cross-checked against the chronological log's claims.
  **One discrepancy found:** HYDRA InsectSound's vs-CPU multiplier is
  inconsistent across sources (49× / 54.3× recomputed / 56×) — flagged above,
  not silently resolved.
