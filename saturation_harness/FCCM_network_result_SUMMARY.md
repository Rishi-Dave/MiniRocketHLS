# FCCM Network-Saturation Result — Summary (2026-07-06/07)

## Headline
A fused MiniRocket inference kernel doing **streaming inference in-fabric over
the network** sustains **0% packet loss to ≥300k pps (~160 Mbps)** — and to
**~700k pps** with a C sender — while a CPU MiniRocket responder over the same
NIC sustains loss-free inference only to **~5k pps**.

**Decisive gap (max loss-free ingestion rate): ~60× (Python sender) to ~140×
(C sender), and the FPGA is not yet saturated** (limited by the sender, not the
kernel — compute ceiling is millions of pps).

**★ UPDATE 2026-07-09 — FPGA-to-FPGA achieved (no host anywhere):** a second U280
(pc161) generating packets *in-fabric* drives the fused DUT to **≥2.24M pps at 0%
loss** — the true kernel-compute ceiling, sender-limit removed. That is **~448×**
the CPU-MiniRocket loss-free ceiling (~5k pps) and 3× the host C-sender's 745k. See
the "FPGA-to-FPGA" section below.

## The numbers (per dataset, ≤1% loss = "sustainable")
| Dataset | FPGA fused (0% loss) | CPU MiniRocket (last ≤1% loss) | gap |
|---|---|---|---|
| InsectSound (L=600, 10-cls) | ≥312k pps / 160 Mbps (700k via C sender) | ~5k pps | ~60–140× |
| FruitFlies (L=5000, 3-cls) | ≥305k pps / 156 Mbps | ~5k pps | ~60× |
| MosquitoSound (L=3750, 6-cls) | ≥300k pps / 154 Mbps | ~5k pps | ~60× |

FPGA processed EVERY packet the Python sender generated (0% loss, all rates,
all datasets); the C sender pushed InsectSound to 700k pps still at 0% loss.
CPU `processed_pps` keeps climbing under overload (Python recv-loop draining the
socket buffer) but with escalating loss (50–88% at high rates) — so the honest
metric is the max loss-free rate (~5k), not the inflated overload throughput.

## Winning FPGA variant
`fpga-network/minirocket_fused/` v4 = **fused CONV+PPV MiniRocket, BUILD=1
free-run, packed single-packet emit, DATAFLOW ingest/compute decouple**.
Built 250 MHz, timing met (WNS +0.016ns). xclbin: `build_dir.hw.fused_v4/`.
Key fixes that got it working (3 rebuilds): packed single-packet emit (fixed a
NetLayer-TX free-run deadlock) + BUILD=1 free-run (removed the single-shot
host-spin ~35k cap). csim bit-exact vs HW golden on all datasets.

## The chart (`runs/tp_2026-07-06_final/plots/bandwidth_vs_pps.{png,svg}`)
Reflects the **C-sender** FPGA sweep (`runs/tp_2026-07-06_csender/fused_csender_sweep.csv`)
— FPGA green curve climbs to ~740–746k pps / ~378–382 Mbps at **0% loss** on all
3 datasets vs CPU red curve going loss-free only to ~5k pps → **~148–149× gap**.
(An earlier version used the Python sweep, which topped at ~312k = the Python
sender's limit; the C-sender version is the accurate, stronger picture.)

## Files to regenerate everything
- FPGA C-sender sweep (CHART SOURCE): `c_sender_sweep_orchestrate.sh` → `runs/tp_2026-07-06_csender/fused_csender_sweep.csv` (0% loss to ~745k)
- FPGA Python sweep (to ~312k, sender-limited): `fused_v4_sweep_orchestrate.sh` → `runs/tp_2026-07-06_fused_v4/fused_v4_sweep.csv`
- CPU baseline: `cpu_sweep_3ds_orchestrate.sh` → `runs/tp_2026-07-06_cpu/cpu_minirocket_3ds.csv`
- C high-rate sender: `hi_rate_sender.c` (pc161; MUST use `--src-port 62178` to match NetLayer socket table)
- Charts: `plot_bandwidth.py --fpga-csv <csender csv> --cpu-csv <cpu csv> --out <dir>`
- Deploy: `deploy_fused_pc160.sh <model.json>`; loader `~/load_minirocket_hbm_fused` on pc160.

## ★ FPGA-to-FPGA (line rate, "no host anywhere") — ACHIEVED 2026-07-09
The stretch goal is DONE. pc161's U280 generates UDP packets **in-fabric**
(`fpga-network/udp_generator/`, hw-built, free-run) → CMAC TX → wire → pc160's
U280 CMAC RX → NetLayer → fused MiniRocket kernel — **zero CPU/NIC/host on either
side.** Rate sweep (`runs/tp_2026-07-09_f2f/ceiling.csv`, pc160 `app_in` =
kernel-processed):

| rate_divider | pc161 gen pps | pc160 app_in pps | loss |
|---|---|---|---|
| 100 | 664k | 664k | 0% |
| 65 | 1.02M | 1.05M | 0% |
| 40 | 1.64M | 1.64M | 0.2% |
| **30** | **2.17M** | **2.24M** | **0%** |
| ≤22 | ≥2.9M offered | 0 | wedge (compute ceiling) |

**Headline: fused DUT sustains ≥2.24M pps @ 0% loss FPGA-to-FPGA** (pc161 TX side
proven to ≥6.1M pps — never the limit). That is **3× the host C-sender's 745k**
(sender-limited) and **~448× the CPU-MiniRocket loss-free ceiling (~5k)**. The
host-sender bottleneck is removed; the new limit is the fused kernel's in-fabric
compute (~2.24–2.9M pps).

Root cause that had blocked originated FPGA TX: `setup_netlayer`'s UDP socket-table
offsets were stale (`kernel.xml` 4-byte stride, +0x10 off). Real HLS map (from
`xudp_hw.h`): **8-byte stride** — `theirIP=0x810+8i`, `theirPort=0x890+8i`,
`myPort=0x910+8i`, `valid=0x990+8i`. `udpTxEngine` silently drops any packet whose
`SocketTable[TDEST].valid!=1`; with wrong offsets `valid` was never set where the
engine reads. (setup_netlayer.cpp/.py still carry the wrong offsets on disk — patch
pending; use tweak2 corrected-offset writes meanwhile.)

## Caveats / future work
- CPU baseline is a Python responder (recv-loop-bound, socket-buffer-drain
  artifact under overload). A C/optimized CPU responder would give a cleaner
  flat ceiling; the max-loss-free-rate comparison (~5k) is the defensible number.
- Host-sender path was sender-limited (Python ~312k, C sender ~700k on the 40G host
  NIC). The FPGA-to-FPGA path above removes that limit entirely.
- FPGA-to-FPGA wedge >~2.24M pps: pc160 NetLayer RX stalls stickily on overrun;
  recover by redeploying the fused DUT between high-rate points. Fixing the fused
  ingest to drop-on-full (like pktDropper) instead of block would convert the wedge
  into graceful loss and let the sweep run past the compute ceiling cleanly.

## SATURATION KNEE (2026-07-09, drop-on-full ingest) — the clean FCCM figure
The fused ingest was rebuilt drop-on-full (guarded stream write at
`minirocket_fused.cpp:370`; 250 MHz, WNS +0.016ns), converting the sticky >2.24M-pps
wedge into graceful loss. FPGA-to-FPGA knee sweep (`f2f_knee_sweep.sh`, no host
anywhere), InsectSound L=600, 64B payloads:

| offered (pps) | processed (pps) | loss | bandwidth |
|---|---|---|---|
| 137k – 2.217M | = offered | 0% | up to 1.135 Gbps |
| 2.98M | 2.749M | 7.9% | 1.407 Gbps |
| 4.29M | 2.749M | 35.9% | 1.407 Gbps |
| 6.24M | 2.748M | 56.0% | 1.407 Gbps |
| 9.80M | 2.746M | 72.0% | 1.406 Gbps |
| 17.1M | 2.746M | 84.0% | 1.406 Gbps |
| 34.3M | 2.750M | 92.0% | 1.408 Gbps |

**Compute ceiling: ~2.75M pps ≈ 1.41 Gbps, dead-flat across a 11.5× overload range,
zero wedges.** Loss-free to 2.22M pps. Ceiling = ~550× the CPU loss-free rate (~5k).
Data: `runs/tp_2026-07-09_f2f_knee/fused_dropfull_knee.csv` (+ `_plotfmt.csv`);
figure: `runs/tp_2026-07-09_f2f_knee/bandwidth_vs_pps.{png,svg}` (plot_bandwidth.py,
PAYLOAD_BYTES=64). Sweep notes: cap at div=3 (34M-pps dwell at div≤1 re-wedges the
NetLayer RX itself, and div=0 wedges the generator); link bring-up/recovery recipe in
memory `network_bandwidth_2026_07_04.md` (xbutil --force program + gtreset ordering).

## ALL-3-DATASET F2F KNEE (2026-07-10) — per-dataset compute ceilings, no host anywhere
Re-ran the drop-on-full F2F knee for every dataset (models swapped on pc160 via
`deploy_dropfull_f2f.sh <model>`; sweep extended down to 5k pps so the ENTIRE FPGA
curve is F2F-measured, replacing the C-sender low-range points):

| dataset | L | 0%-loss through | processed plateau (samples/s) | plateau bandwidth | vs CPU@500k |
|---|---|---|---|---|---|
| InsectSound   | 600  | 2.16M pps | ~2.69M | ~1.38 Gbps | 73× |
| MosquitoSound | 3750 | 6.24M pps | ~7.31M | ~3.74 Gbps | 58× |
| FruitFlies    | 5000 | 6.24M pps | ~7.45M | ~3.81 Gbps | 55× |

**New insight: the compute ceiling in samples/s is NOT L-independent — longer series
amortize per-window overhead** (3750/5000-sample windows sustain ~2.7× the samples/s
of 600-sample windows). Loss-free-rate gap vs CPU (~5k pps ≤1% loss): 432×–1249×.
Data: `runs/tp_2026-07-09_f2f_knee/fused_dropfull_knee_<dataset>.csv` + `sweep_<dataset>.log`.
Figures (reference-style, common 5k-pps start): `fccm_dominance_<dataset>.{png,svg}` +
combined `fccm_dominance_panels.{png,svg}` (`plot_dominance_fccm.py`).

## HYDRA F2F KNEE (2026-07-10) — hydra_axis v6 (v2_fixed-port), all 3 datasets
New network-attached HYDRA kernel: the v2_fixed compute core (ap_fixed<32,16>,
UNROLL=16 dilation-sorted batches, CONV_POOL II=1) ported into the hydra_axis
NetLayer wrapper with the minirocket_fused drop-on-full DATAFLOW ingest +
packed guarded emit. Built @300 MHz, WNS +0.003ns, 0 errors (3h37m link,
`hydra_optimized/hydra_axis/build_dir.hw/krnl.xclbin` → pc160
`~/krnl.xclbin.hydra_dropfull`). csim accuracy = float parity (IS 70.40%,
MS 73.43% vs float 74.83%, FF 89.51% = float exactly; dilation-sort +
per-feature permutation done by `load_hydra_hbm`). Loader spin holds; deploy
via `~/deploy_hydra_f2f.sh <hydra_model.json>`.

| dataset | L | 0%-loss through | processed plateau (samples/s) | plateau bandwidth | vs CPU-HYDRA@500k |
|---|---|---|---|---|---|
| InsectSound   | 600  | 4.24M pps | ~5.39M | ~2.76 Gbps | 49× |
| MosquitoSound | 3750 | 6.17M pps | ~7.88M | ~4.03 Gbps | 66× |
| FruitFlies    | 5000 | 6.01M pps | ~8.0M  | ~4.16 Gbps | 70× |

CPU-HYDRA baseline (`runs/tp_2026-07-10_cpu_hydra/cpu_hydra_3ds.csv`,
hydra_udp_responder over the host NICs): ≤1%-loss ceilings only 2k–10k pps.
HYDRA-FPGA in-fabric rate (5.39M/600 ≈ 8,990 inf/s on IS) even beats the
host-driven v2_fixed m_axi build (6,326 inf/s) — no per-inference dispatch.
Data: `runs/tp_2026-07-09_f2f_knee/hydra_dropfull_knee_<ds>.csv` +
`sweep_hydra_<ds>.log`; figures `fccm_dominance_hydra_<ds>.{png,svg}` +
`fccm_dominance_hydra_panels.{png,svg}` (`plot_dominance_fccm.py hydra`).

## MINIROCKET v5 F2F KNEE (2026-07-11) — facc_t classifier @300 MHz, retakes the lead
The fused kernel's only II defect was its float ridge-classifier accumulation
(FEATURE_LOOP II=4 = 33.6k of ~55.8k cycles/window on IS). v5 accumulates in
ap_fixed<48,24> (float multiplies kept) → FEATURE_LOOP II=1, and links at
300 MHz. Gates before deploy: csim 5/5 argmax parity; csynth all-II=1,
est 2.777 ns; free-run RTL cosim PASS (4/4 windows drained, bit-exact, no
stall); routed WNS **+0.014 ns @300 MHz, 0 failing endpoints** (4h21m link,
`fpga-network/build_dir.hw.fused_dropfull300/krnl.xclbin` → pc160
`~/krnl.xclbin.fused_dropfull300`; deploy `~/deploy_dropfull300_f2f.sh`).

| dataset | L | 0%-loss through | plateau (samples/s) | plateau bw | vs v4-250 | vs HYDRA v6 |
|---|---|---|---|---|---|---|
| InsectSound   | 600  | 4.30M pps | ~5.67M | ~2.90 Gbps | 2.11× | 1.05× |
| MosquitoSound | 3750 | 9.63M pps | ~9.64M | ~4.94 Gbps | 1.32× | 1.22× |
| FruitFlies    | 5000 | 6.21M pps | ~9.33M | ~4.78 Gbps | 1.25× | 1.17× |

MiniRocket leads HYDRA on all 3 datasets again, as predicted by the cycle
model (the two kernels' conv engines were already comparable; the gap was
clock + float classifier). Data: `fused_dropfull300_knee_<ds>.csv` +
`sweep_mr300_<ds>.log`. Headline figures `fccm_dominance_*` now use v5 data
(shared-scale panels); v4-250 figures regenerable via
`plot_dominance_fccm.py minirocket250`.

## MINIROCKET v6 F2F (2026-07-11) — UNROLL=84, one sweep per dilation: 11+ Gbps, generator-limited
UNROLL 28→84 (all 84 kernels per cycle; conv sweeps 27→9/window; weights are
±1/0/2 constants so the widening is adder/LUT area, 694 DSP / 147k LUT).
Gates: csim 5/5 bit-identical; csynth all-II=1 est 2.777ns; freerun RTL cosim
PASS; routed WNS 0.000 @300 MHz, 0 failing endpoints (4h46m link,
`build_dir.hw.fused_dropfull300u84` → pc160 `~/krnl.xclbin.fused_dropfull300u84`,
deploy `~/deploy_dropfull300u84_f2f.sh`).

| dataset | knee / ceiling | plateau bw | note |
|---|---|---|---|
| InsectSound   | 6.17M pps 0-loss; plateau ~8.78M samples/s | ~4.50 Gbps | classifier-bound (next: class-parallel ridge) |
| MosquitoSound | 0% loss to 16.95M; ~21.7M @4.6% loss (div=2) | **~11.1 Gbps** | generator-limited |
| FruitFlies    | 0% loss to 17.37M; ~22.0M @3.6% loss (div=2) | **~11.2 Gbps** | generator-limited |

**div=2 discovery:** the in-fabric generator's true max emit is ~22.8M pps
(not 33M); a 4s div=2 dwell did NOT wedge the dropfull+pktDropper RX —
div=2 is now in the standard sweep list (div<=1 and div=0 remain forbidden).
On MS/FF the DUT ingests everything the generator can offer at <5% loss —
the measurement rig, not the kernel, is now the bottleneck (~23M samples/s
compute ceiling per the cycle model). Headline charts `fccm_dominance_*` = v6
(gaps 239×/172×/161× vs CPU); v5 via plot variant `minirocket_v5`.

## HYDRA u64 F2F (2026-07-12) — UNROLL=64 @200 MHz (300/250 failed routed timing)
Batch width 16→64 (one batch per dilation; shared 9-tap window so widening
costs DSPs only, 2699 used). Timing saga: OOC @300 failed WNS −0.644; full
link @250 failed WNS −0.706 (39k endpoints, NEVER deployed); relink @200 MHz
**passed: WNS +0.016, 0 failing** (4h26m). Deploy `~/deploy_hydra_u64_f2f.sh`,
xclbin pc160 `~/krnl.xclbin.hydra_dropfull_u64`.

| dataset | 0%-loss through | plateau | bw | vs hydra u16@300 |
|---|---|---|---|---|
| InsectSound   | 4.19M pps | ~5.95M samples/s | 3.05 Gbps | 1.10× |
| MosquitoSound | 9.57M pps | ~12.4M samples/s | 6.36 Gbps | 1.57× |
| FruitFlies    | 9.59M pps | ~13.4M samples/s | 6.85 Gbps | 1.67× |

Charts `fccm_dominance_hydra_*` now use u64 data (gaps 56×/104×/116× vs
CPU-HYDRA; u16 variant regenerable via plot variant `hydra_u16`). Lesson:
wide ap_fixed MAC arrays route ~200-250MHz on this design; MiniRocket's
multiplier-free 84-wide closes 300 — algorithm structure sets the
width-vs-clock frontier.
