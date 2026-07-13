# MiniRocketHLS

**FPGA acceleration of convolution-based time series classification (MiniRocket, HYDRA) on Xilinx Alveo U280, including a fully in-fabric, host-free FPGA-to-FPGA inference path.**

[![Platform](https://img.shields.io/badge/Platform-Xilinx_Alveo_U280-orange.svg)](https://www.xilinx.com/products/boards-and-kits/alveo/u280.html)
[![HLS](https://img.shields.io/badge/Vitis_HLS-2023.2-green.svg)](https://www.xilinx.com/products/design-tools/vitis/vitis-hls.html)

This repository accompanies "FPGA Acceleration of Convolution-based Classification
Algorithms for Streaming Time Series" (Dave, Yuvaraj, Brisk — UC Riverside), FCCM
resubmission. Sources: `PaperTexFiles/`.

---

## Three results, one repo

1. **Host-dispatched accelerators (FCCM architecture story).** MiniRocket and
   HYDRA inference kernels on a single U280, PCIe host-dispatched, benchmarked
   against CPU and GPU baselines. Current lead: **MiniRocket fused CONV+PPV,
   1-CU, 3,797 inf/s** on GunPoint (397.6 MHz Fmax, WNS +0.007ns, 300 MHz
   achieved), 0 DSP (ternary weights), 185 BRAM (-68% vs the pre-fusion
   kernel). See [docs/RESULTS.md](docs/RESULTS.md).
2. **The fusion optimization.** Fusing the convolution and PPV-feature stages
   into a single pass — eliminating an intermediate `convolutions[16][8192]`
   array — turns out to matter more than any other single change in the
   MiniRocket line: 6.6-10.9x throughput over the pre-fusion (`v16_fixed`)
   kernel, and the speedup *grows* with series length because the fused
   kernel's loop bound is dynamic. Detail: [docs/RESULTS.md](docs/RESULTS.md#fusion-impact).
3. **Network F2F saturation (the decisive result).** Two U280s, wire-to-wire:
   an in-fabric UDP packet generator on one card feeds a 100G CMAC link
   straight into the CMAC -> NetLayer -> inference pipeline on the other —
   no NIC, no host, no PCIe round-trip on either side. MiniRocket fused v6
   sustains up to **~22M samples/s (≈11.2 Gbps, generator-limited)**; HYDRA
   u64 up to **13.4M samples/s (≈6.85 Gbps)**; the CPU baseline's loss-free
   knee is ~5k pps. Full writeup, caveats, and source CSVs:
   **[docs/NETWORK_RESULTS.md](docs/NETWORK_RESULTS.md)**.

---

## Architecture: the conv-engine ablation ladder

The accelerator variants in this repo form a deliberate ablation, each step
isolating one design decision (see `PaperTexFiles/03_architecture.tex` for
the full argument):

| Step | Variant dir | What it isolates |
|---|---|---|
| Naive + FP (floating point) | `reference_1to1/` | Paper-faithful baseline, full-precision arithmetic |
| Naive + bFP (ternary, no-DSP) | `optimized_version/` | -1/0/+1 kernel weights remove the multiplier from the convolution |
| DP-Reuse + bFP (temporal carry-forward) | `minirocket_modular/` (DP-Reuse build) | Reuses partial sums across dilations; **honest negative result on the network path** — plateaus at ~694 pps, same cliff as naive, because the bottleneck there is the CMAC RX FIFO, not compute (see `docs/NETWORK_RESULTS.md`) |
| Fused CONV+PPV | `minirocket_modular/` (fused build) | Single-pass convolution + feature extraction, 0 DSP, current MiniRocket throughput leader |
| HYDRA fixed-point | `hydra_optimized/` | Same ablation path applied to HYDRA's dictionary-kernel algorithm, `ap_fixed<32,16>` |

---

## Variant status (validated numbers, host-dispatched thread)

| Variant | Best config | Throughput | Status | Source |
|---|---|---|---|---|
| `reference_1to1` | 1-CU, 242 MHz | 45 inf/s | Baseline (Dec 2025) | `results/results_master.csv` |
| `optimized_version` | 4-CU, 404 MHz | 3,468 inf/s (77x vs. CPU) | Historical — superseded by fused kernel below | `results/results_master.csv`; historical framing, see [docs/RESULTS.md](docs/RESULTS.md#superseded) |
| `minirocket_modular` (v16_fixed, pre-fusion) | 1-CU, 300 MHz | 67.9-507.5 inf/s (dataset-dependent) | Superseded by fused kernel | `results/all_data_for_sheets.csv` |
| `minirocket_modular` (**fused CONV+PPV**) | 1-CU, 300 MHz | **3,797 inf/s** (GunPoint) | **Current MiniRocket lead** — timing met (WNS +0.007ns) | `results/minirocket_fused_results.csv` |
| `minirocket_modular` (fused, 2-CU) | 2-CU, 300 MHz | 5,999 inf/s (GunPoint) | Timing met (WNS +0.001ns) | `results/minirocket_fused_results.csv` |
| `minirocket_modular` (fused, 3-CU) | 3-CU, target 300 MHz | 9,585 inf/s (GunPoint) | **Timing FAILED** (WNS -0.380ns), auto-throttled to 269.3 MHz | `results/minirocket_fused_summary.txt` |
| `hydra_optimized` (v2_fixed, `ap_fixed<32,16>`) | 1-CU, 300 MHz | 6,326 inf/s (InsectSound, 69.41% acc, 70.2x vs CPU) | Hardware-validated | `results/hydra_v2_fixed_benchmarks.json` |
| `hydra_optimized` (v2_fixed, 2-CU) | 2-CU, 300 MHz | 8,028 inf/s (InsectSound) | Timing met | `results/all_data_for_sheets.csv` |
| `hydra_optimized` (v2_fixed, 3-CU) | 3-CU, target 300 MHz | 10,573 inf/s (InsectSound) | **Timing FAILED**, auto-throttled | `results/all_data_for_sheets.csv` |
| `hydra_optimized/hydra_axis` | 1-CU, 200 MHz (300/250 MHz failed timing) | see [docs/NETWORK_RESULTS.md](docs/NETWORK_RESULTS.md) | Network-attached HYDRA (F2F thread) | `docs/NETWORK_RESULTS.md` |
| `hydra_stream` | 3-kernel AXI-Stream, 404.86 MHz | 23-193 inf/s | Architectural demo only — per-sample OpenCL dispatch bound, not compute-bound; accuracy delta 0.00% vs Python | `results/all_data_for_sheets.csv` |
| `fpga-network/minirocket_fused` | Network F2F, UNROLL=84, 300 MHz (WNS 0.000) | up to ~22M samples/s ≈ 11.2 Gbps | **Headline network result** | [docs/NETWORK_RESULTS.md](docs/NETWORK_RESULTS.md) |
| `multirocket_optimized` / `multirocket_stream` / `multirocket_fp_optimization` | — | — | **DROPPED 2026-04-08** — host loader bug ran hardware against an all-zero model; all prior accuracy/throughput numbers are invalid. See retraction banners in those directories' READMEs. | `MULTIROCKET_FIXES_SUMMARY.md`, banners added in commit `7f58b99` |

GPU comparison (Tesla T4, batch=256 vs. FPGA 3-CU): 6.1x on InsectSound
(L=600), narrowing to 1.5x on MosquitoSound/FruitFlies (L=3750/5000) — the
GPU's batching advantage shrinks as series get longer. At batch=1, GPU drops
to 55-57 inf/s (HYDRA; PyTorch per-call dispatch overhead across grouped
`conv1d` invocations), so the FPGA wins 26-113x at batch=1. Power: U280
25.7W (static-dominated, <5% fabric utilization, measured via `xbutil
examine --report electrical`), GPU 54-68W (`nvidia-smi`), CPU 90W TDP per
socket (dual-socket Xeon E5-2640 v3, single-threaded workload attributed to
one socket; no RAPL/sudo access for a measured figure — treat as an upper
bound favoring the CPU). Detail and full tables: [docs/RESULTS.md](docs/RESULTS.md).

---

## Repository map

```
MiniRocketHLS/
├── README.md                  # This file
├── docs/                      # Canonical documentation
│   ├── RESULTS.md             #   Host-dispatched thread: consolidated results tables
│   ├── NETWORK_RESULTS.md     #   Network F2F thread: the decisive result (read this)
│   ├── ALGORITHM.md, FPGA_IMPLEMENTATION.md, FILE_STRUCTURE.md   # supporting detail
│   └── DOCUMENTATION_INDEX.md, DOCUMENTATION_SUMMARY.md          # legacy indices (predate 2026 reorg)
├── reference_1to1/             # Paper-faithful 1:1 MiniRocket baseline
├── optimized_version/          # Ternary-weight (-1/0/+1), no-DSP MiniRocket (Dec-2025 story)
├── minirocket_modular/         # DP-Reuse and fused-CONV+PPV MiniRocket builds (current lead)
├── minirocket_stream/          # AXI-Stream MiniRocket pipeline (architectural demo)
├── hydra_optimized/            # HYDRA fixed-point (v2_fixed) + hydra_axis/ (network-attached HYDRA)
├── hydra_stream/               # AXI-Stream HYDRA pipeline (architectural demo)
├── multirocket_optimized/      # DROPPED — invalid results, retraction banners in place
├── multirocket_stream/         # DROPPED — see multirocket_optimized banner
├── multirocket_fp_optimization/# DROPPED — see multirocket_optimized banner
├── fpga-network/               # NetLayer stack: minirocket_fused (F2F kernel), pktDropper
│                                #   (drop-on-full ingest fix), udp_generator (in-fabric sender),
│                                #   NetLayers/, Ethernet/ (CMAC integration)
├── saturation_harness/         # Rate-sweep harness (rate_sweep.py, f2f_knee_sweep.sh) + runs/
│                                #   (per-run JSONL, CSVs, plots — source of NETWORK_RESULTS.md numbers)
├── cpu/, cpu-network/          # C++ / Python CPU baselines (host-dispatched and network)
├── scripts/                    # Figure/table generators, power measurement, log parsers
├── results/                    # results_master.csv, all_data_for_sheets.csv, per-dataset
│                                #   per-sample CSVs, JSON benchmark dumps — primary data source
├── paper-results/               # results_summary.csv, CPU logs feeding the paper tables
├── PaperTexFiles/               # FCCM paper source (fccm.tex, section .tex files, figures)
└── fp_optimization/              # Stale doc copies — see docs/ for canonical content
```

---

## Quick start

**Hardware:** Xilinx Alveo U280 (`xcvu9p-flga2104-2-i`). **Toolchain:** Vitis
HLS / Vitis 2023.2, XRT 2023.2.

Each variant directory has its own `Makefile`. Do not use ad-hoc build
scripts — build targets are:

```bash
cd <variant_dir>            # e.g. minirocket_modular, hydra_optimized
make TARGET=hw_emu          # hardware emulation (~1 hour)
make TARGET=hw              # full hardware build (hours — see that variant's build log for timing)
```

For the network (F2F) variants, see `fpga-network/` and the reproduction
pointers in [docs/NETWORK_RESULTS.md](docs/NETWORK_RESULTS.md#reproduction-pointers)
(`saturation_harness/f2f_knee_sweep.sh`, per-variant deploy scripts).

Training and CPU baselines: `cpu/`, `hydra_optimized/scripts/train_hydra.py`,
`scripts/` (figure/table generation from `results/`).

---

## Documentation index

| Doc | Covers |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | Host-dispatched results tables: MiniRocket variants, HYDRA, GPU comparison, power |
| [docs/NETWORK_RESULTS.md](docs/NETWORK_RESULTS.md) | FPGA-to-FPGA network saturation — the decisive FCCM result |
| [docs/ALGORITHM.md](docs/ALGORITHM.md) | MiniRocket algorithm and FPGA-specific optimizations |
| [docs/FPGA_IMPLEMENTATION.md](docs/FPGA_IMPLEMENTATION.md) | HLS pragmas, HBM banking, timing closure |
| `saturation_harness/FCCM_network_result_SUMMARY.md` | Chronological network-result session log (superseded by NETWORK_RESULTS.md as the current synthesis) |
| `multirocket_optimized/README.md` | MultiRocket retraction banner and what went wrong |

---

## Citation

```bibtex
@inproceedings{dempster2021minirocket,
  title={MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification},
  author={Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={248--257},
  year={2021}
}
```

FCCM paper source: `PaperTexFiles/fccm.tex`.

## License

Apache License 2.0 — see `LICENSE`.
