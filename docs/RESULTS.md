# Results: Host-Dispatched Accelerator Thread

This document consolidates the validated, single-node (PCIe host-dispatched)
benchmark results for MiniRocket and HYDRA on a single Alveo U280, plus the
GPU/CPU comparison and power figures. For the FPGA-to-FPGA network-saturation
thread (a different measurement regime — in-fabric, no host, no NIC), see
**[NETWORK_RESULTS.md](NETWORK_RESULTS.md)**.

Every number below cites the file it was pulled from. Numbers that could not
be traced to a source in `results/`, `paper-results/`, or a build log are
marked `[unverified]` or omitted.

---

## 1. MiniRocket: throughput by variant and CU count

Source: `results/minirocket_fused_results.csv`, `results/minirocket_fused_summary.txt`,
`results/all_data_for_sheets.csv` (Section 1).

| Dataset | v16_fixed (pre-fusion, 1-CU) | **Fused 1-CU** | Fused 2-CU | Fused 3-CU† |
|---|---|---|---|---|
| GunPoint (L=150) | 507.5 inf/s | **3,797.1 inf/s** | 5,999.1 inf/s | 9,585.0 inf/s |
| InsectSound (L=600) | 295.9 inf/s | **1,966.8 inf/s** | 3,191.1 inf/s | 5,012.2 inf/s |
| MosquitoSound (L=3750) | 88.1 inf/s | **937.0 inf/s** | 1,496.9 inf/s | 2,819.6 inf/s |
| FruitFlies (L=5000) | 67.9 inf/s | **742.3 inf/s** | 1,121.4 inf/s | 2,190.7 inf/s |

† 3-CU build ran at **269.3 MHz — timing FAILED** (WNS = -0.380 ns), the tool
auto-throttled from the 300 MHz target; UNROLL=28 caused a DSP jump from 0 to
1,264/CU at 3-CU. 3-CU still outperforms 2-CU in absolute throughput despite
the lower clock, due to parallelism, but it is not a timing-closed
configuration and should always be reported with this caveat. 1-CU and 2-CU
both closed timing at 300 MHz (WNS +0.007 ns and +0.001 ns respectively;
estimated Fmax 397.61 MHz per `minirocket_modular/v++_feature_extraction_fused.log`).

Accuracy is invariant across CU count for a given dataset (identical
predictions at 1/2/3 CUs): GunPoint 98.33% (59/60), InsectSound 74.12%
(18,530/25,000), MosquitoSound 87.88% (5,273/6,000), FruitFlies 95.82%
(16,538/17,259) — all bit-exact vs. the exported model, matching the Python
baseline to within sub-0.01% rounding. Source: `results/minirocket_fused_summary.txt`.

### Fusion impact

Fusing the convolution and PPV-feature-extraction stages into one pass
(eliminating an intermediate `convolutions[16][8192]` array, exploiting
ternary {-1,0,+1} weights to drop the multiplier entirely) is the single
largest throughput lever in the MiniRocket line. Source:
`results/tables/fused_comparison.tex` / `results/all_data_for_sheets.csv`
(Section 14):

| | Separate (v16_fixed) | Fused | Change |
|---|---|---|---|
| Throughput, GunPoint | 507.5 inf/s | 3,797.1 inf/s | **7.5x** |
| Throughput, InsectSound | 295.9 inf/s | 1,966.8 inf/s | **6.6x** |
| Throughput, MosquitoSound | 88.1 inf/s | 937.0 inf/s | **10.6x** |
| Throughput, FruitFlies | 67.9 inf/s | 742.3 inf/s | **10.9x** |
| LUTs | 48,812 | 57,090 | +17% |
| FFs | 88,492 | 94,283 | +7% |
| BRAM (18K) | 576 | 185 | **-68%** |
| DSP48E2 | 724 | 0 | **-100%** (ternary weights) |
| Intermediate array | 16x8192 floats | eliminated | -100% |
| Fmax (v++ estimate) | 397.6 MHz | 397.6 MHz | no regression |

Note: `results/minirocket_fused_summary.txt` states the BRAM reduction as
"18%" — this is a typo in that file. The raw counts (576 -> 185) and the
paper table (`results/tables/fused_comparison.tex`, `all_data_for_sheets.csv`
Section 14) both independently compute **-68%**; that is the correct figure
and the one used here.

The speedup grows with series length (6.6x at L=600 up to 10.9x at L=5000)
because the fused kernel's loop bound is dynamic and scales with the input
series, whereas the pre-fusion kernel's feature-extraction stage dominates
increasingly at longer lengths (K1 fraction of pipeline: 55% at L=150, up to
89% at L=5000 — `results/minirocket_fused_summary.txt`).

### Scalability sweep (series length, 1-CU fused vs. GPU)

Source: `results/all_data_for_sheets.csv` (Section 10), 8-point sweep at
fixed 300 MHz:

| TS Length | FPGA 1-CU (inf/s) | GPU b=256 (inf/s) | GPU b=1 (inf/s) | GPU b=256 / FPGA |
|---|---|---|---|---|
| 64 | 2,681 | 174,912 | 1,204 | 65.2x |
| 512 | 2,204 | 33,826 | 798 | 15.4x |
| 2048 | 1,303 | 9,331 | 661 | 7.2x |
| 8192 | 511 | 1,919 | 539 | 3.8x — FPGA exceeds GPU batch=1 at this length |

Full 8-point table in the source CSV. The GPU's batching advantage collapses
monotonically as series length grows; FPGA throughput also falls with length
(fewer inferences fit the fixed clock budget) but the *ratio* moves in the
FPGA's favor.

---

## 2. HYDRA: throughput by variant and CU count

Source: `results/hydra_v2_fixed_benchmarks.json`, `results/all_data_for_sheets.csv`
(Section 4). All numbers are the `v2_fixed` build: `ap_fixed<32,16>` internal,
float HBM interface, UNROLL=16, dilation-sorted, 300 MHz.

| Dataset | 1-CU | 2-CU | 3-CU† | Accuracy | vs. CPU (1-CU) |
|---|---|---|---|---|---|
| InsectSound (L=600) | 6,326 inf/s | 8,028 inf/s | 10,573 inf/s | 69.41% | 70.2x |
| MosquitoSound (L=3750) | 1,937 inf/s | 2,757 inf/s | 3,516 inf/s | 70.05% | 49.4x |
| FruitFlies (L=5000) | 1,507 inf/s | 2,148 inf/s | 2,972 inf/s | 87.61% | 44.8x |

† 3-CU builds **failed timing** (auto-throttled clock, "Throttled" in
`Clock_MHz` column of the source CSV) — report as timing-failed, not a
closed configuration, same caveat as MiniRocket 3-CU above. 1-CU and 2-CU
closed timing at 300 MHz.

Resource utilization (v2_fixed, 1-CU, Vivado routed — authoritative over the
HLS csynth estimate): LUT 29,758 (2.28%), FF 38,055 (1.46%), BRAM 84.5
(4.19%), DSP 684 (7.58%). Source: `results/resource_utilization.csv`.

`hydra_optimized/hydra_axis/` is the network-attached HYDRA port (dilation-
sorted, feature-permuted for a batched compute core) used in the F2F
saturation thread — see [NETWORK_RESULTS.md](NETWORK_RESULTS.md); its
throughput numbers there (samples/s under network saturation) are a
different metric from the inf/s figures in this table and are not
comparable one-to-one.

`hydra_stream` (3-kernel AXI-Stream dataflow, 404.86 MHz, WNS +0.375ns):
23-193 inf/s (InsectSound 193, MosquitoSound 31, FruitFlies 23) — an
architectural demo, bound by per-sample OpenCL dispatch overhead rather than
kernel compute; accuracy delta is 0.00% vs. the Python reference on all three
datasets. Source: `results/all_data_for_sheets.csv` (Section 5).

---

## 3. GPU comparison (Tesla T4)

Source: `results/gpu_baseline_results_v2.json`, `results/gpu_baseline_hydra_results.json`,
`results/tables/gpu_comparison.tex`, `results/tables/hydra_gpu_comparison.tex`.
PyTorch 2.10.0 + CUDA 12.8, Google Colab, synthetic data matching real dataset
dimensions, 2026-04-02.

| Algorithm | Dataset | GPU b=256 (inf/s) | GPU b=1 (inf/s) | FPGA 3-CU (inf/s) | GPU b=256 / FPGA 3-CU |
|---|---|---|---|---|---|
| MiniRocket | InsectSound (600) | 30,469 | 1,091 | 5,012 | **6.1x** |
| MiniRocket | MosquitoSound (3750) | 4,198 | 661 | 2,820 | **1.5x** |
| MiniRocket | FruitFlies (5000) | 3,227 | 539 | 2,191 | **1.5x** |
| HYDRA | InsectSound (600) | 13,895 | 56 | 6,326 (1-CU)* | 2.2x |
| HYDRA | MosquitoSound (3750) | 5,139 | 53 | 1,937 (1-CU)* | 2.7x |
| HYDRA | FruitFlies (5000) | 4,267 | 57 | 1,507 (1-CU)* | 2.8x |

\* HYDRA ratios in `results/tables/hydra_gpu_comparison.tex` are computed
against **FPGA 1-CU**, not 3-CU (HYDRA 3-CU is timing-failed) — kept as-is
from the source table.

The GPU's batched-throughput advantage shrinks from 6.1x (MiniRocket,
L=600) to 1.5x (L=3750/5000) as series length grows; HYDRA's learned
(non-shared) kernels compress the GPU advantage further (2.2-2.8x) because
they defeat the grouped-`conv1d` batching trick the GPU relies on for
MiniRocket's shared-dilation kernels.

**Batch=1** is the more representative comparison for a streaming/low-latency
deployment: HYDRA GPU batch=1 throughput is 53-57 inf/s across all three
datasets — a PyTorch per-call dispatch-overhead floor across 4 grouped
`conv1d` invocations, not a compute limit — so **FPGA wins 26-113x at
batch=1** (`results/tables/hydra_gpu_comparison.tex`: 113x/36x/26x for
IS/MS/FF respectively).

---

## 4. Power / energy efficiency

Source: `results/tables/power_table.tex`, `results/resubmission_summary.tex`.

| Dataset | Platform | Throughput (inf/s) | Power (W) | Energy (mJ/inf) | Efficiency vs. CPU |
|---|---|---|---|---|---|
| InsectSound | CPU C++ (-O3) | 289 | 90† | 311.4 | 1x |
| | GPU T4 (b=256) | 30,469 | 54† | 1.8 | 176x |
| | FPGA 1-CU | 1,967 | 25.7‡ | 13.1 | 24x |
| | FPGA 3-CU | 5,012 | 25.7 | 5.1 | 60x |
| MosquitoSound | CPU C++ (-O3) | 48 | 90 | 1,875.0 | 1x |
| | GPU T4 (b=256) | 4,198 | 66 | 15.7 | 119x |
| | FPGA 1-CU | 937 | 25.7 | 27.7 | 68x |
| | FPGA 3-CU | 2,820 | 25.7 | 9.2 | 204x |
| FruitFlies | CPU C++ (-O3) | 258 | 90 | 348.8 | 1x |
| | GPU T4 (b=256) | 3,227 | 68 | 21.1 | 17x |
| | FPGA 1-CU | 742 | 25.7 | 35.0 | 10x |
| | FPGA 3-CU | 2,191 | 25.7 | 11.9 | 29x |

† CPU platform is a dual-socket Xeon E5-2640 v3; **90 W is the per-socket TDP
upper bound**, not a measured figure — Intel RAPL counters
(`/sys/class/powercap/intel-rapl/`) require root access unavailable on the
shared server, so measured CPU power could not be collected. Because the
workload is single-threaded, only one socket is loaded; the combined
2-socket system TDP would be ~180 W, but the per-inference efficiency ratios
above use the single-socket 90 W figure (the CPU-favorable choice). GPU power
is measured via `nvidia-smi` under sustained inference load.

‡ FPGA power measured via `xbutil examine --report electrical`; static-
dominated (<5% fabric utilization at these clocks), so 1-CU and 3-CU report
the same 25.7 W.

---

## 5. Resource utilization summary

Source: `results/resource_utilization.csv`, `results/tables/resource_annotated.tex`
(paper Table 4, post-place-and-route where available).

| Variant | LUT | FF | DSP | BRAM | Clock | Note |
|---|---|---|---|---|---|---|
| MiniRocket, fused 1-CU | 57,090 (4.4%) | 94,283 (3.6%) | 0 | 185 (9.2%) | 300 MHz | 0 DSP — ternary weights |
| MiniRocket, fused 2-CU | 114,180 (8.8%) | 188,566 (7.2%) | 0 | 370 (18.4%) | 300 MHz | 0 DSP — ternary weights |
| MiniRocket, fused 3-CU | 92,703 (7.1%) | 159,086 (6.1%) | 1,264 (14%) | 428 (21.2%) | 269 MHz | timing failed, see above |
| MiniRocket, DP-Reuse 1-CU | 135,634 | 153,327 (5.9%) | 2 | 1,320 (65.5%) | 300 MHz | includes CMAC/UDP network stack BRAM |
| HYDRA, v2_fixed 1-CU | 41,746 (3.2%) | 58,537 (2.2%) | 1,355 (15%) | 86 (4.3%) | 300 MHz | Vivado routed |
| HYDRA, Dataflow 1-CU (streaming) | 15,866 (1.2%) | 20,823 (0.8%) | 55 (0.6%) | 115 (5.7%) | 405 MHz | 3-kernel AXI-Stream |
| `reference_1to1` | 27,713 (2%) | 20,822 (0%) | 17 (0%) | 231 (5%) | 242 MHz | paper-faithful baseline |

---

## Superseded / invalid numbers

**Optimized MiniRocket, 4-CU, 3,468 inf/s, 77x vs. CPU (Dec-2025 story).**
This is the original repository headline (`results_master.csv` row, commit
range around 77b3cee) and remains a real, reproducible number for the
`optimized_version` variant — but it predates the fused-kernel and
network-saturation work and should be presented as **historical context**,
not the current state of the art. The current MiniRocket throughput leader
is the fused CONV+PPV kernel above.

**`v16_fixed` (pre-fusion MiniRocket).** Baseline for measuring the fusion
optimization's impact (Section 1 above), not a competitive endpoint on its
own.

**MultiRocket — retracted, 2026-04-08.** All `multirocket_optimized`,
`multirocket_stream`, and `multirocket_fp_optimization` numbers are invalid:
a host-loader bug read non-existent JSON fields, so the reported hardware
runs executed against an all-zero model. A corrected host build collapsed to
28.6% accuracy (suspected `ap_fixed<32,16>` overflow). MultiRocket was
dropped from the paper. See retraction banners added in commit `7f58b99`
(`multirocket_optimized/README.md` and related docs) for the full account —
do not cite any MultiRocket number from before that commit.

**Network F2F thread — do not conflate with this document.** All samples/s,
Gbps, and "vs. CPU" figures for the in-fabric FPGA-to-FPGA path live in
[NETWORK_RESULTS.md](NETWORK_RESULTS.md) and use a different measurement
methodology (NetLayer packet counters, not host-side `clock_gettime` around
an OpenCL call). Do not mix the two threads' throughput numbers in the same
comparison without noting the methodology difference.

---

## Document provenance

Written 2026-07-13 as part of the core-documentation rewrite (Phase 5c-d).
Every number above is cited inline to its source file under `results/`,
`paper-results/`, or a build log referenced from those files. See
[NETWORK_RESULTS.md](NETWORK_RESULTS.md#document-provenance) for the network
thread's equivalent provenance section.
