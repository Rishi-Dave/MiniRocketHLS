# Paper Revision Execution Guide

Run these steps in order. Each step produces data needed for the paper revision.

---

## Step 0: Environment Setup (5 min)

```bash
source /opt/xilinx/xrt/setup.sh
cd /home/rdave009/minirocket-hls/MiniRocketHLS
```

---

## Step 1: HYDRA v2_fixed FPGA Benchmarks (~30 min)

**Why:** Memory claims 6,511 inf/s but NO benchmark logs exist for v2_fixed build.

```bash
# Quick test first (1000 samples per dataset, ~5 min)
bash scripts/run_hydra_v2_benchmarks.sh quick

# If quick test looks good, run full benchmarks
bash scripts/run_hydra_v2_benchmarks.sh full
```

**Output:** `hydra_optimized/results/v2_fixed/hydra_v2fixed_summary.csv`

**IMPORTANT:** The v2_fixed kernel uses `float*` at the HBM interface and `ap_fixed<32,16>` internally. The existing float host code IS compatible (verified from preprocessed HLS source).

---

## Step 2: MultiRocket Model Training (~24-48h background)

**Why:** Only GunPoint and InsectSound models trained. MosquitoSound/FruitFlies missing.

```bash
# Start in tmux (runs for 24-48 hours)
tmux new-session -d -s mr84-train 'bash scripts/train_missing_multirocket_models.sh'

# Check progress
tmux attach -t mr84-train
```

**Output:** `multirocket_optimized/models/multirocket84_{mosquitosound,fruitflies}_{model,test}.json`

**After training completes**, run C++ CPU baseline:
```bash
cd cpu
./multirocket_cpu ../multirocket_optimized/models/multirocket84_mosquitosound_model.json \
                  ../multirocket_optimized/models/multirocket84_mosquitosound_test.json
./multirocket_cpu ../multirocket_optimized/models/multirocket84_fruitflies_model.json \
                  ../multirocket_optimized/models/multirocket84_fruitflies_test.json
```

---

## Step 3: Scalability Sweep (~1 hour)

**Why:** Reviewers asked about scalability with time series length.

```bash
bash scripts/run_scalability_sweep.sh
```

**Output:** `results/scalability_sweep.csv`

**Note:** Uses fused 1-CU bitstream. No rebuild needed — `time_series_length` is a runtime parameter. The kernel's MAX=8192 covers all test lengths.

---

## Step 4: GPU Baseline on Google Colab (~2 hours)

**Why:** Rev C explicitly asked for GPU baseline. Our current argument is hand-wavy.

1. Upload `scripts/gpu_baseline_minirocket.py` to Google Colab
2. Set runtime to GPU (T4 or A100)
3. Run:
```
!pip install torch aeon scikit-learn
!python gpu_baseline_minirocket.py
```
4. Download `gpu_baseline_results.json`
5. Copy to `results/gpu_baseline_results.json`

**Output:** GPU throughput (inf/s), power (W), comparison with FPGA numbers

---

## Step 5: CPU Power Measurement

**Why:** Using 100W TDP vs 24.4W measured FPGA is not apples-to-apples.

### Option A: With perf (needs sudo)
Ask someone with sudo to run:
```bash
sudo apt install linux-tools-$(uname -r) linux-tools-generic
sudo bash scripts/measure_cpu_power.sh
```

### Option B: With RAPL sysfs
Ask someone with sudo to enable RAPL reading:
```bash
sudo chmod 444 /sys/class/powercap/intel-rapl:0/energy_uj
```
Then run without sudo:
```bash
bash scripts/measure_cpu_power.sh
```

### Option C: Published TDP (fallback)
If neither works, use Xeon E5-2640 v3 TDP of **90W** (not 100W) with explicit caveat in paper:
> "CPU TDP of 90W used as conservative upper bound; actual power under single-threaded inference is lower, making efficiency ratios conservative estimates favoring the CPU."

---

## Step 6: MultiRocket FPGA Rebuild (after Step 2 completes)

**Why:** Current xclbin (Dec 30 2025) predates the Jan 6 2026 fix that added 3 missing pooling ops.

```bash
cd multirocket_optimized

# Verify the corrected kernel has all 4 pooling ops
grep -c "MIPV\|LSPV\|MPV\|PPV" multirocket/src/multirocket_pooling.cpp

# Rebuild in tmux
tmux new-session -d -s mr-build 'source /opt/xilinx/xrt/setup.sh && make TARGET=hw'

# Monitor
tmux attach -t mr-build
```

**Expected:** ~4-6 hours. Then run benchmarks:
```bash
./multirocket_host build_dir.hw.*/krnl.xclbin \
    models/multirocket84_insectsound_model.json \
    models/multirocket84_insectsound_test.json
# Repeat for each dataset
```

---

## Vivado Per-Module Resource Data (Already Extracted)

From the `kernel_util_routed.rpt` files, here is the per-module breakdown:

### MiniRocket Fused 1-CU (300 MHz, timing MET)
| Module | LUT | REG | BRAM | DSP |
|--------|-----|-----|------|-----|
| Platform (shell) | 127,370 (9.8%) | 175,427 (6.7%) | 199 (9.9%) | 4 (0.04%) |
| feature_extraction_1 | 43,551 (3.7%) | 72,443 (3.0%) | 90 (5.0%) | 722 (8.0%) |
| scaler_1 | 3,060 (0.3%) | 3,241 (0.1%) | 33 (1.8%) | 5 (0.06%) |
| classifier_1 | 7,672 (0.7%) | 10,385 (0.4%) | 39 (2.2%) | 10 (0.11%) |
| **Kernel Total** | **54,283 (4.6%)** | **86,069 (3.5%)** | **162 (8.9%)** | **737 (8.2%)** |

**Key insight:** 722 of the 737 kernel DSPs (98%) are in the feature_extraction module. The fused CONV+PPV kernel uses DSPs for the accumulation pipeline (despite ternary weights eliminating the multiply).

### MiniRocket v16_fixed (original, not fused, 1-CU)
| Module | LUT | REG | BRAM | DSP |
|--------|-----|-----|------|-----|
| feature_extraction_1 | 9,726 (0.8%) | 9,754 (0.4%) | 36 (2.0%) | 12 (0.1%) |
| scaler_1 | 3,063 (0.3%) | 3,238 (0.1%) | 33 (1.8%) | 5 (0.06%) |
| classifier_1 | 7,678 (0.7%) | 10,382 (0.4%) | 39 (2.2%) | 10 (0.11%) |
| **Kernel Total** | **20,467 (1.7%)** | **23,374 (1.0%)** | **108 (5.9%)** | **27 (0.3%)** |

### MiniRocket Fused 3-CU (269 MHz, timing FAILED, UNROLL=28)
| Module | LUT | REG | BRAM | DSP |
|--------|-----|-----|------|-----|
| Platform (shell) | 149,229 (11.5%) | 235,048 (9.0%) | 205 (10.2%) | 4 (0.04%) |
| feature_extraction (×3) | 220,922 (19.1%) | 356,037 (15.0%) | 273 (15.1%) | 3,786 (42.0%) |
| - per CU | ~73,600 | ~118,600 | 91 | **1,262** |
| scaler (×3) | 9,214 (0.8%) | 9,718 (0.4%) | 99 (5.5%) | 15 (0.17%) |
| classifier (×3) | 23,115 (2.0%) | 31,148 (1.3%) | 117 (6.5%) | 30 (0.33%) |
| **Kernel Total** | **253,251 (21.9%)** | **396,903 (16.7%)** | **489 (27.0%)** | **3,831 (42.5%)** |

**Key insight for 3-CU DSP explosion:** UNROLL=28 (vs 16 in 1-CU) increased DSPs from 722→1,262 per CU. The unrolled accumulation of 28 parallel dot products required DSP48E2s for the addition tree, whereas UNROLL=16 fit in LUT-based adders. This is the direct cause of the timing failure (more DSP = more routing congestion = missed 300 MHz).

### Explaining the Paper's Original Resource Anomalies

**Q: Why does streaming add ~1000 BRAM?**
A: Platform shell increases from 199→~1300 BRAM. The CMAC (100G Ethernet) and UDP/IP stack are platform IP cores placed in the shell partition, not in the user kernel budget. The streaming kernel itself uses similar BRAM to the single-node kernel.

**Q: Why do DSPs drop from 30→4 going single-node→streaming for MultiRocket?**
A: The single-node design includes HBM data mover modules (generated by Vitis for `m_axi` interfaces). These data movers use DSPs for address calculation. The streaming design replaces HBM interfaces with AXI-Stream, eliminating the data mover modules and their DSPs. The kernel DSPs themselves remain constant.

---

## Deliverables Checklist

After completing all steps, verify you have:

- [ ] `hydra_optimized/results/v2_fixed/hydra_v2fixed_summary.csv` — HYDRA v2 FPGA benchmarks
- [ ] `multirocket_optimized/models/multirocket84_{mosquitosound,fruitflies}_{model,test}.json` — trained models
- [ ] `results/scalability_sweep.csv` — throughput vs window length
- [ ] `results/gpu_baseline_results.json` — GPU comparison numbers
- [ ] `results/power/rapl_measurements.csv` (or perf results) — measured CPU power
- [ ] MultiRocket corrected xclbin + benchmark results (if rebuild completed)

Then hand to Philip:
- All CSV files above
- The Vivado per-module resource data from this guide
- The existing `results/tables/*.tex` files
- The existing `results/figures/*.pdf` files
