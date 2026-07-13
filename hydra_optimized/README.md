# HYDRA FPGA Implementation

FPGA acceleration of HYDRA (HYbrid Dictionary Representation Algorithm) for fast time series classification on Xilinx Alveo U280.

## Overview

This implementation provides a complete FPGA-accelerated version of HYDRA, a state-of-the-art dictionary-based time series classification algorithm. The design follows the established patterns from MiniRocket and MultiRocket implementations while incorporating HYDRA's unique dictionary-based approach.

### Key Features

- **512 Dictionary Kernels**: Organized into 8 groups for diversity
- **2 Pooling Operators**: Max pooling + global mean per kernel
- **1,024 Features**: 512 kernels × 2 pooling operators
- **HBM Optimization**: 9 HBM ports for high bandwidth
- **Complete Pipeline**: Training, HLS synthesis, hardware deployment, benchmarking

### Algorithm Characteristics

| Metric | Value |
|--------|-------|
| Number of Kernels | 512 |
| Kernel Groups | 8 |
| Kernel Size | 9 |
| Pooling Operators | 2 (Max + Mean) |
| Features per Sample | 1,024 |
| Max Time Series Length | 512 timesteps |
| Validated Latency (v2_fixed, 1-CU, InsectSound) | 0.158 ms |
| Validated Throughput (v2_fixed, 1-CU, InsectSound) | 6,326 inf/sec |

Numbers above are hardware-validated (2026-04), not estimates — see
[Performance Benchmarking](#performance-benchmarking) below and
`results/hydra_v2_fixed_benchmarks.json`.

## Directory Structure

```
hydra_optimized/
├── hydra/                    # HLS kernel implementation
│   ├── src/
│   │   ├── hydra.cpp        # Main FPGA kernel
│   │   └── hydra_pooling.cpp # Pooling operators
│   ├── include/
│   │   ├── hydra.hpp        # Header & data types
│   │   └── weights.txt      # Dictionary kernels (generated)
│   ├── test/
│   │   ├── test_hydra.cpp   # HLS testbench
│   │   └── hydra_hls_testbench_loader.* # Test data loader
│   ├── CMakeLists.txt       # HLS build configuration
│   └── make.tcl.in          # Vitis HLS script
│
├── host/                     # OpenCL host application
│   ├── src/
│   │   ├── hydra_host.cpp   # Main application
│   │   └── hydra_loader.cpp # Model parameter loader
│   └── include/
│       ├── hydra_host.h
│       └── hydra_loader.h
│
├── scripts/                  # Python training & benchmarking
│   ├── custom_hydra.py      # Python reference implementation
│   ├── train_hydra.py       # Training script
│   └── benchmark_hydra.py   # Benchmarking script
│
├── models/                   # Trained models (generated)
├── docs/                     # Documentation
├── build/                    # Build artifacts (generated)
├── results/                  # Benchmark results (generated)
│
├── CMakeLists.txt           # Root CMake configuration
├── Makefile                 # Vitis build orchestration
├── config.cfg               # HBM connectivity configuration
└── README.md                # This file
```

## Prerequisites

### Hardware

- Xilinx Alveo U280 Data Center Accelerator Card
- PCIe Gen3 x16 or Gen4 x16 slot
- Host machine with 32+ GB RAM

### Software

- **Xilinx Tools**:
  - Vitis HLS 2023.2 or later
  - Vitis 2023.2 or later
  - XRT (Xilinx Runtime) 2023.2 or later

- **Build Tools**:
  - CMake 3.0+
  - GNU Make 4.0+
  - GCC/G++ 7.5+ with C++17 support

- **Python** (for training):
  - Python 3.8+
  - NumPy, scikit-learn
  - Optional: sktime or aeon (for UCR datasets)

## Quick Start

### 1. Environment Setup

```bash
# Source Xilinx tools
source /opt/xilinx/xrt/setup.sh
source /tools/Xilinx/Vitis/2023.2/settings64.sh

# Verify tools
which vitis_hls
which v++
xbutil examine
```

### 2. Train HYDRA Model

```bash
cd scripts

# Train on synthetic data (quick test)
python train_hydra.py --output ../models/hydra_model.json

# Train on UCR dataset (requires sktime)
python train_hydra.py --dataset GunPoint --output ../models/hydra_gunpoint_model.json
```

### 3. HLS C Simulation (Algorithm Validation)

```bash
# Build and run C simulation
cd build
cmake ..
make csim.hydra_inference

# Check output
cat hydra_inference/solution/csim/report/hydra_inference_csim.log
```

### 4. HLS Synthesis

```bash
# Synthesize HLS kernel to RTL
make synthesis.hydra_inference

# Check synthesis report
cat hydra_inference/solution/syn/report/hydra_inference_csynth.rpt
```

### 5. Hardware Build (Full FPGA Bitstream)

```bash
cd ..
# This takes 2-5 hours
make all TARGET=hw PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
```

### 6. Run on FPGA

```bash
cd build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1

./hydra_host krnl.xclbin \
    ../../models/hydra_model.json \
    ../../models/hydra_test.json
```

## Build Targets

### HLS Targets (via CMake)

```bash
make csim.hydra_inference      # C simulation (fast, validates algorithm)
make synthesis.hydra_inference  # HLS synthesis (RTL generation)
make cosim.hydra_inference     # Co-simulation (RTL validation)
make ip.hydra_inference        # Export IP (generates .xo file)
make installip.hydra_inference # Install IP to repository
```

### Vitis Targets (via Makefile)

```bash
make ip                  # HLS synthesis + IP export
make host               # Compile host application
make build TARGET=hw    # Link kernel + generate XClbin (hardware)
make build TARGET=hw_emu # Hardware emulation
make run TARGET=hw      # Build + run on hardware
make clean              # Remove temporary files
make cleanall           # Remove all build artifacts
```

## Testing Pipeline

### Stage 1: C Simulation

Validates algorithm correctness on CPU.

```bash
cd build
make csim.hydra_inference
```

**Expected output:**
```
========================================
  HYDRA HLS Testbench
========================================

Sample 1/10: predicted=0, actual=0 [✓ CORRECT]
Sample 2/10: predicted=1, actual=1 [✓ CORRECT]
...
Accuracy: 8/10 = 80.00%
TEST PASSED
```

### Stage 2: HLS Synthesis

Generates RTL and estimates resources/timing.

```bash
make synthesis.hydra_inference
cat hydra_inference/solution/syn/report/hydra_inference_csynth.rpt
```

**Check for:**
- Clock period: <3.33 ns (300 MHz)
- Latency: <100,000 cycles
- Resources: LUTs <200k, DSPs <400

### Stage 3: Co-Simulation

Validates RTL matches C behavior.

```bash
make cosim.hydra_inference
```

**Expected:**
- RTL output exactly matches C simulation
- No timing violations
- Cycle count matches estimates

### Stage 4: Hardware Build

```bash
cd ..
make build TARGET=hw
```

**Time:** 2-5 hours
**Output:** `build_dir.hw.*/krnl.xclbin`

### Stage 5: Hardware Execution

```bash
cd build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1
./hydra_host krnl.xclbin \
    ../../models/hydra_model.json \
    ../../models/hydra_test.json
```

**Expected output:**
```
========================================
  HYDRA FPGA Host Application
========================================

Found Platform: Xilinx
Device: xilinx_u280_gen3x16_xdma_1_202211_1

Running HYDRA Inference on FPGA
========================================

Sample   1/50: predicted=0, actual=0, time=0.372 ms [✓]
Sample   2/50: predicted=1, actual=1, time=0.368 ms [✓]
...

Results Summary
========================================

Accuracy: 48/50 = 96.00%

Performance:
  Average latency: 0.370 ms
  Throughput: 2,703 inferences/sec
  Total time: 18.5 ms

TEST PASSED
```

## Performance Benchmarking

### Python Benchmark

```bash
cd scripts
python benchmark_hydra.py --num-kernels 512 --samples 100
```

### Validated hardware results (v2_fixed, `ap_fixed<32,16>`, UNROLL=16)

Hardware-validated 2026-04-02, single U280, 300 MHz. Source:
`results/hydra_v2_fixed_benchmarks.json`, `results/all_data_for_sheets.csv`.

| Dataset | 1-CU (inf/s) | 2-CU (inf/s) | 3-CU (inf/s)† | Accuracy | vs. CPU (1-CU) |
|---|---|---|---|---|---|
| InsectSound (L=600) | 6,326 | 8,028 | 10,573 | 69.41% | 70.2x |
| MosquitoSound (L=3750) | 1,937 | 2,757 | 3,516 | 70.05% | 49.4x |
| FruitFlies (L=5000) | 1,507 | 2,148 | 2,972 | 87.61% | 44.8x |

† 3-CU builds **failed timing** and were auto-throttled off the 300 MHz
target — report as timing-failed, not a closed configuration. 1-CU and 2-CU
both closed timing at 300 MHz.

Compare with MiniRocket (fused CONV+PPV, current lead): 3,797-9,585 inf/s
depending on dataset and CU count (GunPoint, 1-3 CU; 3-CU timing-failed at
269 MHz) — see [../docs/RESULTS.md](../docs/RESULTS.md) for the full,
sourced comparison table. Earlier reports on this page citing "MiniRocket
15,000 inf/s" / "MultiRocket 8,000 inf/s" were unsourced estimates and have
been removed; MultiRocket was subsequently dropped entirely (2026-04-08,
invalid results — see `../multirocket_optimized/README.md`).

### Network-attached variant

`hydra_axis/` is a network-attached port of this v2_fixed core (dilation-
sorted kernels, permuted features for a batched compute layout) used in the
FPGA-to-FPGA saturation thread — up to 13.4M samples/s (~6.85 Gbps) on
FruitFlies at 200 MHz (300/250 MHz failed routed timing on the wide
`ap_fixed` MAC arrays). See
[../docs/NETWORK_RESULTS.md](../docs/NETWORK_RESULTS.md) — a different
measurement regime (in-fabric, no host) from the inf/s table above.

## Optimization Strategies

### 1. Multi-Compute-Unit Scaling

Edit `config.cfg`:
```ini
[connectivity]
nk=hydra_inference:2

# CU 1
sp=hydra_inference_1.time_series_input:HBM[0]
...
sp=hydra_inference_1.dilations:HBM[8]

# CU 2
sp=hydra_inference_2.time_series_input:HBM[9]
...
sp=hydra_inference_2.dilations:HBM[17]
```

**Expected:** 2x throughput (5,000+ inf/sec)

### 2. Kernel Weight Caching

Modify `hydra.cpp` to cache weights in BRAM:
```cpp
static data_t kernel_weight_cache[KERNELS_PER_GROUP][KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable=kernel_weight_cache dim=2 complete
```

**Expected:** 20-30% latency reduction

### 3. Batch Processing

Modify kernel to process multiple time series in one invocation.

**Expected:** 2-3x throughput for batch size 8-16

## Troubleshooting

### Issue: "PLATFORM not set"

**Solution:**
```bash
export DEVICE=xilinx_u280_gen3x16_xdma_1_202211_1
# OR
export PLATFORM=/opt/xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1/xilinx_u280_gen3x16_xdma_1_202211_1.xpfm
```

### Issue: HLS synthesis fails

**Check:**
1. Source Vitis HLS: `source /tools/Xilinx/Vitis_HLS/2023.2/settings64.sh`
2. Verify TCL script: `cat build/make.tcl`
3. Check logs: `cat build/hydra_inference/vitis_hls.log`

### Issue: XClbin loading fails

**Check:**
1. XRT installed: `xbutil examine`
2. Device programmed: `xbutil examine -d 0`
3. Correct platform: `xbutil examine -d 0 -r platform`

### Issue: Poor accuracy on FPGA

**Likely causes:**
1. Fixed-point precision issues (verify scaler parameters)
2. Model not trained properly (retrain with more data)
3. Test data mismatch (verify JSON export)

## File Reference

### Core Implementation Files

- `hydra/src/hydra.cpp`: Main FPGA kernel (280 lines)
- `hydra/src/hydra_pooling.cpp`: Pooling operators (90 lines)
- `hydra/include/hydra.hpp`: Data types and constants (200 lines)
- `host/src/hydra_host.cpp`: OpenCL host application (350 lines)
- `scripts/custom_hydra.py`: Python reference implementation (450 lines)

### Build System Files

- `CMakeLists.txt`: Root CMake configuration
- `hydra/CMakeLists.txt`: HLS-specific CMake
- `hydra/make.tcl.in`: Vitis HLS synthesis script
- `Makefile`: Vitis build orchestration
- `config.cfg`: HBM connectivity mapping

### Test Files

- `hydra/test/test_hydra.cpp`: HLS testbench
- `scripts/train_hydra.py`: Model training
- `scripts/benchmark_hydra.py`: Performance benchmarking

## References

- Dempster, A., Schmidt, D. F., & Webb, G. I. (2023). HYDRA: Competing convolutional kernels for fast and accurate time series classification. Data Mining and Knowledge Discovery.
- MiniRocket implementation: `../optimized_version/`
- MultiRocket implementation: `../multirocket_optimized/`

## License

This implementation follows the same license as the parent repository.

## Contact

For issues or questions, please open an issue in the main repository.

---

**Last Updated:** 2026-07-13
**Version:** 2.0.0 (v2_fixed)
**Status:** Hardware-validated 2026-04 (`ap_fixed<32,16>`, 1/2-CU timing met at 300 MHz, 3-CU timing failed); network-attached port (`hydra_axis/`) validated 2026-07 — see [../docs/NETWORK_RESULTS.md](../docs/NETWORK_RESULTS.md)
