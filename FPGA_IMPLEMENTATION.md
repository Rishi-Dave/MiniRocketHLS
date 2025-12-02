# MiniRocket FPGA Implementation Guide

## Overview

This document describes the FPGA implementation of MiniRocket using Xilinx Vitis HLS and provides instructions for building and running the accelerator.

## Architecture

### Hardware Design

The FPGA kernel implements the complete MiniRocket inference pipeline:

```
┌─────────────────────────────────────────────────────────┐
│                    FPGA Kernel (krnl_top)               │
│                                                         │
│  Input TS  →  Feature    →  Scaling   →  Classification │
│  (HBM[0])     Extraction     (Normalize)   (Linear)     │
│                 (PPV)                                    │
│                   ↓                           ↓          │
│              420 features              Class Scores      │
│                                             ↓          │
│                                        Output (HBM[1])   │
└─────────────────────────────────────────────────────────┘
```

### Multi-Kernel Parallelization

The design supports multiple compute units for parallel processing:

```
        PCIe Interface
             ↓
   ┌─────────────────────┐
   │    Host CPU         │
   │  (Scheduling)       │
   └─────────────────────┘
             ↓
   ┌─────────┬─────────┬─────────┬─────────┐
   │  CU 1   │  CU 2   │  CU 3   │  CU 4   │
   │ HBM 0-8 │ HBM 9-17│HBM 18-26│HBM 27-31│
   └─────────┴─────────┴─────────┴─────────┘
```

Each compute unit has independent HBM banks for maximum memory bandwidth.

## Hardware Requirements

### FPGA Platform
- **Xilinx Alveo U280** (or compatible)
- HBM memory (32 banks)
- PCIe Gen3 x16 interface

### Development Tools
- **Vitis 2023.2** (or later)
- **Xilinx Runtime (XRT)** 2023.2
- **GCC 7.5+** with C++17 support

### System Requirements
- Ubuntu 20.04/22.04 LTS
- 64GB+ RAM (for FPGA compilation)
- 200GB+ disk space

## Building the FPGA Accelerator

### 1. Environment Setup

```bash
# Source Vitis and XRT
source /opt/xilinx/xrt/setup.sh
source /tools/Xilinx/Vitis/2023.2/settings64.sh

# Verify tools
which v++
xbutil examine
```

### 2. Kernel Compilation

Navigate to the project directory:

```bash
cd MiniRocketHLS/tcl_template
```

**Step 1: Compile kernel to XO**

```bash
v++ -t hw \
    --platform xilinx_u280_gen3x16_xdma_1_202211_1 \
    -c -k krnl_top \
    -o krnl.xo \
    src/krnl.cpp
```

**Step 2: Link to create xclbin**

For single compute unit:
```bash
v++ -t hw \
    --platform xilinx_u280_gen3x16_xdma_1_202211_1 \
    --config config.cfg \
    -l -o krnl.xclbin \
    krnl.xo
```

For 4 compute units (modify `config.cfg` first - see below):
```bash
v++ -t hw \
    --platform xilinx_u280_gen3x16_xdma_1_202211_1 \
    --config config.cfg \
    -l -o build_dir_4cu.hw.*/krnl.xclbin \
    krnl.xo
```

**Build Time**: ~2-5 hours depending on compute units

### 3. Configuring Compute Units

Edit `config.cfg` to specify number of kernels and memory banks:

**Single Compute Unit:**
```ini
[connectivity]
sp=krnl_top_1.time_series_input:HBM[0]
sp=krnl_top_1.prediction_output:HBM[1]
sp=krnl_top_1.coefficients:HBM[2]
sp=krnl_top_1.intercept:HBM[3]
sp=krnl_top_1.scaler_mean:HBM[4]
sp=krnl_top_1.scaler_scale:HBM[5]
sp=krnl_top_1.dilations:HBM[6]
sp=krnl_top_1.num_features_per_dilation:HBM[7]
sp=krnl_top_1.biases:HBM[8]
```

**Four Compute Units:**
```ini
[connectivity]
nk=krnl_top:4

# Each CU gets its own HBM banks (0-8, 9-17, 18-26, 27-31+DDR)
sp=krnl_top_1.time_series_input:HBM[0]
sp=krnl_top_2.time_series_input:HBM[9]
sp=krnl_top_3.time_series_input:HBM[18]
sp=krnl_top_4.time_series_input:HBM[27]
# ... (see config.cfg for full configuration)
```

### 4. Build in Background (Recommended)

Use tmux to run long builds:

```bash
tmux new-session -s fpga_build
# Inside tmux:
v++ -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 \
    --config config.cfg -c -k krnl_top -o krnl.xo src/krnl.cpp && \
v++ -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 \
    --config config.cfg -l -o krnl.xclbin krnl.xo

# Detach: Ctrl+B, D
# Reattach: tmux attach -t fpga_build
```

## Building Host Applications

### Single-Kernel Host

```bash
g++ -O3 -std=c++17 \
    -I/opt/xilinx/xrt/include \
    -I$XILINX_VIVADO/include \
    -L$XILINX_XRT/lib \
    -o ucr_benchmark \
    ucr_benchmark_host.cpp \
    -lOpenCL -pthread
```

### Multi-Kernel Host (4 CUs)

```bash
g++ -O3 -std=c++17 \
    -I/opt/xilinx/xrt/include \
    -I$XILINX_VIVADO/include \
    -L$XILINX_XRT/lib \
    -o ucr_benchmark_4cu \
    ucr_benchmark_4cu.cpp \
    -lOpenCL -pthread
```

## Training Models for FPGA

### Step 1: Install Dependencies

```bash
pip3 install --user aeon scikit-learn numpy
```

### Step 2: Train on UCR Dataset

```bash
python3 train_minirocket_for_fpga.py <DATASET_NAME> --features 420
```

Example:
```bash
python3 train_minirocket_for_fpga.py CBF --features 420
```

This creates:
- `<dataset>_fpga_model.json` - Model parameters
- `<dataset>_fpga_test.json` - Test data

### Step 3: Run Inference on FPGA

**Single compute unit:**
```bash
./ucr_benchmark \
    build_dir.hw.*/krnl.xclbin \
    cbf_fpga_model.json \
    cbf_fpga_test.json
```

**Four compute units (parallel):**
```bash
./ucr_benchmark_4cu \
    build_dir_4cu.hw.*/krnl.xclbin \
    cbf_fpga_model.json \
    cbf_fpga_test.json
```

## Hardware Design Details

### Kernel Interface

```cpp
void krnl_top(
    const data_t* time_series_input,      // Input time series
    data_t* prediction_output,            // Output class scores
    const data_t* coefficients,           // Classifier weights
    const data_t* intercept,              // Classifier bias
    const data_t* scaler_mean,            // Feature normalization mean
    const data_t* scaler_scale,           // Feature normalization std
    const int_t* dilations,               // Convolution dilations
    const int_t* num_features_per_dilation, // Features per dilation
    const data_t* biases,                 // PPV thresholds
    int time_series_length,               // Input length
    int num_features,                     // Total features (420)
    int num_classes,                      // Number of classes
    int num_dilations                     // Number of dilations (5)
);
```

### HLS Optimizations

1. **Pipeline Directives**: Maximize throughput
   ```cpp
   #pragma HLS PIPELINE II=1
   ```

2. **Array Partitioning**: Parallel memory access
   ```cpp
   #pragma HLS ARRAY_PARTITION variable=features complete
   ```

3. **Dataflow**: Overlap computation stages
   ```cpp
   #pragma HLS DATAFLOW
   ```

4. **Interface Optimization**: Burst transfers from HBM
   ```cpp
   #pragma HLS INTERFACE m_axi port=time_series_input bundle=gmem0
   ```

### Resource Utilization (Single Kernel)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUT | 22,183 | ~1.3M | 1.7% |
| FF | 19,651 | ~2.6M | 0.75% |
| BRAM | 220 | 2,688 | 8.2% |
| DSP | 0 | 9,024 | 0% |

**Low utilization** allows scaling to 8+ compute units.

## Troubleshooting

### Build Issues

**Issue**: `vivado not found`
```bash
# Solution: Source Vitis settings
source /tools/Xilinx/Vitis/2023.2/settings64.sh
```

**Issue**: `Platform not found`
```bash
# Solution: Install platform
sudo /opt/xilinx/xrt/bin/xbmgmt examine
# Download from Xilinx website if missing
```

### Runtime Issues

**Issue**: `No devices found`
```bash
# Solution: Check XRT installation
xbutil examine
sudo /opt/xilinx/xrt/bin/xbmgmt examine
```

**Issue**: `Kernel execution timeout`
```bash
# Solution: Increase timeout in host code or check kernel logic
```

## Performance Tuning

### Maximizing Throughput

1. **Use multiple compute units** (4-8 CUs)
2. **Batch inference** when possible
3. **Cache model parameters** on device
4. **Use out-of-order command queue** for parallel execution

### Memory Bandwidth Optimization

- Assign each CU to different HBM banks
- Use burst transfers for sequential data
- Minimize H2D/D2H transfers

### Expected Performance

| Configuration | Throughput | Latency |
|---------------|-----------|---------|
| 1 CU (single) | 102 inf/sec | 9.8 ms |
| 1 CU (batch) | 1,248 inf/sec | 0.8 ms/sample |
| 4 CU (parallel) | 3,468 inf/sec | 0.29 ms/sample |

## Next Steps

- See [RESULTS.md](RESULTS.md) for detailed performance analysis
- See [ALGORITHM.md](ALGORITHM.md) for MiniRocket theory
- Experiment with different UCR datasets
- Scale to 8 compute units for higher throughput
