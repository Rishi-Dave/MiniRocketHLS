# FPGA Implementation Pipeline

This document details the complete pipeline from Python algorithm to FPGA hardware, including design decisions, optimizations, and verification methodology.

---

## Pipeline Overview

```
Python Training → C++ Reference → HLS Synthesis → Hardware Emulation → FPGA Deployment
    (NumPy)        (Validation)     (RTL Gen)        (Verification)      (U280)
```

---

## Stage 1: Python Training & Model Export

### 1.1 Training Script

**File**: [`tcl_template/train_minirocket.py`](tcl_template/train_minirocket.py)

**Purpose**: Train MiniRocket on time series data and export quantized model for FPGA.

### 1.2 Algorithm Implementation

The Python implementation follows the original MiniRocket numba reference:

```python
# Cumulative convolution (matching numba reference exactly)
A = -_X              # α = -1
G = _X + _X + _X     # γ = +3

# Build C_alpha (cumulative sum with all -1 weights)
C_alpha[:] = A
for gamma_index in range(9 // 2):
    C_alpha[-end:] = C_alpha[-end:] + A[:end]
    end += dilation

# Build C_gamma (individual position contributions with +3 weight)
C_gamma[4] = G  # Center position
for gamma_index in range(9 // 2):
    C_gamma[gamma_index, -end:] = G[:end]

# Combine: Select 3 positions from C_gamma to get final +2 weights
C = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2]
```

**Why γ=3 when paper says +2 weights?**
- C_alpha applies -1 to all 9 positions
- Adding C_gamma (+3) to 3 selected positions: -1 + 3 = **+2** ✓
- Net result: 6 positions with -1, 3 positions with +2

### 1.3 Model Export

The model is exported to JSON with:

```json
{
  "coefficients": [[...], [...]],     // Ridge classifier weights
  "intercept": [...],                 // Classifier biases
  "scaler_mean": [...],               // Feature normalization
  "scaler_scale": [...],              // Feature std dev
  "dilations": [1, 2, 4, 8, ...],     // Dilation values
  "num_features_per_dilation": [...], // Features per dilation
  "biases": [...],                    // PPV quantile thresholds
  "num_features": 840,
  "num_classes": 2,
  "time_series_length": 150
}
```

**Test data**:
```json
{
  "X_test": [[...]],          // Test time series
  "y_test": [0, 1, 0, ...],   // Ground truth labels
  "y_pred": [0, 1, 0, ...],   // Python predictions
  "test_accuracy": 0.9833      // Python baseline
}
```

###  1.4 Supported Datasets

- **Synthetic**: Randomly generated for testing
- **UCR Archive**: arrow_head, gun_point, italy_power
- **Custom**: Any univariate time series

**Example usage**:
```bash
# Train on GunPoint dataset
python3 train_minirocket.py --dataset gun_point

# Generate model files
# → minirocket_model.json
# → minirocket_model_test_data.json
```

---

## Stage 2: C++ Reference Implementation

### 2.1 Testbench Design

**Files**:
- [`tcl_template/src/minirocket_inference_hls.cpp`](tcl_template/src/minirocket_inference_hls.cpp)
- [`tcl_template/src/test_hls.cpp`](tcl_template/src/test_hls.cpp)

**Purpose**: Validate algorithm correctness before HLS synthesis.

### 2.2 Fixed-Point Quantization

**Data type**: `ap_fixed<32, 16>`
- 32 total bits
- 16 integer bits (range: -32768 to +32767)
- 16 fractional bits (precision: 1/65536 ≈ 0.000015)

**Rationale**:
- Sufficient dynamic range for time series values
- Adequate precision for cumulative sums
- Hardware-efficient (32-bit is standard FPGA word size)

### 2.3 Algorithm Verification

Line-by-line correspondence with Python reference:

**Cumulative Convolution** ([minirocket_inference_hls.cpp:48-130](tcl_template/src/minirocket_inference_hls.cpp#L48-L130)):
```cpp
// Exact match with Python train_minirocket.py:97-98
A[i] = -_X[i];                    // A = α * X = -X
G[i] = _X[i] + _X[i] + _X[i];     // G = γ * X = 3X

// Cumulative build (matches Python lines 100-119)
for (int i = 0; i < effective_length; i++) {
    C_alpha[i] = A[i];
}

int end_idx = effective_length - padding;
for (int gamma_index = 0; gamma_index < 4; gamma_index++) {
    for (int i = 0; i < end_idx; i++) {
        C_alpha[effective_length - end_idx + i] += A[i];
    }
    for (int i = 0; i < end_idx; i++) {
        C_gamma[gamma_index][effective_length - end_idx + i] = G[i];
    }
    end_idx += dilation;
}
```

**PPV Feature Extraction** ([minirocket_inference_hls.cpp:147-234](tcl_template/src/minirocket_inference_hls.cpp#L147-L234)):
```cpp
// Count values above bias threshold
int count = 0;
for (int i = 0; i < valid_length; i++) {
    if (C[i] > bias) count++;
}
features[feature_idx] = (data_t)count / (data_t)valid_length;
```

### 2.4 Accuracy Validation

Testbench compares HLS output against ground truth:

```cpp
// Output format shows Python vs HLS comparison
======================================================================
ACCURACY COMPARISON: Python Reference vs HLS Implementation
======================================================================

Implementation                           Accuracy  Correct/Total
----------------------------------------------------------------------
Python (NumPy/Numba)                       98.33%            N/A
HLS (ap_fixed<32,16>)                      98.33%      59/ 60
----------------------------------------------------------------------
Difference (HLS - Python)                 +0.00%
======================================================================

Match Assessment: ✓ EXCELLENT - Within 1% of Python baseline
```

---

## Stage 3: High-Level Synthesis (HLS)

### 3.1 HLS Project Structure

**Files**:
- **Source**: [`tcl_template/src/minirocket_inference_hls.cpp`](tcl_template/src/minirocket_inference_hls.cpp)
- **Kernel Wrapper**: [`tcl_template/src/krnl.cpp`](tcl_template/src/krnl.cpp)
- **Headers**: [`tcl_template/src/minirocket_inference_hls.h`](tcl_template/src/minirocket_inference_hls.h), [`tcl_template/src/krnl.hpp`](tcl_template/src/krnl.hpp)
- **TCL Script**: [`tcl_template/run_hls_sim.tcl`](tcl_template/run_hls_sim.tcl)

**Build command**:
```bash
cd tcl_template
vitis_hls -f run_hls_sim.tcl
```

### 3.2 HLS Pragmas & Optimizations

**Memory Partitioning**:
```cpp
#pragma HLS ARRAY_PARTITION variable=A cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=G cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=C_gamma complete dim=1
```

**Rationale**:
- Cyclic partitioning (factor=8): Enables 8 parallel memory accesses
- Complete partitioning of C_gamma: All 9 positions accessible simultaneously
- Reduces memory bottlenecks in cumulative sum loops

**Pipeline Directives**:
```cpp
#pragma HLS PIPELINE II=1
```
Applied to inner loops to achieve single-cycle throughput.

**Interface Specifications**:
```cpp
#pragma HLS INTERFACE m_axi port=time_series offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=predictions offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=coefficients offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=return
```

- **m_axi**: AXI memory-mapped interface for large data (DDR access)
- **s_axilite**: AXI-Lite for control signals and scalars
- **Multiple bundles**: Allows concurrent memory transactions

### 3.3 Synthesis Results

**Target**: Xilinx Alveo U280 (xcvu9p-flga2104-2-i)

**Clock**: 100 MHz (10 ns period)

**Achieved Frequency**: 136.99 MHz (7.30 ns)
- **37% timing margin** above target

**Resource Utilization**:
| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| **BRAM** | 221 | 4,032 | 5% |
| **DSP** | 17 | 6,840 | <1% |
| **FF** | 15,709 | 2,364,480 | <1% |
| **LUT** | 24,028 | 1,182,240 | 2% |

**Latency**: Variable (depends on time series length and feature count)

---

## Stage 4: Vitis Kernel Compilation

### 4.1 Export HLS IP

Generate Xilinx Object (.xo) file from HLS:

```tcl
# tcl_template/export_ip.tcl
open_project minirocket_hls
open_solution solution1

config_export -format xo
csynth_design
export_design -format xo -output krnl_top.xo
```

**Output**: `build/iprepo/krnl_top.xo` (packaged RTL IP)

### 4.2 Vitis Linking

**Makefile targets**:
```bash
# Hardware emulation (fast, cycle-accurate)
make all TARGET=hw_emu PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1

# Hardware build (actual FPGA, 4-8 hours)
make all TARGET=hw PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
```

**Configuration** ([config.cfg](tcl_template/config.cfg)):
```ini
[connectivity]
nk=krnl_top:1:krnl_top_1

[profile]
data=all:all:all
```

**Build stages**:
1. **v++ --link**: Combine .xo with platform shell → .xclbin.link
2. **v++ --package**: Add platform interfaces → .xclbin

**Output**: `build_dir.TARGET.PLATFORM/krnl.xclbin`

---

## Stage 5: Host Application

### 5.1 OpenCL Host Code

**File**: [`tcl_template/src/minirocket_host.cpp`](tcl_template/src/minirocket_host.cpp)

**Key operations**:
```cpp
// 1. Load model from JSON
load_model_from_json("minirocket_model.json", model);

// 2. Initialize OpenCL
cl::Device device = find_device();
cl::Context context(device);
cl::Program::Binaries bins = import_binary_file("krnl.xclbin");
cl::Program program(context, {device}, bins);
cl::Kernel kernel(program, "krnl_top");

// 3. Create buffers
cl::Buffer buf_input(context, CL_MEM_READ_ONLY, size_in);
cl::Buffer buf_output(context, CL_MEM_WRITE_ONLY, size_out);

// 4. Transfer data
queue.enqueueWriteBuffer(buf_input, CL_TRUE, 0, size_in, input_data);

// 5. Execute kernel
kernel.setArg(0, buf_input);
kernel.setArg(1, buf_output);
queue.enqueueTask(kernel);

// 6. Read results
queue.enqueueReadBuffer(buf_output, CL_TRUE, 0, size_out, output_data);
```

### 5.2 Emulation vs Hardware

**Software Emulation** (`TARGET=sw_emu`):
- Pure C++ execution
- Fast compilation (~minutes)
- Functional verification only

**Hardware Emulation** (`TARGET=hw_emu`):
- Cycle-accurate RTL simulation
- Medium compilation (~1 hour)
- Verifies timing and interfaces

**Hardware** (`TARGET=hw`):
- Actual FPGA bitstream
- Long compilation (4-8 hours)
- Real performance measurements

---

## Stage 6: Verification & Testing

### 6.1 Multi-Level Validation

**Level 1 - C Simulation (csim)**:
- Pure C++ without HLS pragmas
- Validates algorithm correctness
- 100% accuracy match required

**Level 2 - C/RTL Co-Simulation (cosim)**:
- RTL generated from HLS vs C++ testbench
- Cycle-accurate verification
- Optional (csim sufficient if passing)

**Level 3 - Hardware Emulation**:
- Full system with OpenCL host
- Tests memory transfers and kernel execution
- Validates against Python baseline

**Level 4 - Hardware Deployment**:
- Real FPGA execution
- Performance benchmarking
- Power measurements

### 6.2 Accuracy Metrics

Success criteria:
- **Exact match**: HLS == Python (ideal)
- **Excellent**: |HLS - Python| ≤ 1%
- **Acceptable**: |HLS - Python| ≤ 5% OR HLS ≥ 90%
- **Failing**: HLS < 90% AND |HLS - Python| > 5%

**GunPoint UCR Dataset Results**:
```
Python:  98.33% (59/60 correct)
HLS:     98.33% (59/60 correct)
Difference: 0.00%  ← EXCELLENT
```

---

## Design Decisions & Trade-offs

### 1. Fixed-Point vs Floating-Point

**Choice**: `ap_fixed<32,16>`

**Rationale**:
- 10x less DSP usage than float
- Minimal accuracy loss (<0.5%)
- Faster and more power-efficient

### 2. Memory Architecture

**Choice**: AXI memory-mapped for data, AXI-Lite for control

**Rationale**:
- Large model parameters (coefficients, biases) stored in DDR
- Burst transfers amortize memory latency
- Standard Vitis kernel interface

### 3. Parallelization Strategy

**Choice**: Kernel-level parallelism (future work: multi-CU)

**Current**: Single kernel processes all 84 convolutions sequentially
**Future**: 4 compute units (CUs) → 4x throughput

### 4. Padding Strategy

**Choice**: Zero-padding for short time series

**Alternative**: Reflection padding (mirror edges)
**Rationale**: Zero-padding is simpler and sufficient for most datasets

---

## Performance Expectations

### Throughput (Single CU)

**Time series length**: 150 samples
**Features**: 840
**Estimated latency**: ~50,000 cycles @ 200 MHz = **0.25 ms/sample**

**Throughput**: ~4,000 samples/second

### Speedup vs CPU

**CPU (NumPy)**: ~1,000 samples/second (single core)
**FPGA (1 CU)**: ~4,000 samples/second → **4x speedup**
**FPGA (4 CU)**: ~16,000 samples/second → **16x speedup**

### Power Efficiency

**FPGA**: ~25W total (U280)
**CPU**: ~150W (Xeon server)

**Energy per sample**:
- CPU: 150 mJ
- FPGA: 6.25 mJ → **24x more efficient**

---

## File Structure

```
MiniRocketHLS/
├── tcl_template/              # Main implementation directory
│   ├── src/
│   │   ├── minirocket_inference_hls.cpp  # Core algorithm
│   │   ├── minirocket_inference_hls.h     # HLS declarations
│   │   ├── krnl.cpp                        # Vitis kernel wrapper
│   │   ├── krnl.hpp                        # Kernel interface
│   │   ├── minirocket_host.cpp             # OpenCL host
│   │   ├── test_hls.cpp                    # C++ testbench
│   │   └── minirocket_hls_testbench_loader.* # JSON model loader
│   ├── train_minirocket.py                 # Python training script
│   ├── run_hls_sim.tcl                      # HLS synthesis script
│   ├── config.cfg                           # Vitis configuration
│   ├── Makefile                             # Build system
│   └── (generated files)
├── ALGORITHM.md                             # Algorithm description
├── FPGA_IMPLEMENTATION.md                   # This file
├── README.md                                # Repository overview
└── .gitignore                               # Build artifacts exclusion
```

---

## Build Instructions

### Complete Build Flow

```bash
# 1. Train model
cd tcl_template
python3 train_minirocket.py --dataset gun_point

# 2. Run HLS synthesis
vitis_hls -f run_hls_sim.tcl

# 3. Export IP
vitis_hls -f export_ip.tcl

# 4. Build hardware emulation
make all TARGET=hw_emu PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1

# 5. Run emulation
./host build_dir.hw_emu.*/krnl.xclbin minirocket_model.json

# 6. Build for hardware (4-8 hours)
make all TARGET=hw PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1

# 7. Deploy to FPGA
./host build_dir.hw.*/krnl.xclbin minirocket_model.json
```

---

## Future Enhancements

### Multi-Compute Unit (CU) Scaling

Replicate kernel for parallel execution:

```ini
# config.cfg
[connectivity]
nk=krnl_top:4:krnl_top_{1:4}
```

**Expected speedup**: Near-linear with CU count (4 CUs → ~4x)

### Precision Tuning

Explore lower precision:
- `ap_fixed<24,12>`: Smaller, faster, may lose <1% accuracy
- `ap_fixed<16,8>`: Aggressive quantization, needs validation

### Kernel Fusion

Combine feature extraction + normalization + classification into single pipeline to reduce memory transfers.

### Dynamic Reconfiguration

Support runtime model updates without re-compilation (partial reconfiguration).

---

## References

1. Xilinx Vitis HLS User Guide (UG1399)
2. Xilinx Vitis Application Acceleration Development Flow (UG1393)
3. MiniRocket Original Paper: Dempster et al., KDD 2021
