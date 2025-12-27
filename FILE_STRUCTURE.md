# MiniRocket HLS - File Structure Reference

This document provides a comprehensive overview of the project's file structure, describing the role and importance of each component in the 1:1 paper-faithful FPGA implementation of MiniRocket.

## Project Overview

This repository contains a faithful HLS implementation of the MiniRocket algorithm as described in the original paper [Dempster et al., 2020]. The implementation preserves the exact mathematical formulation including cumulative convolution with α=-1 and γ=+3 kernel weights, achieving bit-exact accuracy match with the Python CPU baseline on UCR Time Series datasets.

---

## Repository Structure

```
MiniRocketHLS/
├── README.md                          # Project overview and quick start guide
├── ALGORITHM.md                       # Detailed algorithm documentation
├── RESULTS.md                         # Experimental results and validation
├── FILE_STRUCTURE.md                  # This file - comprehensive structure reference
├── DOCUMENTATION_SUMMARY.md           # Consolidated documentation index
├── FPGA_IMPLEMENTATION.md             # FPGA-specific implementation details
├── .gitignore                         # Git ignore rules for build artifacts
│
├── tcl_template/                      # Main implementation directory
│   ├── src/                           # Source code
│   ├── build/                         # HLS synthesis artifacts
│   ├── config.cfg                     # Vitis v++ build configuration
│   ├── Makefile                       # Build automation
│   ├── *.json                         # Trained models and test data
│   └── build_dir.hw.*/                # FPGA bitstream output
│
├── optimized_version_77b3cee/         # Reference optimized implementation (77x faster)
│
└── comparison_study/                  # Performance comparison data
```

---

## Core Documentation Files

### README.md
**Purpose**: Entry point for the repository
**Content**: Project overview, features, build instructions, usage examples
**Audience**: General users and researchers
**Key Sections**:
- Quick start guide
- Build prerequisites
- Running inference
- UCR dataset validation

### ALGORITHM.md
**Purpose**: Detailed mathematical and algorithmic documentation
**Content**: 1:1 faithful implementation rationale, algorithm specifications
**Importance**: Critical for reproducibility and research validation
**Key Sections**:
- Mathematical formulation of cumulative convolution
- Kernel weight selection (α=-1, γ=+3)
- Feature extraction via Proportion of Positive Values (PPV)
- StandardScaler normalization analysis
- Ridge classifier integration

### RESULTS.md
**Purpose**: Experimental validation and performance metrics
**Content**: UCR benchmark results, accuracy validation, performance analysis
**Key Sections**:
- UCR dataset results (GunPoint, ItalyPowerDemand, ArrowHead)
- Accuracy comparison with Python baseline
- Throughput measurements (inferences/sec)
- FPGA resource utilization

### FILE_STRUCTURE.md (This File)
**Purpose**: Comprehensive reference for all project files
**Content**: Role, importance, and interdependencies of each file
**Audience**: Developers and researchers extending the work

### FPGA_IMPLEMENTATION.md
**Purpose**: FPGA-specific technical details
**Content**: HLS optimization strategies, hardware architecture, timing analysis
**Key Sections**:
- Pipeline design and II (Initiation Interval) optimization
- Memory banking and HBM allocation
- Compute unit configuration
- Clock frequency and timing closure

---

## Source Code Directory (`tcl_template/src/`)

### Core HLS Files

#### `krnl.cpp`
**Purpose**: Top-level kernel wrapper for Vitis HLS synthesis
**Role**: Defines the hardware interface and kernel entry point
**Key Functions**:
- `krnl_top()`: Main kernel function exposed to host via AXI interfaces
- Interface pragmas for memory-mapped access (m_axi)
- Bundle specifications for HBM bank assignment

**Dependencies**: `krnl.hpp`, `minirocket_inference_hls.cpp`
**Synthesis Target**: This file is compiled to RTL by Vitis HLS

#### `minirocket_inference_hls.cpp`
**Purpose**: Core MiniRocket algorithm implementation in HLS C++
**Role**: Contains the 1:1 paper-faithful inference logic
**Importance**: **CRITICAL** - This is the heart of the implementation
**Key Functions**:
- `compute_cumulative_convolution_hls()`: Implements C_alpha and C_gamma arrays
- `minirocket_inference_hls()`: Main inference pipeline
  - Cumulative convolution computation
  - PPV feature extraction
  - StandardScaler normalization
  - Ridge classifier prediction

**Mathematical Fidelity**:
- Exact α=-1, γ=+3 weights as per paper
- Preserves cumulative sum computation C[i] = C[i-1] + value
- Bit-exact float arithmetic matching Python implementation

**HLS Optimizations**:
- `#pragma HLS PIPELINE II=1` for throughput optimization
- `#pragma HLS ARRAY_PARTITION` for parallel memory access
- Loop unrolling where beneficial

**Dependencies**: `krnl.hpp`

#### `krnl.hpp`
**Purpose**: Common header definitions and constants
**Role**: Defines data types, array sizes, and configuration parameters
**Key Definitions**:
- `MAX_TIME_SERIES_LENGTH 512`: Maximum input length
- `MAX_FEATURES 10000`: Maximum number of features
- `MAX_DILATIONS 32`: Maximum dilations supported
- `MAX_CLASSES 10`: Maximum output classes
- `data_t`: Float type for computations
- `int_t`: Integer type for indices

**Importance**: Changing these constants affects hardware resource usage

### Host Application Files

#### `minirocket_host.cpp`
**Purpose**: Host-side application for FPGA inference
**Role**: Manages FPGA device, loads bitstream, executes kernels
**Key Classes**:
- `MiniRocketFPGA`: Main FPGA interface class
  - Device initialization and bitstream loading
  - OpenCL context and queue management
  - Buffer allocation and data transfer
  - Batch inference coordination

**Key Functions**:
- `load_model()`: Loads trained model parameters to FPGA memory
- `predict_batch()`: Executes inference on multiple time series
- Main loop: Argument parsing and benchmark execution

**Memory Management**:
- Allocates HBM-backed cl::Buffer objects for each CU
- Implements pipelining between host-to-device transfer and kernel execution
- Asynchronous execution with event synchronization

**Dependencies**: `minirocket_hls_testbench_loader.h`, OpenCL headers

#### `minirocket_hls_testbench_loader.cpp/h`
**Purpose**: JSON model and test data loading utilities
**Role**: Parses trained models from Python export format
**Key Functions**:
- `load_model_to_hls_arrays()`: Loads model into HLS-compatible arrays
- `load_test_data()`: Loads test samples and labels
- JSON parsing using RapidJSON library

**File Format**: Compatible with `train_minirocket.py` output
**Dependencies**: RapidJSON (header-only library)

#### `test_hls.cpp`
**Purpose**: HLS C-simulation testbench
**Role**: Validates HLS implementation against Python baseline
**Usage**: Run before synthesis to verify correctness
**Key Features**:
- Loads model and test data
- Runs HLS inference function
- Compares predictions with expected labels
- Reports accuracy

**Build**: `vitis_hls -f test_hls.tcl` (separate from FPGA build)

---

## Build Configuration

### `Makefile`
**Purpose**: Automates build process
**Targets**:
- `kernel`: Synthesize HLS kernel to `.xo` object file
- `build`: Link kernel and generate bitstream (`.xclbin`)
- `host`: Compile host application
- `clean`: Remove build artifacts

**Key Variables**:
- `TARGET`: Build target (`hw` for FPGA, `hw_emu` for emulation)
- `PLATFORM`: Xilinx platform file path (e.g., U280)
- `FREQ`: Target clock frequency (300 MHz)

**Invocation**:
```bash
make build TARGET=hw PLATFORM=<platform_path>
```

### `config.cfg`
**Purpose**: Vitis v++ linker configuration
**Role**: Specifies compute units and memory bank connectivity
**Current Configuration** (1-CU):
```ini
[connectivity]
nk=krnl_top:1                               # 1 compute unit

sp=krnl_top_1.time_series_input:HBM[0]      # Input data → HBM bank 0
sp=krnl_top_1.prediction_output:HBM[1]      # Output → HBM bank 1
sp=krnl_top_1.coefficients:HBM[2]           # Model weights → HBM bank 2
sp=krnl_top_1.intercept:HBM[3]              # Intercept → HBM bank 3
sp=krnl_top_1.scaler_mean:HBM[4]            # Normalization mean → HBM bank 4
sp=krnl_top_1.scaler_scale:HBM[5]           # Normalization scale → HBM bank 5
sp=krnl_top_1.dilations:HBM[6]              # Dilation parameters → HBM bank 6
sp=krnl_top_1.num_features_per_dilation:HBM[7]  # Feature counts → HBM bank 7
sp=krnl_top_1.biases:HBM[8]                 # Bias terms → HBM bank 8
```

**Design Note**: 1-CU configuration requires 9 HBM ports. Multi-CU scaling requires XRT native API for explicit bank placement.

---

## Model and Test Data Files

### Trained Models (`.json` format)

#### `minirocket_gunpoint_model.json`
- **Dataset**: GunPoint (UCR Archive)
- **Time Series Length**: 150
- **Classes**: 2 (Gun vs Point gesture)
- **Accuracy**: 98.33% (59/60 correct)
- **Use**: Primary validation benchmark

#### `minirocket_italypower_model.json`
- **Dataset**: ItalyPowerDemand
- **Time Series Length**: 24
- **Classes**: 2 (Oct-Mar vs Apr-Sep)
- **Accuracy**: 97.26% (320/329 correct)
- **Use**: Short time series validation

#### `minirocket_arrowhead_model.json`
- **Dataset**: ArrowHead
- **Time Series Length**: 251
- **Classes**: 3
- **Accuracy**: 83.43% (146/175 correct)
- **Use**: Multi-class validation

### Test Data Files
- `minirocket_*_model_test_data.json`: Corresponding test sets
- Format: `{"samples": [[...]], "labels": [...]}`

### Model Generation
Models are generated using the Python training script:
```python
python train_minirocket.py --dataset GunPoint --output minirocket_gunpoint_model.json
```

---

## Build Artifacts

### `build/` Directory
**Content**: HLS synthesis intermediate files
**Key Subdirectories**:
- `build/iprepo/`: Packaged IP repository
- `build/src/krnl_top_prj/`: Vitis HLS project files
  - `solution1/syn/`: Synthesis reports
  - `solution1/impl/`: RTL implementation
  - `solution1/.autopilot/`: AutoPilot optimization data

**Reports of Interest**:
- `solution1/syn/report/krnl_top_csynth.rpt`: Resource estimates
- `solution1/syn/report/minirocket_inference_hls_csynth.rpt`: Function-level analysis

### `build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/`
**Content**: Final bitstream and metadata
**Key Files**:
- `krnl.xclbin`: FPGA bitstream (47 MB)
- `krnl.link.xclbin.link_summary`: Build summary for Vitis Analyzer
- `krnl.link.ltx`: ILA probe file for debugging

### `_x.hw.*/` Directories
**Content**: Vivado implementation files
**Key Subdirectories**:
- `link/int/`: Intermediate netlist files
- `link/imp/`: Place-and-route results
- `reports/link/`: Timing and utilization reports
- `logs/link/vivado.log`: Vivado implementation log

**Critical Report**:
- `reports/link/imp/impl_1_hw_bb_locked_timing_summary_routed.rpt`:
  Timing analysis, WNS (Worst Negative Slack), achieved frequency

---

##Optimized Reference Implementation

### `optimized_version_77b3cee/`
**Purpose**: Archived optimized implementation for comparison
**Key Differences from 1:1 Reference**:
1. **Kernel Weights**: {-1, 0, +1} instead of {-1, +2}
2. **Convolution Strategy**: Direct convolution (no cumulative arrays)
3. **Performance**: 77x faster (17,267 vs 224 inf/sec on ItalyPowerDemand)
4. **Multi-CU**: Supports 4 compute units with proper buffer allocation

**Structure**: Mirrors `tcl_template/` with modified source files
**Use**: Performance baseline and optimization reference

**Key Insight**: Despite simplified weights, achieves identical accuracy due to StandardScaler normalization and Ridge classifier adaptation.

---

## Comparison and Analysis

### `comparison_study/`
**Purpose**: Detailed performance and algorithmic comparison
**Content**:
- Accuracy comparison tables
- Throughput analysis across datasets
- Resource utilization comparison
- Algorithm difference analysis

---

## Development and Testing Workflow

### 1. HLS Simulation
```bash
cd tcl_template/build/src
vitis_hls -f test_hls.tcl
# Runs C-simulation, verifies accuracy
```

### 2. HLS Synthesis
```bash
make kernel
# Generates build/iprepo/krnl_top.xo
# Review: build/src/krnl_top_prj/solution1/syn/report/*.rpt
```

### 3. Hardware Build
```bash
make build TARGET=hw PLATFORM=<platform>
# Generates build_dir.hw.*/krnl.xclbin (~2 hour build time)
```

### 4. Execution
```bash
make host
./host krnl.xclbin model.json test_data.json
```

---

## Git Configuration

### `.gitignore`
**Purpose**: Exclude build artifacts from version control
**Key Exclusions**:
- `build_dir.*/`: Bitstreams and build outputs
- `_x*/`: Vivado intermediate files
- `*.log`: Build logs
- `build/`: HLS synthesis artifacts (large)

**Tracked Exceptions**:
- UCR model JSON files (for reproducibility)
- Test logs documenting validation results

---

## Key File Dependencies

```
Synthesis Flow:
  krnl.cpp → (includes) → krnl.hpp
            → (calls) → minirocket_inference_hls.cpp
            → (HLS) → build/iprepo/krnl_top.xo
            → (Vivado) → build_dir.*/krnl.xclbin

Host Application Flow:
  minirocket_host.cpp → (links) → minirocket_hls_testbench_loader.cpp
                      → (uses) → OpenCL/XRT libraries
                      → (executable) → host

Execution Flow:
  host → (loads) → krnl.xclbin
       → (reads) → *.json (model + test data)
       → (outputs) → accuracy and throughput
```

---

## Research Reproducibility Checklist

To reproduce results from this implementation:

1. **Platform**: Xilinx Alveo U280 (xcvu9p-flga2104-2-i)
2. **Tools**: Vitis HLS/Vitis 2023.2, XRT 2.16.204
3. **Models**: Use provided UCR trained models (*.json)
4. **Build**: `make build TARGET=hw PLATFORM=<U280 platform>`
5. **Execute**: `./host krnl.xclbin <model>.json <test>.json`
6. **Verify**: Compare accuracy with RESULTS.md benchmarks

**Expected Accuracy**:
- GunPoint: 98.33% (59/60)
- ItalyPowerDemand: 97.26% (320/329)
- ArrowHead: 83.43% (146/175)

**Expected Throughput** (1-CU U280 @ 300 MHz):
- GunPoint: ~46 inf/sec
- ItalyPowerDemand: ~250 inf/sec

---

## Citation and Reference

If using this implementation for research, please cite:

**Original MiniRocket Algorithm**:
```
Dempster, A., Schmidt, D. F., & Webb, G. I. (2020).
MiniRocket: A very fast (almost) deterministic transform for time series classification.
arXiv preprint arXiv:2012.08791.
```

**This HLS Implementation**:
```
[Your Paper Title]
[Authors]
[Conference/Journal, Year]
Repository: https://github.com/[your-repo]/MiniRocketHLS
```

---

## Contact and Contributions

For questions regarding file structure, implementation details, or research collaboration:
- File issues on GitHub repository
- Consult inline code comments for function-specific details
- Review ALGORITHM.md for mathematical formulation
- Check RESULTS.md for validation methodology

---

**Last Updated**: December 26, 2025
**Version**: 1.0 (1-CU Paper-Faithful Reference Implementation)
**Branch**: `1cu-reference-build`
