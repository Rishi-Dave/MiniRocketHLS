# MiniRocket FPGA Project - File Structure and Importance

This document provides a comprehensive overview of all files in this repository, their purpose, and their importance to the FPGA acceleration pipeline.

## Repository Structure

```
MiniRocketHLS/                          # Repository root (this is the git repo)
├── .gitignore                          # Git ignore rules
├── README.md                           # Project overview and quick start
├── FILE_STRUCTURE.md                   # This file - complete repository guide
├── CLEANUP_SUMMARY.md                  # Record of repository cleanup actions
├── ALGORITHM.md                        # MiniRocket algorithm explanation
├── FPGA_IMPLEMENTATION.md              # FPGA build and deployment guide
├── RESULTS.md                          # Performance results and benchmarks
│
├── Core HLS Implementation
│   ├── minirocket_inference_hls.cpp    # **CRITICAL** Main HLS inference engine
│   ├── minirocket_inference_hls.h      # **CRITICAL** HLS inference header
│   ├── minirocket_model_constants.h    # **CRITICAL** Generated model parameters
│   ├── ap_int.h                        # **CRITICAL** Xilinx arbitrary precision integers
│   ├── ap_fixed.h                      # **CRITICAL** Xilinx fixed-point types
│   └── hls_stream.h                    # **CRITICAL** Xilinx HLS streaming library
│
├── Testing & Validation
│   ├── test_hls.cpp                    # **CRITICAL** HLS testbench main
│   ├── minirocket_hls_testbench_loader.cpp  # **CRITICAL** Test data loader
│   ├── minirocket_hls_testbench_loader.h    # **CRITICAL** Test loader header
│   ├── Makefile.hls                    # **CRITICAL** HLS build system
│   ├── minirocket_model.json           # Example trained model
│   └── minirocket_model_test_data.json # Example test data
│
├── Training & Model Generation
│   ├── train_minirocket.py             # **CRITICAL** Model training pipeline
│   └── generate_model_constants.py     # **CRITICAL** Generates C++ model constants
│
└── tcl_template/                       # Vitis build and host applications
    ├── FPGA Kernel (Vitis HLS)
    │   ├── src/
    │   │   ├── krnl.cpp                # **CRITICAL** Main FPGA kernel
    │   │   ├── krnl.hpp                # **CRITICAL** Kernel header
    │   │   ├── minirocket_model_constants.h  # Kernel-specific model params
    │   │   ├── minirocket_hls_testbench_loader.cpp  # Test loader (kernel build)
    │   │   ├── minirocket_hls_testbench_loader.h    # Test loader header
    │   │   ├── test_krnl.cpp           # Kernel testbench
    │   │   ├── ap_int.h                # Xilinx HLS headers (copies)
    │   │   ├── ap_fixed.h              # Xilinx HLS headers (copies)
    │   │   └── hls_stream.h            # Xilinx HLS headers (copies)
    │   └── config.cfg                  # **CRITICAL** Vitis build configuration
    │
    ├── Host Applications
    │   ├── host.h                      # **CRITICAL** Host utilities header
    │   ├── ucr_benchmark_host.cpp      # **IMPORTANT** Single-CU benchmark
    │   ├── ucr_benchmark_4cu.cpp       # **IMPORTANT** 4-CU parallel benchmark
    │   ├── batch_benchmark_host.cpp    # **IMPORTANT** Batch inference host
    │   └── minirocket_host.cpp         # **IMPORTANT** General-purpose host
    │
    ├── Python Utilities
    │   ├── train_minirocket_for_fpga.py  # **CRITICAL** FPGA-specific training
    │   ├── prepare_ucr_data.py         # **IMPORTANT** UCR dataset loader
    │   └── benchmark_cpu.py            # **IMPORTANT** CPU baseline
    │
    ├── Build Scripts & Config
    │   ├── Makefile                    # **CRITICAL** Vitis build automation
    │   ├── build_sim.sh                # **IMPORTANT** HLS simulation script
    │   ├── build_cpp_sim.sh            # **IMPORTANT** C++ simulation script
    │   ├── benchmark_fpga.sh           # **USEFUL** FPGA benchmarking script
    │   ├── CMakeLists.txt              # CMake configuration
    │   └── opencl.mk                   # OpenCL build rules
    │
    └── Build Outputs (gitignored)
        ├── build_dir.hw.*/*.xclbin     # Synthesized FPGA bitstream
        ├── build_dir.hw_emu.*/*.xclbin # Hardware emulation bitstream
        ├── build_dir_4cu.*/*.xclbin    # 4-compute-unit bitstream
        └── build/                      # CMake build artifacts
```

## File Importance Categories

### **CRITICAL** - Essential for Pipeline Functionality
These files are absolutely required for the FPGA acceleration pipeline to work:

#### HLS Core Implementation
- `minirocket_inference_hls.cpp/.h` - Implements the core MiniRocket algorithm in HLS C++
- `minirocket_model_constants.h` - Contains trained model parameters (weights, biases, dilations)
- `ap_int.h`, `ap_fixed.h`, `hls_stream.h` - Xilinx HLS library dependencies

#### FPGA Kernel
- `tcl_template/src/krnl.cpp/.hpp` - Vitis HLS kernel that wraps the inference engine
- `tcl_template/config.cfg` - Configures memory interfaces, compute units, and platform

#### Training & Code Generation
- `train_minirocket.py` - Trains MiniRocket model using aeon library
- `generate_model_constants.py` - Converts trained model to C++ header file
- `train_minirocket_for_fpga.py` - FPGA-optimized training with specific feature counts

#### Testing
- `test_hls.cpp` - HLS testbench for C simulation and cosimulation
- `minirocket_hls_testbench_loader.cpp/.h` - Loads test data from JSON
- `Makefile.hls` - Automates HLS synthesis and simulation

#### Host Applications
- `host.h` - OpenCL/XRT utilities for FPGA communication
- `ucr_benchmark_4cu.cpp` - Multi-compute-unit host (best performance)
- `Makefile` (in tcl_template) - Builds both kernel and host

### **IMPORTANT** - Performance & Validation
These files are important for benchmarking and validation:

- `ucr_benchmark_host.cpp` - Single-CU benchmark for comparison
- `batch_benchmark_host.cpp` - Batch processing benchmark
- `benchmark_cpu.py` - CPU baseline for speedup calculations
- `prepare_ucr_data.py` - Loads UCR datasets for testing
- `build_sim.sh` - Simplifies HLS simulation workflow

### **USEFUL** - Documentation & Convenience
These files improve usability but aren't strictly required:

- `README.md` - Project overview and quick start guide
- `ALGORITHM.md` - Explains MiniRocket algorithm theory
- `FPGA_IMPLEMENTATION.md` - Detailed build instructions
- `RESULTS.md` - Performance benchmarks and analysis
- `benchmark_fpga.sh` - Automates FPGA benchmarking

## Files Removed (No Longer Needed)

The following files were removed during cleanup:

- `fpga_build.pid` - Temporary process ID file
- `MiniRocketHLS/fpga_build.pid` - Duplicate process ID file
- `MiniRocketHLS/fpga_build.log` - Build log (regenerated each build)
- `MiniRocketHLS/commit_changes.sh` - Development-only commit script
- `MiniRocketHLS/RESEARCH_PLAN.md` - Internal planning document

## Understanding the Pipeline Flow

### 1. Model Training (Python)
```
train_minirocket_for_fpga.py
  ↓
minirocket_model.json (model parameters)
  ↓
generate_model_constants.py
  ↓
minirocket_model_constants.h (C++ header)
```

### 2. HLS Development & Validation
```
minirocket_inference_hls.cpp + minirocket_model_constants.h
  ↓
test_hls.cpp (testbench)
  ↓
vitis_hls (C simulation, synthesis, cosimulation)
  ↓
Verified HLS design
```

### 3. FPGA Kernel Synthesis
```
src/krnl.cpp (wraps HLS inference)
  ↓
v++ -c (HLS synthesis to .xo)
  ↓
v++ -l (link to .xclbin)
  ↓
krnl.xclbin (FPGA bitstream)
```

### 4. Host Application Build
```
ucr_benchmark_4cu.cpp + host.h
  ↓
g++ (compile with XRT)
  ↓
ucr_benchmark_4cu (executable)
```

### 5. Execution
```
ucr_benchmark_4cu krnl.xclbin model.json test.json
  ↓
FPGA accelerated inference
  ↓
Performance results
```

## Key Dependencies

### External Libraries
- **Xilinx Vitis HLS 2023.2** - High-level synthesis
- **Xilinx XRT 2023.2** - Runtime for FPGA communication
- **Python packages**: aeon, scikit-learn, numpy

### Hardware
- **Xilinx Alveo U280** - Target FPGA platform
- **HBM2 memory** - High-bandwidth on-chip memory

## Build Artifacts (Gitignored)

These are generated during builds and should not be committed:

### HLS Outputs
- `build_hls_sim/` - HLS synthesis results (~8.8 GB)
- `minirocket_hls_validate/` - Cosimulation waveforms (~51 MB)

### Vitis Outputs
- `build_dir.hw.*/` - Hardware build (~2-5 GB)
- `build_dir.hw_emu.*/` - Hardware emulation build (~1-2 GB)
- `_x*/`, `.ipcache/`, `.run/` - Intermediate files
- `*.log`, `*.jou` - Vivado/Vitis logs

### Host Binaries
- `ucr_benchmark`, `ucr_benchmark_4cu`, `batch_benchmark` - Compiled host apps

### Large Data Files
- `*_fpga_model.json` - Trained models (can be large)
- `*_fpga_test.json` - Test datasets
- `ucr_data/` - UCR time series archive

## Model Files

The repository includes small example models:
- `minirocket_model.json` - Example trained model
- `minirocket_model_test_data.json` - Test data for validation

Larger models (e.g., CBF, GunPoint datasets) are generated locally using the training scripts and are gitignored.

## Configuration Files

### Essential Configurations
- `tcl_template/config.cfg` - Vitis link configuration
  - Platform selection
  - Compute unit count (1 vs 4 CUs)
  - Memory bank connectivity
  - Clock frequency targets

### Build Configurations
- `Makefile` (root) - Host application build
- `tcl_template/Makefile` - Kernel + host build
- `Makefile.hls` - HLS-only build

## Recommended Workflow

1. **First Time Setup**
   ```bash
   cd MiniRocketHLS/tcl_template
   python3 train_minirocket_for_fpga.py CBF --features 420
   ```

2. **HLS Validation**
   ```bash
   cd MiniRocketHLS
   ./build_sim.sh  # Runs HLS C simulation
   ```

3. **FPGA Build** (2-5 hours)
   ```bash
   cd tcl_template
   make build TARGET=hw
   ```

4. **Run Benchmark**
   ```bash
   ./ucr_benchmark_4cu build_dir.hw.*/krnl.xclbin \
       cbf_fpga_model.json cbf_fpga_test.json
   ```

## Notes for Researchers

- All core algorithm code is in `minirocket_inference_hls.cpp`
- Model parameters are in `minirocket_model_constants.h` (auto-generated)
- To modify compute units, edit `nk=krnl_top:N` in `config.cfg`
- Performance results are documented in `RESULTS.md`
- Build logs are in `_x*/logs/` (not committed)

## Version Control Best Practices

### Always Commit
- Source code (.cpp, .h, .py)
- Build scripts (.sh, Makefile, .cfg)
- Documentation (.md)
- Small example models (<10 MB)

### Never Commit
- Build outputs (build_dir.*, _x.*, etc.)
- Large models (>10 MB)
- Logs (.log, .jou)
- Binaries (executables, .xclbin > 100 MB)
- Personal configuration (.vscode/, .idea/)

---

**Maintained by**: MiniRocket FPGA Team
**Last Updated**: December 2025
**For Questions**: See GitHub issues or FPGA_IMPLEMENTATION.md
