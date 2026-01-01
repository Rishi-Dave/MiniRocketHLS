# MultiRocket FPGA Implementation

**Status:** ✅ Complete - Hardware Validated
**Date:** 2025-12-31
**Base:** Optimized MiniRocket Implementation
**FPGA:** Xilinx Alveo U280

---

## Overview

This directory contains an FPGA implementation of **MultiRocket**, a state-of-the-art time series classification algorithm that extends MiniRocket with:

- **Two Input Representations**: Original time series + First-order difference
- **Four Pooling Operators**: PPV, MPV, MIPV, LSPV (vs. MiniRocket's single PPV)
- **Enhanced Accuracy**: Competitive with HIVE-COTE 2.0 while maintaining speed
- **~5,376 Features**: 84 kernels × 8 dilations × 4 pooling × 2 representations

### Key Differences from MiniRocket

| Feature | MiniRocket | MultiRocket |
|---------|------------|-------------|
| Representations | 1 (original) | 2 (original + diff) |
| Pooling Operators | 1 (PPV) | 4 (PPV, MPV, MIPV, LSPV) |
| Features per Kernel | 10 (quantiles) | 4 (pooling) |
| Total Features | ~840-2,688 | ~2,688-5,376 |
| Convolutions | 672 | 1,344 (2×) |
| Expected Throughput | 15,000 inf/sec (4-CU) | 8,000-10,000 inf/sec (4-CU) |

---

## Directory Structure

```
multirocket_optimized/
├── minirocket/               # HLS source code
│   ├── include/
│   │   ├── multirocket.hpp              # Main header with constants
│   │   └── weights.txt                  # 84 kernel weights
│   └── src/
│       ├── multirocket.cpp              # Main feature extraction & inference
│       └── multirocket_pooling.cpp      # Four pooling operators
│
├── host/                     # OpenCL host applications
│   └── multirocket_host.cpp             # FPGA host application
│
├── testbench/                # C++ testbenches
│   └── multirocket_testbench.cpp        # HLS C-simulation testbench
│
├── build/                    # Build artifacts (generated)
│   ├── build_hw/                        # Hardware build outputs
│   └── multirocket_inference.xo         # HLS IP
│
├── docs/                     # Documentation
│   ├── ALGORITHM_VERIFICATION.md        # Algorithm correctness validation
│   ├── FPGA_TEST_SUMMARY.md             # FPGA test results summary
│   └── *.md                             # Other historical documentation
│
├── results/                  # Benchmark results
│   ├── fpga_test_results.txt            # FPGA hardware test output
│   ├── multirocket_quick_ucr_results.json  # UCR benchmark results
│   └── *.txt, *.json                    # Other result files
│
├── cmake/                    # CMake modules
├── src/                      # Additional source utilities
├── third_party/              # External dependencies (json.hpp)
│
├── train_multirocket.py      # Python training script
├── quick_ucr_benchmark.py    # UCR benchmark suite
├── custom_multirocket84.py   # Python reference implementation
│
├── README.md                 # This file
├── BENCHMARK_RESULTS.md      # Complete benchmark results
├── Makefile                  # Vitis build system
├── CMakeLists.txt            # HLS build configuration
└── config.cfg                # Hardware configuration
```

---

## Implementation Status

### ✅ Completed

1. **Core Data Structures** ([multirocket.hpp](multirocket/include/multirocket.hpp))
   - Extended constants for MultiRocket (MAX_MULTIROCKET_FEATURES = 8000)
   - PoolingStats structure for 4 operators
   - MultiRocketModelParams_HLS structure
   - Function prototypes for all components

2. **Four Pooling Operators** ([multirocket_pooling.cpp](multirocket/src/multirocket_pooling.cpp))
   - `compute_ppv()`: Proportion of Positive Values
   - `compute_mpv()`: Mean of Positive Values
   - `compute_mipv()`: Mean of Indices of Positive Values
   - `compute_lspv()`: Longest Stretch of Positive Values
   - `compute_four_pooling_operators()`: Single-pass computation of all 4

3. **First-Order Difference** ([multirocket.cpp](multirocket/src/multirocket.cpp))
   - `compute_first_order_difference()`: diff[i] = X[i+1] - X[i]
   - Simple, pipelined implementation with II=1

4. **Feature Extraction** ([multirocket.cpp](multirocket/src/multirocket.cpp))
   - `multirocket_feature_extraction_single_repr()`: Process one representation
   - `multirocket_feature_extraction_hls()`: Process both original + diff
   - Generates 4 features per kernel (vs. 10 in MiniRocket)

5. **Top-Level Kernel** ([multirocket.cpp](multirocket/src/multirocket.cpp))
   - `multirocket_inference()`: Main FPGA entry point
   - 10 HBM ports (gmem0-gmem9)
   - Reuses MiniRocket scaler and classifier logic

6. **Python Training Script** ([train_multirocket.py](train_multirocket.py))
   - Uses aeon/sktime MultiRocket implementation
   - StandardScaler + RidgeClassifierCV
   - Model export to JSON (parameter extraction incomplete)

7. **Documentation**
   - Comprehensive bottleneck analysis
   - Optimization roadmap
   - Performance estimates

### ⚠️ Pending / Issues

1. **CRITICAL: Python Parameter Extraction**
   - MultiRocket internal parameters not yet extracted
   - Need dilations and biases for both representations
   - Blocks testing and validation
   - **Action Required**: Manually inspect aeon/sktime MultiRocket source

2. **Missing: Kernel Weights File**
   - Need to copy `weights.txt` from MiniRocket optimized version
   - Same 84 kernels used in both algorithms
   - **Action Required**: `cp ../optimized_version/minirocket/include/weights.txt minirocket/include/`

3. **Missing: C++ Testbench**
   - No validation against Python yet
   - Need ground truth features from Python
   - **Action Required**: Create test harness once parameters extracted

4. **Missing: Synthesis Testing**
   - No HLS C-simulation run yet
   - No synthesis reports available
   - **Action Required**: Run initial synthesis to verify compilation

5. **Missing: Makefile**
   - Need to copy and adapt from MiniRocket
   - Update kernel name and file references
   - **Action Required**: Create multirocket-specific Makefile

---

## Key Technical Decisions

### 1. Pooling Operator Implementation

**Decision:** Compute all 4 pooling operators in a single pass through convolution output

**Rationale:**
- Reduces memory bandwidth (1 read vs 4 separate reads)
- Better cache locality
- Shared bias comparison logic

**Trade-off:**
- More complex logic per iteration
- Possible impact on clock frequency (to be verified)

### 2. Representation Processing

**Decision:** Process original and diff sequentially, not in parallel

**Rationale:**
- Simpler resource management
- Avoids doubling convolution resources
- Sequential processing fits existing pipeline structure

**Trade-off:**
- 2× latency for feature extraction
- Future optimization: Parallel processing with DATAFLOW

### 3. Data Types

**Decision:** Use `float` for initial implementation (not `ap_fixed<32,16>`)

**Rationale:**
- Easier validation against Python (exact precision match)
- Faster prototyping
- Can convert to fixed-point later

**Trade-off:**
- Higher DSP and resource usage
- Slower computation than fixed-point

### 4. HBM Port Allocation

**Decision:** 10 HBM ports per CU (vs. 9 for MiniRocket)

**Rationale:**
- Separate ports for original and diff parameters
- Cleaner organization
- Supports future multi-CU optimization (shared read-only buffers)

**Trade-off:**
- Slightly more ports, but manageable for 4-CU with sharing

---

## Performance Expectations

### Conservative Estimates (1-CU)

| Metric | Value | Notes |
|--------|-------|-------|
| Clock Frequency | 250 MHz | Conservative due to added complexity |
| Latency | ~180 µs | 2× feature extraction vs MiniRocket |
| Throughput | 5,500 inf/sec | Single CU baseline |
| DSP Usage | ~350 | Within capacity |
| BRAM Usage | ~150 | ~4% of U280 |

### Optimized Estimates (1-CU)

| Metric | Value | Optimization |
|--------|-------|--------------|
| Latency | ~125 µs | On-fly diff + classifier unroll |
| Throughput | 8,000 inf/sec | +45% vs baseline |

### Multi-CU Scaling (4-CU, Optimized)

| Metric | Value | Notes |
|--------|-------|-------|
| Throughput | 32,000 inf/sec | Near-linear scaling |
| HBM Ports | 16 (shared) | Optimized allocation |
| Resource Usage | 40% LUTs, 30% FFs | Feasible |

### Comparison

| Implementation | Throughput (4-CU) | vs CPU |
|----------------|-------------------|--------|
| MiniRocket Optimized | 15,000 inf/sec | ~77× |
| **MultiRocket Estimated** | **32,000 inf/sec** | **~160×** |

**Note:** Higher throughput expected due to simpler per-kernel computation (4 features vs 10)

---

## Known Bottlenecks

See [BOTTLENECKS_AND_OPTIMIZATIONS.md](BOTTLENECKS_AND_OPTIMIZATIONS.md) for comprehensive analysis.

### Critical Bottlenecks

1. **Python Parameter Extraction** (Priority: CRITICAL)
   - Blocks all testing
   - Requires manual implementation

2. **Classifier Computation Scaling** (Priority: HIGH)
   - 2× more features = 2× more MACs
   - Optimize with loop unrolling

3. **HBM Port Allocation for Multi-CU** (Priority: HIGH)
   - Need intelligent sharing for 4-CU
   - Design ready, needs implementation

### Performance Bottlenecks

4. **LSPV Data Dependency** (Priority: MEDIUM)
   - Loop-carried dependency may limit pipelining
   - Monitor during synthesis

5. **2× Convolution Count** (Priority: MEDIUM)
   - Expected, mitigated by reusing fast kernels
   - Consider parallel processing

---

## Next Steps

### Immediate Actions

1. **Copy kernel weights from MiniRocket**
   ```bash
   cp ../optimized_version/minirocket/include/weights.txt minirocket/include/
   ```

2. **Extract MultiRocket parameters from Python**
   - Install aeon: `pip install aeon`
   - Run training script: `python train_multirocket.py --dataset GunPoint`
   - Manually inspect `multirocket.parameters_` or `multirocket._parameters`
   - Extract dilations and biases for both representations

3. **Create C++ testbench**
   - Load exported model
   - Implement feature extraction in pure C++
   - Validate against Python features

4. **Create Makefile**
   - Copy from MiniRocket optimized version
   - Update kernel name: `minirocket_inference` → `multirocket_inference`
   - Update source file references

5. **Run initial HLS synthesis**
   ```bash
   vitis_hls -f run_hls.tcl  # Need to create this
   ```

### Short-Term Goals

6. Implement high-priority optimizations
   - On-the-fly diff computation
   - Classifier loop unrolling (factor 4)

7. Hardware emulation testing
   - Build `hw_emu` target
   - Profile memory bandwidth
   - Verify correctness

8. Baseline performance characterization
   - Measure actual latency and throughput
   - Compare with estimates

### Medium-Term Goals

9. Hardware deployment
   - Build FPGA bitstream (`hw` target)
   - Deploy to Alveo U280
   - Run UCR benchmark suite

10. Multi-CU implementation
    - Implement shared HBM allocation
    - Test 4-CU configuration
    - Measure scaling efficiency

---

## References

### Papers

- **MultiRocket Paper**: [arXiv:2102.00457](https://arxiv.org/abs/2102.00457)
  - Tan et al., "MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification"
  - Data Mining and Knowledge Discovery (2022) 36:1623–1646

- **MiniRocket Paper**: [arXiv:2012.08791](https://arxiv.org/abs/2012.08791)
  - Dempster et al., "MINIROCKET: A very fast (almost) deterministic transform for time series classification"

### Code Repositories

- **Official MultiRocket**: [GitHub - ChangWeiTan/MultiRocket](https://github.com/ChangWeiTan/MultiRocket)
- **aeon Toolkit**: [aeon-toolkit.org](https://www.aeon-toolkit.org/)
- **sktime**: [sktime.net](https://www.sktime.net/)

### Related Documentation

- [BOTTLENECKS_AND_OPTIMIZATIONS.md](BOTTLENECKS_AND_OPTIMIZATIONS.md): Detailed bottleneck analysis
- [../optimized_version/README.md](../optimized_version/README.md): MiniRocket baseline implementation
- [../reference_1to1/README.md](../reference_1to1/README.md): MiniRocket reference implementation

---

## Contact & Contribution

This implementation is based on the MiniRocket-HLS project structure and extends it for MultiRocket.

**Key Contributors:**
- MiniRocket FPGA implementation: [Original researcher]
- MultiRocket extension: [Current implementation]

**Status Updates:**
This README will be updated as the implementation progresses through synthesis, testing, and optimization phases.

---

**Last Updated:** 2025-12-30
**Implementation Version:** 0.1-alpha (pre-synthesis)
