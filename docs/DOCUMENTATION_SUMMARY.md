# MiniRocketHLS Documentation Overview

## Repository Organization Complete

The MiniRocketHLS repository has been reorganized with comprehensive documentation explaining the research, implementation, and performance results.

---

## Documentation Structure

### 1. [README.md](README.md) - Quick Start & Overview
**Purpose**: Primary entry point for users

**Contents**:
- Project overview and key results (77x speedup, 100% accuracy)
- Quick start guide for building and running on FPGA
- Installation instructions
- Usage examples
- Troubleshooting guide
- Configuration options

**Target Audience**: New users, implementers

---

### 2. [ALGORITHM.md](ALGORITHM.md) - Algorithm Explanation
**Purpose**: Understand MiniRocket and FPGA optimizations

**Contents**:
- Original MiniRocket algorithm explanation
- FPGA-specific optimizations (4 key improvements)
- Side-by-side comparison of 1:1 vs optimized implementations
- Mathematical justification for -1,0,+1 simplification
- Performance breakdown by optimization

**Target Audience**: Researchers, algorithm developers

**Key Insight**: Explains *why* the optimized version is 77x faster while maintaining 100% accuracy

---

### 3. [RESULTS.md](RESULTS.md) - Benchmark Results & Analysis
**Purpose**: Comprehensive performance data and analysis

**Contents**:
- Detailed benchmark results (throughput, latency, accuracy)
- Performance comparison tables
- Resource utilization analysis
- Scalability analysis (1-4 CUs)
- Energy efficiency metrics (1,069x better than CPU)
- Error analysis and debugging journey

**Target Audience**: Performance engineers, system designers

**Key Findings**:
- 1:1 Reference: 45 inf/sec @ 242 MHz (1 CU)
- Optimized: 3,468 inf/sec @ 404 MHz (4 CU)
- Both achieve 100% accuracy

---

### 4. [FPGA_IMPLEMENTATION.md](FPGA_IMPLEMENTATION.md) - Implementation Details
**Purpose**: Technical implementation guide (planned - not created yet)

**Planned Contents**:
- HLS synthesis flow
- Kernel interface design
- Memory architecture
- Host-device communication
- Build process walkthrough

**Target Audience**: FPGA developers, implementers

---

## Additional Documentation

### 5. [../1to1_vs_optimized_comparison.md](../1to1_vs_optimized_comparison.md)
**Purpose**: Detailed comparison study

**Contents**:
- Complete comparison methodology
- Build process documentation
- Interface mismatch debugging
- Memory bank connectivity analysis
- Recommendation for production use

---

## Key Research Findings

### Performance Comparison

| Metric | 1:1 Reference | Optimized | Winner |
|--------|---------------|-----------|---------|
| **Throughput** | 45 inf/sec | 3,468 inf/sec | Optimized (77x) |
| **Clock Freq** | 242 MHz | 404 MHz | Optimized (1.67x) |
| **Active CUs** | 1/4 | 4/4 | Optimized |
| **Accuracy** | 100% | 100% | **Tie** |

### Optimization Techniques

1. **Simplified Kernel Weights**: -1,0,+1 instead of -1,+2
2. **Convolution Placement**: 1× per dilation instead of 84×
3. **Higher Clock**: Simpler logic → better timing
4. **Multi-CU**: All 4 CUs working vs 1

### Validation

**Both implementations achieve 100% accuracy** on UCR test dataset (300 samples, 4 classes), validating that optimizations maintain algorithmic correctness.

---

## Repository Structure

```
MiniRocketHLS/
├── README.md                         # Start here!
├── ALGORITHM.md                      # Algorithm & optimizations
├── RESULTS.md                        # Performance benchmarks
├── FPGA_IMPLEMENTATION.md           # Implementation guide (planned)
├── DOCUMENTATION_SUMMARY.md         # This file
│
├── tcl_template/                    # Main implementation
│   ├── src/
│   │   ├── minirocket_inference_hls.cpp  # Core HLS kernel (1:1 reference)
│   │   ├── minirocket_host.cpp           # Host application
│   │   └── ...
│   ├── config.cfg                   # 4 CU configuration
│   ├── Makefile                     # Build system
│   ├── minirocket_ucr_model.json    # Trained model
│   └── minirocket_ucr_model_test_data.json
│
├── comparison_study/                 # Performance comparison artifacts
├── optimized_version_77b3cee/        # Archived optimized implementation
└── .gitignore

../
└── 1to1_vs_optimized_comparison.md  # Detailed comparison study
```

---

## How to Use This Documentation

### For First-Time Users
1. Start with [README.md](README.md) - Installation and quick start
2. Read [ALGORITHM.md](ALGORITHM.md) - Understand what MiniRocket does
3. Check [RESULTS.md](RESULTS.md) - See performance expectations

### For Researchers
1. [ALGORITHM.md](ALGORITHM.md) - Algorithm theory and optimizations
2. [../1to1_vs_optimized_comparison.md](../1to1_vs_optimized_comparison.md) - Comparison methodology
3. [RESULTS.md](RESULTS.md) - Detailed performance analysis

### For Implementers
1. [README.md](README.md) - Build instructions
2. [FPGA_IMPLEMENTATION.md](FPGA_IMPLEMENTATION.md) - Technical details (planned)
3. [RESULTS.md](RESULTS.md) - Resource utilization and scalability

---

## Files Removed/Archived

**Removed**:
- BUILD_ERROR_REPORT.txt (outdated)
- tcl_template/BUILD_NOTES.md (superseded by README.md)
- tcl_template/1to1_reference_test_output.txt (integrated into RESULTS.md)

**Archived**:
- ALGORITHM.md.backup (previous version, kept for reference)

---

## Next Steps

### For Documentation
- [ ] Create FPGA_IMPLEMENTATION.md with technical details
- [ ] Add diagrams to ALGORITHM.md (convolution visualization)
- [ ] Add power measurement results to RESULTS.md
- [ ] Create BUILD_GUIDE.md for detailed build instructions

### For Implementation
- [ ] Test 8 CU configuration for higher throughput
- [ ] Experiment with ap_fixed<24,12> for lower resource usage
- [ ] Profile energy consumption with Vitis Analyzer
- [ ] Test on additional UCR datasets

---

## Citation

If you use this work in research, please cite:

```bibtex
@inproceedings{dempster2021minirocket,
  title={MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification},
  author={Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
  booktitle={KDD},
  year={2021}
}

@misc{minirockethls2025,
  title={MiniRocket FPGA Accelerator},
  author={Dave, Rohan},
  year={2025},
  url={https://github.com/YOUR_USERNAME/MiniRocketHLS}
}
```

---

**Documentation Created**: December 23, 2025
**Last Updated**: December 23, 2025
