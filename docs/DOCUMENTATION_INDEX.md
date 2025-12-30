# MiniRocket HLS - Documentation Index

This document serves as a comprehensive guide to all documentation in the repository, organized by purpose and audience.

---

## Quick Navigation

| Document | Purpose | Audience | Priority |
|----------|---------|----------|----------|
| [README.md](#readmemd) | Project overview & quick start | All users | ⭐⭐⭐ |
| [FILE_STRUCTURE.md](#file_structuremd) | Complete file reference | Developers, Researchers | ⭐⭐⭐ |
| [ALGORITHM.md](#algorithmmd) | Mathematical specification | Researchers, Algorithm developers | ⭐⭐⭐ |
| [RESULTS.md](#resultsmd) | Experimental validation | Researchers, Users | ⭐⭐ |
| [FPGA_IMPLEMENTATION.md](#fpga_implementationmd) | FPGA technical details | FPGA developers | ⭐⭐ |
| [DOCUMENTATION_SUMMARY.md](#documentation_summarymd) | Legacy summary | Reference only | ⭐ |

---

## Document Descriptions

### README.md
**Location**: Root directory
**Purpose**: Entry point for all users
**Contents**:
- Project overview and motivation
- Quick start guide (build, run, validate)
- Key performance results
- Repository structure overview
- Installation prerequisites
- Usage examples

**When to read**: First-time users, quick reference

**Key Sections**:
1. **Overview**: What is MiniRocket HLS?
2. **Key Results**: Performance benchmarks (1:1 vs optimized)
3. **Quick Start**: Build and run instructions
4. **Training**: How to generate models from UCR datasets
5. **Validation**: Verifying accuracy against Python baseline

---

### FILE_STRUCTURE.md
**Location**: Root directory
**Purpose**: Comprehensive reference for all project files
**Contents**:
- Role and importance of every file in the repository
- File dependencies and relationships
- Build artifact descriptions
- Model and test data specifications
- Development workflow documentation
- Reproducibility checklist

**When to read**:
- Setting up development environment
- Understanding code organization
- Debugging build issues
- Extending the implementation

**Key Sections**:
1. **Core Documentation Files**: Overview of all docs
2. **Source Code Directory**: Detailed HLS and host code descriptions
3. **Build Configuration**: Makefile, config.cfg explanation
4. **Model Files**: JSON format specifications
5. **Build Artifacts**: What gets generated and where
6. **Optimized Reference**: Archived implementation details
7. **Development Workflow**: Step-by-step build process

**Dependencies**:
- Complements README.md with technical depth
- References ALGORITHM.md for mathematical details
- Links to RESULTS.md for validation methodology

---

### ALGORITHM.md
**Location**: Root directory
**Purpose**: Detailed algorithmic and mathematical documentation
**Contents**:
- 1:1 paper-faithful implementation rationale
- Mathematical formulation of cumulative convolution
- Kernel weight analysis (α=-1, γ=+3)
- PPV (Proportion of Positive Values) feature extraction
- StandardScaler normalization impact
- Ridge classifier integration
- Why optimizations preserve accuracy

**When to read**:
- Understanding the MiniRocket algorithm
- Verifying mathematical correctness
- Comparing 1:1 vs optimized implementations
- Research paper writing

**Key Insights**:
1. **Cumulative Convolution**: How C_alpha and C_gamma are computed
2. **Weight Selection**: Why {-1, +2} in paper translates to α=-1, γ=+3
3. **Normalization Effect**: StandardScaler removes scale differences
4. **Classifier Adaptation**: Ridge learns different weights for different feature spaces

**Mathematical Equations**:
- Convolution: `value_alpha = -X[i], value_gamma = 3×X[i]`
- Cumulative Sum: `C[i] = C[i-1] + value`
- PPV Feature: `feature = (sum(C > bias) / length)`
- Normalization: `X_scaled = (X - mean) / std`
- Classification: `y = argmax(coefficients @ features + intercept)`

**Validation**: References Python CPU baseline and UCR benchmark results

---

### RESULTS.md
**Location**: Root directory
**Purpose**: Experimental validation and performance analysis
**Contents**:
- UCR dataset benchmark results
- Accuracy validation against Python baseline
- Throughput measurements (inferences/second)
- FPGA resource utilization
- Build time and frequency analysis
- Comparison with state-of-the-art

**When to read**:
- Evaluating implementation quality
- Comparing with other accelerators
- Writing research papers (results section)
- Debugging accuracy issues

**Key Results Tables**:
1. **Accuracy Validation** (1:1 Reference):
   - GunPoint: 98.33% (59/60 correct)
   - ItalyPowerDemand: 97.26% (320/329 correct)
   - ArrowHead: 83.43% (146/175 correct)

2. **Throughput Benchmarks**:
   - 1-CU Reference: 45.8 - 250 inf/sec (dataset dependent)
   - Optimized (4-CU): 3,468 - 19,267 inf/sec
   - Speedup: 75.7x - 77.1x

3. **Resource Utilization**:
   - LUTs, FFs, BRAMs, DSPs breakdown
   - Clock frequency: 300 MHz (1:1), 404 MHz (optimized)

**Reproducibility**: Instructions for re-running benchmarks

---

### FPGA_IMPLEMENTATION.md
**Location**: Root directory
**Purpose**: FPGA-specific technical documentation
**Contents**:
- HLS optimization strategies
- Pipeline design and II (Initiation Interval) optimization
- Memory architecture and HBM banking
- Compute unit configuration
- Clock frequency and timing closure
- Multi-CU scaling challenges

**When to read**:
- Optimizing HLS code
- Debugging timing violations
- Scaling to multiple CUs
- Understanding resource bottlenecks

**Key Topics**:
1. **HLS Pragmas**:
   - `#pragma HLS PIPELINE II=1`
   - `#pragma HLS ARRAY_PARTITION`
   - `#pragma HLS INTERFACE m_axi`

2. **Memory Banking**:
   - Why 9 HBM ports per CU (1:1 reference)
   - HBM port limitations (32 max on U280)
   - Buffer allocation strategies

3. **Multi-CU Considerations**:
   - XRT native API vs OpenCL
   - Explicit bank placement requirements
   - Host-side parallelization

4. **Timing Optimization**:
   - Critical path analysis
   - Loop unrolling trade-offs
   - Frequency vs resource trade-offs

---

### DOCUMENTATION_SUMMARY.md
**Location**: Root directory
**Status**: Legacy document (superseded by this index)
**Purpose**: Original consolidated documentation
**Note**: Kept for historical reference; prefer DOCUMENTATION_INDEX.md

---

## Documentation by Use Case

### For First-Time Users
1. Start with [README.md](#readmemd)
2. Follow quick start guide to build and run
3. Validate results against [RESULTS.md](#resultsmd)
4. If interested in algorithm details, read [ALGORITHM.md](#algorithmmd)

### For Developers Extending the Code
1. Read [FILE_STRUCTURE.md](#file_structuremd) thoroughly
2. Review [FPGA_IMPLEMENTATION.md](#fpga_implementationmd) for HLS details
3. Consult [ALGORITHM.md](#algorithmmd) for correctness validation
4. Use [RESULTS.md](#resultsmd) as performance baseline

### For Researchers Writing Papers
1. Cite [README.md](#readmemd) for implementation overview
2. Reference [ALGORITHM.md](#algorithmmd) for mathematical formulation
3. Use [RESULTS.md](#resultsmd) for experimental validation
4. Include [FILE_STRUCTURE.md](#file_structuremd) for reproducibility

### For Debugging Build Issues
1. Check [FILE_STRUCTURE.md](#file_structuremd) → Build Artifacts section
2. Review Makefile and config.cfg descriptions
3. Consult [FPGA_IMPLEMENTATION.md](#fpga_implementationmd) for timing issues
4. Verify against [README.md](#readmemd) prerequisites

---

## Additional Resources

### In-Code Documentation
- **Source files**: Inline comments explaining HLS optimizations
- **Header files**: Interface specifications and constants
- **Build scripts**: Comments in Makefile and TCL scripts

### External References
- **Original Paper**: Dempster et al., "MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification," KDD 2021
- **UCR Archive**: [UCR Time Series Classification Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
- **Vitis HLS**: [User Guide UG1399](https://docs.xilinx.com/r/en-US/ug1399-vitis-hls)

### Repository Branches
- `master`: Main development branch (currently has 1:1 reference code)
- `1cu-reference-build`: Stable 1-CU configuration (recommended for research)
- Future: `optimized-multi-cu` (if multi-CU host issues are resolved)

---

## Document Maintenance

### Update Frequency
- **README.md**: Updated when major features or results change
- **FILE_STRUCTURE.md**: Updated when repository structure changes
- **ALGORITHM.md**: Stable (only updated for algorithm changes)
- **RESULTS.md**: Updated after new benchmarks
- **FPGA_IMPLEMENTATION.md**: Updated with new optimizations

### Version Control
All documentation is version-controlled alongside code. Use git history to see evolution:
```bash
git log --oneline -- <filename>.md
```

### Contributing to Documentation
When modifying code:
1. Update relevant sections in FILE_STRUCTURE.md
2. Add performance results to RESULTS.md
3. Update README.md if user-facing changes
4. Document algorithm changes in ALGORITHM.md

---

## Quick Reference Card

### I want to...
| Goal | Document(s) to Read |
|------|---------------------|
| Build the project | README.md (Quick Start) |
| Understand file X | FILE_STRUCTURE.md |
| Validate accuracy | RESULTS.md |
| Modify HLS code | FILE_STRUCTURE.md + FPGA_IMPLEMENTATION.md |
| Understand the algorithm | ALGORITHM.md |
| Reproduce results | FILE_STRUCTURE.md (Reproducibility Checklist) |
| Scale to more CUs | FPGA_IMPLEMENTATION.md (Multi-CU section) |
| Train new models | README.md (Training section) |
| Debug timing issues | FPGA_IMPLEMENTATION.md |
| Write a paper | ALGORITHM.md + RESULTS.md |

---

## Feedback and Improvements

This documentation is actively maintained. If you find:
- Missing information
- Unclear explanations
- Broken links or references
- Opportunities for improvement

Please file an issue on the GitHub repository with tag `documentation`.

---

**Last Updated**: December 27, 2025
**Maintained By**: MiniRocket HLS Development Team
**Version**: 1.0
