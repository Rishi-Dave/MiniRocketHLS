# MiniRocket FPGA Research Plan

## Executive Summary
This document outlines the next steps for conducting comprehensive FPGA inference research with the MiniRocket accelerator on Xilinx Alveo U280.

**Current Status**: ✅ FPGA deployment successful, single inference validated

**Research Goals**:
1. Validate FPGA inference on complex, multi-sample datasets
2. Quantify CPU vs FPGA performance tradeoffs
3. Generate publishable research insights on FPGA acceleration for time series classification

---

## Phase 1: Multi-Dataset Validation (Week 1)

### Objective
Validate FPGA inference accuracy and performance on datasets with varying characteristics.

### Dataset Requirements
- **Small**: 100-1,000 samples, 512-length time series
- **Medium**: 10,000-50,000 samples, 512-1,024 length
- **Large**: 100,000+ samples (batch processing), up to FPGA memory limits
- **Complexity**: Varying number of classes (2, 10, 100)

### FPGA Memory Capacity Analysis
Current U280 HBM configuration:
- **Available**: 220 BRAM, 32 HBM banks
- **Per inference**: ~160 KB (coefficients + intermediate data)
- **Theoretical capacity**: ~1,000 inferences in parallel (if batched)
- **Realistic target**: Process 10K-100K samples sequentially

### Implementation Tasks
1. **Create batch inference host application**
   - Load large datasets from CSV/binary
   - Implement batching strategy (batch size: 1, 10, 100, 1000)
   - Time each batch separately
   - Validate output against CPU reference

2. **Modify host_app.cpp**
   ```cpp
   // Add batch processing loop
   for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
       // Transfer batch to FPGA
       // Execute kernel
       // Retrieve results
       // Validate
   }
   ```

3. **Datasets to test**
   - UCR/UEA Time Series Archive (standard benchmarks)
   - ECG classification (medical)
   - Activity recognition (wearable sensors)
   - Industrial anomaly detection

---

## Phase 2: CPU vs FPGA Benchmarking (Week 1-2)

### Metrics to Measure

#### Throughput Metrics
| Metric | CPU | FPGA | Target Speedup |
|--------|-----|------|----------------|
| Single inference latency (ms) | ? | ~0.135-0.533 | N/A |
| Throughput (inferences/sec) | ? | 1,900-7,400 | >10x |
| Batch-100 throughput | ? | ? | >50x |
| Batch-1000 throughput | ? | ? | >100x |

#### Energy Metrics
| Metric | CPU | FPGA | Target Improvement |
|--------|-----|------|-------------------|
| Power consumption (W) | ~100-200 | ~75 (U280) | 1.3-2.7x |
| Energy per inference (mJ) | ? | ? | >50x |
| Energy efficiency (inferences/J) | ? | ? | >50x |

#### Resource Utilization
| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUT | 22,183 | ~1.3M | 1.7% |
| FF | 19,651 | ~2.6M | 0.75% |
| BRAM | 220 | 2,688 | 8.2% |
| DSP | 0 | 9,024 | 0% |

**Opportunity**: Very low utilization → can replicate kernel multiple times for higher throughput

### Benchmark Implementation

#### CPU Baseline
```bash
# Run Python/C++ CPU inference
time python benchmark_cpu.py --dataset <dataset> --samples 10000
```

#### FPGA Benchmark
```bash
# Run FPGA inference with timing
./host krnl.xclbin dataset.json --batch-size 100 --num-samples 10000 --measure-time
```

#### What to Measure
1. **Latency breakdown**:
   - Data transfer H→D
   - Kernel execution
   - Data transfer D→H
   - Total end-to-end

2. **Throughput scaling**:
   - Single sample
   - Batch sizes: 1, 10, 100, 1000
   - Identify optimal batch size

3. **Accuracy validation**:
   - Compare FPGA vs CPU predictions
   - Acceptable tolerance: <0.1% difference in scores
   - Track classification agreement: >99.9%

---

## Phase 3: Performance Optimization (Week 2-3)

### Identified Bottlenecks

From system estimate report:
- **Achieved frequency**: 404 MHz (target: 300 MHz) → 35% margin
- **Resource utilization**: <2% → massive underutilization
- **Memory bandwidth**: Using HBM but sequential transfers

### Optimization Strategies

#### 1. **Kernel Replication** (Expected: 5-10x speedup)
- Current: 1 compute unit
- Proposed: 4-8 parallel compute units
- Implementation: Modify `config.cfg`:
  ```ini
  [connectivity]
  nk=krnl_top:4
  ```

#### 2. **Data Transfer Pipelining** (Expected: 2-3x speedup)
- Current: Sequential H→D, Execute, D→H
- Proposed: Overlap transfers with execution
- Implementation: Use async XRT APIs + double buffering

#### 3. **On-Chip Caching** (Expected: 1.5-2x speedup)
- Cache frequently reused parameters (scaler, coefficients)
- Reduce DDR/HBM accesses by 50%

#### 4. **Batch Processing in Kernel** (Expected: 3-5x speedup)
- Process multiple samples per kernel invocation
- Amortize kernel launch overhead

### Expected Performance After Optimization
| Scenario | Current | Optimized | Speedup |
|----------|---------|-----------|---------|
| Throughput (inf/sec) | 1,900-7,400 | 50,000-200,000 | 10-30x |
| Latency (single) | 0.135-0.533 ms | 0.050-0.100 ms | 2-5x |
| vs CPU speedup | ? | >100x | N/A |

---

## Phase 4: Research Report Generation (Week 3-4)

### Report Structure

#### 1. Introduction
- Time series classification importance
- MiniRocket algorithm overview
- FPGA acceleration motivation

#### 2. Methodology
- HLS design flow
- Hardware architecture
- Optimization techniques applied

#### 3. Experimental Setup
- Platform: Xilinx Alveo U280
- Datasets: UCR/UEA benchmarks
- Baselines: CPU (Python), GPU (if available)

#### 4. Results
- **Accuracy validation**: FPGA matches CPU to <0.1%
- **Performance comparison**: Tables and graphs
  - Latency vs batch size
  - Throughput vs number of CUs
  - Energy efficiency analysis
- **Resource utilization**: Efficiency of FPGA resources
- **Scalability**: Performance vs dataset size

#### 5. Discussion
- **Tradeoffs identified**:
  - FPGA: High throughput, low latency, energy efficient
  - CPU: Flexible, easier development, better for small batches
  - GPU: Highest peak throughput, but higher power
- **When to use FPGA**: Real-time inference, edge deployment, energy-constrained
- **Limitations**: Fixed-point precision, development complexity

#### 6. Conclusion
- FPGA achieves Xx speedup at Yx better energy efficiency
- Suitable for production deployment in time series applications

### Deliverables
- [ ] IEEE conference paper (6-8 pages)
- [ ] Performance benchmark suite
- [ ] Open-source FPGA implementation
- [ ] Visualization dashboard (latency/throughput plots)

---

## Phase 5: Code Cleanup & Documentation (Week 4)

### Files to Keep

#### Essential Source Files
```
tcl_template/
├── src/
│   ├── krnl.cpp               # Main FPGA kernel
│   ├── krnl.hpp               # Kernel header
│   └── types.h                # Type definitions
├── minirocket_host.cpp        # XRT host application
├── host.h                     # Host header
├── Makefile                   # Build system
├── config.cfg                 # Vitis link config
├── export_ip.tcl              # HLS export script
└── minirocket_model.json      # Model parameters
```

#### Build Outputs (Keep for reference)
```
build_dir.hw.*/krnl.xclbin     # Hardware bitstream
host                           # Compiled host binary
```

#### Documentation (Keep updated versions)
```
README.md                      # Project overview
RESEARCH_PLAN.md               # This file
```

### Files to Remove

#### Redundant Documentation
- [ ] FPGA_SYNTHESIS_GUIDE.md (outdated, replaced by README)
- [ ] OPTIMIZATION_SUMMARY.md (merge into RESEARCH_PLAN)
- [ ] QUICK_START_FPGA.md (merge into README)
- [ ] PRE_FPGA_TEST_REPORT.md (obsolete, FPGA working)
- [ ] GIT_COMMIT_GUIDE.md (standard git practices)
- [ ] HLS_NEWCOMER_GUIDE.md (not project-specific)
- [ ] NEXT_STEPS.md (replaced by RESEARCH_PLAN)
- [ ] READY_FOR_FPGA.md (obsolete)
- [ ] HOW_TO_MONITOR_FPGA_BUILD.md (one-time need)
- [ ] MINIROCKET_PIPELINE_DETAILED.md (merge into README)

#### My Created Files (Unnecessary)
- [ ] tcl_template/minirocket.cfg (wrong config)
- [ ] tcl_template/host_app.cpp (redundant with minirocket_host.cpp)
- [ ] tcl_template/build_fpga.sh (use Makefile)
- [ ] tcl_template/export_and_package.tcl (use export_ip.tcl)

#### Build Artifacts (Add to .gitignore)
- [ ] _x.* directories
- [ ] build_dir.* directories
- [ ] *.log files
- [ ] *.jou files
- [ ] .run/ directory
- [ ] package.* directories

### Updated .gitignore
```gitignore
# HLS Build Outputs
build_hls_sim/
minirocket_hls_validate/
_x*/
build_dir*/
*.backup.log

# Vitis Outputs
package.*/
.run/
.Xil/
*.ltx
*.wdb
*.wcfg

# Logs
*.log
*.jou

# XRT
emulation_debug.log
xcd.log

# Host Binary
host

# Large binaries (optional - decision TBD)
*.xclbin

# Temporary files
*.swp
*~
.DS_Store
```

### Updated README Structure
```markdown
# MiniRocket FPGA Accelerator

## Overview
Time series classification accelerator using MiniRocket algorithm on Xilinx Alveo U280.

## Quick Start
- Build HLS: `make build TARGET=hw_emu`
- Run simulation: `make run TARGET=hw_emu`
- Build hardware: `make build TARGET=hw` (4-6 hours)
- Run on FPGA: `make run TARGET=hw`

## Performance
- Throughput: 1,900-7,400 inferences/second
- Latency: 0.135-0.533 ms
- Clock: 300 MHz (404 MHz achieved)
- Resources: <2% LUT/FF utilization

## Project Structure
[Clean tree view]

## Development
[Build commands, testing workflow]

## Research
See RESEARCH_PLAN.md for benchmarking methodology.

## Citation
[Publication details when available]
```

---

## Implementation Timeline

### Week 1: Validation & Initial Benchmarking
- Day 1-2: Implement batch inference in host
- Day 3-4: Test on 3-5 datasets (small/medium/large)
- Day 5: CPU baseline measurements
- Day 6-7: Initial FPGA vs CPU comparison

### Week 2: Deep Benchmarking & Optimization
- Day 1-2: Comprehensive timing analysis
- Day 3-4: Implement kernel replication (4 CUs)
- Day 5: Test pipelining optimizations
- Day 6-7: Measure optimized performance

### Week 3: Analysis & Writing
- Day 1-2: Generate all plots/tables
- Day 3-5: Draft research paper
- Day 6-7: Code cleanup & documentation

### Week 4: Finalization
- Day 1-2: Final validation
- Day 3-4: Complete paper
- Day 5: Git cleanup & commit
- Day 6-7: Buffer/review

---

## Success Criteria

### Validation
- ✅ FPGA matches CPU accuracy within 0.1% on all datasets
- ✅ No crashes or errors on 100K+ sample runs
- ✅ Consistent performance across multiple runs

### Performance
- ✅ Achieve >10x speedup vs CPU for large batches
- ✅ <1ms latency for single inference
- ✅ >50x better energy efficiency

### Documentation
- ✅ Clean, professional codebase
- ✅ Reproducible benchmarks
- ✅ Publication-quality results

---

## Open Questions for Discussion

1. **Dataset Selection**: Which specific UCR/UEA datasets to use?
2. **CPU Baseline**: Use Python (slower) or C++ (fairer comparison)?
3. **GPU Comparison**: Do you have GPU access for comparison?
4. **Optimization Priority**: Focus on throughput or latency?
5. **Publication Target**: Conference (FCCM, FPL) or journal (TRETS, TCAD)?
6. **Model Complexity**: Test with different MiniRocket model sizes?
7. **Precision**: Compare fixed-point vs floating-point impact?

---

## Next Immediate Steps (This Week)

1. **Review this plan** - Confirm research direction
2. **Create batch inference host** - Modify minirocket_host.cpp
3. **Select 3 datasets** - Small, medium, large
4. **Run CPU baseline** - Establish comparison metrics
5. **Clean up codebase** - Remove unnecessary files
6. **Update documentation** - README + this plan

---

## Resources Needed

- [ ] Access to UCR/UEA Time Series Archive
- [ ] CPU baseline implementation (Python or C++)
- [ ] Plotting tools (Python: matplotlib)
- [ ] LaTeX for paper writing
- [ ] Git repository for code sharing

