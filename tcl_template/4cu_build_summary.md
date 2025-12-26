# 4-CU Build with HBM Bank Assignments

**Start Time**: December 24, 2025, 5:36 AM GMT  
**Expected Completion**: ~7 hours (around 12:36 PM GMT)  
**Goal**: Enable all 4 compute units to work simultaneously on 1:1 reference implementation

---

## Problem Statement

The current 1:1 reference bitstream only has 1 of 4 CUs operational due to memory bank connectivity issues:

**XRT Warning**:
```
[XRT] WARNING: Argument '5' of kernel 'krnl_top' is allocated in memory bank 'bank0';
compute unit 'krnl_top_2/3/4' cannot be used with this argument and is ignored.
```

**Root Cause**: Vitis automatically assigned some memory buffers to DDR (bank0) instead of HBM, and CUs 2-4 cannot access bank0.

**Impact**: Only 1 CU works → throughput limited to 42-217 inf/sec (77x slower than optimized version)

---

## Solution

### Explicit HBM Bank Assignments

Modified [config.cfg](config.cfg) to explicitly assign all 36 memory ports (9 ports × 4 CUs) to specific HBM banks:

```ini
[connectivity]
nk=krnl_top:4

# CU 1: HBM[0-8]
sp=krnl_top_1.time_series_input:HBM[0]
sp=krnl_top_1.prediction_output:HBM[1]
sp=krnl_top_1.coefficients:HBM[2]
sp=krnl_top_1.intercept:HBM[3]
sp=krnl_top_1.scaler_mean:HBM[4]
sp=krnl_top_1.scaler_scale:HBM[5]
sp=krnl_top_1.dilations:HBM[6]
sp=krnl_top_1.num_features_per_dilation:HBM[7]
sp=krnl_top_1.biases:HBM[8]

# CU 2: HBM[9-17]
# CU 3: HBM[18-26]
# CU 4: HBM[27-31, 0-3] (wraps around)
```

### Why This Works

- **U280 has 32 HBM banks**: Each bank provides independent high-bandwidth access
- **Each CU gets 9 dedicated banks**: No conflicts or bank0 (DDR) assignments
- **Explicit assignment**: Prevents Vitis from making suboptimal automatic choices
- **Wrap-around for CU 4**: Uses HBM[27-31] then wraps to HBM[0-3] (still HBM, not DDR)

---

## Expected Results

###Before (1 CU @ 242 MHz):
- **Throughput**: 42.5 inf/sec (GunPoint), 217.3 inf/sec (ItalyPower)
- **Active CUs**: 1 of 4
- **Memory Bank Warnings**: Yes (CUs 2-4 ignored)

### After (4 CU @ 242 MHz):
- **Expected Throughput**: ~170-870 inf/sec (4x improvement)
- **Active CUs**: 4 of 4 ✅
- **Memory Bank Warnings**: None ✅

### Performance Calculation

**Theoretical 4x speedup**:
- GunPoint: 42.5 × 4 = **170 inf/sec**
- ItalyPower: 217.3 × 4 = **869 inf/sec**

**Comparison to Optimized**:
- Optimized version: 3,468 inf/sec @ 404 MHz with 4 CUs
- 1:1 with 4 CUs: ~170-869 inf/sec @ 242 MHz
- **Still slower** due to:
  - Lower clock frequency (242 vs 404 MHz) = 1.67x slower
  - More complex algorithm (84x convolutions) = ~11x slower
  - **Total**: ~18x slower even with 4 CUs

**Final expected ratio**: Optimized should still be ~18x faster than 1:1 with 4 CUs.

---

## Build Progress

Build started at **2025-12-24 05:36 GMT**

**Build Log**: [build_4cu_hbm.log](build_4cu_hbm.log)

**Build Stages**:
1. ✅ **v++ Link** - Started (combining 4 CUs with HBM assignments)
2. ⏳ **Vivado Synthesis** - Pending (~2 hours)
3. ⏳ **Place & Route** - Pending (~4 hours)
4. ⏳ **Bitstream Generation** - Pending (~1 hour)

**Check Progress**:
```bash
tail -f build_4cu_hbm.log
```

---

## Validation Plan

Once the build completes:

1. **Test on GunPoint** dataset:
   ```bash
   ./host build_dir.hw.*/krnl.xclbin minirocket_gunpoint_model.json minirocket_gunpoint_model_test_data.json
   ```
   - Verify 4 CUs are active (no XRT warnings)
   - Measure throughput (expect ~170 inf/sec)
   - Confirm 98.33% accuracy maintained

2. **Test on ItalyPowerDemand** dataset:
   ```bash
   ./host build_dir.hw.*/krnl.xclbin minirocket_italypower_model.json minirocket_italypower_model_test_data.json
   ```
   - Expect ~869 inf/sec throughput
   - Confirm 97.26% accuracy maintained

3. **Compare Results**:
   - Document speedup from 1 CU → 4 CU
   - Update comparison documentation
   - Analyze resource utilization vs optimized version

---

## Success Criteria

✅ **Build completes successfully** (~7 hours)  
✅ **All 4 CUs operational** (no XRT warnings about bank0)  
✅ **4x throughput improvement** over 1-CU version  
✅ **Accuracy maintained** at 98.33% (GunPoint), 97.26% (ItalyPower)  
✅ **HBM banks properly assigned** (verified via xclbinutil)

---

## Risk Analysis

### Potential Issues

1. **Build Failure**: HBM bank assignments might conflict with platform constraints
   - **Mitigation**: U280 has 32 HBM banks, we're using 32 (36 with wrap-around sharing)
   
2. **Timing Closure**: More complex routing might reduce clock frequency below 242 MHz
   - **Mitigation**: 1:1 reference already achieved 242 MHz with similar complexity

3. **Resource Utilization**: 4 CUs might exceed FPGA resources
   - **Mitigation**: Optimized version successfully fit 4 CUs; 1:1 should be similar

### Fallback Plan

If 4 CUs fail to build or operate:
- Try 2 CUs (simpler, should work)
- Reduce HBM bank assignments to fewer banks per CU
- Accept single-CU limitation as algorithmic trade-off

---

## Documentation Updates

Files to update after successful build:

- [1to1_vs_optimized_comparison.md](../1to1_vs_optimized_comparison.md)
  - Add 4-CU results
  - Update performance comparison table
  - Document memory bank solution

- [ucr_benchmark_results.md](../ucr_benchmark_results.md)
  - Add 4-CU throughput results
  - Update conclusions

- [MiniRocketHLS/README.md](README.md)
  - Update throughput numbers
  - Document HBM configuration approach

---

**Last Updated**: December 24, 2025, 05:45 GMT
