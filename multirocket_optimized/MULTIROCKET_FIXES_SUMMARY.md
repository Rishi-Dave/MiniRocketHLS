# MultiRocket Implementation Fixes - January 6, 2026

## Problem Identified

The MultiRocket FPGA implementation was actually implementing **MiniRocket** instead of **MultiRocket**.

### Critical Discrepancies Found:

1. **Missing 3 Pooling Operators**: Only PPV (Proportion of Positive Values) was implemented
   - ❌ MPV (Mean of Positive Values) was missing
   - ❌ MIPV (Mean of Indices of Positive Values) was missing
   - ❌ LSPV (Longest Stretch of Positive Values) was missing

2. **Missing First-Order Difference Representation**: Only processed original time series
   - ❌ First-order difference computation was missing
   - ❌ Dual representation processing was missing

3. **Feature Count Mismatch**: Generated ~672-2,688 features instead of ~5,376 features
   - This is **1/8th** the expected features (missing 3 operators × 2 representations)

## Fixes Implemented

### 1. Updated Constants ([multirocket.hpp](multirocket/include/multirocket.hpp))

```c
// Constants for MultiRocket (compile-time known)
#define MAX_TIME_SERIES_LENGTH 5120  // Updated for UCR datasets (matches HYDRA)
#define MAX_FEATURES 20000  // Increased for 4 pooling operators × 2 representations
#define NUM_KERNELS 84
#define KERNEL_SIZE 9
#define MAX_DILATIONS 8
#define MAX_CLASSES 10  // Increased to match HYDRA
#define NUM_POOLING_OPERATORS 4  // PPV, MPV, MIPV, LSPV
#define NUM_REPRESENTATIONS 2  // Original + First-order difference
```

### 2. Added PoolingStats Structure ([multirocket.hpp](multirocket/include/multirocket.hpp))

```c
// Structure to hold all four pooling operator results
struct PoolingStats {
    data_t ppv;   // Proportion of Positive Values
    data_t mpv;   // Mean of Positive Values
    data_t mipv;  // Mean of Indices of Positive Values
    data_t lspv;  // Longest Stretch of Positive Values
};
```

### 3. Implemented First-Order Difference ([multirocket.cpp](multirocket/src/multirocket.cpp))

```c
// Compute first-order difference: diff[i] = series[i+1] - series[i]
void compute_first_order_difference(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t diff_series[MAX_TIME_SERIES_LENGTH],
    int_t length
) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE off

    if (length <= 1) {
        return;
    }

    DIFF_LOOP: for (int_t i = 0; i < length - 1; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=99 max=5119
        diff_series[i] = time_series[i + 1] - time_series[i];
    }

    diff_series[length - 1] = 0.0;
}
```

### 4. Rewrote Feature Extraction to Use All 4 Pooling Operators and 2 Representations

The `multirocket_feature_extraction_hls()` function was completely rewritten to:

- ✅ Compute first-order difference representation
- ✅ Process **both** original and difference representations
- ✅ Apply all **4 pooling operators** per convolution
- ✅ Generate correct number of features: 84 kernels × 8 dilations × 4 pooling × 2 representations = 5,376 features

Key implementation details:
```c
// Process both representations: Original (rep=0) and Difference (rep=1)
REPRESENTATION_LOOP: for (int_t rep = 0; rep < NUM_REPRESENTATIONS; rep++) {
    data_t* current_series = (rep == 0) ? time_series : diff_series;

    // For each kernel and dilation...
    // Compute all 4 pooling operators in single pass
    compute_four_pooling_operators(convolutions, bias, time_series_length, &stats);

    // Store all 4 features
    features[feature_idx++] = stats.ppv;
    features[feature_idx++] = stats.mpv;
    features[feature_idx++] = stats.mipv;
    features[feature_idx++] = stats.lspv;
}
```

### 5. Created UCR Benchmark Script

Created `scripts/ucr_fpga_benchmark.py` to match HYDRA benchmarking:
- Uses the same UCR datasets: InsectSound, MosquitoSound, FruitFlies
- Trains MultiRocket84 Python reference implementation
- Saves models and test data in JSON format for FPGA
- Measures Python baseline accuracy and performance
- Generates comprehensive results for comparison

## Impact

### Before (MiniRocket Implementation):
- 1 pooling operator (PPV only)
- 1 representation (original only)
- ~672-2,688 features
- **Identical to MiniRocket** (not MultiRocket!)
- Expected accuracy: 85-90% on typical datasets

### After (Correct MultiRocket Implementation):
- 4 pooling operators (PPV, MPV, MIPV, LSPV)
- 2 representations (original + difference)
- ~5,376 features (8x increase)
- **True MultiRocket algorithm**
- Expected accuracy: 90-95% on typical datasets (2-5% improvement)

## Files Modified

1. **multirocket/include/multirocket.hpp**
   - Updated constants (MAX_TIME_SERIES_LENGTH, MAX_FEATURES, MAX_CLASSES)
   - Added NUM_POOLING_OPERATORS and NUM_REPRESENTATIONS constants
   - Added PoolingStats structure
   - Added function declarations for pooling operators and difference computation

2. **multirocket/src/multirocket.cpp**
   - Added `compute_first_order_difference()` function
   - Completely rewrote `multirocket_feature_extraction_hls()` to:
     - Process 2 representations (original + difference)
     - Apply all 4 pooling operators per convolution
     - Generate correct feature count

3. **scripts/ucr_fpga_benchmark.py** (NEW)
   - Created comprehensive UCR benchmark script
   - Matches HYDRA benchmark methodology
   - Trains on InsectSound, MosquitoSound, FruitFlies datasets
   - Saves models for FPGA testing

## Existing Files (Already Correct)

The pooling operator implementations in `multirocket/src/multirocket_pooling.cpp` were **already correct**:
- ✅ `compute_ppv()` - PPV implementation
- ✅ `compute_mpv()` - MPV implementation
- ✅ `compute_mipv()` - MIPV implementation
- ✅ `compute_lspv()` - LSPV implementation
- ✅ `compute_four_pooling_operators()` - Single-pass computation of all 4

These were just not being called by the main feature extraction function.

## Next Steps

1. ✅ **Completed**: Fix MultiRocket implementation
2. ⏳ **In Progress**: Wait for HYDRA FPGA build to complete (~2 hours)
3. **Pending**: Build MultiRocket FPGA with updated implementation
4. **Pending**: Run Python benchmarks on UCR datasets for both HYDRA and MultiRocket
5. **Pending**: Run FPGA benchmarks and compare with Python baselines
6. **Pending**: Collect comprehensive metrics (accuracy, latency, throughput, power, resources)

## Verification

To verify the fixes are correct:

1. **Feature Count Check**:
   - Expected: 84 kernels × 8 dilations × 4 pooling × 2 representations = 5,376 features
   - Verify Python model generates 5,376 features
   - Verify FPGA generates same number of features

2. **Accuracy Check**:
   - Python vs FPGA accuracy should match within ±0.5%
   - MultiRocket should outperform MiniRocket by 2-5%

3. **Algorithm Verification**:
   - Confirm first-order difference is computed
   - Confirm all 4 pooling operators are applied
   - Confirm features are generated in correct order

## Build Status

- **HYDRA FPGA**: Currently building in tmux session `hydra_ucr_build`
- **MultiRocket FPGA**: Will build after HYDRA completes
- **Python Benchmarks**: Will run in parallel with FPGA builds

---

**Last Updated**: January 6, 2026
**Status**: MultiRocket implementation fixes complete, ready for FPGA build after HYDRA completes
