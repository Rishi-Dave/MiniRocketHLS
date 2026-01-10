# HYDRA FPGA Accuracy Fix - Complete Summary

## Problem Identified

HYDRA FPGA was showing **50% accuracy** instead of the expected **~69.4%** for InsectSound dataset (matching Python model performance).

### Root Cause
The host code in `hydra_optimized/host/src/hydra_loader.cpp` contained **stub implementations** that:
- Hardcoded 2 classes (instead of loading 10 from model)
- Hardcoded 150 time series length (instead of 600)
- Generated synthetic sine wave test data (instead of loading real InsectSound data)
- Only tested 50 samples (instead of 25,000)

The JSON files existed but were completely ignored!

## Solution Implemented

### 1. Created Real JSON Loader
**Files Created:**
- `hydra_optimized/host/include/hydra_loader_v2.h`
- `hydra_optimized/host/src/hydra_loader_v2.cpp`

**Features:**
- Custom JSON parser (no external dependencies - adapted from MultiRocket)
- Parses model parameters: kernel weights, biases, dilations, scaler, coefficients
- Handles string label arrays (InsectSound has string labels like "aedes_male")
- Maps string labels to integers (0-9)
- Handles 1D flat coefficients array (10,240 elements) and reshapes to 2D

### 2. Updated Host Application
**File Modified:** `hydra_optimized/host/src/hydra_host.cpp`

**Changes:**
- Replaced stub `hydra_loader.h` with real `hydra_loader_v2.h`
- Uses HLS-style arrays instead of vectors for model parameters
- Dynamically loads all dimensions from JSON files
- Properly handles loaded test data

### 3. Fixed Constants for Massive Datasets
**File Modified:** `hydra_optimized/host/include/hydra_loader_v2.h`

Updated constants to support large UCR datasets:
```cpp
#define MAX_TIME_SERIES_LENGTH 8192  // MosquitoSound=3750, FruitFlies=5000
#define MAX_FEATURES 1024            // HYDRA: 512 kernels * 2 pooling
#define MAX_CLASSES 10               // InsectSound has 10 classes
```

### 4. Recompiled and Tested
Successfully rebuilt HYDRA host executable with new JSON loader.

## Results

### Quick Validation Test (1,000 samples)
```
Accuracy: 699/1000 = 69.90%
Average latency: 5.169 ms
Throughput: 193 inferences/sec
Test time: 5.17 seconds
```

### Full Test Results (25,000 samples)
```
Accuracy: 17,359/25,000 = 69.44%
Python model accuracy: 69.41%
FPGA vs Python difference: +0.03%
Average latency: 5.163 ms
Throughput: 194 inferences/sec
Total time: 129.07 seconds
```

**✅ FPGA accuracy matches Python model perfectly!**

### Performance Metrics
- **Latency:** ~5.2 ms per inference
- **Throughput:** ~194 inferences/second
- **Hardware:** Xilinx Alveo U280 FPGA @ 300 MHz

## Comparison: Before vs After

| Metric | Before (Stub) | After (Real Loader) |
|--------|---------------|---------------------|
| Classes | 2 (hardcoded) | 10 (loaded from JSON) ✓ |
| Time Series Length | 150 (hardcoded) | 600 (loaded from JSON) ✓ |
| Test Samples | 50 (synthetic) | 25,000 (real data) ✓ |
| Test Data | Sine waves | InsectSound audio ✓ |
| Accuracy | 50% (random) | 69.44% (matches Python: 69.41%) ✓ |
| Latency | ~5 ms | ~5 ms ✓ |

## Files Modified/Created

### Created:
1. `hydra_optimized/host/include/hydra_loader_v2.h` - JSON loader header
2. `hydra_optimized/host/src/hydra_loader_v2.cpp` - JSON loader implementation
3. `hydra_optimized/host/include/npy_loader.h` - NPY loader for future use
4. `hydra_optimized/host/src/npy_loader.cpp` - NPY loader implementation
5. `hydra_optimized/scripts/train_and_save_models_fast.py` - Fast training with .npy
6. `hydra_optimized/scripts/create_sample_test.py` - Create test subsets
7. `hydra_optimized/run_full_fpga_test.sh` - Full test script

### Modified:
1. `hydra_optimized/host/src/hydra_host.cpp` - Use new loader
2. `hydra_optimized/host/include/hydra_loader_v2.h` - Updated constants

## How to Run

### Quick Test (1,000 samples, ~5 seconds):
```bash
cd /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized
DEVICE=xilinx_u280_gen3x16_xdma_1_202211_1 make run-quick
```

### Full Test (25,000 samples, ~2 minutes):
```bash
cd /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized
DEVICE=xilinx_u280_gen3x16_xdma_1_202211_1 make run
```

### Build Host Executable:
```bash
cd /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized
DEVICE=xilinx_u280_gen3x16_xdma_1_202211_1 make host
```

## Next Steps: MosquitoSound & FruitFlies

### Problem
These datasets are too large for JSON:
- **MosquitoSound**: 139,780 × 3,750 = ~10 GB JSON (hours to serialize/parse)
- **FruitFlies**: 17,259 × 5,000 = ~1.6 GB JSON (slow)

### Solution
1. Use `.npy` binary format for test data (implemented in `train_and_save_models_fast.py`)
2. Implement NPY loader in C++ (already created: `npy_loader.h/cpp`)
3. Update host code to support both JSON and NPY formats

### File Size Comparison
```
InsectSound:  25,000 × 600   = 286 MB JSON vs  57 MB NPY (5× smaller)
MosquitoSound: 139,780 × 3,750 = 10 GB JSON vs 2.0 GB NPY (5× smaller)
FruitFlies:   17,259 × 5,000  = 1.6 GB JSON vs 329 MB NPY (5× smaller)
```

## Status

✅ **HYDRA FPGA InsectSound - WORKING PERFECTLY!**
- JSON loader implemented and tested
- **Accuracy: 69.44% (FPGA) vs 69.41% (Python) - Perfect match!**
- Performance excellent: 5.2ms latency, 194 inferences/sec
- **Note:** The ~79% expectation was incorrect - actual Python model achieves 69.41%

⏳ **MosquitoSound & FruitFlies - Ready to implement**
- NPY loader code written
- Fast training script created
- Just need to integrate and test

## Makefile Improvements (Jan 8, 2026)

Updated Makefile to provide simple `make run` commands similar to MiniRocket:

### Changes Made:
1. Updated `HOST_SRCS` to use new loaders:
   - `host/src/hydra_loader_v2.cpp` (real JSON loader)
   - `host/src/npy_loader.cpp` (for future NPY support)
2. Added variables for test files:
   - `TEST_FILE_FULL`: Full 25,000 sample test
   - `TEST_FILE_QUICK`: Quick 1,000 sample test
3. Created new targets:
   - `make run`: Full test (25,000 samples)
   - `make run-quick`: Quick test (1,000 samples)
4. Updated help text to document new commands

### Benefits:
- Simple, memorable commands (`make run` vs complex bash script)
- Consistent with MiniRocket workflow
- Easy to maintain and extend
- Clear separation between quick validation and full testing

## Conclusion

The HYDRA FPGA hardware was working perfectly all along. The issue was purely in the host software - the stub loader wasn't loading the real trained models and test data. With the proper JSON loader implemented and simple Makefile commands, HYDRA now correctly loads and runs inference on real InsectSound data with expected accuracy levels.
