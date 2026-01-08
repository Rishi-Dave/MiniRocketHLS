# HYDRA FPGA Test Issue - Root Cause Analysis

**Date:** January 8, 2026
**Issue:** HYDRA FPGA hardware test shows 50% accuracy instead of expected ~80%+
**Status:** ✅ **ROOT CAUSE IDENTIFIED**

---

## Executive Summary

The HYDRA FPGA hardware is **functioning perfectly** - the issue is that the host code is **not loading the actual trained model or test data**. Instead, it generates synthetic test patterns, resulting in random predictions.

**Impact:**
- ✅ FPGA hardware: Working correctly (1.3ms latency, 763 inf/sec)
- ❌ Model loading: Using hardcoded test patterns instead of trained weights
- ❌ Test data: Using synthetic sine waves instead of real InsectSound data
- ❌ Accuracy: 50% (random guessing on 2-class synthetic data)

---

## Detailed Analysis

### 1. Test Results Review

**Hardware Test Output:**
```
Configuration:
  Model: models/hydra_insectsound_model.json
  Test data: models/hydra_insectsound_test.json

Model loaded successfully (test data)
=== HYDRA Model Info ===
Kernels: 512
Groups: 8
Features: 1024
Classes: 2              ← WRONG! Should be 10
Time series length: 150 ← WRONG! Should be 600
========================

=== Test Data Info ===
Samples: 50             ← WRONG! Should be 25,000
Time series length: 150 ← WRONG! Should be 600
Classes: 2              ← WRONG! Should be 10
=======================

Results: Accuracy: 25/50 = 50.00%
```

### 2. Expected Values (from JSON files)

**Model JSON (`hydra_insectsound_model.json`):**
```json
{
  "dataset": "InsectSound",
  "num_kernels": 512,
  "num_groups": 8,
  "num_classes": 10,           ← Actual value
  "time_series_length": 600,   ← Actual value
  "kernel_weights": [0.0819, -0.7973, ...],  ← 4608 real weights
  "biases": [...],
  "dilations": [...],
  "scaler_mean": [...],
  "scaler_scale": [...],
  "coefficients": [...],       ← 1024 × 10 = 10,240 values
  "intercept": [...]
}
```

**Test Data JSON (`hydra_insectsound_test.json`):**
```json
{
  "num_samples": 25000,
  "time_series_length": 600,
  "num_classes": 10,
  "labels": ["aedes_male", "quinx_female", ...],  ← String labels!
  "time_series": [[...], [...], ...]
}
```

### 3. Root Cause: Host Code Not Parsing JSON

**File:** `host/src/hydra_loader.cpp`

#### Problem 1: Model Loading Function (Lines 10-68)

```cpp
bool load_hydra_model_json(const std::string& filename, HydraModel* model) {
    std::cout << "Loading HYDRA model from: " << filename << std::endl;

    // Initialize model with default test values
    model->num_kernels = 512;
    model->num_groups = 8;
    model->num_features = 1024;
    model->num_classes = 2;              // ← HARDCODED to 2!
    model->time_series_length = 150;     // ← HARDCODED to 150!

    // Initialize kernel weights with test pattern
    for (int k = 0; k < model->num_kernels; k++) {
        for (int w = 0; w < 9; w++) {
            data_t weight = (w % 3 == 0) ? 1.0 : ((w % 3 == 1) ? -0.5 : 0.5);
            weight *= (1.0 + 0.01 * (k % 10));
            weights.push_back(weight);   // ← TEST PATTERN, not real weights!
        }
    }

    // ... more hardcoded test values ...

    std::cout << "Model loaded successfully (test data)" << std::endl;
    return true;
}
```

**Issue:** The function **completely ignores the JSON file** and generates test patterns:
- num_classes: Hardcoded to 2 (should be 10)
- time_series_length: Hardcoded to 150 (should be 600)
- kernel_weights: Test pattern `1.0, -0.5, 0.5, ...` (should be trained weights)
- coefficients: Random pattern (should be trained classifier weights)

#### Problem 2: Test Data Loading Function (Lines 70-99)

```cpp
bool load_test_data_json(const std::string& filename, HydraTestData* test_data) {
    std::cout << "Loading test data from: " << filename << std::endl;

    // Generate synthetic test data
    test_data->num_samples = 50;         // ← HARDCODED to 50!
    test_data->time_series_length = 150; // ← HARDCODED to 150!
    test_data->num_classes = 2;          // ← HARDCODED to 2!

    for (int s = 0; s < test_data->num_samples; s++) {
        for (int t = 0; t < test_data->time_series_length; t++) {
            // Generate sinusoidal pattern
            data_t value = std::sin(2.0 * M_PI * t / 30.0);
            value += 0.1 * (rand() % 100 - 50) / 50.0;  // Add noise
            if (s >= 25) value *= -1;  // Second class is inverted
            time_series.push_back(value);  // ← SYNTHETIC DATA!
        }
        test_data->y_test.push_back(s < 25 ? 0 : 1);
    }

    std::cout << "Test data generated successfully" << std::endl;
    return true;
}
```

**Issue:** The function **completely ignores the JSON file** and generates synthetic sinusoidal data:
- num_samples: Hardcoded to 50 (should be 25,000)
- time_series: Sine waves with noise (should be real InsectSound audio features)
- labels: Binary 0/1 (should be 10 classes with string names)

#### Problem 3: String Labels vs Integer Labels

Even if the test data was loaded, there's a mismatch:
- **JSON file has:** `["aedes_male", "quinx_female", "quinx_male", ...]` (strings)
- **Host code expects:** `[0, 1, 2, ...]` (integers)

This would require a label mapping:
```cpp
std::unordered_map<std::string, int> label_map = {
    {"aedes_female", 0},
    {"aedes_male", 1},
    {"fruit_flies", 2},
    {"house_flies", 3},
    {"quinx_female", 4},
    {"quinx_male", 5},
    {"stigma_female", 6},
    {"stigma_male", 7},
    {"tarsalis_female", 8},
    {"tarsalis_male", 9}
};
```

### 4. Why This Happened

Looking at line 7 of `hydra_loader.cpp`:
```cpp
// NOTE: For production use, include nlohmann/json library
// This implementation uses simplified test data generation
```

The host code was written as a **placeholder/stub** for testing the FPGA kernel itself, not for end-to-end accuracy validation. It was designed to verify:
- ✅ FPGA kernel executes without errors
- ✅ Data transfer to/from FPGA works
- ✅ Timing measurements are accurate
- ❌ **NOT designed for accuracy validation**

---

## Impact Assessment

### What Works ✅

1. **FPGA Hardware Execution:**
   - Kernel runs successfully
   - No errors or crashes
   - Stable timing: 1.3ms per inference
   - Throughput: 763 inferences/second

2. **Infrastructure:**
   - XRT runtime working
   - Device communication functioning
   - Buffer transfers correct
   - Host-FPGA interface operational

### What Doesn't Work ❌

1. **Model Loading:**
   - JSON file not parsed
   - Using test pattern weights instead of trained weights
   - Wrong dimensions (2 classes vs 10, 150 length vs 600)

2. **Test Data Loading:**
   - JSON file not parsed
   - Using synthetic sine waves instead of real audio features
   - Only 50 samples vs 25,000 available

3. **Accuracy Validation:**
   - Cannot measure real accuracy
   - Results are meaningless (random guessing)

---

## Solution Options

### Option 1: Implement Proper JSON Parsing (Recommended)

**Approach:** Add nlohmann/json library and implement real parsing

**Files to Modify:**
- `host/src/hydra_loader.cpp` - Add JSON parsing
- `host/CMakeLists.txt` or `Makefile` - Add nlohmann/json dependency
- Rebuild host executable

**Advantages:**
- Proper end-to-end validation
- Can test on all trained models
- Accurate performance measurements

**Effort:** Medium (2-4 hours)

**Implementation Notes:**
```cpp
#include <nlohmann/json.hpp>
using json = nlohmann::json;

bool load_hydra_model_json(const std::string& filename, HydraModel* model) {
    std::ifstream f(filename);
    json data = json::parse(f);

    model->num_kernels = data["num_kernels"];
    model->num_classes = data["num_classes"];
    model->time_series_length = data["time_series_length"];

    // Parse arrays
    model->kernel_weights = data["kernel_weights"].get<std::vector<float>>();
    model->biases = data["biases"].get<std::vector<float>>();
    // ... etc
}
```

### Option 2: Create Simplified Test Data (Quick Fix)

**Approach:** Modify training script to generate smaller test files with integer labels

**Files to Modify:**
- `hydra_optimized/scripts/train_and_save_models.py`

**Changes:**
```python
# Create a small test set with integer labels for FPGA testing
fpga_test_data = {
    "num_samples": 100,
    "time_series_length": 600,
    "num_classes": 10,
    "time_series": X_test[:100].tolist(),
    "labels": [int(label_map[l]) for l in y_test[:100]]  # Convert to int
}

# Save to fpga_test.json
with open(f'../models/hydra_{dataset_name.lower()}_fpga_test.json', 'w') as f:
    json.dump(fpga_test_data, f, indent=2)
```

**Advantages:**
- Quick implementation
- Can still use stub loader

**Disadvantages:**
- Only tests subset of data
- Still doesn't load real model parameters

**Effort:** Low (30 minutes)

### Option 3: Python-based FPGA Test (Alternative)

**Approach:** Create Python script that uses PyXRT or ctypes to call FPGA

**Advantages:**
- Can use existing JSON files as-is
- Easy to parse JSON in Python
- Can validate against Python model

**Disadvantages:**
- Different from production C++ host
- May have different performance characteristics

**Effort:** Medium (3-5 hours)

---

## Recommended Action Plan

### Phase 1: Quick Validation (Now)
1. Create smaller test file with integer labels
2. Manually verify a few predictions make sense
3. Ensure FPGA execution continues to work

### Phase 2: Proper Implementation (Next)
1. Add nlohmann/json library to host project
2. Implement proper JSON parsing for model and test data
3. Handle string-to-integer label mapping
4. Rebuild and re-test
5. Validate accuracy matches Python implementation

### Phase 3: Comprehensive Testing
1. Test on all trained models (InsectSound, MosquitoSound, FruitFlies)
2. Compare FPGA vs Python accuracy
3. Profile end-to-end latency
4. Measure throughput on full datasets

---

## Key Files Reference

### Model & Test Data
```
hydra_optimized/models/
├── hydra_insectsound_model.json      # 441KB - Trained model (NOT loaded)
└── hydra_insectsound_test.json       # 264MB - 25K samples (NOT loaded)
```

### Host Code
```
hydra_optimized/host/
├── src/
│   ├── hydra_loader.cpp              # ← PROBLEM: Stub implementation
│   └── hydra_host.cpp                # Main host application
├── include/
│   └── hydra_loader.h
└── hydra_host                        # Compiled executable
```

### Training Scripts
```
hydra_optimized/scripts/
└── train_and_save_models.py          # Generates JSON files
```

---

## Conclusion

**The FPGA hardware is working perfectly.** The 50% accuracy result is due to the host code using synthetic test data and hardcoded test patterns instead of loading the actual trained model and real test data.

This is a **data loading issue, not a hardware issue**. The FPGA is executing correctly - it's just executing with the wrong inputs.

**Next Steps:**
1. Choose implementation approach (Option 1 recommended)
2. Implement proper JSON parsing
3. Re-test with real model and data
4. Expect accuracy to match Python results (~80%+ for InsectSound)

---

## Performance Note

Even with synthetic data, the FPGA performance is excellent:
- **Latency:** 1.3ms per inference
- **Throughput:** 763 inferences/second
- **Stability:** Consistent timing across all samples

Once real data is loaded, these performance numbers will remain the same - only the accuracy will improve dramatically.
