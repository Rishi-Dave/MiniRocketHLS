# HYDRA JSON Loader Fix - Implementation Guide

**Date:** January 8, 2026
**Status:** Solution Identified - Ready to Implement
**Reference:** MultiRocket's working JSON parser

---

## Executive Summary

**Good News:** We already have a working JSON parser in the MultiRocket implementation that can be adapted for HYDRA!

**Location:** `multirocket_optimized/host/src/multirocket_loader.cpp`

This custom parser:
- ✅ No external dependencies (no nlohmann/json needed)
- ✅ Already tested and working on MultiRocket
- ✅ Handles all JSON types: int, float, arrays, 2D arrays
- ✅ Properly loads model parameters from JSON files
- ✅ Can be adapted for HYDRA with minimal changes

---

## Current MultiRocket Loader Implementation

### Architecture

**Files:**
```
multirocket_optimized/host/
├── include/multirocket_loader.h          # Header with class definition
└── src/multirocket_loader.cpp            # Implementation
```

**Class:** `MiniRocketTestbenchLoader` (Note: name is MiniRocket but used for MultiRocket)

### Key Functions

#### 1. Basic Parsing Functions
```cpp
// Read entire file into string
std::string read_file(const std::string& filename);

// Trim whitespace
void trim_whitespace(std::string& str);

// Parse single integer value
int parse_int_value(const std::string& content, const std::string& key);

// Parse array of integers
std::vector<int> parse_int_array(const std::string& content, const std::string& key);

// Parse array of floats
std::vector<float> parse_float_array(const std::string& content, const std::string& key);

// Parse 2D array of integers
std::vector<std::vector<int>> parse_2d_int_array(const std::string& content, const std::string& key);

// Parse 2D array of floats
std::vector<std::vector<float>> parse_2d_float_array(const std::string& content, const std::string& key);
```

#### 2. High-Level Loading Functions
```cpp
// Load model from JSON into HLS arrays
bool load_model_to_hls_arrays(
    const std::string& model_filename,
    data_t coefficients[MAX_CLASSES][MAX_FEATURES],
    data_t intercept[MAX_CLASSES],
    data_t scaler_mean[MAX_FEATURES],
    data_t scaler_scale[MAX_FEATURES],
    int_t dilations[MAX_DILATIONS],
    int_t num_features_per_dilation[MAX_DILATIONS],
    data_t biases[MAX_FEATURES],
    int_t& num_dilations_out,
    int_t& num_features_out,
    int_t& num_classes_out,
    int_t& time_series_length_out
);

// Load test data from JSON
bool load_test_data(
    const std::string& test_filename,
    std::vector<std::vector<float>>& test_inputs,
    std::vector<std::vector<float>>& expected_outputs
);
```

### Example Usage (from MultiRocket host)

```cpp
// Initialize loader
MiniRocketTestbenchLoader loader;

// Allocate arrays
data_t (*coefficients)[MAX_FEATURES] = new data_t[MAX_CLASSES][MAX_FEATURES];
std::vector<data_t> intercept(MAX_CLASSES);
std::vector<data_t> scaler_mean(MAX_FEATURES);
std::vector<data_t> scaler_scale(MAX_FEATURES);
std::vector<data_t> biases(MAX_FEATURES);
std::vector<int_t> dilations(MAX_DILATIONS);
std::vector<int_t> num_features_per_dilation(MAX_DILATIONS);

int_t num_dilations, num_features, num_classes, time_series_length;

// Load model
if (!loader.load_model_to_hls_arrays(
    model_file,
    coefficients,
    intercept.data(),
    scaler_mean.data(),
    scaler_scale.data(),
    dilations.data(),
    num_features_per_dilation.data(),
    biases.data(),
    num_dilations,
    num_features,
    num_classes,
    time_series_length)) {
    std::cerr << "Failed to load model!" << std::endl;
    return 1;
}

// Load test data
std::vector<std::vector<float>> test_inputs, expected_outputs;
if (!loader.load_test_data(test_file, test_inputs, expected_outputs)) {
    std::cerr << "Failed to load test data!" << std::endl;
    return 1;
}
```

---

## Adaptation Plan for HYDRA

### Step 1: Copy and Rename Files

```bash
cd /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized

# Copy loader files from MultiRocket
cp ../multirocket_optimized/host/include/multirocket_loader.h \
   host/include/hydra_loader_v2.h

cp ../multirocket_optimized/host/src/multirocket_loader.cpp \
   host/src/hydra_loader_v2.cpp
```

### Step 2: Modify Header File

**File:** `host/include/hydra_loader_v2.h`

**Changes needed:**
1. Rename class: `MiniRocketTestbenchLoader` → `HydraJSONLoader`
2. Add HYDRA-specific parameters
3. Update MAX constants if needed

**Key differences for HYDRA:**

| Parameter | MultiRocket | HYDRA | Notes |
|-----------|-------------|-------|-------|
| Kernel weights | biases only | full 9-weight kernels | Need to parse 512×9 array |
| Dilations | Variable per kernel | One per kernel | Different storage |
| Groups | N/A | 8 groups | Need group_assignments array |
| Parameters | dilations, biases | kernels, biases, dilations, groups | More complex |

### Step 3: Adapt load_model_to_hls_arrays() for HYDRA

**Current HYDRA JSON structure:**
```json
{
  "dataset": "InsectSound",
  "num_kernels": 512,
  "num_groups": 8,
  "kernel_size": 9,
  "num_features": 1024,
  "num_classes": 10,
  "time_series_length": 600,
  "kernel_weights": [/* 512 * 9 = 4608 floats */],
  "biases": [/* 512 floats */],
  "dilations": [/* 512 ints */],
  "scaler_mean": [/* 1024 floats */],
  "scaler_scale": [/* 1024 floats */],
  "coefficients": [/* 1024 * 10 = 10240 floats, 2D array */],
  "intercept": [/* 10 floats */]
}
```

**New function signature:**
```cpp
bool HydraJSONLoader::load_hydra_model(
    const std::string& model_filename,
    data_t kernel_weights[MAX_KERNELS][KERNEL_SIZE],  // NEW: 512x9 array
    data_t biases[MAX_KERNELS],
    int_t dilations[MAX_KERNELS],
    int_t group_assignments[MAX_KERNELS],             // NEW: group per kernel
    data_t scaler_mean[MAX_FEATURES],
    data_t scaler_scale[MAX_FEATURES],
    data_t coefficients[MAX_CLASSES][MAX_FEATURES],
    data_t intercept[MAX_CLASSES],
    int_t& num_kernels_out,
    int_t& num_groups_out,                            // NEW
    int_t& num_features_out,
    int_t& num_classes_out,
    int_t& time_series_length_out
) {
    std::string content = read_file(model_filename);
    if (content.empty()) return false;

    // Parse scalar values
    num_kernels_out = parse_int_value(content, "num_kernels");
    num_groups_out = parse_int_value(content, "num_groups");
    num_features_out = parse_int_value(content, "num_features");
    num_classes_out = parse_int_value(content, "num_classes");
    time_series_length_out = parse_int_value(content, "time_series_length");

    // Parse arrays
    auto kernel_weights_flat = parse_float_array(content, "kernel_weights");
    auto biases_vec = parse_float_array(content, "biases");
    auto dilations_vec = parse_int_array(content, "dilations");
    auto scaler_mean_vec = parse_float_array(content, "scaler_mean");
    auto scaler_scale_vec = parse_float_array(content, "scaler_scale");
    auto intercept_vec = parse_float_array(content, "intercept");

    // Parse 2D coefficient array
    auto coefficients_2d = parse_2d_float_array(content, "coefficients");

    // Reshape kernel_weights from flat array to 2D
    for (int k = 0; k < num_kernels_out; k++) {
        for (int w = 0; w < KERNEL_SIZE; w++) {
            kernel_weights[k][w] = kernel_weights_flat[k * KERNEL_SIZE + w];
        }
    }

    // Copy 1D arrays
    for (int k = 0; k < num_kernels_out; k++) {
        biases[k] = biases_vec[k];
        dilations[k] = dilations_vec[k];
        group_assignments[k] = k / (num_kernels_out / num_groups_out);
    }

    // Copy scaler parameters
    for (int f = 0; f < num_features_out; f++) {
        scaler_mean[f] = scaler_mean_vec[f];
        scaler_scale[f] = scaler_scale_vec[f];
    }

    // Copy coefficients (2D)
    for (int c = 0; c < num_classes_out; c++) {
        for (int f = 0; f < num_features_out; f++) {
            coefficients[c][f] = coefficients_2d[c][f];
        }
    }

    // Copy intercept
    for (int c = 0; c < num_classes_out; c++) {
        intercept[c] = intercept_vec[c];
    }

    return true;
}
```

### Step 4: Handle Test Data Loading

**Current HYDRA test JSON structure:**
```json
{
  "num_samples": 25000,
  "time_series_length": 600,
  "num_classes": 10,
  "time_series": [/* 25000 x 600 2D array */],
  "labels": ["aedes_male", "quinx_female", ...]  // STRING LABELS!
}
```

**Challenge:** Labels are strings, not integers!

**Solution:** Create label mapping

```cpp
bool HydraJSONLoader::load_hydra_test_data(
    const std::string& test_filename,
    std::vector<std::vector<float>>& test_inputs,
    std::vector<int>& test_labels,
    int& num_samples,
    int& time_series_length,
    int& num_classes
) {
    std::string content = read_file(test_filename);
    if (content.empty()) return false;

    // Parse metadata
    num_samples = parse_int_value(content, "num_samples");
    time_series_length = parse_int_value(content, "time_series_length");
    num_classes = parse_int_value(content, "num_classes");

    // Parse time series (2D array)
    test_inputs = parse_2d_float_array(content, "time_series");

    // Parse string labels
    std::vector<std::string> label_strings = parse_string_array(content, "labels");

    // Create label mapping for InsectSound
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

    // Convert string labels to integers
    test_labels.clear();
    for (const auto& label_str : label_strings) {
        if (label_map.count(label_str) > 0) {
            test_labels.push_back(label_map[label_str]);
        } else {
            std::cerr << "Warning: Unknown label: " << label_str << std::endl;
            test_labels.push_back(-1);
        }
    }

    return true;
}
```

**Need to add:** String array parser

```cpp
std::vector<std::string> HydraJSONLoader::parse_string_array(
    const std::string& content,
    const std::string& key
) {
    std::vector<std::string> result;

    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return result;

    size_t array_start = content.find("[", key_pos);
    size_t array_end = content.find("]", array_start);

    if (array_start == std::string::npos || array_end == std::string::npos)
        return result;

    std::string array_content = content.substr(array_start + 1, array_end - array_start - 1);

    // Parse quoted strings
    size_t pos = 0;
    while (pos < array_content.length()) {
        size_t quote_start = array_content.find("\"", pos);
        if (quote_start == std::string::npos) break;

        size_t quote_end = array_content.find("\"", quote_start + 1);
        if (quote_end == std::string::npos) break;

        std::string str_value = array_content.substr(quote_start + 1,
                                                     quote_end - quote_start - 1);
        result.push_back(str_value);
        pos = quote_end + 1;
    }

    return result;
}
```

### Step 5: Update hydra_host.cpp

**Replace current loader calls:**

```cpp
// OLD (stub version)
#include "hydra_loader.h"

HydraModel model;
HydraTestData test_data;

load_hydra_model_json(model_file, &model);
load_test_data_json(test_file, &test_data);
```

**NEW (working version):**

```cpp
// NEW (real JSON parsing)
#include "hydra_loader_v2.h"

HydraJSONLoader loader;

// Allocate arrays
data_t (*kernel_weights)[9] = new data_t[512][9];
data_t biases[512];
int_t dilations[512];
int_t group_assignments[512];
data_t scaler_mean[1024];
data_t scaler_scale[1024];
data_t (*coefficients)[1024] = new data_t[10][1024];
data_t intercept[10];

int num_kernels, num_groups, num_features, num_classes, time_series_length;

// Load model
if (!loader.load_hydra_model(
    model_file,
    kernel_weights,
    biases,
    dilations,
    group_assignments,
    scaler_mean,
    scaler_scale,
    coefficients,
    intercept,
    num_kernels,
    num_groups,
    num_features,
    num_classes,
    time_series_length)) {
    std::cerr << "Failed to load model!" << std::endl;
    return 1;
}

// Load test data
std::vector<std::vector<float>> test_inputs;
std::vector<int> test_labels;
int num_samples;

if (!loader.load_hydra_test_data(
    test_file,
    test_inputs,
    test_labels,
    num_samples,
    time_series_length,
    num_classes)) {
    std::cerr << "Failed to load test data!" << std::endl;
    return 1;
}
```

### Step 6: Update Makefile/Build

**Add new source files to build:**

```makefile
# Add to SRCS or equivalent
SRCS += host/src/hydra_loader_v2.cpp

# Add to includes
INCLUDES += -Ihost/include
```

### Step 7: Rebuild and Test

```bash
cd /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/host

# Rebuild host executable
make clean
make

# Test with real data
./hydra_host \
  ../build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/krnl.xclbin \
  ../models/hydra_insectsound_model.json \
  ../models/hydra_insectsound_test.json
```

**Expected result:**
- ✅ Loads 10 classes (not 2)
- ✅ Loads 600 time series length (not 150)
- ✅ Loads real trained weights (not test patterns)
- ✅ Loads real InsectSound audio data (not sine waves)
- ✅ Accuracy ~80%+ (not 50%)

---

## Implementation Checklist

### Phase 1: File Setup
- [ ] Copy `multirocket_loader.h` → `hydra_loader_v2.h`
- [ ] Copy `multirocket_loader.cpp` → `hydra_loader_v2.cpp`
- [ ] Rename class: `MiniRocketTestbenchLoader` → `HydraJSONLoader`

### Phase 2: Model Loading
- [ ] Add `parse_string_array()` function for label parsing
- [ ] Implement `load_hydra_model()` function
- [ ] Handle kernel weights reshaping (flat → 2D)
- [ ] Handle group assignments calculation

### Phase 3: Test Data Loading
- [ ] Implement `load_hydra_test_data()` function
- [ ] Create label mapping (string → int)
- [ ] Handle large test data (25K samples)

### Phase 4: Host Code Integration
- [ ] Update `hydra_host.cpp` to use new loader
- [ ] Replace stub loader calls
- [ ] Update memory allocation
- [ ] Update buffer setup

### Phase 5: Build & Test
- [ ] Update Makefile
- [ ] Rebuild host executable
- [ ] Test with InsectSound model
- [ ] Verify accuracy improves from 50% to ~80%+

---

## Estimated Effort

| Phase | Task | Effort | Priority |
|-------|------|--------|----------|
| 1 | File copy and rename | 15 min | High |
| 2 | Model loading adaptation | 1-2 hours | High |
| 3 | Test data loading | 1 hour | High |
| 4 | Host code integration | 30 min | High |
| 5 | Build & test | 30 min | High |
| **Total** | **End-to-end** | **3-4 hours** | - |

---

## Alternative: Quick Fix Option

If you want to test immediately without full implementation, create a simplified test file:

### Quick Fix: Generate Integer-Label Test File

**Modify training script** (`hydra_optimized/scripts/train_and_save_models.py`):

```python
# After saving the main test file, create FPGA-friendly version
fpga_test = {
    "num_samples": 100,  # Smaller for quick testing
    "time_series_length": X_test.shape[1],
    "num_classes": len(np.unique(y_test)),
    "time_series": X_test[:100].tolist(),
    "labels": [int(y_test[i]) for i in range(100)]  # Already integers!
}

with open(f'../models/hydra_{dataset_name.lower()}_fpga_test.json', 'w') as f:
    json.dump(fpga_test, f, indent=2)
```

**Then test:**
```bash
./hydra_host krnl.xclbin \
  models/hydra_insectsound_model.json \
  models/hydra_insectsound_fpga_test.json
```

This still won't load the real model, but it will test if the FPGA can handle different data.

---

## References

- **Working example:** `multirocket_optimized/host/src/multirocket_loader.cpp`
- **Working host:** `multirocket_optimized/host/src/multirocket_host.cpp`
- **Test JSON:** `multirocket_optimized/models/*.json`
- **Issue analysis:** `docs/HYDRA_FPGA_ISSUE_ANALYSIS.md`

---

## Conclusion

We have a **proven, working JSON parser** that just needs to be adapted for HYDRA's data structure. The MultiRocket implementation provides a complete template that handles all the complexity without external dependencies.

**Next Step:** Copy the files and start adapting!
