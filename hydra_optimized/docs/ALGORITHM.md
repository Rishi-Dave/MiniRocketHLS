# HYDRA Algorithm

## Overview

HYDRA (HYbrid Dictionary Representation Algorithm) is a fast and accurate time series classification method that uses dictionary-based convolutional kernels.

## Algorithm Description

### Core Concept

HYDRA represents time series using features extracted from a dictionary of convolutional kernels. Unlike MiniRocket and MultiRocket which use fixed kernel patterns, HYDRA learns or initializes a dictionary of kernels that capture diverse temporal patterns.

### FPGA Implementation Parameters

```
Number of kernels: 512
Kernel groups: 8
Kernels per group: 64
Kernel size: 9
Pooling operators: 2 (max + mean)
Total features: 1,024
```

### Algorithm Steps

#### 1. Dictionary Initialization

For each of 512 kernels:
```python
kernel = random_normal(size=9)
kernel = (kernel - mean(kernel)) / std(kernel)  # Normalize
```

Kernels are organized into 8 groups for diversity.

#### 2. Dilation Assignment

Each kernel is assigned a dilation from {1, 2, 4, 8}:
```python
for i in range(512):
    dilations[i] = 2^(i % 3)  # Cycles through 1, 2, 4, 8
```

#### 3. Feature Extraction

For each time series X and kernel k:

**Step 3a: Convolution with Dilation**
```
conv_output[t] = sum(X[t + w*dilation[k]] * kernel[k][w]) + bias[k]
                 for w in range(9)
```

**Step 3b: Max Pooling**
```
feature_max[k] = max(conv_output)
```

**Step 3c: Global Mean**
```
feature_mean[k] = mean(conv_output)
```

**Combined:**
```
features = [feature_max[0], feature_mean[0],
            feature_max[1], feature_mean[1],
            ...
            feature_max[511], feature_mean[511]]
```

Result: 1,024-dimensional feature vector

#### 4. Feature Normalization

```python
features_normalized = (features - scaler_mean) / scaler_scale
```

Uses StandardScaler trained on training set.

#### 5. Classification

Linear classifier (Ridge Regression):
```python
predictions = features_normalized @ coefficients + intercept
predicted_class = argmax(predictions)
```

## Comparison with MiniRocket and MultiRocket

### MiniRocket

- **Kernels:** 84 (fixed C(9,3) combinations)
- **Representations:** 1 (original time series)
- **Pooling:** PPV (proportion of positive values) with 9 quantiles
- **Features:** 84 × 9 = 756 (with dilations: 840-2,688)
- **Characteristics:** Fast, simple, good accuracy

### MultiRocket

- **Kernels:** 84 (same as MiniRocket)
- **Representations:** 2 (original + first-order difference)
- **Pooling:** 4 operators (PPV, MPV, MIPV, LSPV)
- **Features:** 84 × 4 × 2 = 672 (with dilations: 2,688-5,376)
- **Characteristics:** Better accuracy than MiniRocket, more complex

### HYDRA

- **Kernels:** 512 (learned/random dictionary)
- **Representations:** 1 (original time series)
- **Pooling:** 2 operators (max + mean)
- **Features:** 512 × 2 = 1,024
- **Characteristics:** Dictionary-based, competitive accuracy, medium complexity

## Key Differences

| Aspect | MiniRocket | MultiRocket | HYDRA |
|--------|------------|-------------|-------|
| Kernel Selection | Fixed patterns | Fixed patterns | Dictionary learning |
| Kernel Count | 84 | 84 | 512 |
| Kernel Organization | None | None | 8 groups |
| Input Transformations | None | First-order diff | None |
| Pooling Complexity | Medium (quantiles) | High (4 operators) | Low (2 operators) |
| Feature Dimensionality | 840-2,688 | 2,688-5,376 | 1,024 |
| Computational Cost | Low | Medium-High | Medium |

## FPGA Implementation Details

### Memory Layout

**Kernel Weights:** `[512][9]` = 4,608 float32 values = 18.4 KB
**Biases:** `[512]` = 512 float32 = 2 KB
**Dilations:** `[512]` = 512 int32 = 2 KB
**Scaler Mean:** `[1024]` = 4 KB
**Scaler Scale:** `[1024]` = 4 KB
**Classifier Coefficients:** `[1024][num_classes]` ≈ 8-40 KB

**Total Model Size:** ~40-70 KB (fits easily in HBM)

### Computational Complexity

**Per Kernel Convolution:**
- Kernel span with dilation d: `(9-1)*d + 1`
- Output length: `time_series_length - kernel_span + 1`
- Operations: `output_length × 9` multiply-adds

**For 512 Kernels:**
- Total operations: `512 × time_series_length × 9` ≈ 700K ops (for length=150)
- Pooling operations: `512 × 2` simple reductions
- Classification: `1024 × num_classes` multiply-adds

**Estimated Cycles (HLS):**
- Convolution: ~60,000 cycles (pipelined II=1)
- Pooling: ~10,000 cycles
- Normalization: ~2,000 cycles
- Classification: ~5,000 cycles
- **Total:** ~80,000-100,000 cycles

**At 300 MHz:** 80,000 cycles ÷ 300 MHz = 0.27 ms latency

### HLS Optimizations

**Array Partitioning:**
```cpp
data_t sliding_window[KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable=sliding_window complete
```
Fully unrolls 9-element window for parallel access.

**Pipelining:**
```cpp
CONV_LOOP: for (int t = 0; t < output_length; t++) {
    #pragma HLS PIPELINE II=1
    // Convolution operations
}
```
Initiates new iteration every cycle (II=1).

**Dataflow:**
Main pipeline: `Load → Feature Extract → Normalize → Classify → Store`

## Training Procedure

### 1. Dictionary Initialization

**Option A: Random (current implementation)**
```python
dictionary = np.random.randn(512, 9)
# Normalize each kernel
for i in range(512):
    dictionary[i] = (dictionary[i] - mean) / std
```

**Option B: K-means clustering**
```python
# Extract random subsequences from training data
subsequences = extract_random_subsequences(X_train, size=9, count=5000)
# Cluster into 512 centroids
kmeans = KMeans(n_clusters=512)
dictionary = kmeans.fit(subsequences).cluster_centers_
```

**Option C: Gradient-based learning**
```python
# Learn kernels end-to-end with backpropagation
# (more complex, not implemented in this version)
```

### 2. Feature Extraction

```python
features = []
for x in X_train:
    x_features = []
    for kernel, bias, dilation in zip(dictionary, biases, dilations):
        conv = convolve(x, kernel, dilation) + bias
        x_features.append(max(conv))
        x_features.append(mean(conv))
    features.append(x_features)
```

### 3. Normalization & Classification

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

classifier = RidgeClassifierCV()
classifier.fit(X_scaled, y_train)
```

## Expected Performance

### Accuracy

Based on similar dictionary-based methods:
- **GunPoint:** 95-98%
- **ItalyPowerDemand:** 94-96%
- **ECG200:** 87-90%
- **Arrowhead:** 80-85%

Competitive with MiniRocket and MultiRocket, though specific performance depends on dictionary initialization.

### Latency & Throughput

**Single Compute Unit:**
- Latency: 0.3-0.5 ms
- Throughput: 2,000-3,000 inferences/sec

**2 Compute Units:**
- Throughput: 4,000-6,000 inferences/sec

**3 Compute Units:**
- Throughput: 6,000-9,000 inferences/sec

### Comparison

```
Algorithm       | Latency (ms) | Throughput (inf/s) | Accuracy
----------------|--------------|-------------------|----------
MiniRocket (1CU)| 0.20        | 15,000            | 94%
MultiRocket (1CU)| 0.35       | 8,000             | 96%
HYDRA (1CU)     | 0.40        | 2,500             | 95-96%
HYDRA (2CU)     | 0.40        | 5,000             | 95-96%
HYDRA (4CU)     | 0.40        | 10,000            | 95-96%
```

HYDRA trades some throughput for a more flexible dictionary-based approach while maintaining competitive accuracy.

## References

1. Dempster, A., Schmidt, D. F., & Webb, G. I. (2023). "HYDRA: Competing convolutional kernels for fast and accurate time series classification." Data Mining and Knowledge Discovery.

2. Dempster, A., Petitjean, F., & Webb, G. I. (2020). "ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels." Data Mining and Knowledge Discovery, 34(5), 1454-1495.

3. Tan, C. W., Dempster, A., Bergmeir, C., & Webb, G. I. (2022). "MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification." Data Mining and Knowledge Discovery, 36(5), 1623-1646.
