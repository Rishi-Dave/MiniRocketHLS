# MiniRocket Algorithm

## Overview

**MiniRocket** (MINImally RandOm Convolutional KErnel Transform) is a fast, deterministic time series classification algorithm that achieves state-of-the-art accuracy with minimal computational cost.

**Key Innovation**: Fixed random convolution kernels eliminate the need for gradient-based learning, enabling:
- Training in seconds vs hours for deep learning
- ~94% average accuracy on UCR benchmark
- Excellent hardware acceleration potential

**Reference**: Dempster, A., Schmidt, D.F., Webb, G.I. (2021). "MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification." KDD 2021.

---

## Algorithm Pipeline

```
Time Series → Convolution Transform → Feature Extraction → Normalization → Linear Classifier → Prediction
```

---

## 1. Convolution Transform

### 1.1 Fixed Kernel Structure

Each kernel has **9 weights** restricted to two values:
- **6 positions**: weight = **-1**
- **3 positions**: weight = **+2**

The 3 positions with weight +2 are selected from C(9,3) = **84 combinations**, giving 84 fixed kernels.

### 1.2 Dilated Convolution

Multiple **dilation** values scale the effective kernel size:

```
Dilation=1: [w₀ w₁ w₂ w₃ w₄ w₅ w₆ w₇ w₈]
Dilation=2: [w₀ _ w₁ _ w₂ _ w₃ _ w₄ _ w₅ _ w₆ _ w₇ _ w₈]
Dilation=4: [w₀ _ _ _ w₁ _ _ _ w₂ ...]
```

Dilations capture patterns at different time scales without increasing kernel length.

### 1.3 Cumulative Convolution (Implementation Optimization)

Instead of computing full convolutions, MiniRocket uses cumulative sums:

**Step 1**: Compute base cumulative arrays:
```python
A = -X              # α * X where α = -1
G = X + X + X       # γ * X where γ = +3 = 2 - (-1)
```

**Step 2**: Build cumulative convolution:
```python
C_alpha[i] = A[i] + A[i-d] + A[i-2d] + ...  # Sum over all 9 positions
C_gamma[j][i] = G[i-j*d]                     # Value at position j
```

**Step 3**: Combine for final convolution:
```python
# For a kernel with +2 weights at positions i₀, i₁, i₂:
C = C_alpha + C_gamma[i₀] + C_gamma[i₁] + C_gamma[i₂]
```

**Why this works**:
- C_alpha contains contributions from all 9 positions with weight -1
- Adding C_gamma at 3 positions effectively changes those weights from -1 to -1+3 = +2
- Result: Equivalent to full convolution but **much faster**

---

## 2. Feature Extraction

### 2.1 Proportion of Positive Values (PPV)

For each kernel and dilation, extract **10 features** using:

```python
PPV(C, q) = count(C > quantile(C, q)) / length(C)
```

Where quantiles: `q ∈ {0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99}`

**Total features**: 84 kernels × 10 features = **840 features** (with default dil ations)

### 2.2 Bias Selection

The quantile values serve as **thresholds (biases)** for the comparison `C > bias`.

**Why quantiles?**: Automatically adapt to the data distribution without parameter tuning.

---

## 3. Feature Normalization

Apply **StandardScaler** to ensure zero mean and unit variance:

```python
X_scaled = (X - mean) / std
```

This ensures all features contribute equally to classification.

---

## 4. Linear Classification

Use **Ridge Regression** (L2-regularized linear model):

```python
score[c] = Σᵢ (coefficient[c][i] × feature[i]) + intercept[c]
prediction = argmax(score)
```

**Why Ridge?**: Handles multicollinearity well and regularization prevents overfitting.

---

## Mathematical Formulation

### Convolution at position t

For kernel weights **w** and dilation **d**:

```
C[t] = Σₖ w[k] × X[t - k×d]
     = Σₖ∈α w[k] × X[t - k×d] + Σₖ∈γ w[k] × X[t - k×d]
```

Where:
- α = set of 6 positions with weight -1
- γ = set of 3 positions with weight +2

### Cumulative Optimization

```
C_alpha[t] = Σₖ₌₀⁸ (-1) × X[t - k×d] = -Σₖ₌₀⁸ X[t - k×d]

C_gamma[j][t] = 3 × X[t - j×d]    for j ∈ {0, 1, ..., 8}

C[t] = C_alpha[t] + Σⱼ∈γ C_gamma[j][t]
```

**Net effect**: 6 positions with -1, 3 selected positions with +2

---

## Why MiniRocket Works

### 1. Universal Feature Extraction
- **Fixed kernels** capture diverse temporal patterns without learning
- **Multiple dilations** handle patterns at different time scales
- **PPV metric** provides invariance to amplitude scaling

### 2. Computational Efficiency
- No backpropagation or gradient computation
- Cumulative convolution is O(n) per kernel
- Only linear classifier requires training

### 3. Hardware-Friendly
- Fixed-point arithmetic compatible (no floating gradients)
- Highly parallelizable (independent kernels)
- Deterministic execution (no stochastic training)

---

## Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Kernels** | 84 | C(9,3) combinations of +2 positions |
| **Kernel Length** | 9 | Fixed kernel size |
| **Dilations** | [1, 2, 4, 8, ...] | Powers of 2 up to max(32, ⌊T/16⌋) |
| **Features per Kernel** | 10 | PPV at different quantiles |
| **Quantiles** | 0.1 to 0.99 | Threshold values for PPV |
| **Ridge Alpha** | Auto (CV) | Regularization strength |

---

## Performance Characteristics

| Metric | MiniRocket | Deep Learning |
|--------|------------|---------------|
| **Training Time** | Seconds | Hours |
| **UCR Accuracy** | ~94% | ~95% |
| **Inference Speed** | Very Fast | Moderate |
| **Memory** | Low | High (gradients) |
| **Hardware Acceleration** | Excellent | Moderate |

---

## Implementation Considerations

### Fixed-Point Arithmetic
For FPGA implementation, the algorithm supports fixed-point:
- Recommended: `ap_fixed<32,16>` (16 integer, 16 fractional bits)
- Minimal accuracy loss (<0.5% typically)

### Padding
Time series shorter than kernel span require padding:
- Use zero-padding or reflection padding
- Adjust PPV calculation to exclude padded regions

### Parallelization
- **Kernel-level**: Each of 84 kernels can compute independently
- **Dilation-level**: Different dilations can process in parallel
- **Feature-level**: PPV calculations are independent

---

## References

1. Dempster, A., Schmidt, D.F., Webb, G.I. (2021). "MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification." KDD 2021.

2. Dempster, A., Petitjean, F., Webb, G.I. (2020). "ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels." Data Mining and Knowledge Discovery 34, 1454-1495.

3. UCR Time Series Classification Archive: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
