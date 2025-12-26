# MiniRocket Algorithm for Time Series Classification

## Overview

MiniRocket (Mini RandOm Convolutional KErnel Transform) is a state-of-the-art algorithm for time series classification that achieves competitive accuracy with exceptional computational efficiency.

## Why MiniRocket Works for Time Series

### The Core Insight

Traditional deep learning approaches for time series require:
- Large amounts of training data
- Extensive hyperparameter tuning
- Long training times
- Complex architectures

MiniRocket takes a radically different approach:
1. **Fixed random features** - No learning of convolutional kernels
2. **Universal feature extraction** - Same kernels work across all datasets
3. **Simple linear classifier** - Only the final layer is trained

This combination makes MiniRocket:
- **Fast to train** (seconds vs hours)
- **Accurate** (~94% average on UCR benchmark)
- **Hardware-friendly** (no backpropagation, fixed computation)

## Algorithm Pipeline

```
Input Time Series → Feature Extraction → Normalization → Linear Classification → Prediction
```

### Stage 1: Feature Extraction (Convolution)

MiniRocket uses **84 fixed convolutional kernels** with varying dilations:

```
Kernel Pattern: [-1, -1, -1, 1, 1, 1, 1, 1, 1]  (9 weights)
Dilations: [1, 2, 4, 8, 16, ...]

For dilation=1: kernel applied to consecutive elements
For dilation=2: kernel applied to every 2nd element
For dilation=4: kernel applied to every 4th element
```

**Key Innovation**: Instead of extracting thousands of features per kernel, MiniRocket extracts only **2 features per kernel**:
- PPV (Proportion of Positive Values): `count(conv(x) > bias) / length`

**Example**:
```
Time series: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
Kernel: [-1, -1, -1, 1, 1, 1, 1, 1, 1]
Dilation: 1
Bias: 0.5

Convolution:
conv[0] = -1*1 - 1*2 - 1*3 + 1*4 + 1*5 + 1*6 + 1*7 + 1*8 + 1*9 = 33
conv[1] = -1*2 - 1*3 - 1*4 + 1*5 + 1*6 + 1*7 + 1*8 + 1*9 = 30
...

PPV = count(conv > 0.5) / num_windows = 0.85  ← This is the feature!
```

With 84 kernels × 5 dilations = 420 features total.

### Stage 2: Normalization

Standardize features to zero mean and unit variance:

```python
normalized_feature[i] = (feature[i] - mean[i]) / std[i]
```

This ensures all features contribute equally to classification.

### Stage 3: Linear Classification

A simple Ridge regression classifier:

```
score[class_c] = Σ(coef[c][i] * feature[i]) + intercept[c]
prediction = argmax(scores)
```

## Why This Works

### 1. Universal Feature Extraction

The random kernels capture diverse temporal patterns:
- **Different dilations** = different time scales
- **Fixed weights** = computational efficiency
- **PPV metric** = invariant to amplitude scaling

### 2. Simplicity = Generalization

By not learning convolutional weights, MiniRocket:
- Avoids overfitting on small datasets
- Generalizes across diverse time series types
- Requires minimal training data

### 3. Computational Efficiency

No gradients to compute:
- Feature extraction is pure forward pass
- Only linear classifier needs training
- Perfect for hardware acceleration

## Performance Characteristics

| Metric | MiniRocket | Deep Learning |
|--------|------------|---------------|
| **Training Time** | Seconds | Hours |
| **Accuracy** | ~94% (UCR avg) | ~95% (UCR avg) |
| **Inference Speed** | Very Fast | Moderate |
| **Training Data Needed** | Small | Large |
| **Hardware Acceleration** | Excellent | Moderate |

## Application Domains

MiniRocket excels at:

### Medical/Healthcare
- ECG/EEG classification
- Activity recognition from wearables
- Fall detection

### Industrial
- Predictive maintenance (vibration analysis)
- Quality control (sensor data)
- Anomaly detection

### IoT/Edge
- Gesture recognition
- Environmental monitoring
- Smart home automation

## References

1. Dempster, A., Schmidt, D.F., Webb, G.I. (2021). "MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification." KDD 2021.

2. UCR Time Series Classification Archive: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
