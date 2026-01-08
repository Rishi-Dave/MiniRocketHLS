# UCR Dataset Train-Test Split Information

**Date:** January 8, 2026
**Project:** MiniRocket-HLS

---

## Overview

We are using the **standard UCR Time Series Archive splits** as provided by the aeon library. These are pre-defined train/test splits that come with the UCR datasets.

---

## Train-Test Split Details

### InsectSound Dataset

| Split | Samples | Percentage | Usage |
|-------|---------|------------|-------|
| **Train** | 25,000 | 50% | Model training |
| **Test** | 25,000 | 50% | Evaluation & FPGA validation |
| **Total** | 50,000 | 100% | - |

**Characteristics:**
- **Perfect 50/50 split** (1:1 ratio)
- 10 classes (insect species)
- Time series length: 600
- All test samples saved for FPGA testing

### Dataset Loading Code

**Location:** All training scripts use the same loading pattern

```python
from aeon.datasets import load_classification

# Load with standard UCR splits
X_train, y_train = load_classification(name, split="train", extract_path=extract_path)
X_test, y_test = load_classification(name, split="test", extract_path=extract_path)
```

**Files:**
- `hydra_optimized/scripts/train_and_save_models.py` (lines 27-28)
- `reference_1to1/scripts/train_and_save_models.py` (lines 29-30)
- `multirocket_optimized/scripts/train_and_save_models.py` (lines 29-30)

---

## What Gets Saved

### Training Phase

**Model trained on:**
- Train split: 25,000 samples
- Used for fitting HYDRA/MiniRocket/MultiRocket transformers
- Used for training Ridge classifier

### Test Data Saved for FPGA

**Current implementation saves ALL test samples:**

```python
# From train_and_save_models.py (line 131-138)
test_data = {
    "dataset": dataset_name,
    "num_samples": len(X_test),           # 25,000 for InsectSound
    "time_series_length": X_test.shape[1], # 600 for InsectSound
    "num_classes": len(np.unique(y_test)), # 10 for InsectSound
    "time_series": X_test.tolist(),        # ALL 25,000 samples
    "labels": y_test.tolist()              # ALL 25,000 labels
}
```

**Result:**
- Test file size: 264 MB (for InsectSound)
- Contains all 25,000 test samples
- Ready for comprehensive FPGA validation

---

## Other UCR Datasets

Based on UCR Archive standards, the splits vary by dataset:

### MosquitoSound (Expected)
```
Train: ~69,890 samples (50%)
Test:  ~69,890 samples (50%)
Total: ~139,780 samples
```

### FruitFlies (Expected)
```
Train: ~8,630 samples (50%)
Test:  ~8,629 samples (50%)
Total: ~17,259 samples
```

**Note:** Exact splits will be confirmed when training completes.

---

## Why This Split?

### Advantages of UCR Standard Splits

1. **Reproducibility:**
   - Standardized across research community
   - Enables fair comparison with published results
   - Well-established benchmark

2. **Pre-defined:**
   - No random splitting needed
   - Consistent across all runs
   - No seed dependencies

3. **Realistic Evaluation:**
   - Test set never seen during training
   - True generalization performance
   - Proper ML methodology

4. **Community Standard:**
   - Used in ROCKET, MiniRocket, MultiRocket papers
   - Used in HYDRA paper
   - Allows comparison with literature

---

## Cross-Validation?

**We are NOT using cross-validation.** Instead:

- Using UCR standard train/test split
- Model trained once on train split
- Evaluated once on test split
- **RidgeClassifierCV** does use cross-validation internally for hyperparameter (alpha) selection, but only on the training set

```python
# From training scripts
from sklearn.linear_model import RidgeClassifierCV

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_scaled, y_train)  # CV only on train set
```

---

## FPGA Testing Strategy

### Current Approach

**All test samples available:**
- Model JSON: Contains trained parameters
- Test JSON: Contains all 25,000 test samples
- FPGA can validate on any subset or all samples

### Typical FPGA Testing Scenarios

**Scenario 1: Quick Validation (Current)**
- Use first 50-100 samples
- Fast sanity check
- Verify FPGA execution works

**Scenario 2: Full Accuracy Validation**
- Use all 25,000 samples
- Compare FPGA vs Python accuracy
- Comprehensive verification

**Scenario 3: Performance Benchmarking**
- Use subset (e.g., 1000 samples)
- Measure throughput
- Profile latency distribution

---

## Implications for HYDRA FPGA Issue

### Why 50% Accuracy Was Suspicious

With 10 balanced classes:
- Random guessing: 10% accuracy
- **Observed: 50% accuracy**
- This suggested binary classification (2 classes)

**Root cause:** Stub loader hardcoded 2 classes instead of 10

### Expected Accuracy with Real Data

Based on saved metrics in model JSON:
```json
{
  "train_accuracy": 0.8710,  // 87.10%
  "test_accuracy": 0.7912    // 79.12%
}
```

**Once JSON loader is fixed:**
- FPGA should achieve ~79% accuracy
- Matching Python implementation
- On same 25,000 test samples

---

## Summary

| Aspect | Details |
|--------|---------|
| **Split Type** | UCR standard pre-defined |
| **Train/Test Ratio** | 50/50 (1:1) |
| **Train Samples** | 25,000 |
| **Test Samples** | 25,000 |
| **Total Samples** | 50,000 |
| **Cross-Validation** | No (standard split) |
| **Test Data Saved** | All 25,000 samples |
| **File Size** | 264 MB (InsectSound) |
| **Expected Accuracy** | ~79% (InsectSound) |
| **FPGA Flexibility** | Can test on any subset |

---

## References

- **UCR Time Series Archive:** https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
- **Aeon library:** https://www.aeon-toolkit.org/
- **Training scripts:**
  - `hydra_optimized/scripts/train_and_save_models.py`
  - `reference_1to1/scripts/train_and_save_models.py`
  - `multirocket_optimized/scripts/train_and_save_models.py`
