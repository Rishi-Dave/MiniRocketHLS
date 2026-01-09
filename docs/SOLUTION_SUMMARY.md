# MiniRocket/MultiRocket Training Issues - Comprehensive Solution

## Date: 2026-01-08

## Problems Identified and Fixed

### Problem 1: MiniRocket/MultiRocket API - Kernel Parameters Not Exposed

**Issue:**
- The aeon library's MiniRocket and MultiRocket don't expose `kernels_`, `biases_`, `dilations_` attributes directly
- Instead, they have `parameters` (MiniRocket) and `parameter`/`parameter1` (MultiRocket) tuples after fitting
- Training scripts were trying to access non-existent attributes, causing failures

**Root Cause:**
The aeon implementation of MiniRocket/MultiRocket uses a tuple-based parameter structure instead of individual attributes:

**MiniRocket:**
- `parameters` is a tuple of 5 elements:
  - `parameters[0]`: n_channels_per_combination (array)
  - `parameters[1]`: channel_indices (array)
  - `parameters[2]`: dilations (array)
  - `parameters[3]`: n_features_per_dilation (array)
  - `parameters[4]`: biases (array)

**MultiRocket:**
- `parameter`: tuple of 3 elements for original series
  - `parameter[0]`: dilations (array)
  - `parameter[1]`: n_features_per_dilation (array)
  - `parameter[2]`: biases (array)
- `parameter1`: tuple of 3 elements for differenced series
  - `parameter1[0]`: dilations (array)
  - `parameter1[1]`: n_features_per_dilation (array)
  - `parameter1[2]`: biases (array)

**Solution:**
Updated all training scripts to correctly extract parameters from the tuple structure:

1. **MiniRocket Training Script** (`/home/rdave009/minirocket-hls/MiniRocketHLS/reference_1to1/scripts/train_and_save_models.py`):
   - Extracts all 5 parameter elements from `minirocket.parameters`
   - Saves them with descriptive names in the JSON model file
   - Documents that MiniRocket uses 84 fixed kernel patterns (not learned weights)

2. **MultiRocket Training Script** (`/home/rdave009/minirocket-hls/MiniRocketHLS/multirocket_optimized/scripts/train_and_save_models.py`):
   - Extracts parameters from both `multirocket.parameter` and `multirocket.parameter1`
   - Saves separate parameter sets for original and differenced series
   - Documents the 4 pooling operators used (PPV, MPV, MIPV, LSPV)

3. **HYDRA Training Script** (`/home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/scripts/train_and_save_models.py`):
   - Already correct - uses custom HYDRA implementation with proper attributes
   - No changes needed

### Problem 2: Dataset Loading Issues

#### Issue 2a: MosquitoSound - Nested Directory Structure

**Issue:**
- Files were in nested directory `MosquitoSound/MosquitoSound/` instead of `MosquitoSound/`
- Files were in `.arff` format, but dataset loader expected them in parent directory

**Solution:**
```bash
# Moved arff files from nested directory to parent
mv /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/datasets/ucr_data/MosquitoSound/MosquitoSound/*.arff \
   /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/datasets/ucr_data/MosquitoSound/

# Removed empty nested directory
rm -rf /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/datasets/ucr_data/MosquitoSound/MosquitoSound/
```

**Result:**
- MosquitoSound_TRAIN.arff and MosquitoSound_TEST.arff now in correct location
- Dataset loader can find and load the files

#### Issue 2b: FruitFlies - ZIP Extraction Failure

**Issue:**
- ZIP extraction was failing with "compression method not supported" when using Python's zipfile
- The directory was empty, preventing training

**Solution:**
```bash
# Used system unzip command instead of Python's zipfile
cd /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/datasets/ucr_data/FruitFlies
unzip -q /home/rdave009/minirocket-hls/tmp/tmpor5stevi/FruitFlies.zip
```

**Result:**
- Successfully extracted FruitFlies_TRAIN.arff, FruitFlies_TEST.arff, and .ts files
- Dataset now loadable for training

## Important Notes for FPGA Implementation

### MiniRocket Kernel Structure
- MiniRocket does NOT store learned kernel weights
- Uses 84 fixed kernel patterns generated from `combinations(9, 3)`
- Kernel weights are always: 6 values of `-1` and 3 values of `2`
- The 3 positions with value `2` are determined by the combination indices
- Example: combination (0,1,2) means positions 0,1,2 have value 2, rest have -1

### MultiRocket Structure
- Uses same 84 fixed kernel patterns as MiniRocket
- Applies 4 pooling operators instead of just PPV:
  1. PPV - Percentage of Positive Values
  2. MPV - Mean of Positive Values
  3. MIPV - Mean of Indices of Positive Values
  4. LSPV - Longest Stretch of Positive Values
- Processes both original series AND first-order differenced series
- Total features: 84 kernels × 4 pooling × 2 representations = 672 features per dilation

### HYDRA Structure
- Uses custom dictionary-based kernels (learned)
- 512 kernels with actual weight values (not fixed patterns)
- 2 pooling operators per kernel (max + mean)
- Total features: 512 × 2 = 1,024

## Files Modified

1. `/home/rdave009/minirocket-hls/MiniRocketHLS/reference_1to1/scripts/train_and_save_models.py`
2. `/home/rdave009/minirocket-hls/MiniRocketHLS/multirocket_optimized/scripts/train_and_save_models.py`
3. Dataset directories:
   - `/home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/datasets/ucr_data/MosquitoSound/`
   - `/home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/datasets/ucr_data/FruitFlies/`

## Next Steps

### 1. Test the Updated Scripts

Run training for each algorithm to verify the fixes:

```bash
# Set TMPDIR to avoid /tmp space limitations
export TMPDIR=/home/rdave009/minirocket-hls/tmp

# Test HYDRA training
cd /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/scripts
python3 train_and_save_models.py --dataset InsectSound

# Test MiniRocket training
cd /home/rdave009/minirocket-hls/MiniRocketHLS/reference_1to1/scripts
python3 train_and_save_models.py --dataset InsectSound

# Test MultiRocket training
cd /home/rdave009/minirocket-hls/MiniRocketHLS/multirocket_optimized/scripts
python3 train_and_save_models.py --dataset InsectSound
```

### 2. Train All Datasets

Once verified, train all three datasets for each algorithm:

```bash
export TMPDIR=/home/rdave009/minirocket-hls/tmp

# HYDRA - all datasets
cd /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/scripts
python3 train_and_save_models.py --all

# MiniRocket - all datasets
cd /home/rdave009/minirocket-hls/MiniRocketHLS/reference_1to1/scripts
python3 train_and_save_models.py --all

# MultiRocket - all datasets
cd /home/rdave009/minirocket-hls/MiniRocketHLS/multirocket_optimized/scripts
python3 train_and_save_models.py --all
```

### 3. Verify Model Files

Check that the model JSON files contain the correct parameter structure:

```bash
# Example: Check MiniRocket model
cat /home/rdave009/minirocket-hls/MiniRocketHLS/reference_1to1/models/minirocket_insectsound_model.json | python3 -m json.tool | head -50

# Verify it contains:
# - dilations
# - n_features_per_dilation
# - biases
# - scaler parameters
# - classifier coefficients
```

## Investigation Scripts Created

Two investigation scripts were created to understand the aeon API:

1. `/home/rdave009/minirocket-hls/investigate_parameters.py` - Basic structure exploration
2. `/home/rdave009/minirocket-hls/investigate_parameters2.py` - Detailed parameter analysis

These can be deleted after verification or kept for future reference.

## References

- MiniRocket implementation: `/home/rdave009/.local/lib/python3.10/site-packages/aeon/transformations/collection/convolution_based/_minirocket.py`
- MultiRocket implementation: `/home/rdave009/.local/lib/python3.10/site-packages/aeon/transformations/collection/convolution_based/_multirocket.py`
- Custom HYDRA: `/home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/scripts/custom_hydra.py`
