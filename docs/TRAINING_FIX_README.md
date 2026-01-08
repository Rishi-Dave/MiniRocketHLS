# MiniRocket/MultiRocket Training Fix - Quick Start Guide

## What Was Fixed

This fix resolves two critical issues that prevented training:

1. **API Compatibility Issue**: MiniRocket and MultiRocket from aeon library don't expose `kernels_`, `biases_`, `dilations_` attributes. Instead, they use tuple-based `parameters` attributes.

2. **Dataset Issues**:
   - MosquitoSound files were in nested directory
   - FruitFlies ZIP extraction was failing

## Files Changed

### Training Scripts Updated
- `/home/rdave009/minirocket-hls/MiniRocketHLS/reference_1to1/scripts/train_and_save_models.py`
- `/home/rdave009/minirocket-hls/MiniRocketHLS/multirocket_optimized/scripts/train_and_save_models.py`

### Dataset Fixes Applied
- `/home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/datasets/ucr_data/MosquitoSound/`
- `/home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/datasets/ucr_data/FruitFlies/`

## Quick Start

### Option 1: Run All Training (Recommended)

```bash
# Run the master script that trains all three algorithms on all datasets
cd /home/rdave009/minirocket-hls
./run_all_training.sh
```

This will:
- Set TMPDIR to avoid /tmp space issues
- Train HYDRA on InsectSound, MosquitoSound, FruitFlies
- Train MiniRocket on InsectSound, MosquitoSound, FruitFlies
- Train MultiRocket on InsectSound, MosquitoSound, FruitFlies
- Print a comprehensive summary

### Option 2: Run Individual Training

```bash
# Set environment
export TMPDIR=/home/rdave009/minirocket-hls/tmp
mkdir -p $TMPDIR

# HYDRA
cd /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/scripts
python3 train_and_save_models.py --all

# MiniRocket
cd /home/rdave009/minirocket-hls/MiniRocketHLS/reference_1to1/scripts
python3 train_and_save_models.py --all

# MultiRocket
cd /home/rdave009/minirocket-hls/MiniRocketHLS/multirocket_optimized/scripts
python3 train_and_save_models.py --all
```

### Option 3: Test Individual Dataset First

```bash
export TMPDIR=/home/rdave009/minirocket-hls/tmp

# Test with InsectSound first (smallest dataset)
cd /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/scripts
python3 train_and_save_models.py --dataset InsectSound
```

## Verification

### Test the Fixes
```bash
cd /home/rdave009/minirocket-hls
python3 test_fixes.py
```

This runs a quick test to verify parameter extraction works correctly.

### Check Model Files

After training, verify the model files were created:

```bash
# HYDRA models
ls -lh /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/models/

# MiniRocket models
ls -lh /home/rdave009/minirocket-hls/MiniRocketHLS/reference_1to1/models/

# MultiRocket models
ls -lh /home/rdave009/minirocket-hls/MiniRocketHLS/multirocket_optimized/models/
```

Each should contain:
- `*_model.json` - Model parameters
- `*_test.json` - Test data

### Inspect a Model File

```bash
# Example: View MiniRocket model structure
python3 -m json.tool /home/rdave009/minirocket-hls/MiniRocketHLS/reference_1to1/models/minirocket_insectsound_model.json | head -100
```

## Understanding the Parameter Structure

### MiniRocket (`parameters` tuple)
```python
parameters[0]  # n_channels_per_combination
parameters[1]  # channel_indices
parameters[2]  # dilations
parameters[3]  # n_features_per_dilation
parameters[4]  # biases
```

Note: MiniRocket uses 84 fixed kernel patterns (not learned weights).
Kernel weights are always: 6 values of -1, 3 values of 2 at specific positions.

### MultiRocket (`parameter` and `parameter1` tuples)
```python
# For original series
parameter[0]   # dilations
parameter[1]   # n_features_per_dilation
parameter[2]   # biases

# For first-order differenced series
parameter1[0]  # dilations
parameter1[1]  # n_features_per_dilation
parameter1[2]  # biases
```

Note: MultiRocket uses same 84 fixed patterns but with 4 pooling operators
(PPV, MPV, MIPV, LSPV) on both original and differenced series.

### HYDRA (custom implementation)
```python
hydra.dictionary_  # [512, 9] learned kernel weights
hydra.biases_      # [512] bias values
hydra.dilations_   # [512] dilation values
```

Note: HYDRA uses learned dictionary kernels, not fixed patterns.

## Expected Output

### Training Time Estimates
- **InsectSound** (~220 samples): 1-5 minutes per algorithm
- **MosquitoSound** (~24k samples): 30-60 minutes per algorithm
- **FruitFlies** (~2k samples): 5-15 minutes per algorithm

### Model File Sizes
- HYDRA models: ~200KB - 1MB
- MiniRocket models: ~50KB - 500KB
- MultiRocket models: ~100KB - 1MB

## Troubleshooting

### Issue: TMPDIR space error
```bash
# Solution: Clean up temp directory
rm -rf /home/rdave009/minirocket-hls/tmp/*
# Or set to a different location with more space
export TMPDIR=/path/to/large/disk/tmp
```

### Issue: Dataset not found
```bash
# Verify datasets exist
ls /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/datasets/ucr_data/

# Should show:
# - InsectSound/
# - MosquitoSound/
# - FruitFlies/
```

### Issue: Import errors
```bash
# Verify aeon is installed
python3 -c "import aeon; print(aeon.__version__)"

# Should print version (e.g., 0.9.0 or similar)
```

### Issue: AttributeError on parameters
This should be fixed by the updated scripts. If you still see this:
1. Verify you're running the updated scripts
2. Check the modification dates:
```bash
ls -l /home/rdave009/minirocket-hls/MiniRocketHLS/reference_1to1/scripts/train_and_save_models.py
ls -l /home/rdave009/minirocket-hls/MiniRocketHLS/multirocket_optimized/scripts/train_and_save_models.py
```

## Documentation

For detailed information, see:
- **SOLUTION_SUMMARY.md** - Comprehensive technical details
- **investigate_parameters.py** - Parameter structure exploration script
- **test_fixes.py** - Verification test script

## Next Steps After Training

1. **Verify accuracy**: Check the `train_accuracy` and `test_accuracy` in model JSON files
2. **Compare algorithms**: Compare performance across HYDRA, MiniRocket, and MultiRocket
3. **FPGA synthesis**: Use the model JSON files for FPGA implementation
4. **Benchmarking**: Run benchmark scripts to measure inference performance

## Contact

For issues or questions about this fix, refer to the investigation scripts and
SOLUTION_SUMMARY.md for detailed technical information.
