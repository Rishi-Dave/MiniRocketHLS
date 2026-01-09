# MiniRocket/MultiRocket Training Issues - Solution Deliverables

## Date: 2026-01-08

## Summary

This solution comprehensively fixes the MiniRocket and MultiRocket training issues caused by:
1. API incompatibility with aeon library's parameter structure
2. Dataset loading problems (nested directories and extraction failures)

All issues have been resolved and tested. The training scripts are now ready to use.

## Deliverables

### 1. Fixed Training Scripts

#### Updated Files:
- **MiniRocket**: `/home/rdave009/minirocket-hls/MiniRocketHLS/reference_1to1/scripts/train_and_save_models.py`
  - Correctly extracts 5-element `parameters` tuple
  - Saves all kernel configuration data to JSON
  - Documents fixed kernel pattern structure

- **MultiRocket**: `/home/rdave009/minirocket-hls/MiniRocketHLS/multirocket_optimized/scripts/train_and_save_models.py`
  - Correctly extracts `parameter` and `parameter1` tuples
  - Saves separate configurations for original and differenced series
  - Documents 4 pooling operators (PPV, MPV, MIPV, LSPV)

- **HYDRA**: `/home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/scripts/train_and_save_models.py`
  - Already correct (uses custom HYDRA with proper attributes)
  - No changes required

### 2. Fixed Datasets

#### MosquitoSound:
- **Location**: `/home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/datasets/ucr_data/MosquitoSound/`
- **Issue Fixed**: Moved ARFF files from nested directory to correct location
- **Files Available**:
  - `MosquitoSound_TRAIN.arff`
  - `MosquitoSound_TEST.arff`

#### FruitFlies:
- **Location**: `/home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/datasets/ucr_data/FruitFlies/`
- **Issue Fixed**: Successfully extracted ZIP file using system unzip
- **Files Available**:
  - `FruitFlies_TRAIN.arff`
  - `FruitFlies_TRAIN.ts`
  - `FruitFlies_TEST.arff`
  - `FruitFlies_TEST.ts`

### 3. Automation Scripts

#### Master Training Script:
- **File**: `/home/rdave009/minirocket-hls/run_all_training.sh`
- **Purpose**: One-command training for all algorithms and datasets
- **Features**:
  - Sets TMPDIR environment variable
  - Runs HYDRA, MiniRocket, MultiRocket sequentially
  - Colored output with status tracking
  - Comprehensive error reporting
  - Final summary of all training runs

#### Usage:
```bash
cd /home/rdave009/minirocket-hls
./run_all_training.sh
```

### 4. Testing and Validation Scripts

#### Parameter Extraction Test:
- **File**: `/home/rdave009/minirocket-hls/test_fixes.py`
- **Purpose**: Verify parameter extraction works correctly
- **Tests**:
  - MiniRocket 5-element tuple extraction
  - MultiRocket dual-tuple extraction
  - Transform operations
- **Status**: All tests passing

#### Investigation Scripts:
- **File 1**: `/home/rdave009/minirocket-hls/investigate_parameters.py`
  - Basic parameter structure exploration
  - Identifies available attributes

- **File 2**: `/home/rdave009/minirocket-hls/investigate_parameters2.py`
  - Detailed parameter analysis
  - Shows data types and shapes

### 5. Documentation

#### Quick Start Guide:
- **File**: `/home/rdave009/minirocket-hls/TRAINING_FIX_README.md`
- **Content**:
  - Quick start instructions
  - Three usage options (automated, individual, test)
  - Parameter structure reference
  - Troubleshooting guide
  - Expected outputs and timing

#### Technical Documentation:
- **File**: `/home/rdave009/minirocket-hls/SOLUTION_SUMMARY.md`
- **Content**:
  - Detailed problem analysis
  - Root cause explanation
  - Complete solution description
  - Parameter structure documentation
  - Implementation notes for FPGA
  - Next steps guide

## Key Technical Insights

### MiniRocket Parameter Structure
```python
# From aeon implementation
parameters = (
    n_channels_per_combination,  # [N] channel configuration
    channel_indices,              # [M] selected channels
    dilations,                    # [D] dilation values
    n_features_per_dilation,      # [D] features per dilation
    biases                        # [84] bias values
)
```

**Important**: MiniRocket does NOT store kernel weights. It uses 84 fixed patterns:
- Kernel size: 9
- Weights: 6 values of -1, 3 values of 2
- Patterns: All combinations of choosing 3 positions from 9 for the 2s

### MultiRocket Parameter Structure
```python
# Two separate tuples for two representations
parameter = (dilations, n_features_per_dilation, biases)    # Original series
parameter1 = (dilations, n_features_per_dilation, biases)   # Diff series
```

**Important**: MultiRocket uses:
- Same 84 fixed kernel patterns as MiniRocket
- 4 pooling operators: PPV, MPV, MIPV, LSPV
- Both original and first-order differenced series
- Total features: 84 × 4 pooling × 2 representations = 672 per dilation

### HYDRA Structure (For Reference)
```python
# Custom implementation with learned kernels
dictionary_   # [512, 9] actual kernel weights (learned)
biases_       # [512] bias values
dilations_    # [512] dilation values
```

## Verification Checklist

- [x] MosquitoSound dataset files in correct location
- [x] FruitFlies dataset extracted successfully
- [x] MiniRocket training script updated for `parameters` tuple
- [x] MultiRocket training script updated for `parameter`/`parameter1` tuples
- [x] HYDRA training script verified (already correct)
- [x] Test script created and passing
- [x] Master training script created
- [x] Documentation complete

## Usage Examples

### Run All Training:
```bash
cd /home/rdave009/minirocket-hls
./run_all_training.sh
```

### Test Before Full Training:
```bash
python3 /home/rdave009/minirocket-hls/test_fixes.py
```

### Train Single Dataset:
```bash
export TMPDIR=/home/rdave009/minirocket-hls/tmp
cd /home/rdave009/minirocket-hls/MiniRocketHLS/reference_1to1/scripts
python3 train_and_save_models.py --dataset InsectSound
```

### Verify Model Output:
```bash
# Check model was created
ls -lh /home/rdave009/minirocket-hls/MiniRocketHLS/reference_1to1/models/

# View model structure
python3 -m json.tool minirocket_insectsound_model.json | head -50
```

## Expected Model Files After Training

### HYDRA Models:
```
/home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/models/
├── hydra_insectsound_model.json
├── hydra_insectsound_test.json
├── hydra_mosquitosound_model.json
├── hydra_mosquitosound_test.json
├── hydra_fruitflies_model.json
└── hydra_fruitflies_test.json
```

### MiniRocket Models:
```
/home/rdave009/minirocket-hls/MiniRocketHLS/reference_1to1/models/
├── minirocket_insectsound_model.json
├── minirocket_insectsound_test.json
├── minirocket_mosquitosound_model.json
├── minirocket_mosquitosound_test.json
├── minirocket_fruitflies_model.json
└── minirocket_fruitflies_test.json
```

### MultiRocket Models:
```
/home/rdave009/minirocket-hls/MiniRocketHLS/multirocket_optimized/models/
├── multirocket_insectsound_model.json
├── multirocket_insectsound_test.json
├── multirocket_mosquitosound_model.json
├── multirocket_mosquitosound_test.json
├── multirocket_fruitflies_model.json
└── multirocket_fruitflies_test.json
```

## Next Steps

1. **Run Training**: Execute `./run_all_training.sh` to train all models
2. **Verify Results**: Check model JSON files for accuracy metrics
3. **FPGA Synthesis**: Use the model parameters for FPGA implementation
4. **Benchmarking**: Compare performance across algorithms
5. **Optimization**: Fine-tune based on FPGA constraints

## Support Files

All scripts and documentation are in: `/home/rdave009/minirocket-hls/`

- `TRAINING_FIX_README.md` - Quick start guide
- `SOLUTION_SUMMARY.md` - Technical details
- `DELIVERABLES.md` - This file
- `run_all_training.sh` - Master training script
- `test_fixes.py` - Validation script
- `investigate_parameters.py` - API exploration
- `investigate_parameters2.py` - Detailed analysis

## Success Criteria

All success criteria have been met:

1. ✅ Dataset structure issues resolved
2. ✅ MiniRocket parameter extraction working
3. ✅ MultiRocket parameter extraction working
4. ✅ HYDRA verified (already working)
5. ✅ Training scripts updated and tested
6. ✅ Automation scripts created
7. ✅ Comprehensive documentation provided
8. ✅ Verification tests passing

## Conclusion

The solution is complete and ready for use. All training scripts have been updated to correctly handle the aeon library's parameter structure, and all dataset issues have been resolved. The training can now proceed without errors.
