# HYDRA Training Summary - All Datasets

## Status Overview (as of Jan 9, 2026)

### InsectSound ✅ COMPLETE
- **Status**: Fully trained and tested on FPGA
- **Train Accuracy**: 70.75%
- **Test Accuracy**: 69.41% (Python) / 69.44% (FPGA)
- **FPGA Performance**:
  - Latency: 5.163 ms per inference
  - Throughput: 194 inferences/second
  - Test time: 129 seconds for 25,000 samples
- **Dataset Size**:
  - Train: 25,000 samples
  - Test: 25,000 samples
  - Time series length: 600
  - Classes: 10
- **Model File**: `models/hydra_insectsound_model.json` (334 KB)
- **Test Files**:
  - Full: `models/hydra_insectsound_test.json`
  - Quick (1,000 samples): `models/hydra_insectsound_test_1000.json`

**Makefile Commands**:
```bash
cd /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized
DEVICE=xilinx_u280_gen3x16_xdma_1_202211_1 make run         # Full test
DEVICE=xilinx_u280_gen3x16_xdma_1_202211_1 make run-quick  # Quick test
```

---

### MosquitoSound ✅ COMPLETE
- **Status**: Training completed successfully
- **Train Accuracy**: 70.40%
- **Test Accuracy**: 70.05%
- **Training Completed**: Jan 8, 21:51
- **Dataset Size**:
  - Train: 139,780 samples (largest in UCR archive)
  - Test: 139,780 samples
  - Time series length: 3,750
  - Classes: 6
- **Model File**: `models/hydra_mosquitosound_model.json` (334 KB)
- **Notes**: Used ARFF loader approach to avoid JSON serialization bottleneck

**Next Steps**:
- Create test data files (JSON or NPY format)
- Test on FPGA (expected time: ~12 minutes for full test set)

---

### FruitFlies ⏳ IN PROGRESS
- **Status**: Training in progress (started with auto-download)
- **Training Log**: `/tmp/fruitflies_training_v2.log`
- **Expected Dataset Size**:
  - Train: 17,259 samples
  - Test: 17,259 samples
  - Time series length: 5,000
  - Classes: (to be determined)
- **Model File**: `models/hydra_fruitflies_model.json` (pending)

**Check Progress**:
```bash
tail -f /tmp/fruitflies_training_v2.log
```

---

## Key Findings

### HYDRA Accuracy Analysis

**Issue Identified**: The custom HYDRA implementation (`scripts/custom_hydra.py`) uses **random dictionary initialization** rather than learning dictionaries from data. This results in suboptimal accuracy.

**Comparison**:
- Custom HYDRA (random kernels): ~69-70% accuracy
- Official HYDRA (learned kernels): Expected to be higher
- Official benchmark (InsectWingbeatSound): 65.99% from HYDRA paper

**FPGA Validation**: The FPGA implementation is working **perfectly** - it matches Python accuracy exactly (69.44% vs 69.41% = 0.03% difference for InsectSound).

### Performance Summary

| Dataset | Train Samples | Test Samples | Length | Classes | Test Acc | FPGA Status |
|---------|--------------|--------------|---------|---------|----------|-------------|
| InsectSound | 25,000 | 25,000 | 600 | 10 | 69.41% | ✅ Tested |
| MosquitoSound | 139,780 | 139,780 | 3,750 | 6 | 70.05% | ⏳ Pending |
| FruitFlies | 17,259 | 17,259 | 5,000 | TBD | TBD | ⏳ Training |

---

## Files Created

### Model Files (JSON format)
- `models/hydra_insectsound_model.json` - 334 KB
- `models/hydra_mosquitosound_model.json` - 334 KB
- `models/hydra_fruitflies_model.json` - pending

### Test Data Files
- `models/hydra_insectsound_test.json` - Full test set
- `models/hydra_insectsound_test_1000.json` - Quick validation set

### Code Files
- `host/src/hydra_loader_v2.cpp` - Real JSON loader (replaces stub)
- `host/include/hydra_loader_v2.h` - Loader header
- `host/src/npy_loader.cpp` - NPY binary format loader
- `host/include/npy_loader.h` - NPY loader header
- `scripts/train_and_save_models.py` - Training script (aeon loader)
- `scripts/create_sample_test.py` - Create test subsets

### Documentation
- `docs/HYDRA_FPGA_FIX_SUMMARY.md` - Complete fix documentation
- `Makefile` - Updated with simple run commands

---

## Next Steps

1. **Wait for FruitFlies** training to complete
2. **Create test data files** for MosquitoSound and FruitFlies:
   - Option A: JSON format (simple but large)
   - Option B: NPY format (fast but requires NPY loader integration)
3. **FPGA Testing** (optional - can wait for build as you mentioned):
   - MosquitoSound: ~12 minutes for full test
   - FruitFlies: ~3 minutes for full test
4. **Consider retraining** with official HYDRA implementation for better accuracy

---

## Commands Reference

### Training New Datasets
```bash
cd /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized
python3 scripts/train_and_save_models.py --dataset <DatasetName> --num-kernels 512
```

### FPGA Testing
```bash
cd /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized
DEVICE=xilinx_u280_gen3x16_xdma_1_202211_1 make run        # Full test
DEVICE=xilinx_u280_gen3x16_xdma_1_202211_1 make run-quick # Quick test
```

### Build Host Executable
```bash
DEVICE=xilinx_u280_gen3x16_xdma_1_202211_1 make host
```

---

## Notes

- All models use 512 kernels, 8 groups, 1024 features (512 × 2 pooling)
- Random seed: 42 (for reproducibility)
- Ridge classifier with CV (alphas: 10^-3 to 10^3)
- StandardScaler for feature normalization
