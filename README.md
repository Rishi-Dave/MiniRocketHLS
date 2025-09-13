# MiniRocket HLS Implementation

A complete implementation of MiniRocket time series classification algorithm optimized for FPGA acceleration using Xilinx HLS.

## Overview

This project implements the MiniRocket algorithm in three stages:
1. **Python Training**: Train the model and export parameters
2. **C++ Implementation**: Verify correctness against Python
3. **HLS Acceleration**: FPGA-optimized implementation

## Project Structure

```
.
├── train_minirocket.py          # Python training script
├── minirocket_inference.cpp     # C++ verification implementation
├── minirocket_inference_hls.h   # HLS header definitions
├── minirocket_inference_hls.cpp # HLS implementation
├── test_hls.cpp                 # HLS testbench
├── tcl_template/                # Xilinx build infrastructure
│   ├── src/
│   │   ├── krnl.hpp            # HLS kernel header
│   │   └── krnl.cpp            # HLS kernel implementation
│   ├── minirocket_host.cpp     # Host application
│   ├── Makefile                # Build system
│   └── config.cfg              # Memory configuration
├── minirocket_model.json       # Trained model parameters
└── minirocket_model_test_data.json # Test data for verification
```

## Usage

### 1. Train the Model

```bash
# Activate Python environment
source venv/bin/activate

# Train model and export parameters
python train_minirocket.py --samples 1000 --length 128 --classes 4
```

This generates:
- `minirocket_model.json`: Model parameters for C++/HLS
- `minirocket_model_test_data.json`: Test data for verification

### 2. Verify C++ Implementation

```bash
# Build and test C++ implementation
make clean && make
make test
```

Expected output: "SUCCESS: C++ implementation matches Python results!"

### 3. Test HLS Implementation

```bash
# Build and test HLS simulation
make -f Makefile.hls clean && make -f Makefile.hls
make -f Makefile.hls test
```

### 4. Build for FPGA

```bash
cd tcl_template

# Software emulation
make clean
make run TARGET=sw_emu PLATFORM=<your_platform>

# Hardware emulation  
make run TARGET=hw_emu PLATFORM=<your_platform>

# Hardware (requires FPGA board)
make run TARGET=hw PLATFORM=<your_platform>
```

## Algorithm Details

### MiniRocket Features

MiniRocket extracts features using:
- **84 fixed kernels**: Combinations of 3 indices from positions 0-8
- **Multiple dilations**: Exponentially increasing (1, 2, 4, 8, ...)
- **PPV (Positive Proportion Values)**: Fraction of convolution outputs > bias
- **Linear classification**: Ridge regression on extracted features

### HLS Optimizations

The FPGA implementation includes:
- **Fixed-point arithmetic**: `ap_fixed<32,16>` for 32-bit precision
- **Pipeline optimization**: II=1 for critical loops
- **Memory partitioning**: Cyclic/block partitioning for parallel access
- **Loop unrolling**: For small inner loops
- **Streaming interfaces**: Efficient data movement

## Performance

- **Accuracy**: 94% on synthetic 4-class time series data
- **C++ vs Python**: 100% agreement on test predictions
- **HLS**: Functionally verified through simulation

## Requirements

### Software
- Python 3.7+ with scikit-learn, numpy
- C++ compiler with C++11 support
- Xilinx Vitis HLS (for FPGA synthesis)
- jsoncpp library (for C++ implementation)

### Hardware
- Xilinx Alveo U280 or compatible FPGA board
- Host system with PCIe slot

## Model Parameters

The trained model exports the following parameters:
- **Kernel indices**: 84 × 3 fixed combinations
- **Dilations**: Variable number (typically 5-8)
- **Biases**: One per feature (typically 420 features)
- **Scaler parameters**: Mean and scale for standardization
- **Classifier weights**: Ridge regression coefficients and intercepts

## Memory Layout

The FPGA implementation uses 9 separate memory banks:
- HBM[0]: Input time series
- HBM[1]: Output predictions  
- HBM[2]: Classifier coefficients
- HBM[3]: Classifier intercepts
- HBM[4-5]: Scaler mean/scale
- HBM[6-7]: Dilations and feature counts
- HBM[8]: Bias values

## Extending the Implementation

To modify for different datasets:
1. Adjust constants in header files (`MAX_TIME_SERIES_LENGTH`, etc.)
2. Retrain with your data using `train_minirocket.py`
3. Update memory depths in `config.cfg` if needed
4. Rebuild and test

## References

- Dempster, A., et al. "MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification." KDD 2021.
- Xilinx Vitis HLS User Guide (UG1399)