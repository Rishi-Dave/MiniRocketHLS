# MiniRocket FPGA Accelerator

High-performance FPGA implementation of the MiniRocket algorithm for time series classification on Xilinx Alveo U280.

## Overview

This project implements MiniRocket (Mini RandOm Convolutional KErnel Transform), a state-of-the-art time series classification algorithm, on FPGA hardware for dramatically improved performance and energy efficiency.

### Key Results

- **3,468 inferences/second** with 4 compute units
- **178x speedup** over CPU Python implementation
- **250x energy efficiency** improvement
- **Exact accuracy match** with CPU reference (functional equivalence verified)

## Quick Start

```bash
# 1. Install dependencies
pip3 install aeon scikit-learn numpy

# 2. Train a model on UCR dataset
cd MiniRocketHLS/tcl_template
python3 train_minirocket_for_fpga.py CBF --features 420

# 3. Build host application
g++ -O3 -std=c++17 \
    -I/opt/xilinx/xrt/include \
    -o ucr_benchmark_4cu \
    ucr_benchmark_4cu.cpp \
    -lOpenCL -pthread

# 4. Run inference on FPGA (requires pre-built xclbin)
./ucr_benchmark_4cu \
    build_dir_4cu.hw.*/krnl.xclbin \
    cbf_fpga_model.json \
    cbf_fpga_test.json
```

## Documentation

- **[ALGORITHM.md](ALGORITHM.md)** - Detailed explanation of the MiniRocket algorithm and why it works for time series classification
- **[FPGA_IMPLEMENTATION.md](FPGA_IMPLEMENTATION.md)** - Complete guide to building and running the FPGA accelerator
- **[RESULTS.md](RESULTS.md)** - Comprehensive performance results and analysis

## Hardware Requirements

- **FPGA**: Xilinx Alveo U280 (or compatible with HBM)
- **Development Tools**: Vitis 2023.2, XRT 2023.2
- **System**: Ubuntu 20.04/22.04, 64GB+ RAM (for FPGA builds)

## Project Structure

```
minirocket-hls/
├── README.md                    # This file
├── ALGORITHM.md                 # MiniRocket algorithm explanation
├── FPGA_IMPLEMENTATION.md       # FPGA build and usage guide
├── RESULTS.md                   # Performance results
├── .gitignore                   # Git ignore rules
│
└── MiniRocketHLS/tcl_template/
    ├── src/                     # FPGA kernel source code
    │   ├── krnl.cpp            # Main FPGA kernel
    │   └── krnl.hpp            # Kernel header
    │
    ├── host.h                   # Host application utilities
    ├── ucr_benchmark_host.cpp   # Single-kernel host
    ├── ucr_benchmark_4cu.cpp    # 4-kernel parallel host
    ├── batch_benchmark_host.cpp # Batch inference host
    │
    ├── train_minirocket_for_fpga.py  # Training pipeline
    ├── prepare_ucr_data.py          # UCR dataset loader
    ├── benchmark_cpu.py              # CPU baseline
    │
    ├── config.cfg               # Vitis build configuration
    ├── Makefile                 # Build automation
    │
    └── build_sim.sh             # HLS simulation script
```

## Performance Summary

| Configuration | Throughput | Speedup vs CPU | Latency |
|---------------|-----------|----------------|---------|
| CPU (Python) | 19.5 inf/sec | 1x | 51.3 ms |
| FPGA (1 CU) | 1,248 inf/sec | 64x | 0.8 ms |
| **FPGA (4 CU)** | **3,468 inf/sec** | **178x** | **0.29 ms** |

### Accuracy Validation

| Dataset | Difficulty | CPU Accuracy | FPGA Accuracy |
|---------|-----------|--------------|---------------|
| CBF | Easy | 99.89% | ✓ 99.89% |
| GunPoint | Easy | 99.33% | ✓ 99.33% |
| ItalyPowerDemand | Medium | 95.63% | ✓ 95.63% |
| ECG200 | Hard | 88.00% | ✓ 88.00% |

**Finding**: FPGA achieves bit-accurate functional equivalence with CPU.

### Energy Efficiency

- **CPU**: ~100W, 0.2 inf/J
- **FPGA**: 24.4W, 142 inf/J
- **Advantage**: **250-700x more energy efficient**

## Applications

This accelerator is ideal for:

- **Medical Devices**: ECG/EEG classification, fall detection
- **Industrial IoT**: Predictive maintenance, anomaly detection
- **Edge Computing**: Wearable activity recognition
- **Data Centers**: High-throughput time series analytics

## Building the FPGA Bitstream

**Note**: FPGA builds take 2-5 hours. See [FPGA_IMPLEMENTATION.md](FPGA_IMPLEMENTATION.md) for detailed instructions.

```bash
# Source Xilinx tools
source /opt/xilinx/xrt/setup.sh
source /tools/Xilinx/Vitis/2023.2/settings64.sh

# Build kernel (single CU)
v++ -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 \
    --config config.cfg -c -k krnl_top -o krnl.xo src/krnl.cpp

v++ -t hw --platform xilinx_u280_gen3x16_xdma_1_202211_1 \
    --config config.cfg -l -o krnl.xclbin krnl.xo
```

For 4 compute units, modify `config.cfg` to set `nk=krnl_top:4` (see [FPGA_IMPLEMENTATION.md](FPGA_IMPLEMENTATION.md#3-configuring-compute-units)).

## Citation

**Original MiniRocket Paper**:
```bibtex
@inproceedings{dempster2021minirocket,
  title={MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification},
  author={Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={248--257},
  year={2021}
}
```

## License

MIT License

## Acknowledgments

- MiniRocket algorithm by Dempster et al.
- UCR Time Series Classification Archive
- Xilinx Vitis HLS and Runtime tools

---

**Status**: Research implementation
**Platform**: Xilinx Alveo U280
**Last Updated**: December 2025
