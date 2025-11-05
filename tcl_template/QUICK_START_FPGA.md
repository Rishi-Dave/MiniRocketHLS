# MiniRocket HLS - FPGA Quick Start

## Ready for FPGA Synthesis! ✅

Your design has been optimized and validated. Follow these steps to deploy to FPGA.

---

## Option 1: Export for Vivado IP Integration (Fastest)

```bash
cd /home/rdave009/minirocket-hls/MiniRocketHLS/tcl_template

# Create export script
cat > export_ip.tcl << 'EOF'
open_project build_hls_sim/minirocket_hls
open_solution "solution1"
export_design -format ip_catalog
exit
EOF

# Run export
vitis_hls -f export_ip.tcl

# IP location:
# build_hls_sim/minirocket_hls/solution1/impl/ip/
```

**Use Case**: Integrate into existing Vivado block design

---

## Option 2: Build Vitis Kernel for Alveo (Recommended)

```bash
cd /home/rdave009/minirocket-hls/MiniRocketHLS/tcl_template

# Package as Vitis kernel
cat > package_xo.tcl << 'EOF'
open_project build_hls_sim/minirocket_hls
open_solution "solution1"
export_design -format xo -output minirocket_kernel.xo
exit
EOF

vitis_hls -f package_xo.tcl

# Build bitstream for Alveo U250 (4-6 hours)
v++ -t hw \
    --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 \
    -l -o minirocket.xclbin \
    minirocket_kernel.xo

# For faster testing, use emulation:
v++ -t hw_emu --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 \
    -l -o minirocket_emu.xclbin minirocket_kernel.xo
```

**Use Case**: Alveo accelerator cards with XRT runtime

---

## Option 3: Quick Validation (No Hardware Needed)

```bash
cd /home/rdave009/minirocket-hls/MiniRocketHLS/tcl_template

# Run synthesis validation only (1-2 minutes)
vitis_hls -f validate_synthesis.tcl

# Check results
cat minirocket_hls_validate/solution1/syn/report/csynth.rpt | grep -A 5 "Estimated Fmax"
```

**Expected Output**:
```
INFO: [HLS 200-789] **** Estimated Fmax: 136.99 MHz
```

---

## Key Design Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Target Device** | xcvu9p-flga2104-2-i | Virtex UltraScale+ |
| **Clock Period** | 10ns (100 MHz) | Can run up to 137 MHz |
| **Latency** | ~138K cycles | 1.38ms @ 100MHz |
| **Throughput** | 725 inf/sec @ 100MHz | 989 inf/sec @ 137MHz |
| **BRAM** | 220 (5% of VU9P) | Very efficient |
| **DSP** | 14 (<1%) | Minimal compute |
| **FF** | 17,434 (<1%) | Low logic usage |
| **LUT** | 21,286 (2%) | Plenty of headroom |

---

## Memory Interface Summary

The kernel has **9 AXI4 Master ports** for memory access:

| Port | Purpose | Size | Direction | Burst |
|------|---------|------|-----------|-------|
| gmem0 | Time series input | 512 elem | Read | 256 |
| gmem1 | Prediction output | 4 elem | Write | 16 |
| gmem2 | Coefficients | 40K elem | Read | 256 |
| gmem3 | Intercept | 4 elem | Read | 16 |
| gmem4 | Scaler mean | 10K elem | Read | 256 |
| gmem5 | Scaler scale | 10K elem | Read | 256 |
| gmem6 | Dilations | 8 elem | Read | 16 |
| gmem7 | (unused) | - | - | - |
| gmem8 | Biases | 10K elem | Read | 256 |

**Total Memory Footprint per Inference**:
- Input: 2 KB (512 × 32-bit)
- Model: ~280 KB (coefficients, scalers, biases)
- Output: 16 bytes (4 × 32-bit)

---

## Control Interface (AXI-Lite)

```
Base Address: 0x00
Registers:
  0x00: CTRL (start/done/idle)
  0x10-0x14: time_series_input pointer
  0x1C-0x20: prediction_output pointer
  0x28-0x2C: coefficients pointer
  0x34-0x38: intercept pointer
  0x40-0x44: scaler_mean pointer
  0x4C-0x50: scaler_scale pointer
  0x58-0x5C: dilations pointer
  0x70-0x74: biases pointer
  0x7C: time_series_length
  0x84: num_features
  0x8C: num_classes
  0x94: num_dilations
```

---

## Optimizations Applied ✅

1. ✅ **Loop Flattening**: Improved burst inference for coefficient access
2. ✅ **Burst Length Optimization**: 256-word bursts for large arrays
3. ✅ **Trip Count Hints**: Better latency estimation
4. ✅ **Timing Optimization**: Achieved 137 MHz Fmax (37% above target)

---

## Troubleshooting

### Build Fails - Platform Not Found
```bash
# List available platforms
v++ --list-platforms

# Use correct platform name from list
```

### Timing Fails in Implementation
Reduce clock period in export script:
```tcl
# Change from 10ns to 12ns (83.3 MHz)
create_clock -period 12 -name default
```

### Out of Memory During Build
```bash
# Use smaller FPGA or reduce MAX_FEATURES in src/krnl.hpp
# Current: #define MAX_FEATURES 10000
# Try: #define MAX_FEATURES 5000
```

---

## Documentation

- **Full Synthesis Guide**: [`../FPGA_SYNTHESIS_GUIDE.md`](../FPGA_SYNTHESIS_GUIDE.md)
- **Optimization Details**: [`../OPTIMIZATION_SUMMARY.md`](../OPTIMIZATION_SUMMARY.md)
- **HLS Reports**: `build_hls_sim/minirocket_hls/solution1/syn/report/`
- **Simulation Results**: `build_hls_sim/minirocket_hls/solution1/sim/report/`

---

## Contact & Support

For issues or questions:
1. Check HLS synthesis reports in `build_hls_sim/minirocket_hls/solution1/syn/report/`
2. Review Vitis HLS User Guide: UG1399
3. Consult Vitis Application Acceleration Guide: UG1393

---

**Status**: ✅ Design ready for FPGA synthesis
**Last Validated**: November 4, 2025
**Tools Version**: Vitis HLS 2023.2
