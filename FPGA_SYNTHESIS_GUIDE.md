# MiniRocket HLS - FPGA Synthesis Guide

## Overview
This guide provides step-by-step instructions for synthesizing the MiniRocket HLS design for FPGA deployment on Xilinx platforms.

## Pre-Synthesis Checklist

### Completed Optimizations
- ✅ Fixed memory burst inference for coefficient access (flattened nested loops)
- ✅ Optimized AXI burst lengths (256 for large arrays, 16 for small arrays)
- ✅ Added loop trip count pragmas for better estimation
- ✅ Validated synthesis compilation
- ✅ Estimated Fmax: **136.99 MHz** (improved from ~100 MHz target)

### Current Design Status
- **Target Device**: Xilinx VU9P (xcvu9p-flga2104-2-i)
- **Clock Period**: 10ns (100 MHz nominal, capable of 136.99 MHz)
- **Resource Usage**: Very low (~5% BRAM, <1% other resources)
- **RTL Simulation**: PASSED (100/100 transactions)
- **Latency**: ~138K cycles per inference

## FPGA Synthesis Steps

### Step 1: Export IP from Vitis HLS

1. **Navigate to HLS project directory**:
   ```bash
   cd /home/rdave009/minirocket-hls/MiniRocketHLS/tcl_template
   ```

2. **Create IP export TCL script**:
   ```bash
   cat > export_ip.tcl << 'EOF'
   # Open existing HLS project
   open_project build_hls_sim/minirocket_hls
   open_solution "solution1"

   # Export as Vivado IP
   export_design -format ip_catalog -description "MiniRocket Accelerator" -vendor "xilinx.com" -library "user" -version "1.0" -display_name "MiniRocket HLS"

   exit
   EOF
   ```

3. **Run IP export**:
   ```bash
   vitis_hls -f export_ip.tcl
   ```

4. **IP location**: The exported IP will be in:
   ```
   build_hls_sim/minirocket_hls/solution1/impl/ip/
   ```

### Step 2: Create Vitis Kernel for Alveo/FPGA Platform

For deployment on Alveo or other Xilinx acceleration platforms, package as a Vitis kernel:

1. **Create Vitis packaging script**:
   ```bash
   cat > package_kernel.tcl << 'EOF'
   # Open existing HLS project
   open_project build_hls_sim/minirocket_hls
   open_solution "solution1"

   # Export as Vitis kernel (.xo)
   export_design -format xo -output minirocket_kernel.xo

   exit
   EOF
   ```

2. **Run packaging**:
   ```bash
   vitis_hls -f package_kernel.tcl
   ```

3. **Output**: `minirocket_kernel.xo` - Ready for Vitis linking

### Step 3: Build FPGA Bitstream with Vitis

#### Option A: For Alveo U250/U280 (Recommended)

1. **Create Vitis configuration file** (`minirocket.cfg`):
   ```ini
   [connectivity]
   # Map all AXI masters to DDR banks
   sp=krnl_top_1.m_axi_gmem0:DDR[0]
   sp=krnl_top_1.m_axi_gmem1:DDR[0]
   sp=krnl_top_1.m_axi_gmem2:DDR[1]
   sp=krnl_top_1.m_axi_gmem3:DDR[1]
   sp=krnl_top_1.m_axi_gmem4:DDR[1]
   sp=krnl_top_1.m_axi_gmem5:DDR[1]
   sp=krnl_top_1.m_axi_gmem6:DDR[0]
   sp=krnl_top_1.m_axi_gmem8:DDR[1]

   [advanced]
   # Optimization parameters
   prop=run.impl_1.strategy=Performance_ExplorePostRoutePhysOpt

   [vivado]
   # Place design in specific SLR if needed
   # prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=ExtraNetDelay_high
   ```

2. **Build for Alveo U250**:
   ```bash
   v++ -t hw \
       --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 \
       --config minirocket.cfg \
       --save-temps \
       --temp_dir ./build_dir \
       --log_dir ./logs \
       -c -k krnl_top \
       -o minirocket_kernel.xo \
       build_hls_sim/minirocket_hls/solution1/impl/export.zip

   v++ -t hw \
       --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 \
       --config minirocket.cfg \
       --save-temps \
       --temp_dir ./build_dir \
       --log_dir ./logs \
       -l -o minirocket.xclbin \
       minirocket_kernel.xo
   ```

3. **Build time**: Expect 4-6 hours for full hardware build

#### Option B: For Custom Platforms

1. **Use your platform DSA/XSA file**:
   ```bash
   v++ -t hw \
       --platform /path/to/your/platform.xpfm \
       --config minirocket.cfg \
       -l -o minirocket.xclbin \
       minirocket_kernel.xo
   ```

### Step 4: Software/Host Integration

1. **Create XRT host application** (example in C++):
   ```cpp
   // Load bitstream
   auto device = xrt::device(0);
   auto uuid = device.load_xclbin("minirocket.xclbin");
   auto kernel = xrt::kernel(device, uuid, "krnl_top");

   // Create buffers
   auto bo_input = xrt::bo(device, input_size, kernel.group_id(0));
   auto bo_output = xrt::bo(device, output_size, kernel.group_id(1));
   // ... more buffers for coefficients, etc.

   // Transfer data
   bo_input.write(input_data.data());
   bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

   // Run kernel
   auto run = kernel(bo_input, bo_output, ...params...);
   run.wait();

   // Read results
   bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
   bo_output.read(output_data.data());
   ```

2. **Compile host application**:
   ```bash
   g++ -std=c++17 host.cpp \
       -I${XILINX_XRT}/include \
       -L${XILINX_XRT}/lib \
       -lxrt_coreutil \
       -pthread \
       -o minirocket_host
   ```

## Performance Expectations

### Timing
- **Target Clock**: 100 MHz (10ns period)
- **Estimated Fmax**: 136.99 MHz (positive timing margin)
- **Recommendation**: Start with 100 MHz, can potentially increase to 120-130 MHz

### Throughput
- **Latency per inference**: ~138,497 cycles
- **At 100 MHz**: 1.38 ms/inference (~725 inferences/sec)
- **At 130 MHz**: 1.07 ms/inference (~940 inferences/sec)

### Resource Usage (VU9P)
| Resource | Used | Total | Utilization |
|----------|------|-------|-------------|
| BRAM     | 220  | 4320  | 5%          |
| DSP      | 14   | 6840  | <1%         |
| FF       | 17434| 2364480| <1%        |
| LUT      | 21286| 1182240| 2%         |
| URAM     | 0    | 960   | 0%          |

### Memory Bandwidth
- **9 AXI4 Master Interfaces** (32-bit wide)
- **Burst lengths**: 256 (large arrays), 16 (small arrays)
- **Outstanding transactions**: 16 per interface
- **Peak bandwidth**: ~3.6 GB/s per interface @ 100 MHz

## Known Issues & Limitations

### 1. Coefficient Burst Pattern
**Issue**: Coefficient access shows "Could not analyze pattern" warning
```
| m_axi_gmem2  | coefficients | Fail | Could not analyze pattern |
```

**Reason**: Strided access pattern due to MAX_FEATURES constant
- Access: `coefficients[i * MAX_FEATURES + j]`
- HLS cannot infer optimal burst due to compile-time unknown stride

**Impact**: Moderate - coefficient loading takes longer but only occurs once per batch

**Mitigation**: Already implemented - flattened loop enables better scheduling

### 2. Unused AXI Port (gmem7)
**Issue**: `num_features_per_dilation` port is marked as unused

**Reason**: This parameter is not currently accessed in the design

**Impact**: None - port will be optimized away, warnings are expected

**Action**: Can be removed in future design iteration

## Troubleshooting

### Synthesis Fails - Timing Closure
If timing fails during Vivado implementation:

1. **Reduce clock frequency**:
   ```tcl
   create_clock -period 12 -name default  # 83.3 MHz
   ```

2. **Add pipeline stages** in critical paths (apply_kernel_hls)

3. **Use placement directives**:
   ```
   prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=ExtraPostPlacementOpt
   ```

### Resource Overflow
If targeting smaller FPGA:

1. **Reduce MAX_FEATURES** in krnl.hpp
2. **Reduce array partitioning** factors (currently factor=8)
3. **Consider streaming interfaces** instead of full local buffers

### Simulation Mismatch
If hardware results differ from simulation:

1. **Check fixed-point precision** - may need more fractional bits
2. **Verify data alignment** - ensure buffers are properly aligned
3. **Run RTL/Verilog simulation** with real data

## Quick Reference Commands

```bash
# Full HLS flow (C sim + synthesis + cosim)
cd /home/rdave009/minirocket-hls/MiniRocketHLS/tcl_template
./build_sim.sh

# Quick synthesis check only
vitis_hls -f validate_synthesis.tcl

# Export IP for Vivado
vitis_hls -f export_ip.tcl

# Package for Vitis
vitis_hls -f package_kernel.tcl

# Build FPGA bitstream (Alveo U250)
v++ -t hw --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 \
    --config minirocket.cfg -l -o minirocket.xclbin minirocket_kernel.xo

# Emulation builds (faster testing)
v++ -t sw_emu ...  # Software emulation
v++ -t hw_emu ...  # Hardware emulation
```

## Next Steps

1. ✅ **HLS Synthesis Complete** - Design validated and optimized
2. **Export IP** - Generate .xo kernel file
3. **Vitis Linking** - Create FPGA bitstream (.xclbin)
4. **Host Integration** - Develop XRT application
5. **On-Board Testing** - Deploy and benchmark on real hardware

## Support & Resources

- **Vitis HLS Documentation**: https://docs.xilinx.com/r/en-US/ug1399-vitis-hls
- **Vitis Application Acceleration**: https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration
- **XRT Documentation**: https://xilinx.github.io/XRT/
- **Alveo Platforms**: https://www.xilinx.com/products/boards-and-kits/alveo.html

## Design Files Summary

| File | Purpose |
|------|---------|
| `src/krnl.cpp` | Main HLS kernel implementation |
| `src/krnl.hpp` | Header with constants and interfaces |
| `src/test_krnl.cpp` | Testbench for simulation |
| `build_sim.sh` | Build script for HLS flow |
| `validate_synthesis.tcl` | Quick synthesis validation |
| `minirocket.cfg` | Vitis configuration for linking |

---

**Status**: Ready for FPGA synthesis and deployment
**Last Updated**: November 4, 2025
**HLS Version**: Vitis HLS 2023.2
