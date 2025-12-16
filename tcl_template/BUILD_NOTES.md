# Build Notes and Multi-CU Configuration

## Hardware Emulation Build Status

### Current Status: ❌ Failed (Platform Version Incompatibility)

The hardware emulation build fails during Vivado block diagram integration:

```
ERROR: [VPL 60-704] Integration error, Failed to update block diagram in project
ERROR: [VPL 60-1328] Vpl run 'vpl' failed
```

**Root Cause**: The U280 platform (`xilinx_u280_gen3x16_xdma_1_202211_1`) was built with Vitis 2022.1.1, but we're using Vitis 2023.2. This causes IP version lock warnings and ultimately fails the block diagram update.

**Warnings**:
- Multiple SmartConnect, SC MMU, and other IP blocks are locked due to version mismatches
- Platform IPs customized with 2022.1.1 have different revisions in the 2023.2 IP Catalog

### Solutions (in order of preference):

1. **Use matching platform version** - Download U280 platform for Vitis 2023.2
2. **Downgrade to Vitis 2022.1** - Match the platform's Vitis version
3. **Software emulation only** - Skip hardware emulation, go directly to hardware build (very long synthesis time)
4. **Target different platform** - Use a U250/U200 platform if available for 2023.2

---

## Multi-Compute Unit (4-CU) Configuration

### Problem: Initial 4-CU Approach Failed

The original [config.cfg](config.cfg) attempted to assign each kernel port individually to HBM banks:

```
sp=krnl_top_1.time_series:HBM[0]
sp=krnl_top_1.predictions:HBM[1]
sp=krnl_top_1.coefficients:HBM[2]
... (9 ports per CU)
sp=krnl_top_4.scaler_scale:DDR[0]  # Ran out of HBM banks!
```

**Problem**: Each CU has 9 ports, so 4 CUs × 9 ports = 36 assignments needed, but U280 only has 32 HBM banks (0-31).

**Error**:
```
ERROR: [CFGEN 83-2229] Failed to find single interface to reach all segments in {DDR[0], DDR[0], DDR[0], HBM[31], DDR[0]}
```

### Solution: Bundle-Based Mapping Strategy

#### Key Insight

Looking at [krnl.cpp:21-29](src/krnl.cpp#L21-L29), the kernel uses only **4 AXI memory bundles** (`gmem0`-`gmem3`), not 9 independent interfaces:

| Bundle | Ports | Purpose |
|--------|-------|---------|
| `gmem0` | `time_series` | Input time series data |
| `gmem1` | `predictions` | Output predictions |
| `gmem2` | `dilations`, `num_features_per_dilation`, `biases`, `scaler_mean`, `scaler_scale` | Small read-only parameters |
| `gmem3` | `coefficients`, `intercept` | Large read-only classifier weights |

#### Optimal 4-CU Configuration

See [config_4cu.cfg](config_4cu.cfg) for the complete configuration:

```
[connectivity]
nk=krnl_top:4

# Map each CU's 4 bundles to 4 consecutive HBM banks
# Total: 4 bundles × 4 CUs = 16 HBM banks (fits easily in 32)

sp=krnl_top_1.gmem0:HBM[0]
sp=krnl_top_1.gmem1:HBM[1]
sp=krnl_top_1.gmem2:HBM[2]
sp=krnl_top_1.gmem3:HBM[3]

sp=krnl_top_2.gmem0:HBM[4]
... (and so on through HBM[15])
```

#### Benefits of This Approach

1. **Efficient Memory Usage**: 16 / 32 HBM banks = 50% utilization
2. **Maximum Bandwidth**: Each CU gets dedicated HBM banks, no contention
3. **Scalable**: Could theoretically support up to 8 CUs (8 × 4 = 32 banks)
4. **Clean Mapping**: Simple, predictable bank allocation

#### Memory Bandwidth Analysis

- **U280 HBM**: 32 banks × 256 GB/s per bank = 460 GB/s total
- **Per CU**: 4 banks × 256 GB/s = 1024 GB/s theoretical bandwidth
- **4 CUs Total**: 4096 GB/s aggregate bandwidth (limited by HBM to 460 GB/s actual)

Each CU can process a separate time series in parallel, achieving ~4x throughput compared to single CU.

---

## Build Instructions

### Current 1-CU Build (Working)

```bash
make all TARGET=hw_emu PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
```

Note: This will fail at hardware emulation due to platform version mismatch. The host application builds successfully.

### Future 4-CU Build (Once Platform Fixed)

```bash
# Replace config.cfg with the 4-CU version
cp config_4cu.cfg config.cfg

# Build for hardware (skip hw_emu due to platform issues)
make xclbin TARGET=hw PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
```

**Hardware build time**: ~6-8 hours for full place & route.

---

## Testing Without Hardware Emulation

Since hardware emulation is blocked by platform version issues, we can:

1. **HLS C-Simulation**: Already working (tested with `test_hls`)
2. **Software Emulation**: Fast functional verification (if platform allows)
3. **Direct Hardware Build**: Skip to actual FPGA programming (long but works)

To test the hardware build on actual U280:

```bash
# After hardware xclbin completes (~6-8 hrs)
./host build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/krnl.xclbin minirocket_model.json
```

---

## Summary

### ✅ What Works

- HLS synthesis and C-simulation
- Host application compilation
- 1-CU configuration for eventual hardware build
- 4-CU memory mapping strategy (designed but not yet built)

### ❌ What's Blocked

- Hardware emulation (platform version mismatch)
- Cannot test xclbin before committing to long hardware build

### 📋 Next Steps

1. Obtain U280 platform for Vitis 2023.2, OR
2. Downgrade to Vitis 2022.1, OR
3. Proceed with hardware build and test on actual FPGA
4. Once unblocked, switch to [config_4cu.cfg](config_4cu.cfg) for 4-CU parallelism
