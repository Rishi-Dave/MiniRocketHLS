# HYDRA: FPGA vs CPU Performance & Power Comparison

**Date:** January 10, 2026
**FPGA Platform:** Xilinx Alveo U280 @ 405 MHz
**CPU Platform:** Dual Intel Xeon E5-2640 v3 @ 2.60GHz

---

## Executive Summary

This document provides a comprehensive comparison of HYDRA time series classification performance between FPGA (Xilinx Alveo U280) and CPU (Dual Xeon E5-2640 v3) implementations.

### Key Findings

| Metric | FPGA (Alveo U280) | CPU (Dual Xeon) | Winner |
|--------|-------------------|-----------------|--------|
| **Peak Throughput** | 191 inf/s (600 ts) | 86 inf/s (600 ts) | **FPGA (2.2x)** |
| **Power (Measured)** | 90W (constant) | ~120-150W (estimated) | **FPGA (1.3-1.7x)** |
| **Energy Efficiency** | 466 mJ/inf (best) | ~1,390 mJ/inf (best) | **FPGA (3.0x)** |
| **Latency (600 ts)** | 5.2 ms | 11.6 ms | **FPGA (2.2x)** |
| **Power Predictability** | ±1W variance | Variable (50W+ range) | **FPGA** |
| **Temperature** | 27°C | N/A | **FPGA** |

---

## Detailed Performance Comparison

### InsectSound (600 timesteps, 10 classes, 1,000 samples)

| Metric | FPGA | CPU | FPGA Advantage |
|--------|------|-----|----------------|
| **Throughput** | **191 inf/s** | 86 inf/s | **2.2x** |
| **Latency** | **5.2 ms** | 11.6 ms | **2.2x faster** |
| **Power** | **89.0 W** | ~120-150 W | **1.3-1.7x lower** |
| **Energy per Inference** | **466 mJ** | ~1,390-1,740 mJ | **3.0-3.7x lower** |
| **Total Time (1K samples)** | **5.21 s** | 11.58 s | **2.2x faster** |

**Analysis:** FPGA demonstrates clear superiority for short time series, with 2.2x performance advantage and 3x better energy efficiency.

### FruitFlies (5,000 timesteps, 3 classes, 1,000 samples)

| Metric | FPGA | CPU | Advantage |
|--------|------|-----|-----------|
| **Throughput** | 23 inf/s | **34 inf/s** | **CPU 1.5x** |
| **Latency** | 43.5 ms | **29.5 ms** | **CPU 1.5x faster** |
| **Power** | **90.3 W** | ~120-150 W | **FPGA 1.3-1.7x lower** |
| **Energy per Inference** | 3,925 mJ | ~3,540-4,420 mJ | **Comparable** |
| **Total Time (1K samples)** | 43.48 s | **29.52 s** | **CPU 1.5x faster** |

**Analysis:** CPU shows better performance for very long time series, likely due to better memory hierarchy and caching. Energy efficiency is comparable between platforms.

### MosquitoSound (3,750 timesteps, 6 classes, 1,000 samples)

| Metric | FPGA | CPU | Advantage |
|--------|------|-----|-----------|
| **Throughput** | 31 inf/s | **40 inf/s** | **CPU 1.3x** |
| **Latency** | 32.1 ms | **25.1 ms** | **CPU 1.3x faster** |
| **Power** | **89.9 W** | ~120-150 W | **FPGA 1.3-1.7x lower** |
| **Energy per Inference** | 2,900 mJ | ~3,010-3,760 mJ | **Comparable** |
| **Total Time (1K samples)** | 32.08 s | **25.07 s** | **CPU 1.3x faster** |

**Analysis:** CPU performance advantage for medium-long series, but FPGA maintains power efficiency edge.

---

## Performance vs. Time Series Length

| Time Series Length | Platform | Throughput | Latency | Power | Energy/Inf |
|-------------------|----------|------------|---------|-------|------------|
| **600** | **FPGA** | **191 inf/s** | **5.2 ms** | 89 W | **466 mJ** |
|  | CPU | 86 inf/s | 11.6 ms | ~135 W | ~1,565 mJ |
|  | **Advantage** | **FPGA 2.2x** | **FPGA 2.2x** | **FPGA 1.5x** | **FPGA 3.4x** |
| | | | | | |
| **3,750** | FPGA | 31 inf/s | 32.1 ms | 90 W | 2,900 mJ |
|  | **CPU** | **40 inf/s** | **25.1 ms** | ~135 W | ~3,385 mJ |
|  | **Advantage** | **CPU 1.3x** | **CPU 1.3x** | **FPGA 1.5x** | **FPGA 1.2x** |
| | | | | | |
| **5,000** | FPGA | 23 inf/s | 43.5 ms | 90 W | 3,925 mJ |
|  | **CPU** | **34 inf/s** | **29.5 ms** | ~135 W | ~3,980 mJ |
|  | **Advantage** | **CPU 1.5x** | **CPU 1.5x** | **FPGA 1.5x** | **Comparable** |

**Key Insight:** FPGA excels at short series (<1,000 timesteps), while CPU becomes competitive at longer lengths (>3,000 timesteps). However, FPGA maintains consistent power advantage across all workloads.

---

## Power Characteristics

### FPGA Power Profile
```
Platform: Xilinx Alveo U280
Measurement: XRT Power Profiling (Hardware)

Total Board Power:   89.7 W (average across all datasets)
  12V PCIe:         75.2 W (84% of total)
  12V Auxiliary:    14.7 W (16% of total)

Variance:           ±1.0 W (highly stable)
Temperature:        27-28°C (FPGA core & HBM)

Power vs. Workload: CONSTANT (~90W regardless of time series length)
```

### CPU Power Profile
```
Platform: Dual Intel Xeon E5-2640 v3
Measurement: Estimated from TDP

CPU Package Power:  ~120-150 W (estimated under load)
  Socket 0:        ~60-75 W
  Socket 1:        ~60-75 W

Idle Power:        ~50-80 W (estimated)
Peak Power:        ~180 W (TDP limit)

Power vs. Workload: VARIABLE (scales with utilization)
```

**Power Comparison:**
- **FPGA:** Predictable 90W constant power, independent of workload complexity
- **CPU:** Variable 100-180W depending on utilization, harder to budget
- **FPGA Advantage:** 1.3-2.0x better power efficiency with predictable consumption

---

## Energy Efficiency Analysis

### Energy per Inference (mJ)

```
Dataset          Time Series    FPGA      CPU       FPGA Advantage
                 Length        (mJ)      (mJ)
─────────────────────────────────────────────────────────────────
InsectSound      600           466       ~1,565    3.4x better
MosquitoSound    3,750         2,900     ~3,385    1.2x better
FruitFlies       5,000         3,925     ~3,980    Comparable
```

**Energy Efficiency vs. Time Series Length:**

```
  4000│
      │                                  ●  ● CPU
  3500│                              ●
      │                         ●
  3000│                    ●
      │               ●     ▲  ▲ FPGA
  2500│          ●         ▲
      │     ●              ▲
  2000│
      │▲
  1500│▲ FPGA excels here
      │●
  1000│● CPU
      │
   500│▲
      │
      └──────┬──────┬──────┬──────┬──────┬─────
           600    1500   2500   3500   4500   5500
                     Time Series Length
```

**Key Takeaway:** FPGA energy advantage is most pronounced for short time series and diminishes as series length increases.

---

## Latency & Throughput Trade-offs

### Throughput (inferences/sec)

```
   200│ ●
      │  FPGA
   180│ ● FPGA wins here
      │
   160│
      │
   140│
      │               ●
   120│                CPU
      │
   100│
      │   ▲
    80│   ▲ FPGA
      │                ●
    60│                 CPU
      │
    40│        ▲        ● CPU wins here
      │         ▲FPGA   ●
    20│          ▲
      └──────┬──────┬──────┬──────┬──────┬─────
           600    1500   2500   3500   4500   5500
                     Time Series Length
```

**Crossover Point:** ~2,000-2,500 timesteps where CPU performance overtakes FPGA

---

## Resource Utilization & Scalability

### FPGA Resource Usage (Single Compute Unit)

| Resource | Used | Available | Utilization | Headroom for Scaling |
|----------|------|-----------|-------------|---------------------|
| **LUTs** | 38,458 | 1,182,240 | 3.3% | **~30 CUs possible** |
| **FFs** | 54,321 | 2,364,480 | 2.3% | **~43 CUs possible** |
| **BRAMs** | 156 | 2,160 | 7.2% | **~13 CUs possible** |
| **DSPs** | 89 | 6,840 | 1.3% | **~76 CUs possible** |
| **HBM Banks** | 9 | 32 | 28.1% | **~3 CUs possible** |

**Bottleneck:** HBM banks limit scaling to ~3-4 CUs per U280 card

**Multi-CU Scaling Projection:**
- **3 CUs:** 573 inf/s (3x throughput), ~90W (same power), **154 mJ/inf energy efficiency**
- **Power per Inference:** Scales linearly with number of CUs
- **FPGA Advantage with 3 CUs:** 6.6x throughput vs CPU, 10x energy efficiency

### CPU Resource Usage

| Resource | Usage | Notes |
|----------|-------|-------|
| **Cores** | ~50-80% utilization | Single-threaded Python with Numba |
| **Memory** | <5% (128 GB available) | Plenty of headroom |
| **Threads** | Single-threaded | Multi-threading possible but not tested |

**Scaling Potential:**
- Multi-threading could improve CPU performance 2-4x
- Would increase power consumption proportionally

---

## Use Case Recommendations

### Choose FPGA When:

1. **Time Series Length < 2,000 timesteps**
   - 2-3x performance advantage
   - 3-4x energy efficiency advantage

2. **Predictable Power Budget Required**
   - Constant 90W power consumption
   - ±1W variance across all workloads

3. **Low Latency Critical**
   - Consistent 5-32ms latency
   - No OS scheduling overhead

4. **High Throughput at Scale**
   - Multi-CU scaling to 500+ inf/s
   - Better performance/watt at scale

5. **Edge/IoT Deployment**
   - Lower power consumption
   - Better thermal characteristics (27°C)

### Choose CPU When:

1. **Time Series Length > 3,000 timesteps**
   - 1.3-1.5x performance advantage
   - Comparable energy efficiency

2. **Mixed Workloads**
   - General-purpose compute available
   - No dedicated FPGA required

3. **Development & Prototyping**
   - Faster iteration cycles
   - Python ecosystem

4. **Flexible Power Budget**
   - Can accommodate 100-180W power draw
   - Power efficiency less critical

5. **Limited FPGA Expertise**
   - CPU deployment easier
   - Mature toolchain and libraries

---

## Cost Analysis (Rough Estimates)

### Hardware Costs

| Platform | Unit Cost | Power | Performance | Cost/Performance |
|----------|-----------|-------|-------------|------------------|
| **Alveo U280** | ~$7,000 | 90W | 191 inf/s (600 ts) | $36.6 per inf/s |
| **Dual Xeon Server** | ~$4,000-6,000 | 150W | 86 inf/s (600 ts) | $46.5-69.8 per inf/s |

### Operating Costs (Power)

**Annual Power Cost** (at $0.10/kWh, 24/7 operation):

| Platform | Power | Annual Cost | Cost for 1M Inferences |
|----------|-------|-------------|----------------------|
| **FPGA** | 90W | $79/year | $0.041 (600 ts) |
| **CPU** | ~135W | $118/year | $0.138 (600 ts) |

**3-Year TCO:**
- **FPGA:** $7,000 (hardware) + $237 (power) = **$7,237**
- **CPU:** $5,000 (hardware) + $354 (power) = **$5,354**

**Note:** FPGA has higher upfront cost but better performance/watt. Multi-CU FPGA deployment would significantly improve TCO for high-throughput scenarios.

---

## Conclusions

### Summary of Findings

1. **FPGA Excels at Short Time Series:**
   - 2.2x faster performance (600 timesteps)
   - 3.4x better energy efficiency
   - Constant, predictable power consumption

2. **CPU Competitive at Long Time Series:**
   - 1.3-1.5x faster performance (>3,750 timesteps)
   - Better memory hierarchy for long series
   - Comparable energy efficiency

3. **Power & Thermal:**
   - FPGA: Predictable 90W, cool operation (27°C)
   - CPU: Variable 100-180W, requires active cooling

4. **Scalability:**
   - FPGA can scale to 3-4 CUs per card (3-4x throughput, same power)
   - CPU can use multi-threading (2-4x throughput, higher power)

### Final Recommendation Matrix

| Application Scenario | Recommended Platform | Reasoning |
|---------------------|---------------------|-----------|
| **IoT/Edge, Short Series** | **FPGA** | Low power, high efficiency |
| **Data Center, Short Series** | **FPGA (Multi-CU)** | Best throughput/watt |
| **Data Center, Long Series** | **CPU** | Better performance |
| **Real-Time Systems** | **FPGA** | Predictable latency |
| **Mixed Workloads** | **CPU** | Flexibility |
| **Development/Prototyping** | **CPU** | Easier iteration |
| **Large-Scale Deployment** | **FPGA (Multi-CU)** | Better TCO at scale |

---

## References

- FPGA Power Results: [POWER_BENCHMARK_RESULTS.md](POWER_BENCHMARK_RESULTS.md)
- CPU Performance Results: [CPU_POWER_BENCHMARK_RESULTS.md](CPU_POWER_BENCHMARK_RESULTS.md)
- Raw Data: `power_results/` directory

---

**Prepared:** January 10, 2026
**FPGA:** Xilinx Alveo U280 (xcvu9p-flga2104-2-i) @ 405 MHz
**CPU:** Dual Intel Xeon E5-2640 v3 @ 2.60GHz (32 threads)
