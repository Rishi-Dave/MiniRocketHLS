# MiniRocket HLS Guide for Newcomers

## What is High-Level Synthesis (HLS)?

High-Level Synthesis (HLS) is a technology that allows you to implement hardware designs using C/C++ instead of traditional hardware description languages (HDL) like Verilog or VHDL. HLS tools automatically convert your C++ code into optimized hardware that can run on FPGAs (Field-Programmable Gate Arrays).

### Key Benefits of HLS:
- **Faster Development**: Write hardware in familiar C/C++ syntax
- **High Performance**: Achieve parallel processing and custom optimizations
- **Portability**: Same code can target different FPGA platforms
- **Optimization**: Automatic hardware optimizations with pragma directives

## Understanding FPGAs vs CPUs

| Aspect | CPU | FPGA |
|--------|-----|------|
| **Execution** | Sequential (mostly) | Massively Parallel |
| **Memory** | Large cache, shared | Distributed, custom |
| **Optimization** | Fixed architecture | Custom hardware for each algorithm |
| **Latency** | Higher due to instruction overhead | Ultra-low with pipelining |
| **Power** | Higher per operation | Much lower for specific tasks |

## MiniRocket Algorithm Overview

MiniRocket is a time series classification algorithm that:

1. **Feature Extraction**: Applies 84 different convolution kernels to input time series
2. **Pooling**: Computes "Positive Proportion Values" (PPV) for each convolution
3. **Scaling**: Normalizes features using pre-computed mean/scale values
4. **Classification**: Uses linear classifier to predict final class

### Why MiniRocket is Perfect for FPGAs:
- **Parallel Convolutions**: All 84 kernels can run simultaneously
- **Fixed-Point Math**: Deterministic computation without floating-point overhead
- **Streaming Data**: Natural fit for time series processing
- **Low Latency**: Real-time classification in microseconds

## HLS Pragma Directives Explained

Pragmas are special comments that guide the HLS tool on how to optimize your hardware:

### Memory Interface Pragmas
```cpp
#pragma HLS INTERFACE m_axi port=time_series_input bundle=gmem0 depth=512
```
- `m_axi`: Creates AXI4 memory interface for external memory access
- `bundle=gmem0`: Groups related ports for efficient memory access
- `depth=512`: Tells HLS the maximum array size for optimization

### Control Interface Pragmas
```cpp
#pragma HLS INTERFACE s_axilite port=num_features bundle=control
```
- `s_axilite`: Creates lightweight control interface for parameters
- Used for scalar inputs and control signals

### Loop Optimization Pragmas
```cpp
#pragma HLS PIPELINE II=1
```
- `PIPELINE`: Enables pipelined execution (overlapping iterations)
- `II=1`: Initiation Interval of 1 (start new iteration every clock cycle)

```cpp
#pragma HLS UNROLL
```
- `UNROLL`: Completely unrolls loop (parallel execution of all iterations)
- Use for small, fixed-size loops

```cpp
#pragma HLS LOOP_TRIPCOUNT min=100 max=10000
```
- Provides loop iteration bounds for optimization estimation

### Array Optimization Pragmas
```cpp
#pragma HLS ARRAY_PARTITION variable=array type=cyclic factor=8
```
- `ARRAY_PARTITION`: Splits arrays into multiple memories
- `cyclic factor=8`: Every 8th element goes to same memory bank
- Enables parallel access to array elements

```cpp
#pragma HLS ARRAY_PARTITION variable=predictions type=complete
```
- `complete`: Each array element gets its own memory (full parallelism)

### Function Pragmas
```cpp
#pragma HLS INLINE off
```
- Controls whether function is inlined or kept as separate module
- `off`: Keep as separate module for better hierarchy

## Project Structure

```
tcl_template/                    # Main HLS project directory
├── src/                        # Source code directory
│   ├── krnl.hpp               # Main kernel header (interface definitions)
│   ├── krnl.cpp               # Main kernel implementation 
│   ├── test_krnl.cpp          # Testbench for simulation
│   ├── minirocket_hls_testbench_loader.h/.cpp  # Test data loading
│   ├── ap_fixed.h             # Fixed-point arithmetic types
│   ├── ap_int.h               # Integer types for HLS
│   └── hls_stream.h           # Streaming interfaces
├── build_sim.sh               # Automated build and simulation script
├── minirocket_model.json      # Pre-trained model parameters
├── minirocket_model_test_data.json  # Test data for validation
├── Makefile                   # Build system for hardware generation
└── config.cfg                 # Memory connectivity configuration
```

## Running Your First Simulation

### Prerequisites
1. **Xilinx Vitis HLS** installed and licensed
2. **Environment Setup**: Source the Vitis HLS tools
   ```bash
   source /tools/Xilinx/Vitis_HLS/2023.1/settings64.sh
   ```

### Step 1: Navigate to Project
```bash
cd tcl_template/
```

### Step 2: Run Simulation
```bash
./build_sim.sh
```

This script will:
1. Create a build directory
2. Copy all necessary files
3. Generate HLS project
4. Run C simulation (functional verification)
5. Run synthesis (generate hardware)
6. Run co-simulation (verify hardware matches C behavior)

### Step 3: Check Results
- **C Simulation**: `build_hls_sim/minirocket_hls/solution1/csim/report/`
- **Synthesis Report**: `build_hls_sim/minirocket_hls/solution1/syn/report/`
- **Co-simulation**: `build_hls_sim/minirocket_hls/solution1/sim/report/`

## Understanding HLS Data Types

### Fixed-Point Types
```cpp
typedef ap_fixed<32,16> data_t;  // 32 total bits: 16 integer, 16 fractional
```
- **Why Fixed-Point?** Deterministic, faster than floating-point, lower resource usage
- **Format**: `ap_fixed<width, integer_width>`
- **Range**: Our type represents values from -32768.0 to +32767.99998

### Integer Types  
```cpp
typedef ap_int<32> int_t;        // 32-bit signed integer
typedef ap_uint<8> idx_t;        // 8-bit unsigned integer
```
- **ap_int**: Arbitrary-width signed integers
- **ap_uint**: Arbitrary-width unsigned integers
- Use smaller widths to save hardware resources

## Key HLS Concepts

### 1. Pipelining
Traditional software executes loops sequentially:
```
Iteration 1: [Load][Compute][Store]
Iteration 2:                      [Load][Compute][Store]
```

HLS pipelining overlaps operations:
```
Iteration 1: [Load][Compute][Store]
Iteration 2:       [Load][Compute][Store]
Iteration 3:              [Load][Compute][Store]
```

### 2. Parallelism
HLS can execute multiple operations simultaneously:
- **Loop Unrolling**: Execute multiple loop iterations in parallel
- **Array Partitioning**: Access multiple array elements simultaneously
- **Function Pipelining**: Overlap function calls

### 3. Memory Architecture
FPGAs have distributed memory, not a single large memory like CPUs:
- **Block RAM (BRAM)**: Fast on-chip memory blocks
- **UltraRAM**: Larger on-chip memory (newer FPGAs)
- **External Memory**: DDR/HBM for large datasets

## Performance Optimization Tips

### 1. Memory Access Patterns
- **Good**: Sequential access with regular patterns
- **Bad**: Random access with irregular patterns
- **Solution**: Use array partitioning and proper bundling

### 2. Loop Structure
- **Good**: Fixed bounds, simple increment
- **Bad**: Variable bounds, complex control flow
- **Solution**: Restructure with fixed maximum bounds

### 3. Data Dependencies
- **Good**: Independent operations that can run in parallel
- **Bad**: Each operation depends on the previous result
- **Solution**: Redesign algorithm to reduce dependencies

## Debugging HLS Designs

### 1. C Simulation Fails
- Check array bounds and memory allocation
- Verify test data loading
- Use standard C++ debugging techniques

### 2. Synthesis Fails
- Check pragma syntax
- Verify data types are HLS-compatible
- Simplify complex C++ constructs

### 3. Co-simulation Fails
- Timing issues: Check if design meets clock constraints
- Memory issues: Verify interface pragmas match actual usage
- Precision issues: Fixed-point precision may differ from floating-point

## Next Steps

1. **Run the provided simulation** to get familiar with the tools
2. **Experiment with pragma settings** to see their effect on performance
3. **Modify the MiniRocket parameters** to understand the trade-offs
4. **Try implementing your own simple HLS design**

## Resources

- **Xilinx HLS User Guide**: Official documentation
- **HLS Optimization Methodology**: Best practices guide  
- **Vitis HLS Pragmas Reference**: Complete pragma documentation
- **MiniRocket Paper**: Original algorithm description

This guide provides the foundation for understanding and working with HLS. The MiniRocket implementation demonstrates real-world application of these concepts for machine learning acceleration.