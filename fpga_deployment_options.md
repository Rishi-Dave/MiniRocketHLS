# FPGA MiniRocket Deployment Options

## Option 1: Embedded Constants (Recommended for Your Model)
**✅ Best for models <50KB**

```cpp
// All parameters compiled into bitstream
#include "minirocket_model_constants.h"
extern "C" void minirocket_fpga_inference(data_t* input, data_t* output, int_t length);
```

**Pros:**
- Fastest access (BRAM/LUT)
- No external memory needed
- Deterministic performance
- Your model: only 6KB

**Cons:**
- Static model (resynthesize to update)
- Limited by FPGA memory

## Option 2: DDR/HBM External Memory
**For larger models (>1MB)**

```cpp
// Load model from DDR at runtime
extern "C" void minirocket_load_model(data_t* ddr_model_base);
extern "C" void minirocket_inference(data_t* input, data_t* output, int_t length);
```

**Implementation:**
- Host CPU loads JSON → DDR memory
- FPGA reads model parameters via AXI
- Slower but supports larger models

## Option 3: Model Streaming
**For very large models**

```cpp
// Stream model parameters as needed
extern "C" void minirocket_inference_stream(
    data_t* input, 
    data_t* output,
    data_t* model_stream,  // Stream coefficients
    int_t length
);
```

## Option 4: Flash/NVME Storage
**For model updates without resynthesize**

```cpp
// Load from on-board flash
extern "C" void minirocket_load_from_flash(uint32_t model_address);
extern "C" void minirocket_inference(data_t* input, data_t* output, int_t length);
```

## Recommendation for Your Project

**Use Option 1 (Embedded Constants)** because:
- ✅ Your model is only 6KB (very small)
- ✅ Fits comfortably in BRAM
- ✅ Maximum performance
- ✅ Simplest deployment

**Generated files:**
- `minirocket_model_constants.h` - All model parameters
- `minirocket_fpga_embedded.cpp` - FPGA-ready inference

**Usage:**
```bash
# Generate constants from trained model
python generate_model_constants.py minirocket_model.json minirocket_model_constants.h

# Synthesize with embedded model
vitis_hls -f build_fpga.tcl
```

**Host application just needs to:**
1. Send time series data to FPGA
2. Read back predictions
3. No model loading required!