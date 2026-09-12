# cpu/ — MiniRocket host baselines (CPU / GPU / streaming)

Branch lineage: **`feature/gpu_cpp`** → streaming work on **`feature/gpu-streaming-naive`**.

## What's in `cpu/`

| Target | Source | Description |
|--------|--------|-------------|
| `minirocket_cpu` | `minirocket_cpu_inference.cpp` | Single-thread CPU ST baseline |
| `minirocket_cpu_mt` | `minirocket_cpu_inference_mt.cpp` | OpenMP multithreaded CPU |
| `minirocket_gpu` | `minirocket_gpu_inference.cu` | Offline GPU batch/latency inference |
| `minirocket_gpu_streams` | `minirocket_gpu_inference_streams.cu` | CUDA streams multi-batch GPU |
| `minirocket_gpu_udp_server` | `minirocket_gpu_udp_server.cu` | **Naive** streaming UDP server (GPU) |
| `minirocket_udp_client` | `minirocket_udp_client.cpp` | UDP client for correctness tests |

> **`minirocket_stream/`** (repo root) is the FPGA AXI-Stream HLS design — **unrelated** to this host UDP streaming code.

## Naive streaming (phase a)

The UDP server does a **full MiniRocket recompute** on the entire sliding window of length `L = time_series_length` after every new sample. There is **no DP-Reuse** (no incremental convolution / PPV). DP-Reuse is future work; this phase is for correctness against offline GPU/CPU inference.

### Build

```bash
# RTX 2080 Ti on cerebro (default)
make gpu-stream
# or explicitly:
make CUDA_ARCH=sm_75 gpu-stream

# Individual targets
make stream-server
make stream-client
```

Compile hints (also in file headers):

```bash
nvcc -O3 -arch=sm_75 -std=c++17 -o minirocket_gpu_udp_server minirocket_gpu_udp_server.cu
g++ -O3 -std=c++17 -o minirocket_udp_client minirocket_udp_client.cpp
```

Override architecture for other GPUs: `make CUDA_ARCH=sm_80 ...` (A100), `sm_86` (RTX 30xx), etc.

### Two-process localhost example

Terminal 1 (server):

```bash
./minirocket_gpu_udp_server ../fpga-network/minirocket_fused/InsectSound_minirocket_model.json 127.0.0.1 9000
```

Terminal 2 (client):

```bash
./minirocket_udp_client ../fpga-network/minirocket_fused/InsectSound_test_data_compact.json 127.0.0.1 9000 0
```

Only the **final** prediction (after `L` samples) is comparable to offline whole-series inference — early windows are zero-padded.

### Protocol (IPv4 UDP)

| Direction | Payload |
|-----------|---------|
| Client → server | exactly **4 bytes**, little-endian **float32** sample |
| Server → client | exactly **4 bytes**, little-endian **int32** predicted class (`model.classes[argmax]`) |

Window starts as zeros; each packet: `memmove` left by 1, append sample, **NAIVE full recompute**, scaler + Ridge classify, reply to source address.

### Model / test JSON

Generate model JSON (gitignore may hide `*.json`):

```bash
python export_minirocket_model.py InsectSound
```

Typical paths under `../fpga-network/minirocket_fused/`:

- `InsectSound_minirocket_model.json`
- `InsectSound_test_data_compact.json`

Model fields used by the GPU code: `num_kernels`, `num_dilations`, `num_features`, `num_classes`, `time_series_length`, `kernel_indices`, `dilations`, `num_features_per_dilation`, `biases`, `scaler_mean`, `scaler_scale`, `classifier_coef`, `classifier_intercept`, `classes`.
