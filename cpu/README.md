# MiniRocket host streaming (GPU / CPU UDP) — `feature/gpu-streaming-naive`

Branch lineage: **`feature/gpu_cpp`** → streaming work on **`feature/gpu-streaming-naive`**.

## Naming (do not confuse)

| Name | What it is |
|------|------------|
| **Offline GPU** (`minirocket_gpu`) | Batch / per-series host→GPU inference; prints Accuracy, Throughput, latency Mean/P50/P95/P99 |
| **UDP streaming (this tree)** | Client sends samples over UDP; server keeps a sliding window of length `L`, **NAIVE full recompute** each step, replies with class |
| **FPGA `minirocket_stream/`** | AXI-Stream HLS design — **unrelated** to this host UDP code; no FPGA integration here |

DP-Reuse (incremental convolution / PPV) is **future work**. This phase is correctness + rate-sweep curves.

## Paper framing

- **Table V (ArrowHead streaming RTT experiment):** MiniRocket CPU **11.0 ms** vs FPGA **4.12 ms** (**2.65×**). Client sends UDP datapoints; server maintains sliding window, runs inference, replies with class; client measures **RTT**.
- Paper conclusion: practical streams arrive faster than the RTT experiment; **amount/rate of streamed data** determines gains.
- **NETWORK_RESULTS:** rate sweeps (offered pps vs processed pps, loss%, knee/plateau).
- **Expectation:** **GPU lands between CPU and FPGA** on the plateau curve.
  - CPU baseline: ~Table V 11 ms RTT / ~5k pps loss-free in NETWORK_RESULTS CPU baselines
  - FPGA: Table V 4.12 ms RTT; **F2F FPGA multi-million pps is a different (much higher) regime** than host UDP

## Targets

| Target | Source | Description |
|--------|--------|-------------|
| `minirocket_gpu_udp_server` | `minirocket_gpu_udp_server.cu` | Naive GPU UDP server (full recompute) |
| `minirocket_cpu_udp_server` | `minirocket_cpu_udp_server.cpp` | Same protocol, host MiniRocket (single-thread; MT later) |
| `minirocket_udp_client` | `minirocket_udp_client.cpp` | One-series correctness client |
| `minirocket_udp_bench` | `minirocket_udp_bench.cpp` | Full test-set streaming bench (offline-style report) |
| `minirocket_udp_rate_sweep` | `minirocket_udp_rate_sweep.cpp` | Offered-pps knee/plateau harness (`rr` / `openloop`) |

## Build

```bash
# RTX 2080 Ti on cerebro (default sm_75)
make gpu-stream
# or:
make CUDA_ARCH=sm_75 gpu-stream

# Individuals
make stream-server        # GPU UDP server
make stream-cpu-server    # CPU UDP server
make stream-client
make stream-bench
make stream-rate
```

Compile hints:

```bash
nvcc -O3 -arch=sm_75 -std=c++17 -o minirocket_gpu_udp_server minirocket_gpu_udp_server.cu
g++ -O3 -march=native -std=c++17 -o minirocket_cpu_udp_server minirocket_cpu_udp_server.cpp
g++ -O3 -std=c++17 -o minirocket_udp_client minirocket_udp_client.cpp
g++ -O3 -std=c++17 -o minirocket_udp_bench minirocket_udp_bench.cpp
g++ -O3 -std=c++17 -o minirocket_udp_rate_sweep minirocket_udp_rate_sweep.cpp
```

## Protocol (IPv4 UDP, little-endian)

| Packet | Payload |
|--------|---------|
| **Sample** (`len==4`) | Client → server: LE **float32** sample. Server → client: LE **int32** class. Window: memmove left by 1, append, **NAIVE full recompute**, scaler + Ridge, reply. |
| **Control** (`len==8`) | LE `uint32 magic` + LE `uint32 cmd`. `magic = 0x4D525354` (`'MRST'`). **`cmd=1` RESET**: zero sliding window; reply int32 **0** ACK. Unknown cmd → reply **-1**. |

Window starts as zeros. **RESET between independent series** is required for correctness benches (otherwise leftover samples contaminate the next series). Early in-window predictions are **not** scored — only the prediction after `L` samples is offline-equivalent.

## How to run

### 1. Correctness client (one series)

Terminal 1 (GPU server):

```bash
./minirocket_gpu_udp_server ../fpga-network/minirocket_fused/InsectSound_minirocket_model.json 127.0.0.1 9000
```

Terminal 2:

```bash
./minirocket_udp_client ../fpga-network/minirocket_fused/InsectSound_test_data_compact.json 127.0.0.1 9000 0
```

Only the **final** prediction (after `L` samples) is comparable to offline whole-series inference. (This simple client does not send RESET; restart the server or use the full bench for multi-series independence.)

### 2. Full bench (accuracy like offline GPU)

With GPU server running on `:9000`:

```bash
./minirocket_udp_bench ../fpga-network/minirocket_fused/InsectSound_test_data_compact.json \
    127.0.0.1 9000 0 bench_out.csv
# args: test_data [host] [port] [max_series=0→all] [csv_out]
```

Prints Accuracy, Throughput (series/sec **and** samples/sec), and RTT distributions for (1) final-after-L only and (2) all per-sample RTTs — matching offline Mean/P50/P95/P99 style.

Same bench against the **CPU** server (apples-to-apples):

```bash
# Terminal 1
./minirocket_cpu_udp_server ../fpga-network/minirocket_fused/InsectSound_minirocket_model.json 127.0.0.1 9001
# Terminal 2
./minirocket_udp_bench ... 127.0.0.1 9001 0 bench_cpu.csv
```

### 3. Rate sweep (knee/plateau) vs GPU and vs CPU

```bash
# Open-loop offered-pps sweep (default) → CSV for NETWORK_RESULTS-style curve
./minirocket_udp_rate_sweep ../fpga-network/minirocket_fused/InsectSound_test_data_compact.json \
    127.0.0.1 9000 sweep_gpu.csv \
    --rates 100,500,1000,2000,5000,10000,20000 --samples 5000 --mode openloop --warmup 200

# Request-response (Table V style RTT)
./minirocket_udp_rate_sweep ... 127.0.0.1 9000 sweep_rr.csv --mode rr --samples 2000

# Same client against CPU server on another port for fair CPU-vs-GPU plateau
./minirocket_udp_rate_sweep ... 127.0.0.1 9001 sweep_cpu.csv --mode openloop
```

CSV columns: `rate_target_pps,mode,offered,received,loss_pct,processed_pps,rtt_mean_ms,rtt_p50_ms,rtt_p95_ms,rtt_p99_ms`.

**Openloop RTT caveat:** protocol has no sequence number; RTT is approximated by FIFO-pairing replies with a send-timestamp ring. Assumption: **in-order localhost UDP**. Do not trust openloop RTT under loss/reorder.

## Model / test JSON

Typical paths under `../fpga-network/minirocket_fused/`:

- `InsectSound_minirocket_model.json`
- `InsectSound_test_data_compact.json`

Model fields: `num_kernels`, `num_dilations`, `num_features`, `num_classes`, `time_series_length`, `kernel_indices`, `dilations`, `num_features_per_dilation`, `biases`, `scaler_mean`, `scaler_scale`, `classifier_coef`, `classifier_intercept`, `classes`.
