#!/usr/bin/env python3
"""
GPU Scalability Sweep — MiniRocket throughput vs time series length
===================================================================
Run on Google Colab with T4 GPU. Matches the FPGA scalability sweep
(results/scalability_sweep.csv) at the same 8 window lengths.

Produces a direct GPU vs FPGA comparison at each length for the paper's
scalability figure.

Usage on Colab:
  !python gpu_scalability_sweep.py
"""

import time
import json
import numpy as np

import torch
import torch.nn.functional as F

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def generate_minirocket_kernels():
    from itertools import combinations
    kernels = []
    for combo in combinations(range(9), 3):
        kernel = np.full(9, -1.0, dtype=np.float32)
        for idx in combo:
            kernel[idx] = 2.0
        kernels.append(kernel)
    return np.array(kernels, dtype=np.float32)


def compute_dilations(ts_len):
    p = int(np.log2((ts_len - 1) / 8))
    return [int(2**i) for i in range(p + 1)]


def minirocket_gpu(ts_batch, device='cuda'):
    """MiniRocket feature extraction on GPU."""
    B, L = ts_batch.shape
    kernels = generate_minirocket_kernels()
    dilations = compute_dilations(L)
    ts = torch.from_numpy(ts_batch).float().to(device).unsqueeze(1)
    features = []
    for d in dilations:
        if 1 + 8 * d > L:
            continue
        w = torch.from_numpy(kernels).float().to(device).unsqueeze(1)
        out = F.conv1d(ts, w, dilation=d)
        ppv = (out > 0).float().mean(dim=2)
        features.append(ppv)
    return torch.cat(features, dim=1).cpu().numpy()


def minirocket_gpu_batched(ts, device='cuda', batch_size=256):
    n = len(ts)
    parts = []
    for i in range(0, n, batch_size):
        parts.append(minirocket_gpu(ts[i:i+batch_size], device))
    return np.vstack(parts)


def benchmark_length(length, n_samples=1000, device='cuda',
                     num_warmup=3, num_runs=10):
    """Benchmark at a specific time series length."""
    X = np.random.randn(n_samples, length).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        minirocket_gpu(X[:100], device)
    torch.cuda.synchronize()

    # Batched (batch=256)
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        minirocket_gpu_batched(X, device, batch_size=256)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    batch256_tp = n_samples / np.mean(times)
    batch256_lat = np.mean(times) / n_samples * 1000

    # Single-sample (batch=1)
    n_b1 = min(200, n_samples)
    # warmup
    for i in range(3):
        minirocket_gpu(X[i:i+1], device)
    torch.cuda.synchronize()

    lats = []
    for i in range(n_b1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        minirocket_gpu(X[i:i+1], device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        lats.append(t1 - t0)
    lats_ms = np.array(lats) * 1000
    batch1_tp = 1000.0 / np.mean(lats_ms)
    batch1_p50 = np.median(lats_ms)

    # GPU power
    try:
        import subprocess
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        power = float(r.stdout.strip())
    except Exception:
        power = None

    return {
        'length': length,
        'batch256_throughput': round(batch256_tp, 1),
        'batch256_latency_ms': round(batch256_lat, 3),
        'batch1_throughput': round(batch1_tp, 1),
        'batch1_p50_ms': round(batch1_p50, 3),
        'gpu_power_w': power,
        'num_dilations': len([d for d in compute_dilations(length) if 1 + 8*d <= length]),
    }


# FPGA reference data from results/scalability_sweep.csv
FPGA_DATA = {
    64: 2681.0, 128: 2457.7, 256: 2512.1, 512: 2203.9,
    1024: 1744.9, 2048: 1302.5, 4096: 793.6, 8192: 511.1,
}

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    print(f"\n{'='*80}")
    print(f"  MiniRocket GPU Scalability Sweep")
    print(f"  Lengths: {lengths}")
    print(f"{'='*80}\n")

    results = []
    for length in lengths:
        print(f"  Length {length:>5} ... ", end='', flush=True)
        r = benchmark_length(length, device=device)
        results.append(r)
        print(f"b256={r['batch256_throughput']:>10,.1f} inf/s  "
              f"b1={r['batch1_throughput']:>8,.1f} inf/s  "
              f"dilations={r['num_dilations']}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"  GPU vs FPGA Scalability Comparison")
    print(f"{'='*90}")
    print(f"{'Length':<8} {'GPU b256':<12} {'GPU b1':<10} {'FPGA 1CU':<12} "
          f"{'GPU/FPGA':<10} {'b1/FPGA':<10} {'Dilations'}")
    print(f"{'-'*80}")
    for r in results:
        fpga = FPGA_DATA.get(r['length'], 0)
        ratio256 = r['batch256_throughput'] / fpga if fpga else 0
        ratio1 = r['batch1_throughput'] / fpga if fpga else 0
        print(f"{r['length']:<8} {r['batch256_throughput']:<12,.1f} "
              f"{r['batch1_throughput']:<10,.1f} {fpga:<12,.1f} "
              f"{ratio256:<10.2f}x {ratio1:<10.2f}x {r['num_dilations']}")

    # Save results
    output = {
        'metadata': {
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'pytorch_version': torch.__version__,
            'date': time.strftime('%Y-%m-%d'),
            'batch_size': 256,
            'num_samples': 1000,
        },
        'results': results,
        'fpga_reference': FPGA_DATA,
    }
    with open('gpu_scalability_sweep.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: gpu_scalability_sweep.json")

    # CSV for easy plotting
    with open('gpu_scalability_sweep.csv', 'w') as f:
        f.write('length,gpu_batch256_inf_per_s,gpu_batch1_inf_per_s,fpga_1cu_inf_per_s,'
                'gpu_batch256_latency_ms,gpu_batch1_p50_ms,num_dilations,gpu_power_w\n')
        for r in results:
            fpga = FPGA_DATA.get(r['length'], 0)
            pw = r['gpu_power_w'] if r['gpu_power_w'] else ''
            f.write(f"{r['length']},{r['batch256_throughput']},{r['batch1_throughput']},"
                    f"{fpga},{r['batch256_latency_ms']},{r['batch1_p50_ms']},"
                    f"{r['num_dilations']},{pw}\n")
    print(f"Saved to: gpu_scalability_sweep.csv")
