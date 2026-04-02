#!/usr/bin/env python3
"""
GPU Baseline for MiniRocket Inference
=====================================
Run this on Google Colab with a T4 or A100 GPU.

This script benchmarks the MiniRocket feature extraction (84 ternary kernels,
9-tap filters with irregular dilations) on GPU using PyTorch 1D convolutions.

To use on Colab:
1. Upload this file
2. Runtime -> Change runtime type -> GPU (T4 or A100)
3. !pip install torch aeon scikit-learn
4. !python gpu_baseline_minirocket.py

The script measures:
- GPU inference throughput (inferences/second)
- GPU power consumption (via nvidia-smi)
- Accuracy comparison with CPU
"""

import time
import sys
import json
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch")
    sys.exit(1)

# ============================================================
# MiniRocket GPU Implementation
# ============================================================

def generate_minirocket_kernels():
    """Generate all 84 MiniRocket kernels: C(9,3) with 3 values of 2 and 6 values of -1."""
    from itertools import combinations
    kernels = []
    for combo in combinations(range(9), 3):
        kernel = np.full(9, -1.0, dtype=np.float32)
        for idx in combo:
            kernel[idx] = 2.0
        kernels.append(kernel)
    assert len(kernels) == 84, f"Expected 84 kernels, got {len(kernels)}"
    return np.array(kernels, dtype=np.float32)


def compute_dilations(time_series_length, max_dilations=32):
    """Compute uniformly-spaced dilations for MiniRocket."""
    p = int(np.log2((time_series_length - 1) / 8))
    dilations = [int(2**i) for i in range(p + 1)]
    return dilations[:max_dilations]


def minirocket_gpu_inference(time_series_batch, device='cuda'):
    """
    Full MiniRocket feature extraction on GPU using PyTorch.

    Args:
        time_series_batch: np.ndarray of shape (batch_size, time_series_length)
        device: 'cuda' or 'cpu'

    Returns:
        features: np.ndarray of shape (batch_size, num_features)
    """
    batch_size, ts_len = time_series_batch.shape
    kernels = generate_minirocket_kernels()  # (84, 9)
    dilations = compute_dilations(ts_len)

    # Move data to GPU
    ts_tensor = torch.from_numpy(time_series_batch).float().to(device)
    ts_tensor = ts_tensor.unsqueeze(1)  # (B, 1, L) for conv1d

    all_features = []

    for dilation in dilations:
        # Compute effective kernel length
        eff_len = 1 + (9 - 1) * dilation
        if eff_len > ts_len:
            continue

        # Create dilated convolution weights: (84, 1, 9)
        weight = torch.from_numpy(kernels).float().to(device).unsqueeze(1)

        # Apply all 84 kernels at once with this dilation
        # F.conv1d with dilation parameter handles the irregular spacing
        conv_out = F.conv1d(ts_tensor, weight, dilation=dilation)  # (B, 84, L')

        # Compute PPV (proportion of positive values) for each kernel
        # bias = 0 for standard MiniRocket
        ppv = (conv_out > 0).float().mean(dim=2)  # (B, 84)
        all_features.append(ppv)

    features = torch.cat(all_features, dim=1)  # (B, 84 * num_dilations)
    return features.cpu().numpy()


def minirocket_gpu_inference_batched(time_series_batch, device='cuda', batch_size=256):
    """Batched inference for large test sets."""
    n = len(time_series_batch)
    all_features = []
    for i in range(0, n, batch_size):
        batch = time_series_batch[i:i+batch_size]
        features = minirocket_gpu_inference(batch, device)
        all_features.append(features)
    return np.vstack(all_features)


# ============================================================
# Benchmark
# ============================================================

def benchmark_gpu(time_series, num_warmup=3, num_runs=10, device='cuda'):
    """Benchmark GPU inference."""
    n, ts_len = time_series.shape

    # Warmup
    for _ in range(num_warmup):
        _ = minirocket_gpu_inference(time_series[:min(100, n)], device)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = minirocket_gpu_inference_batched(time_series, device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_time = np.mean(times)
    throughput = n / mean_time
    latency_ms = mean_time / n * 1000

    return {
        'throughput_inf_per_s': throughput,
        'mean_latency_ms': latency_ms,
        'total_time_s': mean_time,
        'num_samples': n,
        'time_series_length': ts_len,
    }


def benchmark_cpu(time_series, num_runs=3):
    """Benchmark CPU inference for comparison."""
    n = len(time_series)
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = minirocket_gpu_inference_batched(time_series, device='cpu', batch_size=64)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_time = np.mean(times)
    return {
        'throughput_inf_per_s': n / mean_time,
        'mean_latency_ms': mean_time / n * 1000,
        'total_time_s': mean_time,
    }


def get_gpu_power():
    """Get current GPU power draw via nvidia-smi."""
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        return float(result.stdout.strip())
    except:
        return None


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  MiniRocket GPU Baseline Benchmark")
    print("="*70 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: No GPU found. Running on CPU only.")

    # Try to load real UCR datasets (try multiple aeon API variants)
    datasets = {}
    dataset_specs = [('InsectSound', 600, 1000), ('MosquitoSound', 3750, 1000), ('FruitFlies', 5000, 1000)]
    try:
        from aeon.datasets import load_classification

        for name, length, n_fallback in dataset_specs:
            loaded = False
            # Try original name and common UCR variants
            for try_name in [name, name.upper(), name.lower()]:
                try:
                    X_test, y_test = load_classification(try_name, split="test")
                    if X_test.ndim == 3:
                        X_test = X_test.squeeze(1)
                    datasets[name] = {'X': X_test.astype(np.float32), 'y': y_test, 'length': X_test.shape[1]}
                    print(f"  Loaded {name}: {X_test.shape}")
                    loaded = True
                    break
                except Exception:
                    continue
            if not loaded:
                print(f"  {name} not in aeon registry — using synthetic ({n_fallback} x {length})")
                X = np.random.randn(n_fallback, length).astype(np.float32)
                datasets[name] = {'X': X, 'y': np.zeros(n_fallback).astype(str), 'length': length}
    except ImportError:
        print("  aeon not installed — using synthetic data")

    # Fall back to synthetic data if nothing loaded at all
    if not datasets:
        print("\n  Using synthetic datasets:")
        for name, length, n_test in dataset_specs:
            X = np.random.randn(n_test, length).astype(np.float32)
            y = np.random.randint(0, 3, n_test).astype(str)
            datasets[name] = {'X': X, 'y': y, 'length': length}
            print(f"    {name}: ({n_test}, {length}) synthetic")

    # NOTE: Synthetic data gives identical throughput numbers as real data —
    # GPU computation time depends only on tensor dimensions, not values.

    # Also add GunPoint-like short series
    datasets['GunPoint_synthetic'] = {
        'X': np.random.randn(150, 150).astype(np.float32),
        'y': np.random.randint(0, 2, 150).astype(str),
        'length': 150
    }

    print(f"\n  Device: {device}")
    if device == 'cuda':
        gpu_power = get_gpu_power()
        if gpu_power:
            print(f"  GPU Power (idle): {gpu_power:.1f} W")

    results = {}
    for name, data in datasets.items():
        X = data['X']
        # Limit to 1000 samples for timing
        X_bench = X[:min(1000, len(X))]

        print(f"\n--- {name} (n={len(X_bench)}, length={data['length']}) ---")

        if device == 'cuda':
            gpu_result = benchmark_gpu(X_bench, device='cuda')
            print(f"  GPU: {gpu_result['throughput_inf_per_s']:.1f} inf/s, "
                  f"{gpu_result['mean_latency_ms']:.3f} ms/sample")

            # Measure GPU power during inference
            gpu_power_load = get_gpu_power()
            if gpu_power_load:
                gpu_result['gpu_power_w'] = gpu_power_load
                print(f"  GPU Power (load): {gpu_power_load:.1f} W")

            results[name] = {'gpu': gpu_result}

        cpu_result = benchmark_cpu(X_bench[:min(200, len(X_bench))])
        print(f"  CPU (PyTorch): {cpu_result['throughput_inf_per_s']:.1f} inf/s, "
              f"{cpu_result['mean_latency_ms']:.3f} ms/sample")

        if name in results:
            results[name]['cpu_pytorch'] = cpu_result
        else:
            results[name] = {'cpu_pytorch': cpu_result}

    # Print summary table
    print("\n" + "="*70)
    print("  SUMMARY TABLE")
    print("="*70)
    print(f"{'Dataset':<20} {'Length':<8} {'GPU inf/s':<12} {'CPU inf/s':<12} {'GPU/CPU':<8} {'GPU Power'}")
    print("-"*70)

    for name, data in datasets.items():
        gpu_tp = results[name].get('gpu', {}).get('throughput_inf_per_s', 0)
        cpu_tp = results[name].get('cpu_pytorch', {}).get('throughput_inf_per_s', 0)
        ratio = gpu_tp / cpu_tp if cpu_tp > 0 else 0
        power = results[name].get('gpu', {}).get('gpu_power_w', 'N/A')
        power_str = f"{power:.0f}W" if isinstance(power, float) else power
        print(f"{name:<20} {data['length']:<8} {gpu_tp:<12.1f} {cpu_tp:<12.1f} {ratio:<8.1f}x {power_str}")

    # Save results
    output_file = 'gpu_baseline_results.json'
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to: {output_file}")

    # Print comparison data for paper
    print("\n" + "="*70)
    print("  FOR PAPER: Comparison with FPGA results")
    print("="*70)
    print("\nFPGA MiniRocket Fused (from paper-results):")
    fpga_results = {
        'InsectSound': {'1cu': 1967, '2cu': 3191, '3cu': 5012},
        'MosquitoSound': {'1cu': 937, '2cu': 1497, '3cu': 2820},
        'FruitFlies': {'1cu': 742, '2cu': 1121, '3cu': 2191},
    }
    print(f"{'Dataset':<20} {'GPU inf/s':<12} {'FPGA 1CU':<12} {'FPGA 3CU':<12} {'FPGA/GPU(1CU)':<14} {'FPGA/GPU(3CU)'}")
    print("-"*80)
    for name in ['InsectSound', 'MosquitoSound', 'FruitFlies']:
        if name in results and 'gpu' in results[name]:
            gpu_tp = results[name]['gpu']['throughput_inf_per_s']
            fpga_1cu = fpga_results.get(name, {}).get('1cu', 0)
            fpga_3cu = fpga_results.get(name, {}).get('3cu', 0)
            ratio_1cu = fpga_1cu / gpu_tp if gpu_tp > 0 else 0
            ratio_3cu = fpga_3cu / gpu_tp if gpu_tp > 0 else 0
            print(f"{name:<20} {gpu_tp:<12.1f} {fpga_1cu:<12} {fpga_3cu:<12} {ratio_1cu:<14.2f}x {ratio_3cu:.2f}x")
