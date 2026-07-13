#!/usr/bin/env python3
"""
GPU Baseline for HYDRA Inference
=================================
Run this on Google Colab with a T4 GPU.

HYDRA uses 512 random 9-tap convolutional kernels with learned weights,
grouped into 64 groups. Features are extracted via max-pooling and
global mean pooling (2 features per kernel = 1024 features), then
classified with Ridge regression.

To use on Colab:
1. Upload this file
2. Runtime -> Change runtime type -> GPU (T4)
3. !pip install torch aeon scikit-learn
4. !python gpu_baseline_hydra.py
5. Copy the printed JSON results

Measures:
- GPU inference throughput at batch=256 and batch=1
- GPU power via nvidia-smi
"""

import time
import sys
import json
import subprocess
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch")
    sys.exit(1)

# ============================================================
# HYDRA GPU Implementation
# ============================================================

NUM_KERNELS = 512
KERNEL_SIZE = 9
NUM_GROUPS = 64
KERNELS_PER_GROUP = NUM_KERNELS // NUM_GROUPS  # 8

class HydraGPU(nn.Module):
    """HYDRA feature extraction + classification on GPU."""

    def __init__(self, num_kernels=512, kernel_size=9, num_groups=64,
                 num_features=1024, num_classes=10, time_series_length=600):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.num_features = num_features
        self.num_classes = num_classes

        # Random convolutional kernels (grouped conv1d)
        # Each group has kernels_per_group=8 kernels
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=num_kernels,
            kernel_size=kernel_size,
            bias=True,
            padding=0,
            groups=1  # HYDRA uses standard conv, not grouped conv1d
        )

        # Initialize with random weights (HYDRA uses learned random kernels)
        nn.init.normal_(self.conv.weight, 0, 1)
        nn.init.normal_(self.conv.bias, 0, 1)

        # Scaler parameters
        self.scaler_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.scaler_scale = nn.Parameter(torch.ones(num_features), requires_grad=False)

        # Ridge classifier (linear layer)
        self.classifier = nn.Linear(num_features, num_classes, bias=True)

    def extract_features(self, x):
        """Extract HYDRA features: max pooling + global mean pooling per kernel."""
        # x: (batch, 1, length)
        conv_out = self.conv(x)  # (batch, num_kernels, out_len)

        # Max pooling (global max over time dimension)
        max_pool = conv_out.max(dim=2).values  # (batch, num_kernels)

        # Global mean pooling
        mean_pool = conv_out.mean(dim=2)  # (batch, num_kernels)

        # Interleave: [max_0, mean_0, max_1, mean_1, ...]
        batch_size = x.shape[0]
        features = torch.empty(batch_size, self.num_features, device=x.device)
        features[:, 0::2] = max_pool
        features[:, 1::2] = mean_pool

        return features

    def forward(self, x):
        """Full pipeline: feature extraction -> scaling -> classification."""
        features = self.extract_features(x)
        # StandardScaler
        features = (features - self.scaler_mean) / self.scaler_scale
        # Ridge classifier
        logits = self.classifier(features)
        return logits


def get_gpu_power():
    """Get current GPU power draw via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        return float(result.stdout.strip().split('\n')[0])
    except Exception:
        return None


def benchmark_hydra_gpu(X, model, device, batch_sizes=[256, 1], warmup=50, n_iters=500):
    """Benchmark HYDRA inference on GPU at different batch sizes."""
    results = {}
    model.eval()

    for bs in batch_sizes:
        n_samples = min(len(X), max(bs * 10, 1000))
        X_bench = torch.tensor(X[:n_samples], dtype=torch.float32, device=device).unsqueeze(1)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                idx = torch.randint(0, n_samples, (bs,))
                _ = model(X_bench[idx])

        torch.cuda.synchronize()

        # Timed iterations
        latencies = []
        total_samples = 0
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(n_iters):
                idx = torch.randint(0, n_samples, (bs,))
                t0 = time.perf_counter()
                _ = model(X_bench[idx])
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append(t1 - t0)
                total_samples += bs

        elapsed = time.perf_counter() - start
        throughput = total_samples / elapsed
        mean_latency_ms = np.mean(latencies) * 1000

        results[f'batch_{bs}'] = {
            'batch_size': bs,
            'throughput_inf_per_s': round(throughput, 1),
            'mean_latency_ms': round(mean_latency_ms, 4),
            'total_samples': total_samples,
            'total_time_s': round(elapsed, 2)
        }

        print(f"    batch={bs}: {throughput:.1f} inf/s, {mean_latency_ms:.3f} ms/batch")

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print("WARNING: No GPU detected. Results will be CPU-only.")

    # Load datasets
    print("\nLoading datasets...")
    datasets = {}
    dataset_specs = [
        ('InsectSound', 600, 10, 25000),
        ('MosquitoSound', 3750, 2, 6000),
        ('FruitFlies', 5000, 3, 17259)
    ]

    try:
        from aeon.datasets import load_classification

        for name, length, num_classes, n_test in dataset_specs:
            try:
                X_test, y_test = load_classification(name, split="test")
                if X_test.ndim == 3:
                    X_test = X_test.squeeze(1)
                datasets[name] = {
                    'X': X_test.astype(np.float32),
                    'length': length,
                    'num_classes': num_classes,
                    'n_test': len(X_test)
                }
                print(f"  Loaded {name}: {X_test.shape}")
            except Exception as e:
                print(f"  {name}: aeon load failed ({e}), using synthetic")
                X = np.random.randn(min(n_test, 2000), length).astype(np.float32)
                datasets[name] = {'X': X, 'length': length, 'num_classes': num_classes, 'n_test': len(X)}
    except ImportError:
        print("  aeon not installed, using synthetic data")
        for name, length, num_classes, n_test in dataset_specs:
            X = np.random.randn(min(n_test, 2000), length).astype(np.float32)
            datasets[name] = {'X': X, 'length': length, 'num_classes': num_classes, 'n_test': len(X)}

    # Benchmark each dataset
    all_results = {}

    for name, data in datasets.items():
        print(f"\n--- {name} (length={data['length']}, classes={data['num_classes']}) ---")

        model = HydraGPU(
            num_kernels=NUM_KERNELS,
            kernel_size=KERNEL_SIZE,
            num_groups=NUM_GROUPS,
            num_features=NUM_KERNELS * 2,  # 1024
            num_classes=data['num_classes'],
            time_series_length=data['length']
        ).to(device)

        gpu_result = benchmark_hydra_gpu(data['X'], model, device)

        gpu_power = get_gpu_power()
        if gpu_power:
            gpu_result['gpu_power_w'] = gpu_power
            print(f"    GPU Power: {gpu_power:.1f} W")

        all_results[name] = gpu_result

    # Summary
    print("\n" + "=" * 70)
    print("  HYDRA GPU BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<20} {'Length':<8} {'b=256 inf/s':<15} {'b=1 inf/s':<15} {'GPU Power'}")
    print("-" * 70)

    for name, data in datasets.items():
        b256 = all_results[name].get('batch_256', {}).get('throughput_inf_per_s', 0)
        b1 = all_results[name].get('batch_1', {}).get('throughput_inf_per_s', 0)
        power = all_results[name].get('gpu_power_w', 'N/A')
        power_str = f"{power:.0f}W" if isinstance(power, float) else power
        print(f"{name:<20} {data['length']:<8} {b256:<15.1f} {b1:<15.1f} {power_str}")

    # Cross-check: b=256 numbers should approximately match paper table values
    print("\n--- CROSS-CHECK (b=256 should match paper values) ---")
    paper_values = {'InsectSound': 13895, 'MosquitoSound': 5139, 'FruitFlies': 4267}
    for name, expected in paper_values.items():
        if name in all_results:
            actual = all_results[name].get('batch_256', {}).get('throughput_inf_per_s', 0)
            ratio = actual / expected if expected > 0 else 0
            status = "OK" if 0.5 < ratio < 2.0 else "CHECK"
            print(f"  {name}: paper={expected}, measured={actual:.0f}, ratio={ratio:.2f} [{status}]")

    # Save results
    output = {
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'pytorch_version': torch.__version__,
        'results': all_results
    }

    output_file = 'gpu_baseline_hydra_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"\nResults saved to {output_file}")

    # Print for easy copy-paste into paper
    print("\n--- VALUES FOR gpu_comparison.tex (HYDRA rows, GPU b=1 column) ---")
    for name in ['InsectSound', 'MosquitoSound', 'FruitFlies']:
        if name in all_results:
            b1 = all_results[name].get('batch_1', {}).get('throughput_inf_per_s', 0)
            print(f"  {name}: {b1:.0f} inf/s")


if __name__ == '__main__':
    main()
