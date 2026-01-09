#!/usr/bin/env python3
"""
Benchmark HYDRA on UCR datasets

Compares HYDRA accuracy and performance across multiple datasets.
"""

import argparse
import time
import numpy as np
from custom_hydra import Hydra


def generate_test_data(length=150, n_samples=100):
    """Generate synthetic test data"""
    np.random.seed(42)
    X = []
    y = []

    for i in range(n_samples):
        if i < n_samples // 2:
            t = np.arange(length)
            x = np.sin(2 * np.pi * t / 30) + 0.1 * np.random.randn(length)
            X.append(x)
            y.append(0)
        else:
            t = np.arange(length)
            x = -np.sin(2 * np.pi * t / 15) + 0.1 * np.random.randn(length)
            X.append(x)
            y.append(1)

    return np.array(X), np.array(y)


def benchmark_hydra(X_train, y_train, X_test, y_test, num_kernels=512):
    """Benchmark HYDRA on a dataset"""
    print("\nRunning benchmark...")

    # Training
    start_time = time.time()
    hydra = Hydra(num_kernels=num_kernels, random_state=42)
    hydra.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Testing
    start_time = time.time()
    y_pred = hydra.predict(X_test)
    test_time = time.time() - start_time

    # Accuracy
    train_acc = hydra.score(X_train, y_train)
    test_acc = np.mean(y_pred == y_test)

    # Throughput
    throughput = len(X_test) / test_time

    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_time': train_time,
        'test_time': test_time,
        'throughput': throughput,
        'num_samples': len(X_test)
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark HYDRA')
    parser.add_argument('--num-kernels', type=int, default=512,
                        help='Number of kernels (default: 512)')
    parser.add_argument('--length', type=int, default=150,
                        help='Time series length (default: 150)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of test samples (default: 100)')

    args = parser.parse_args()

    print("=" * 70)
    print("HYDRA Benchmark")
    print("=" * 70)

    # Generate data
    print(f"\nGenerating synthetic data...")
    print(f"  Time series length: {args.length}")
    print(f"  Number of samples: {args.samples}")

    X_train, y_train = generate_test_data(args.length, args.samples)
    X_test, y_test = generate_test_data(args.length, args.samples // 2)

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Benchmark
    results = benchmark_hydra(X_train, y_train, X_test, y_test, args.num_kernels)

    # Print results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    print(f"\nAccuracy:")
    print(f"  Training: {results['train_accuracy']:.4f} ({results['train_accuracy']*100:.2f}%)")
    print(f"  Test: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")

    print(f"\nTiming:")
    print(f"  Training time: {results['train_time']:.3f} s")
    print(f"  Test time: {results['test_time']:.3f} s")

    print(f"\nThroughput:")
    print(f"  {results['throughput']:.0f} inferences/sec")
    print(f"  {1000.0 / results['throughput']:.2f} ms/inference")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
