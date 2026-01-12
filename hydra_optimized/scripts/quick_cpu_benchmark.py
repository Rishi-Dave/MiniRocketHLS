#!/usr/bin/env python3
"""
Quick CPU benchmark - just timing, no complex power estimation
"""

import json
import numpy as np
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from custom_hydra import Hydra

def load_model_and_test(model_file, test_file):
    """Load model and test data"""
    with open(model_file, 'r') as f:
        model_data = json.load(f)

    with open(test_file, 'r') as f:
        test_data = json.load(f)

    X_test = np.array(test_data['time_series'])
    y_test = np.array(test_data['labels'])

    # Reconstruct HYDRA
    hydra = Hydra(
        num_kernels=model_data['num_kernels'],
        num_groups=model_data['num_groups'],
        kernel_size=model_data['kernel_size']
    )

    # Restore parameters
    hydra.num_features_ = model_data['num_features']
    hydra.classes_ = np.arange(model_data['num_classes'])
    hydra.dictionary_ = np.array(model_data['kernel_weights']).reshape(
        model_data['num_kernels'], model_data['kernel_size']
    )
    hydra.biases_ = np.array(model_data['biases'])
    hydra.dilations_ = np.array(model_data['dilations'])
    hydra.group_assignments_ = np.array(model_data.get('group_assignments',
                                        np.repeat(np.arange(model_data['num_groups']),
                                                 model_data['num_kernels'] // model_data['num_groups'])))

    # Restore scaler
    from sklearn.preprocessing import StandardScaler
    hydra.scaler_ = StandardScaler()
    hydra.scaler_.mean_ = np.array(model_data['scaler_mean'])
    hydra.scaler_.scale_ = np.array(model_data['scaler_scale'])

    # Restore classifier
    from sklearn.linear_model import RidgeClassifierCV
    hydra.classifier_ = RidgeClassifierCV()
    dummy_X = np.random.randn(model_data['num_classes'], model_data['num_features'])
    dummy_y = np.arange(model_data['num_classes'])
    hydra.classifier_.fit(dummy_X, dummy_y)
    hydra.classifier_.coef_ = np.array(model_data['coefficients']).reshape(-1, model_data['num_features'])
    hydra.classifier_.intercept_ = np.array(model_data['intercept'])

    return hydra, X_test, y_test

def benchmark_dataset(dataset_name, model_file, test_file):
    """Benchmark a dataset"""
    print(f"\n{'='*70}")
    print(f"CPU Benchmark: {dataset_name}")
    print(f"{'='*70}")

    # Load model and data
    print("Loading model and test data...")
    hydra, X_test, y_test = load_model_and_test(model_file, test_file)

    print(f"Samples: {len(X_test)}")
    print(f"Time series length: {X_test.shape[1]}")
    print(f"Classes: {len(np.unique(y_test))}")

    # Warmup
    print("\nWarmup (100 samples)...")
    _ = hydra.predict(X_test[:100])

    # Full benchmark
    print(f"Running full inference on {len(X_test)} samples...")
    start_time = time.time()
    predictions = hydra.predict(X_test)
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    throughput = len(X_test) / total_time
    latency_per_sample = total_time / len(X_test)
    accuracy = np.mean(predictions == y_test)

    # Estimate power (conservative)
    estimated_power = 80.0  # Watts (typical CPU during compute)
    energy_per_inference = estimated_power * latency_per_sample

    print(f"\n{'='*70}")
    print(f"RESULTS - {dataset_name}")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Latency per sample: {latency_per_sample*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} inferences/second")
    print(f"Estimated power: {estimated_power:.2f} W")
    print(f"Energy per inference: {energy_per_inference*1000:.2f} mJ")
    print(f"Accuracy: {accuracy*100:.2f}%")

    return {
        'dataset': dataset_name,
        'samples': len(X_test),
        'time_series_length': X_test.shape[1],
        'total_time_s': total_time,
        'latency_per_sample_ms': latency_per_sample * 1000,
        'latency_per_sample_s': latency_per_sample,
        'throughput_infer_per_s': throughput,
        'estimated_power_w': estimated_power,
        'energy_per_inference_mj': energy_per_inference * 1000,
        'accuracy_pct': accuracy * 100
    }

if __name__ == "__main__":
    models_dir = Path("models")

    datasets = [
        ('InsectSound', 'hydra_insectsound_model.json', 'hydra_insectsound_test.json'),
        ('FruitFlies', 'hydra_fruitflies_model.json', 'hydra_fruitflies_test.json'),
        ('MosquitoSound', 'hydra_mosquitosound_model.json', 'hydra_mosquitosound_test.json')
    ]

    results = []

    for dataset_name, model_file, test_file in datasets:
        model_path = models_dir / model_file
        test_path = models_dir / test_file

        if not model_path.exists():
            print(f"WARNING: Model file not found: {model_path}")
            continue
        if not test_path.exists():
            print(f"WARNING: Test file not found: {test_path}")
            continue

        result = benchmark_dataset(dataset_name, model_path, test_path)
        results.append(result)

    # Save results
    output_file = models_dir / "cpu_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("SUMMARY - CPU BENCHMARKS")
    print("="*70)
    print(f"\n{'Dataset':<20} {'Samples':<10} {'Latency (ms)':<15} {'Throughput':<20} {'Power (W)':<12}")
    print("-" * 90)
    for r in results:
        print(f"{r['dataset']:<20} {r['samples']:<10} {r['latency_per_sample_ms']:<15.2f} "
              f"{r['throughput_infer_per_s']:<20.2f} {r['estimated_power_w']:<12.2f}")

    print(f"\nResults saved to: {output_file}")
