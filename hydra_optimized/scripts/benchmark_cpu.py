#!/usr/bin/env python3
"""
Benchmark CPU performance of HYDRA on all three datasets
Measures latency, throughput, and power consumption
"""

import json
import numpy as np
import time
from pathlib import Path
from custom_hydra import Hydra
import os

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not available, using estimated power consumption")

def estimate_cpu_power():
    """
    Estimate CPU power consumption during inference
    This is a rough estimation based on CPU utilization
    Typical CPU TDP: 65-150W, we'll use average active power
    """
    if HAS_PSUTIL:
        # Get CPU utilization
        cpu_percent = psutil.cpu_percent(interval=0.1)
        # Assume typical server CPU draws ~100W at full load, ~20W idle
        idle_power = 20.0  # Watts
        max_power = 100.0  # Watts
        estimated_power = idle_power + (max_power - idle_power) * (cpu_percent / 100.0)
        return estimated_power
    else:
        # Conservative estimate: assume 80% CPU utilization during inference
        # Typical server CPU: ~80W average during compute
        return 80.0

def benchmark_dataset(dataset_name, model_file, test_file):
    """Benchmark a single dataset"""
    print(f"\n{'='*70}")
    print(f"Benchmarking {dataset_name}")
    print(f"{'='*70}")

    # Load model
    print(f"Loading model from {model_file}...")
    with open(model_file, 'r') as f:
        model_data = json.load(f)

    # Load test data
    print(f"Loading test data from {test_file}...")
    with open(test_file, 'r') as f:
        test_data = json.load(f)

    X_test = np.array(test_data['time_series'])
    y_test = np.array(test_data['labels'])

    print(f"Dataset: {dataset_name}")
    print(f"  Samples: {len(X_test)}")
    print(f"  Time series length: {X_test.shape[1]}")
    print(f"  Classes: {len(np.unique(y_test))}")

    # Reconstruct model
    print("Reconstructing HYDRA model...")
    hydra = Hydra(
        num_kernels=model_data['num_kernels'],
        num_groups=model_data['num_groups'],
        kernel_size=model_data['kernel_size']
    )

    # Restore dictionary and parameters
    hydra.num_features_ = model_data['num_features']
    hydra.classes_ = np.arange(model_data['num_classes'])  # Infer classes from num_classes
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
    # Fit with dummy data to initialize classes_
    dummy_X = np.random.randn(model_data['num_classes'], model_data['num_features'])
    dummy_y = np.arange(model_data['num_classes'])
    hydra.classifier_.fit(dummy_X, dummy_y)
    # Now override with actual learned parameters
    hydra.classifier_.coef_ = np.array(model_data['coefficients']).reshape(-1, model_data['num_features'])
    hydra.classifier_.intercept_ = np.array(model_data['intercept'])

    print("\n" + "="*70)
    print("RUNNING INFERENCE BENCHMARK")
    print("="*70)

    # Warmup run (small subset)
    print("Warmup run (100 samples)...")
    warmup_samples = min(100, len(X_test))
    _ = hydra.predict(X_test[:warmup_samples])

    # Measure power before test
    print("Measuring baseline CPU power...")
    baseline_power = estimate_cpu_power()
    print(f"Baseline CPU power: {baseline_power:.2f} W")

    # Full benchmark
    print(f"\nRunning full inference on {len(X_test)} samples...")
    start_time = time.time()

    # Measure power during inference
    power_samples = []
    predictions = []

    # Process in chunks and measure power
    chunk_size = 100
    num_chunks = (len(X_test) + chunk_size - 1) // chunk_size

    for i in range(0, len(X_test), chunk_size):
        chunk = X_test[i:i+chunk_size]

        # Measure power at start of chunk
        power_samples.append(estimate_cpu_power())

        # Run inference
        preds = hydra.predict(chunk)
        predictions.extend(preds)

        if (i + chunk_size) % 1000 == 0:
            progress = min(i + chunk_size, len(X_test))
            print(f"  Progress: {progress}/{len(X_test)} ({100*progress//len(X_test)}%)")

    end_time = time.time()
    predictions = np.array(predictions)

    # Calculate metrics
    total_time = end_time - start_time
    throughput = len(X_test) / total_time
    latency_per_sample = total_time / len(X_test)
    avg_power = np.mean(power_samples)
    energy_per_inference = avg_power * latency_per_sample

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} inferences/second")
    print(f"Latency per sample: {latency_per_sample*1000:.2f} ms")
    print(f"Average CPU power: {avg_power:.2f} W")
    print(f"Energy per inference: {energy_per_inference*1000:.2f} mJ")
    print(f"Accuracy: {accuracy*100:.2f}%")

    return {
        'dataset': dataset_name,
        'samples': len(X_test),
        'time_series_length': X_test.shape[1],
        'total_time_s': total_time,
        'throughput_infer_per_s': throughput,
        'latency_per_sample_ms': latency_per_sample * 1000,
        'latency_per_sample_s': latency_per_sample,
        'avg_power_w': avg_power,
        'energy_per_inference_mj': energy_per_inference * 1000,
        'energy_per_inference_j': energy_per_inference,
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
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Dataset':<20} {'Samples':<10} {'Latency (ms)':<15} {'Throughput':<15} {'Power (W)':<12} {'Accuracy':<10}")
    print("-" * 90)
    for r in results:
        print(f"{r['dataset']:<20} {r['samples']:<10} {r['latency_per_sample_ms']:<15.2f} "
              f"{r['throughput_infer_per_s']:<15.2f} {r['avg_power_w']:<12.2f} {r['accuracy_pct']:<10.2f}%")

    print(f"\nResults saved to: {output_file}")
