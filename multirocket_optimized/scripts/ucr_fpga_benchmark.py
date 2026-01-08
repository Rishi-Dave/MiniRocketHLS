#!/usr/bin/env python3
"""
Comprehensive MultiRocket FPGA Benchmark for UCR Datasets
Measures accuracy, performance, and compares Python vs FPGA implementations
"""

import argparse
import json
import time
import numpy as np
from custom_multirocket84 import MultiRocket84
from aeon.datasets import load_classification

# Target UCR datasets (matching HYDRA benchmarks)
DATASETS = ["InsectSound", "MosquitoSound", "FruitFlies"]

def load_ucr_dataset(name):
    """Load UCR dataset using aeon"""
    print(f"\nLoading {name} dataset...")

    # Load and download if necessary
    try:
        X_train, y_train = load_classification(name, split="train", extract_path=None)
        X_test, y_test = load_classification(name, split="test", extract_path=None)
    except FileNotFoundError:
        print(f"  Downloading {name}...")
        X_train, y_train = load_classification(name, split="train", extract_path=None)
        X_test, y_test = load_classification(name, split="test", extract_path=None)

    # Convert to numpy arrays if needed
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(X_test, 'values'):
        X_test = X_test.values

    # Ensure 2D array format
    if len(X_train.shape) == 3:
        X_train = X_train.squeeze()
    if len(X_test.shape) == 3:
        X_test = X_test.squeeze()

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Time series length: {X_train.shape[1]}")
    print(f"  Classes: {len(np.unique(y_train))}")

    return X_train, y_train, X_test, y_test


def save_model_for_fpga(multirocket, dataset_name, X_test, y_test):
    """Save trained MultiRocket model in JSON format for FPGA"""

    # Extract model parameters
    num_classes = len(np.unique(y_test))
    num_features = multirocket.num_features

    # Prepare biases - need to separate by representation and dilation
    # For FPGA, we'll organize biases in the order they're used:
    # [rep0_dil0_biases, rep0_dil1_biases, ..., rep1_dil0_biases, ...]
    all_biases = []

    # Original representation biases
    for bias_list in multirocket.biases_orig:
        all_biases.extend(bias_list.tolist())

    # Difference representation biases
    for bias_list in multirocket.biases_diff:
        all_biases.extend(bias_list.tolist())

    model_params = {
        "num_kernels": multirocket.num_kernels,
        "num_groups": 1,  # MultiRocket doesn't use groups
        "kernel_size": 9,
        "num_features": num_features,
        "num_classes": num_classes,
        "time_series_length": X_test.shape[1],
        "num_pooling_operators": 4,
        "num_representations": 2,

        # Dictionary kernel weights [84, 9]
        "kernel_weights": multirocket.weights.flatten().tolist(),

        # Dilations for both representations
        "dilations_orig": multirocket.dilations_orig.tolist(),
        "dilations_diff": multirocket.dilations_diff.tolist(),

        # Number of biases per dilation
        "num_features_per_dilation_orig": [len(b) for b in multirocket.biases_orig],
        "num_features_per_dilation_diff": [len(b) for b in multirocket.biases_diff],

        # All biases flattened
        "biases": all_biases,

        # StandardScaler parameters
        "scaler_mean": multirocket.scaler.mean_.tolist(),
        "scaler_scale": multirocket.scaler.scale_.tolist(),

        # Ridge classifier coefficients
        "coefficients": multirocket.classifier.coef_.T.flatten().tolist(),  # [features, classes]
        "intercept": multirocket.classifier.intercept_.tolist()
    }

    # Save model
    model_file = f"../models/multirocket_{dataset_name.lower()}_model.json"
    with open(model_file, 'w') as f:
        json.dump(model_params, f, indent=2)

    # Save test data (first 100 samples for FPGA testing)
    n_test = min(100, len(X_test))
    test_data = {
        "num_samples": n_test,
        "time_series_length": X_test.shape[1],
        "num_classes": num_classes,
        "time_series": X_test[:n_test].tolist(),
        "labels": y_test[:n_test].tolist()
    }

    test_file = f"../models/multirocket_{dataset_name.lower()}_test.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"  Saved model to: {model_file}")
    print(f"  Saved test data to: {test_file}")

    return model_file, test_file


def benchmark_python(X_train, y_train, X_test, y_test, max_dilations=8):
    """Benchmark MultiRocket on Python"""
    print("\n" + "="*70)
    print("Python MultiRocket Benchmark")
    print("="*70)

    # Training
    print("\nTraining...")
    start_time = time.time()
    multirocket = MultiRocket84(max_dilations_per_kernel=max_dilations, random_state=42)
    multirocket.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Total features: {multirocket.num_features}")

    # Testing - measure per-sample latency
    print("\nTesting...")
    start_time = time.time()
    y_pred = multirocket.predict(X_test)
    test_time = time.time() - start_time

    # Calculate metrics
    train_acc = multirocket.score(X_train, y_train)
    test_acc = np.mean(y_pred == y_test)
    avg_latency_ms = (test_time / len(X_test)) * 1000
    throughput = len(X_test) / test_time

    print(f"\nResults:")
    print(f"  Train accuracy: {train_acc*100:.2f}%")
    print(f"  Test accuracy: {test_acc*100:.2f}%")
    print(f"  Avg latency: {avg_latency_ms:.3f} ms")
    print(f"  Throughput: {throughput:.1f} inferences/sec")

    return {
        'model': multirocket,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_time': train_time,
        'test_time': test_time,
        'avg_latency_ms': avg_latency_ms,
        'throughput': throughput,
        'num_samples': len(X_test),
        'num_features': multirocket.num_features
    }


def main():
    parser = argparse.ArgumentParser(description='MultiRocket FPGA Comprehensive Benchmark')
    parser.add_argument('--max-dilations', type=int, default=8,
                        help='Maximum dilations per kernel (default: 8)')
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                        help=f'Datasets to benchmark (default: {DATASETS})')

    args = parser.parse_args()

    print("="*70)
    print("MultiRocket FPGA Comprehensive Benchmark")
    print("="*70)
    print(f"Datasets: {args.datasets}")
    print(f"Max Dilations: {args.max_dilations}")

    results = {}

    for dataset_name in args.datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")

        try:
            # Load dataset
            X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_name)

            # Python benchmark
            python_results = benchmark_python(X_train, y_train, X_test, y_test, args.max_dilations)

            # Save model for FPGA
            model_file, test_file = save_model_for_fpga(
                python_results['model'], dataset_name, X_test, y_test
            )

            # Store results
            results[dataset_name] = {
                'python': python_results,
                'model_file': model_file,
                'test_file': test_file,
                'dataset_info': {
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'length': X_train.shape[1],
                    'classes': len(np.unique(y_train))
                }
            }

            print(f"\n✓ {dataset_name} completed")
            print(f"  Python accuracy: {python_results['test_accuracy']*100:.2f}%")
            print(f"  Features generated: {python_results['num_features']}")
            print(f"  Model files saved for FPGA testing")

        except Exception as e:
            print(f"\n✗ Error with {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary
    summary_file = '../results/multirocket_python_benchmark_summary.json'
    summary = {}
    for dataset, res in results.items():
        summary[dataset] = {
            'python_accuracy': res['python']['test_accuracy'],
            'python_latency_ms': res['python']['avg_latency_ms'],
            'python_throughput': res['python']['throughput'],
            'num_features': res['python']['num_features'],
            'dataset_info': res['dataset_info'],
            'model_file': res['model_file'],
            'test_file': res['test_file']
        }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("Python Benchmark Complete")
    print(f"{'='*70}")
    print(f"Summary saved to: {summary_file}")
    print(f"\nNext step: Run FPGA benchmark using the generated model files")
    print(f"Example:")
    for dataset in results.keys():
        model_file = results[dataset]['model_file']
        test_file = results[dataset]['test_file']
        print(f"  ./host/multirocket_host build_dir.hw.*/krnl.xclbin {model_file} {test_file}")


if __name__ == "__main__":
    main()
