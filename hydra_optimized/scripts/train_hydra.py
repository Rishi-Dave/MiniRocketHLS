#!/usr/bin/env python3
"""
Train HYDRA model on UCR Time Series datasets

This script trains a HYDRA model and exports it to JSON format for FPGA deployment.

Usage:
    python train_hydra.py --dataset GunPoint --output models/hydra_gunpoint_model.json
"""

import argparse
import json
import numpy as np
from custom_hydra import Hydra

# Try to import sktime for UCR datasets
try:
    from sktime.datasets import load_UCR_UEA_dataset
    SKTIME_AVAILABLE = True
except ImportError:
    print("Warning: sktime not available. Using synthetic data.")
    SKTIME_AVAILABLE = False


def load_ucr_dataset(dataset_name):
    """Load UCR dataset using sktime"""
    if not SKTIME_AVAILABLE:
        print("Generating synthetic data instead...")
        return generate_synthetic_data()

    try:
        X_train, y_train = load_UCR_UEA_dataset(dataset_name, split='train', return_X_y=True)
        X_test, y_test = load_UCR_UEA_dataset(dataset_name, split='test', return_X_y=True)

        # Convert to numpy arrays
        X_train = np.array([x.values.flatten() for x in X_train])
        X_test = np.array([x.values.flatten() for x in X_test])

        # Convert labels to integers
        unique_labels = np.unique(np.concatenate([y_train, y_test]))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y_train = np.array([label_map[y] for y in y_train])
        y_test = np.array([label_map[y] for y in y_test])

        return X_train, y_train, X_test, y_test

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Generating synthetic data instead...")
        return generate_synthetic_data()


def generate_synthetic_data(n_train=50, n_test=150, length=150):
    """Generate synthetic time series data for testing"""
    np.random.seed(42)

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # Class 0: Sinusoidal with period 30
    for i in range(n_train // 2):
        t = np.arange(length)
        x = np.sin(2 * np.pi * t / 30) + 0.1 * np.random.randn(length)
        X_train.append(x)
        y_train.append(0)

    for i in range(n_test // 2):
        t = np.arange(length)
        x = np.sin(2 * np.pi * t / 30) + 0.1 * np.random.randn(length)
        X_test.append(x)
        y_test.append(0)

    # Class 1: Sinusoidal with period 15 (inverted)
    for i in range(n_train // 2):
        t = np.arange(length)
        x = -np.sin(2 * np.pi * t / 15) + 0.1 * np.random.randn(length)
        X_train.append(x)
        y_train.append(1)

    for i in range(n_test // 2):
        t = np.arange(length)
        x = -np.sin(2 * np.pi * t / 15) + 0.1 * np.random.randn(length)
        X_test.append(x)
        y_test.append(1)

    return (np.array(X_train), np.array(y_train),
            np.array(X_test), np.array(y_test))


def export_test_data(X_test, y_test, filename):
    """Export test data to JSON"""
    test_data = {
        'X_test': X_test.tolist(),
        'y_test': y_test.tolist(),
        'num_samples': len(y_test),
        'time_series_length': X_test.shape[1],
        'num_classes': len(np.unique(y_test))
    }

    with open(filename, 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"Test data exported to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Train HYDRA model')
    parser.add_argument('--dataset', type=str, default='GunPoint',
                        help='UCR dataset name (default: GunPoint)')
    parser.add_argument('--output', type=str, default='models/hydra_model.json',
                        help='Output model file (default: models/hydra_model.json)')
    parser.add_argument('--test-output', type=str, default='models/hydra_test.json',
                        help='Output test data file (default: models/hydra_test.json)')
    parser.add_argument('--num-kernels', type=int, default=512,
                        help='Number of dictionary kernels (default: 512)')
    parser.add_argument('--num-groups', type=int, default=8,
                        help='Number of kernel groups (default: 8)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    print("=" * 70)
    print("HYDRA Training for FPGA")
    print("=" * 70)

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    X_train, y_train, X_test, y_test = load_ucr_dataset(args.dataset)

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Time series length: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")

    # Train HYDRA
    print(f"\nTraining HYDRA model...")
    print(f"  Kernels: {args.num_kernels}")
    print(f"  Groups: {args.num_groups}")

    hydra = Hydra(
        num_kernels=args.num_kernels,
        num_groups=args.num_groups,
        random_state=args.random_state
    )

    hydra.fit(X_train, y_train)

    # Evaluate
    print("\nEvaluating model...")
    train_accuracy = hydra.score(X_train, y_train)
    test_accuracy = hydra.score(X_test, y_test)

    print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Export model
    print(f"\nExporting model to {args.output}")
    hydra.export_to_json(args.output)

    # Export test data (first 50 samples)
    n_export = min(50, len(X_test))
    print(f"\nExporting test data to {args.test_output} ({n_export} samples)")
    export_test_data(X_test[:n_export], y_test[:n_export], args.test_output)

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)

    print("\nTo test on FPGA:")
    print(f"  1. Build FPGA binary: make build TARGET=hw")
    print(f"  2. Run inference: ./hydra_host krnl.xclbin {args.output} {args.test_output}")


if __name__ == "__main__":
    main()
