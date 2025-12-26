#!/usr/bin/env python3
"""
Train MiniRocket model on a UCR dataset and export for FPGA inference.

This script:
1. Loads a UCR dataset
2. Trains MiniRocket with a fixed number of features (to match FPGA kernel)
3. Exports the model in the format expected by the FPGA host
4. Validates accuracy on test set

Usage:
    python train_minirocket_for_fpga.py <dataset_name> [--features 420] [--output model.json]
"""

import argparse
import json
import numpy as np
from aeon.datasets import load_classification

# MiniRocket constants (must match FPGA kernel)
NUM_KERNELS = 84
KERNEL_SIZE = 9
MAX_DILATIONS = 8
MAX_FEATURES = 10000


def compute_dilations(time_series_length, num_dilations=None):
    """Compute dilations for MiniRocket based on time series length."""
    # Standard MiniRocket dilation computation
    max_dilation = max(1, (time_series_length - 1) // (KERNEL_SIZE - 1))

    if num_dilations is None:
        # Auto-compute number of dilations
        num_dilations = min(MAX_DILATIONS, int(np.log2(max_dilation)) + 1)

    # Generate dilations as powers of 2, capped at max_dilation
    dilations = [min(2**i, max_dilation) for i in range(num_dilations)]
    return dilations


def minirocket_transform_cpu(X, dilations, biases):
    """
    CPU implementation of MiniRocket feature extraction.
    This matches the FPGA kernel logic exactly.
    """
    n_samples = X.shape[0]
    ts_length = X.shape[1]
    num_dilations = len(dilations)
    num_features = NUM_KERNELS * num_dilations

    # Fixed kernel indices (84 combinations of 3 from 9)
    kernel_indices = []
    for i in range(9):
        for j in range(i+1, 9):
            for k in range(j+1, 9):
                kernel_indices.append([i, j, k])

    features = np.zeros((n_samples, num_features))

    for sample_idx in range(n_samples):
        time_series = X[sample_idx]
        feature_idx = 0

        for dil_idx, dilation in enumerate(dilations):
            for kernel_idx in range(NUM_KERNELS):
                # Compute convolution output length
                output_length = ts_length - (KERNEL_SIZE - 1) * dilation
                if output_length <= 0:
                    features[sample_idx, feature_idx] = 0
                    feature_idx += 1
                    continue

                # Apply kernel (indices give positions, weights are -1, 0, 1)
                indices = kernel_indices[kernel_idx]
                conv_output = np.zeros(output_length)

                for j in range(output_length):
                    # Weight pattern: -1, 0, 1 for the three selected positions
                    pos0 = j + indices[0] * dilation
                    pos1 = j + indices[1] * dilation
                    pos2 = j + indices[2] * dilation

                    if pos2 < ts_length:
                        conv_output[j] = -time_series[pos0] + time_series[pos2]

                # Compute PPV (Proportion of Positive Values)
                bias = biases[feature_idx] if feature_idx < len(biases) else 0
                ppv = np.mean(conv_output > bias)
                features[sample_idx, feature_idx] = ppv
                feature_idx += 1

    return features


def train_minirocket_model(dataset_name, target_features=420):
    """Train MiniRocket model on a UCR dataset."""
    print(f"Loading {dataset_name}...")
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")

    # Squeeze channel dimension
    X_train = X_train.squeeze()
    X_test = X_test.squeeze()

    if X_train.ndim == 1:
        X_train = X_train.reshape(1, -1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    ts_length = X_train.shape[1]
    print(f"  Time series length: {ts_length}")
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Map class labels to 0-indexed integers
    unique_classes = np.unique(np.concatenate([y_train, y_test]))
    class_map = {c: i for i, c in enumerate(unique_classes)}
    y_train_mapped = np.array([class_map[c] for c in y_train])
    y_test_mapped = np.array([class_map[c] for c in y_test])
    num_classes = len(unique_classes)

    print(f"  Classes: {num_classes} ({list(unique_classes)})")

    # Compute dilations to achieve target feature count
    # Features = NUM_KERNELS * num_dilations = 84 * num_dilations
    num_dilations = max(1, min(MAX_DILATIONS, target_features // NUM_KERNELS))
    actual_features = NUM_KERNELS * num_dilations

    dilations = compute_dilations(ts_length, num_dilations)
    print(f"  Dilations: {dilations} ({num_dilations} dilations, {actual_features} features)")

    # Generate random biases (standard MiniRocket uses quantiles of training data)
    # For reproducibility, we use fixed random biases
    np.random.seed(42)
    biases = np.random.uniform(-1, 1, actual_features).astype(np.float32)

    # Extract features
    print("Extracting features (CPU reference)...")
    X_train_features = minirocket_transform_cpu(X_train, dilations, biases)
    X_test_features = minirocket_transform_cpu(X_test, dilations, biases)

    # Standardize features
    print("Standardizing features...")
    scaler_mean = X_train_features.mean(axis=0)
    scaler_scale = X_train_features.std(axis=0)
    scaler_scale[scaler_scale == 0] = 1  # Avoid division by zero

    X_train_scaled = (X_train_features - scaler_mean) / scaler_scale
    X_test_scaled = (X_test_features - scaler_mean) / scaler_scale

    # Train linear classifier
    print("Training Ridge classifier...")
    from sklearn.linear_model import RidgeClassifierCV
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_scaled, y_train_mapped)

    # Evaluate
    train_acc = classifier.score(X_train_scaled, y_train_mapped)
    test_acc = classifier.score(X_test_scaled, y_test_mapped)

    print(f"\n  Train accuracy: {train_acc*100:.2f}%")
    print(f"  Test accuracy:  {test_acc*100:.2f}%")

    # Prepare model for export
    # Reshape coefficients to match FPGA format [num_classes][MAX_FEATURES]
    coef = classifier.coef_
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    # Pad to MAX_FEATURES
    coef_padded = np.zeros((num_classes, MAX_FEATURES), dtype=np.float32)
    coef_padded[:coef.shape[0], :coef.shape[1]] = coef

    intercept = classifier.intercept_
    if not hasattr(intercept, '__len__'):
        intercept = [intercept]

    # Pad arrays
    scaler_mean_padded = np.zeros(MAX_FEATURES, dtype=np.float32)
    scaler_mean_padded[:len(scaler_mean)] = scaler_mean

    scaler_scale_padded = np.ones(MAX_FEATURES, dtype=np.float32)
    scaler_scale_padded[:len(scaler_scale)] = scaler_scale

    biases_padded = np.zeros(MAX_FEATURES, dtype=np.float32)
    biases_padded[:len(biases)] = biases

    # Features per dilation (all same for our implementation)
    num_features_per_dilation = [NUM_KERNELS] * num_dilations

    model = {
        "dataset_name": dataset_name,
        "num_features": actual_features,
        "num_classes": num_classes,
        "num_dilations": num_dilations,
        "time_series_length": ts_length,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "dilations": dilations,
        "num_features_per_dilation": num_features_per_dilation,
        "biases": biases_padded.tolist(),
        "scaler_mean": scaler_mean_padded.tolist(),
        "scaler_scale": scaler_scale_padded.tolist(),
        "classifier_coef": coef_padded.flatten().tolist(),
        "classifier_intercept": list(intercept),
        "class_map": {str(k): v for k, v in class_map.items()},
    }

    return model, X_test, y_test_mapped


def export_model(model, output_file):
    """Export model to JSON for FPGA."""
    with open(output_file, 'w') as f:
        json.dump(model, f, indent=2)
    print(f"Model exported to {output_file}")


def export_test_data(X_test, y_test, output_file):
    """Export test data for FPGA validation."""
    data = {
        "num_samples": len(X_test),
        "time_series_length": X_test.shape[1],
        "samples": X_test.tolist(),
        "labels": y_test.tolist(),
    }
    with open(output_file, 'w') as f:
        json.dump(data, f)
    print(f"Test data exported to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniRocket for FPGA")
    parser.add_argument("dataset", help="UCR dataset name (e.g., CBF, ECG200)")
    parser.add_argument("--features", type=int, default=420,
                        help="Target number of features (default: 420)")
    parser.add_argument("--output", default=None,
                        help="Output model file (default: <dataset>_fpga_model.json)")
    args = parser.parse_args()

    output_file = args.output or f"{args.dataset.lower()}_fpga_model.json"
    test_file = f"{args.dataset.lower()}_fpga_test.json"

    print("=" * 60)
    print(f"Training MiniRocket for FPGA - {args.dataset}")
    print("=" * 60)

    model, X_test, y_test = train_minirocket_model(args.dataset, args.features)
    export_model(model, output_file)
    export_test_data(X_test, y_test, test_file)

    print("\n" + "=" * 60)
    print("DONE! Files created:")
    print(f"  Model:     {output_file}")
    print(f"  Test data: {test_file}")
    print(f"\nTo run on FPGA:")
    print(f"  ./ucr_benchmark build_dir.hw.*/krnl.xclbin {output_file} {test_file}")
    print("=" * 60)
