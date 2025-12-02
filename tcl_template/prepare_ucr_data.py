#!/usr/bin/env python3
"""
Prepare UCR datasets for FPGA testing using aeon library.
"""

import json
import numpy as np
from aeon.datasets import load_classification

# Datasets suitable for FPGA testing (short time series)
DATASETS = [
    "CBF",                # 128 length, 3 classes - matches our test setup!
    "GunPoint",           # 150 length, 2 classes - classic benchmark
    "ECG200",             # 96 length, 2 classes
    "TwoLeadECG",         # 82 length, 2 classes
    "Coffee",             # 286 length, 2 classes
    "SyntheticControl",   # 60 length, 6 classes
]


def prepare_dataset(name, output_dir="ucr_data"):
    """Download and prepare a UCR dataset for FPGA testing."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {name}...")
    X_train, y_train = load_classification(name, split="train")
    X_test, y_test = load_classification(name, split="test")

    # Convert from 3D (n_samples, n_channels, n_timesteps) to 2D
    X_train = X_train.squeeze()  # Remove channel dimension
    X_test = X_test.squeeze()

    # Combine train and test
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    # Normalize class labels to 0-indexed integers
    unique_classes = np.unique(y_all)
    class_map = {c: i for i, c in enumerate(unique_classes)}
    y_all = np.array([class_map[c] for c in y_all])

    info = {
        "name": name,
        "num_samples": len(X_all),
        "time_series_length": X_all.shape[1],
        "num_classes": len(unique_classes),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    print(f"  {name}: {info['time_series_length']} length, "
          f"{info['num_classes']} classes, {info['num_samples']} total samples")

    # Export to JSON for FPGA testing
    output_file = os.path.join(output_dir, f"{name.lower()}_fpga.json")
    data = {
        "dataset_name": name,
        "num_samples": len(X_all),
        "time_series_length": int(X_all.shape[1]),
        "num_classes": int(len(unique_classes)),
        "samples": X_all.tolist(),
        "labels": y_all.tolist(),
    }

    with open(output_file, 'w') as f:
        json.dump(data, f)

    print(f"  Exported to {output_file}")
    return info


if __name__ == "__main__":
    print("UCR Dataset Preparation for FPGA Testing")
    print("=" * 50)

    results = []
    for name in DATASETS:
        try:
            info = prepare_dataset(name)
            results.append(info)
        except Exception as e:
            print(f"  Error loading {name}: {e}")

    print("\n" + "=" * 50)
    print("Summary of prepared datasets:")
    print("=" * 50)
    for info in results:
        print(f"  {info['name']}: {info['time_series_length']}×{info['num_samples']} "
              f"({info['num_classes']} classes)")

    print("\nDatasets ready for FPGA testing in ucr_data/")
