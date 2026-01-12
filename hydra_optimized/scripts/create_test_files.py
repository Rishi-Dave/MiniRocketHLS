#!/usr/bin/env python3
"""
Create test JSON files for MosquitoSound and FruitFlies datasets
to enable FPGA inference testing.
"""

import json
import numpy as np
from pathlib import Path
from aeon.datasets import load_from_arff_file

def load_real_dataset(dataset_name='ArrowHead'):
    """Load real time series datasets from local ARFF files"""
    base_path = Path.home() / f".local/lib/python3.10/site-packages/aeon/datasets/local_data/{dataset_name}"

    # Check for nested directory structure (e.g., MosquitoSound/MosquitoSound/)
    nested_path = base_path / dataset_name
    if nested_path.exists():
        base_path = nested_path

    train_file = base_path / f"{dataset_name}_TRAIN.arff"
    test_file = base_path / f"{dataset_name}_TEST.arff"

    print(f"Attempting to load dataset: {dataset_name} from local ARFF files")
    print(f"  Train file: {train_file}")
    print(f"  Test file: {test_file}")

    X_train, y_train = load_from_arff_file(str(train_file))
    X_test, y_test = load_from_arff_file(str(test_file))

    # Remove extra dimension if present
    X_train = np.squeeze(X_train)
    X_test = np.squeeze(X_test)

    # Convert labels to integers
    unique_labels_train = sorted(set(y_train))
    label_to_int_train = {label: i for i, label in enumerate(unique_labels_train)}
    y_numpy_train = np.array([label_to_int_train[label] for label in y_train])

    unique_labels_test = sorted(set(y_test))
    label_to_int_test = {label: i for i, label in enumerate(unique_labels_test)}
    y_numpy_test = np.array([label_to_int_test[label] for label in y_test])

    return X_train, y_numpy_train, X_test, y_numpy_test


def create_test_file(dataset_name, output_file, sample_size=None):
    """Create test JSON file for a dataset"""
    print(f"\n{'='*70}")
    print(f"Creating test file for {dataset_name}")
    print(f"{'='*70}")

    # Load dataset
    X_train, y_train, X_test, y_test = load_real_dataset(dataset_name)

    print(f"Dataset loaded:")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Time series length: {X_test.shape[1]}")
    print(f"  Classes: {len(np.unique(y_test))}")

    # Sample if requested
    if sample_size and sample_size < len(X_test):
        print(f"\nSampling {sample_size} test samples...")
        indices = np.linspace(0, len(X_test) - 1, sample_size, dtype=int)
        X_test = X_test[indices]
        y_test = y_test[indices]

    # Convert to JSON format
    print(f"\nConverting to JSON format...")
    test_data = {
        "test_data": X_test.tolist(),
        "test_labels": y_test.tolist()
    }

    # Save to file
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(test_data, f)

    # Check file size
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"File saved successfully!")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Samples: {len(X_test)}")

    return file_size_mb


if __name__ == "__main__":
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    # MosquitoSound - Create quick test (1000 samples) and full test
    print("\n" + "="*70)
    print("MOSQUITOSOUND TEST FILES")
    print("="*70)

    create_test_file(
        "MosquitoSound",
        output_dir / "hydra_mosquitosound_test_1000.json",
        sample_size=1000
    )

    # Full test - check size first
    X_train, y_train, X_test, y_test = load_real_dataset("MosquitoSound")
    estimated_size_mb = len(X_test) * X_test.shape[1] * 8 / (1024 * 1024)  # rough estimate
    print(f"\nFull MosquitoSound test would be ~{estimated_size_mb:.0f} MB")

    if estimated_size_mb > 1000:
        print(f"Skipping full test file (too large)")
        print(f"Creating 10,000 sample test instead...")
        create_test_file(
            "MosquitoSound",
            output_dir / "hydra_mosquitosound_test.json",
            sample_size=10000
        )
    else:
        create_test_file(
            "MosquitoSound",
            output_dir / "hydra_mosquitosound_test.json"
        )

    # FruitFlies - Create quick test (1000 samples) and full test
    print("\n" + "="*70)
    print("FRUITFLIES TEST FILES")
    print("="*70)

    create_test_file(
        "FruitFlies",
        output_dir / "hydra_fruitflies_test_1000.json",
        sample_size=1000
    )

    create_test_file(
        "FruitFlies",
        output_dir / "hydra_fruitflies_test.json"
    )

    print("\n" + "="*70)
    print("ALL TEST FILES CREATED SUCCESSFULLY!")
    print("="*70)
