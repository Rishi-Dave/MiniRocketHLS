#!/usr/bin/env python3
"""
Create quick 1000-sample test files for MosquitoSound and FruitFlies
in the same JSON format as InsectSound
"""

import json
import numpy as np
from pathlib import Path
from aeon.datasets import load_from_arff_file

def load_real_dataset_test_only(dataset_name):
    """Load only test data from ARFF files"""
    base_path = Path.home() / f".local/lib/python3.10/site-packages/aeon/datasets/local_data/{dataset_name}"

    # Check for nested directory structure
    test_file = base_path / f"{dataset_name}_TEST.arff"
    if not test_file.exists():
        nested_path = base_path / dataset_name
        test_file = nested_path / f"{dataset_name}_TEST.arff"

    print(f"Loading {dataset_name} test data from: {test_file}")

    X_test, y_test = load_from_arff_file(str(test_file))
    X_test = np.squeeze(X_test)

    # Convert labels to integers
    unique_labels = sorted(set(y_test))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    y_test = np.array([label_to_int[label] for label in y_test])

    return X_test, y_test


def create_quick_test(dataset_name, output_file, num_samples=1000):
    """Create 1000-sample test file"""
    print(f"\n{'='*70}")
    print(f"Creating {num_samples}-sample test for {dataset_name}")
    print(f"{'='*70}")

    # Load test data
    X_test, y_test = load_real_dataset_test_only(dataset_name)
    print(f"Loaded {len(X_test)} test samples")

    # Sample evenly
    if num_samples < len(X_test):
        indices = np.linspace(0, len(X_test) - 1, num_samples, dtype=int)
        X_test = X_test[indices]
        y_test = y_test[indices]
        print(f"Sampled {num_samples} samples")

    # Create JSON in same format as InsectSound
    test_data = {
        "dataset": dataset_name,
        "num_samples": len(X_test),
        "time_series_length": X_test.shape[1],
        "num_classes": len(np.unique(y_test)),
        "time_series": X_test.tolist(),
        "labels": y_test.tolist()
    }

    # Save
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(test_data, f)

    size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"Saved! Size: {size_mb:.1f} MB")
    print(f"  Samples: {len(X_test)}")
    print(f"  Time series length: {X_test.shape[1]}")
    print(f"  Classes: {len(np.unique(y_test))}")


if __name__ == "__main__":
    output_dir = Path("models")

    # MosquitoSound - 1000 samples
    create_quick_test(
        "MosquitoSound",
        output_dir / "hydra_mosquitosound_test_1000.json",
        num_samples=1000
    )

    # FruitFlies - 1000 samples
    create_quick_test(
        "FruitFlies",
        output_dir / "hydra_fruitflies_test_1000.json",
        num_samples=1000
    )

    print("\n" + "="*70)
    print("QUICK TEST FILES CREATED!")
    print("="*70)
