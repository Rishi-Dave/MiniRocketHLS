#!/usr/bin/env python3
"""Create full test files for FruitFlies and MosquitoSound"""

import json
import numpy as np
from pathlib import Path
from aeon.datasets import load_from_arff_file
import sys

def create_full_test(dataset_name, output_file):
    """Create full test file for dataset"""
    print(f"\n{'='*70}")
    print(f"Creating full test file for {dataset_name}")
    print(f"{'='*70}")

    # Determine path
    base_path = Path.home() / f".local/lib/python3.10/site-packages/aeon/datasets/local_data/{dataset_name}"
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

    print(f"Loaded {len(X_test)} samples")
    print(f"Time series length: {X_test.shape[1]}")
    print(f"Creating JSON structure...")

    # Create JSON
    test_data = {
        "dataset": dataset_name,
        "num_samples": int(len(X_test)),
        "time_series_length": int(X_test.shape[1]),
        "num_classes": int(len(np.unique(y_test))),
        "time_series": X_test.tolist(),
        "labels": y_test.tolist()
    }

    # Save
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(test_data, f, indent=2)

    size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"Saved! Size: {size_mb:.1f} MB")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        if dataset == "FruitFlies":
            create_full_test("FruitFlies", "models/hydra_fruitflies_test.json")
        elif dataset == "MosquitoSound":
            create_full_test("MosquitoSound", "models/hydra_mosquitosound_test.json")
    else:
        # Create both
        create_full_test("FruitFlies", "models/hydra_fruitflies_test.json")
        print("\n")
        create_full_test("MosquitoSound", "models/hydra_mosquitosound_test.json")

    print("\n" + "="*70)
    print("FULL TEST FILES CREATED!")
    print("="*70)
