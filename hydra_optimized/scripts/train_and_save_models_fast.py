#!/usr/bin/env python3
"""
Train HYDRA models on UCR datasets and save them for FPGA testing - FAST VERSION.
Uses .npy format for test data to avoid slow JSON serialization of massive arrays.
"""

import argparse
import json
import time
import numpy as np
from custom_hydra import Hydra
from aeon.datasets import load_classification
import sys
import os

def load_ucr_dataset(name):
    """Load UCR dataset using aeon"""
    print(f"\n{'='*70}")
    print(f"Loading {name} dataset...")
    print(f"{'='*70}")

    start_load = time.time()

    # Use project directory for dataset extraction (not /tmp which is limited to 8GB)
    extract_path = "../datasets/ucr_data"

    # Load dataset
    X_train, y_train = load_classification(name, split="train", extract_path=extract_path)
    X_test, y_test = load_classification(name, split="test", extract_path=extract_path)

    # Convert to numpy arrays
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(X_test, 'values'):
        X_test = X_test.values

    # Ensure 2D array format
    if len(X_train.shape) == 3:
        X_train = X_train.squeeze()
    if len(X_test.shape) == 3:
        X_test = X_test.squeeze()

    load_time = time.time() - start_load

    print(f"Dataset loaded in {load_time:.2f}s")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Time series length: {X_train.shape[1]}")
    print(f"  Classes: {len(np.unique(y_train))}")

    return X_train, y_train, X_test, y_test


def train_and_save(dataset_name, num_kernels=512):
    """Train HYDRA model and save it along with test data"""

    print(f"\n{'='*70}")
    print(f"TRAINING HYDRA FOR {dataset_name}")
    print(f"{'='*70}")

    try:
        # Load dataset
        X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_name)

        # Train HYDRA
        print(f"\nTraining HYDRA with {num_kernels} kernels...")
        print(f"Training samples: {len(X_train)}, Length: {X_train.shape[1]}")

        start_train = time.time()
        hydra = Hydra(num_kernels=num_kernels, random_state=42)
        hydra.fit(X_train, y_train, verbose=True)
        train_time = time.time() - start_train

        print(f"\nTraining completed in {train_time:.2f}s")

        # Evaluate
        print(f"\nEvaluating...")
        train_acc = hydra.score(X_train, y_train)

        start_test = time.time()
        y_pred = hydra.predict(X_test)
        test_time = time.time() - start_test

        test_acc = np.mean(y_pred == y_test)

        print(f"\nResults:")
        print(f"  Train accuracy: {train_acc*100:.2f}%")
        print(f"  Test accuracy: {test_acc*100:.2f}%")
        print(f"  Test time: {test_time:.2f}s")

        # Save model parameters for FPGA (JSON - small file)
        model_file = f"../models/hydra_{dataset_name.lower()}_model.json"

        model_params = {
            "dataset": dataset_name,
            "num_kernels": hydra.num_kernels,
            "num_groups": hydra.num_groups,
            "kernel_size": 9,
            "num_features": hydra.num_kernels * 2,
            "num_classes": len(np.unique(y_test)),
            "time_series_length": X_test.shape[1],

            # Dictionary kernel weights [512, 9]
            "kernel_weights": hydra.dictionary_.flatten().tolist(),

            # Biases [512]
            "biases": hydra.biases_.tolist(),

            # Dilations [512]
            "dilations": hydra.dilations_.tolist(),

            # StandardScaler parameters
            "scaler_mean": hydra.scaler_.mean_.tolist(),
            "scaler_scale": hydra.scaler_.scale_.tolist(),

            # Ridge classifier coefficients
            "coefficients": hydra.classifier_.coef_.T.flatten().tolist(),
            "intercept": hydra.classifier_.intercept_.tolist(),

            # Accuracy metrics
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc)
        }

        print(f"\nSaving model to {model_file}...")
        start_save = time.time()
        with open(model_file, 'w') as f:
            json.dump(model_params, f, indent=2)
        save_time = time.time() - start_save
        print(f"Model saved in {save_time:.2f}s")

        # Save test data in NumPy binary format (.npy) - MUCH FASTER!
        # For FPGA C++ loader, we'll need to implement .npy reader
        test_data_npy = f"../models/hydra_{dataset_name.lower()}_test_X.npy"
        test_labels_npy = f"../models/hydra_{dataset_name.lower()}_test_y.npy"

        print(f"\nSaving test data to .npy format (FAST)...")
        start_save = time.time()

        # Save as float32 to match FPGA data type and reduce size
        np.save(test_data_npy, X_test.astype(np.float32))
        np.save(test_labels_npy, y_test.astype(np.int32))

        save_time = time.time() - start_save
        print(f"Test data saved in {save_time:.2f}s (vs 10+ minutes for JSON!)")

        # Also save a small metadata JSON
        test_meta_file = f"../models/hydra_{dataset_name.lower()}_test_meta.json"
        test_metadata = {
            "dataset": dataset_name,
            "num_samples": int(X_test.shape[0]),
            "time_series_length": int(X_test.shape[1]),
            "num_classes": int(len(np.unique(y_test))),
            "data_file": os.path.basename(test_data_npy),
            "labels_file": os.path.basename(test_labels_npy),
            "data_dtype": "float32",
            "labels_dtype": "int32"
        }

        with open(test_meta_file, 'w') as f:
            json.dump(test_metadata, f, indent=2)

        print(f"\n{'='*70}")
        print(f"SUCCESS: {dataset_name} model and test data saved!")
        print(f"{'='*70}")
        print(f"Model file: {model_file}")
        print(f"Test data: {test_data_npy}")
        print(f"Test labels: {test_labels_npy}")
        print(f"Test metadata: {test_meta_file}")
        print(f"Train accuracy: {train_acc*100:.2f}%")
        print(f"Test accuracy: {test_acc*100:.2f}%")
        print(f"Total time: {train_time + test_time:.2f}s")

        return True

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: Failed to train {dataset_name}")
        print(f"{'='*70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Train and save HYDRA models (FAST version with .npy)')
    parser.add_argument('--dataset', type=str, nargs='+',
                        help='Dataset name(s) (InsectSound, MosquitoSound, FruitFlies). Can specify multiple.')
    parser.add_argument('--num-kernels', type=int, default=512,
                        help='Number of kernels (default: 512)')
    parser.add_argument('--all', action='store_true',
                        help='Train all three datasets: InsectSound, MosquitoSound, FruitFlies')

    args = parser.parse_args()

    # Determine which datasets to train
    if args.all:
        datasets = ['InsectSound', 'MosquitoSound', 'FruitFlies']
    elif args.dataset:
        datasets = args.dataset
    else:
        print("Error: Must specify either --dataset or --all")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"HYDRA MODEL TRAINING - FAST VERSION")
    print(f"{'='*70}")
    print(f"Datasets to train: {', '.join(datasets)}")
    print(f"Number of kernels: {args.num_kernels}")
    print(f"Output format: JSON (model) + NPY (test data)")
    print(f"{'='*70}\n")

    # Train each dataset iteratively
    results = {}
    for dataset_name in datasets:
        success = train_and_save(dataset_name, args.num_kernels)
        results[dataset_name] = success

    # Print summary
    print(f"\n\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    for dataset_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {dataset_name}: {status}")
    print(f"{'='*70}\n")

    # Exit with error if any failed
    all_success = all(results.values())
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
