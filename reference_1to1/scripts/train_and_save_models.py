#!/usr/bin/env python3
"""
Train MiniRocket models on UCR datasets and save them for FPGA testing.
This script trains once and saves the models so we don't need to retrain.
"""

import argparse
import json
import time
import numpy as np
from aeon.datasets import load_classification
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
import sys

def load_ucr_dataset(name):
    """Load UCR dataset using aeon"""
    print(f"\n{'='*70}")
    print(f"Loading {name} dataset...")
    print(f"{'='*70}")

    start_load = time.time()

    # Use shared dataset directory (same as HYDRA)
    extract_path = "../../hydra_optimized/datasets/ucr_data"

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


def train_and_save(dataset_name, num_kernels=10000):
    """Train MiniRocket model and save it along with test data"""

    print(f"\n{'='*70}")
    print(f"TRAINING MINIROCKET FOR {dataset_name}")
    print(f"{'='*70}")

    try:
        # Load dataset
        X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_name)

        # Train MiniRocket
        print(f"\nTraining MiniRocket with {num_kernels} kernels...")
        print(f"Training samples: {len(X_train)}, Length: {X_train.shape[1]}")

        start_train = time.time()
        minirocket = MiniRocket(n_kernels=num_kernels, random_state=42)
        minirocket.fit(X_train)
        X_train_transform = minirocket.transform(X_train)

        # Train classifier
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_transform)

        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        classifier.fit(X_train_scaled, y_train)
        train_time = time.time() - start_train

        print(f"\nTraining completed in {train_time:.2f}s")

        # Evaluate
        print(f"\nEvaluating...")
        train_acc = classifier.score(X_train_scaled, y_train)

        start_test = time.time()
        X_test_transform = minirocket.transform(X_test)
        X_test_scaled = scaler.transform(X_test_transform)
        y_pred = classifier.predict(X_test_scaled)
        test_time = time.time() - start_test

        test_acc = np.mean(y_pred == y_test)

        print(f"\nResults:")
        print(f"  Train accuracy: {train_acc*100:.2f}%")
        print(f"  Test accuracy: {test_acc*100:.2f}%")
        print(f"  Test time: {test_time:.2f}s")

        # Save model parameters for FPGA
        model_file = f"../models/minirocket_{dataset_name.lower()}_model.json"

        # Extract parameters from the aeon MiniRocket implementation
        # parameters is a tuple: (n_channels_per_combination, channel_indices,
        #                         dilations, n_features_per_dilation, biases)
        n_channels_per_combination = minirocket.parameters[0].tolist()
        channel_indices = minirocket.parameters[1].tolist()
        dilations = minirocket.parameters[2].tolist()
        n_features_per_dilation = minirocket.parameters[3].tolist()
        biases = minirocket.parameters[4].tolist()

        # MiniRocket uses 84 fixed kernel patterns (combinations of 9 indices choosing 3)
        # The kernel weights are not stored but are implicitly -1 and 2
        # We need to document this for FPGA implementation

        model_params = {
            "dataset": dataset_name,
            "num_kernels": minirocket.n_kernels,
            "num_features": minirocket.n_kernels * 2,
            "num_classes": len(np.unique(y_test)),
            "time_series_length": X_test.shape[1],

            # MiniRocket uses 84 fixed kernel patterns
            # Kernel structure: weights are [-1, -1, -1, -1, -1, -1, 2, 2, 2]
            # with the three 2s at positions determined by combinations(9, 3)
            "num_base_kernels": 84,
            "kernel_size": 9,

            # Parameters from aeon MiniRocket
            "n_channels_per_combination": n_channels_per_combination,
            "channel_indices": channel_indices,
            "dilations": dilations,
            "n_features_per_dilation": n_features_per_dilation,
            "biases": biases,

            # StandardScaler parameters
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),

            # Ridge classifier coefficients
            "coefficients": classifier.coef_.T.flatten().tolist(),
            "intercept": classifier.intercept_.tolist(),

            # Accuracy metrics
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),

            # Note for FPGA implementation
            "note": "MiniRocket uses 84 fixed kernel patterns from combinations(9,3). "
                    "Kernel weights are 6 values of -1 and 3 values of 2 at specific indices."
        }

        print(f"\nSaving model to {model_file}...")
        with open(model_file, 'w') as f:
            json.dump(model_params, f, indent=2)

        # Save test data (all samples for comprehensive testing)
        test_file = f"../models/minirocket_{dataset_name.lower()}_test.json"

        test_data = {
            "dataset": dataset_name,
            "num_samples": len(X_test),
            "time_series_length": X_test.shape[1],
            "num_classes": len(np.unique(y_test)),
            "time_series": X_test.tolist(),
            "labels": y_test.tolist()
        }

        print(f"Saving test data to {test_file}...")
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)

        print(f"\n{'='*70}")
        print(f"SUCCESS: {dataset_name} model and test data saved!")
        print(f"{'='*70}")
        print(f"Model file: {model_file}")
        print(f"Test file: {test_file}")
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
    parser = argparse.ArgumentParser(description='Train and save MiniRocket models')
    parser.add_argument('--dataset', type=str, nargs='+',
                        help='Dataset name(s) (InsectSound, MosquitoSound, FruitFlies). Can specify multiple.')
    parser.add_argument('--num-kernels', type=int, default=10000,
                        help='Number of kernels (default: 10000)')
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
    print(f"MINIROCKET MODEL TRAINING")
    print(f"{'='*70}")
    print(f"Datasets to train: {', '.join(datasets)}")
    print(f"Number of kernels: {args.num_kernels}")
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
