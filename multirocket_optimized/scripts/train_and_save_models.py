#!/usr/bin/env python3
"""
Train MultiRocket models on UCR datasets and save them for FPGA testing.
This script trains once and saves the models so we don't need to retrain.
"""

import argparse
import json
import time
import numpy as np
from aeon.datasets import load_classification
from aeon.transformations.collection.convolution_based import MultiRocket
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


def train_and_save(dataset_name, num_kernels=6250):
    """Train MultiRocket model and save it along with test data"""

    print(f"\n{'='*70}")
    print(f"TRAINING MULTIROCKET FOR {dataset_name}")
    print(f"{'='*70}")

    try:
        # Load dataset
        X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_name)

        # Train MultiRocket
        # MultiRocket generates: num_kernels × 4 pooling × 2 representations
        # For 6,250 kernels: 6,250 × 4 × 2 = 50,000 features
        print(f"\nTraining MultiRocket with {num_kernels} kernels...")
        print(f"Training samples: {len(X_train)}, Length: {X_train.shape[1]}")

        start_train = time.time()
        multirocket = MultiRocket(n_kernels=num_kernels, random_state=42)
        multirocket.fit(X_train)
        X_train_transform = multirocket.transform(X_train)

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
        X_test_transform = multirocket.transform(X_test)
        X_test_scaled = scaler.transform(X_test_transform)
        y_pred = classifier.predict(X_test_scaled)
        test_time = time.time() - start_test

        test_acc = np.mean(y_pred == y_test)

        print(f"\nResults:")
        print(f"  Train accuracy: {train_acc*100:.2f}%")
        print(f"  Test accuracy: {test_acc*100:.2f}%")
        print(f"  Test time: {test_time:.2f}s")

        # Save model parameters for FPGA
        model_file = f"../models/multirocket_{dataset_name.lower()}_model.json"

        # Extract parameters from the aeon MultiRocket implementation
        # MultiRocket has two parameter tuples:
        # - parameter: (dilations, n_features_per_dilation, biases) for original series
        # - parameter1: (dilations, n_features_per_dilation, biases) for diff series

        dilations_original = multirocket.parameter[0].tolist()
        n_features_per_dilation_original = multirocket.parameter[1].tolist()
        biases_original = multirocket.parameter[2].tolist()

        dilations_diff = multirocket.parameter1[0].tolist()
        n_features_per_dilation_diff = multirocket.parameter1[1].tolist()
        biases_diff = multirocket.parameter1[2].tolist()

        # MultiRocket has more complex structure than MiniRocket
        model_params = {
            "dataset": dataset_name,
            "num_kernels": multirocket.n_kernels,
            "num_features": multirocket.n_kernels * 8,  # 4 pooling × 2 representations
            "num_classes": len(np.unique(y_test)),
            "time_series_length": X_test.shape[1],

            # MultiRocket uses 84 fixed kernel patterns (same as MiniRocket)
            "num_base_kernels": 84,
            "kernel_size": 9,
            "n_features_per_kernel": multirocket.n_features_per_kernel,

            # Parameters for original series transformation
            "parameter_original": {
                "dilations": dilations_original,
                "n_features_per_dilation": n_features_per_dilation_original,
                "biases": biases_original
            },

            # Parameters for first-order differenced series transformation
            "parameter_diff": {
                "dilations": dilations_diff,
                "n_features_per_dilation": n_features_per_dilation_diff,
                "biases": biases_diff
            },

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
            "note": "MultiRocket uses 84 fixed kernel patterns from combinations(9,3) "
                    "with 4 pooling operators (PPV, MPV, MIPV, LSPV) on both original "
                    "and first-order differenced series."
        }

        print(f"\nSaving model to {model_file}...")
        with open(model_file, 'w') as f:
            json.dump(model_params, f, indent=2)

        # Save test data (all samples for comprehensive testing)
        test_file = f"../models/multirocket_{dataset_name.lower()}_test.json"

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
    parser = argparse.ArgumentParser(description='Train and save MultiRocket models')
    parser.add_argument('--dataset', type=str, nargs='+',
                        help='Dataset name(s) (InsectSound, MosquitoSound, FruitFlies). Can specify multiple.')
    parser.add_argument('--num-kernels', type=int, default=6250,
                        help='Number of kernels (default: 6250)')
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
    print(f"MULTIROCKET MODEL TRAINING")
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
