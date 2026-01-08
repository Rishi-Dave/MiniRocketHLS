#!/usr/bin/env python3
"""
Train HYDRA models on UCR datasets and save them for FPGA testing.
This script trains once and saves the models so we don't need to retrain.
"""

import argparse
import json
import time
import numpy as np
from custom_hydra import Hydra
from aeon.datasets import load_classification
import sys

def load_ucr_dataset(name):
    """Load UCR dataset using aeon"""
    print(f"\n{'='*70}")
    print(f"Loading {name} dataset...")
    print(f"{'='*70}")

    start_load = time.time()

    # Load dataset
    X_train, y_train = load_classification(name, split="train", extract_path=None)
    X_test, y_test = load_classification(name, split="test", extract_path=None)

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

        # Save model parameters for FPGA
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
        with open(model_file, 'w') as f:
            json.dump(model_params, f, indent=2)

        # Save test data (all samples for comprehensive testing)
        test_file = f"../models/hydra_{dataset_name.lower()}_test.json"

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
    parser = argparse.ArgumentParser(description='Train and save HYDRA models')
    parser.add_argument('dataset', type=str,
                        help='Dataset name (InsectSound, MosquitoSound, or FruitFlies)')
    parser.add_argument('--num-kernels', type=int, default=512,
                        help='Number of kernels (default: 512)')

    args = parser.parse_args()

    success = train_and_save(args.dataset, args.num_kernels)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
