#!/usr/bin/env python3
"""
Train MultiRocket84 models on the 3 benchmark datasets and export for C++ CPU baseline.
Uses custom_multirocket84.py which matches the FPGA kernel structure.

Usage: python3 train_multirocket84_benchmarks.py [--dataset InsectSound] [--all]
"""

import sys
import os
import json
import time
import numpy as np

# Add parent directory for custom_multirocket84 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom_multirocket84 import MultiRocket84

from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from aeon.datasets import load_classification

DATASETS = ['InsectSound', 'MosquitoSound', 'FruitFlies']
DATA_PATH = "../../hydra_optimized/datasets/ucr_data"
MODEL_DIR = "../models"


def train_and_export(dataset_name):
    print(f"\n{'='*70}")
    print(f"Training MultiRocket84 on {dataset_name}")
    print(f"{'='*70}")

    X_train, y_train = load_classification(dataset_name, split="train", extract_path=DATA_PATH)
    X_test, y_test = load_classification(dataset_name, split="test", extract_path=DATA_PATH)

    if X_train.ndim == 3:
        X_train = X_train.squeeze(1)
    if X_test.ndim == 3:
        X_test = X_test.squeeze(1)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {len(np.unique(y_train))}")

    t0 = time.time()
    mr = MultiRocket84(random_state=42)
    mr.fit(X_train)

    X_train_feat = mr.transform(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)

    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(X_train_scaled, y_train)
    train_time = time.time() - t0

    X_test_feat = mr.transform(X_test)
    X_test_scaled = scaler.transform(X_test_feat)
    y_pred = clf.predict(X_test_scaled)
    test_acc = np.mean(y_pred == y_test)
    train_acc = clf.score(X_train_scaled, y_train)

    print(f"  Train accuracy: {train_acc*100:.2f}%")
    print(f"  Test accuracy: {test_acc*100:.2f}%")
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Features: {mr.num_features}")

    # Export model
    model_file = os.path.join(MODEL_DIR, f"multirocket84_{dataset_name.lower()}_model.json")
    mr.export_for_fpga(scaler, clf, model_file)

    # Add time_series_length to model file
    with open(model_file) as f:
        model_data = json.load(f)
    model_data['time_series_length'] = int(X_test.shape[1])
    with open(model_file, 'w') as f:
        json.dump(model_data, f, indent=2)

    # Export test data (limit to 500 samples for reasonable JSON size)
    max_samples = min(500, len(X_test))
    test_file = os.path.join(MODEL_DIR, f"multirocket84_{dataset_name.lower()}_test.json")
    test_data = {
        "dataset": dataset_name,
        "num_samples": max_samples,
        "time_series_length": int(X_test.shape[1]),
        "num_classes": int(len(np.unique(y_test))),
        "time_series": X_test[:max_samples].tolist(),
        "labels": y_test[:max_samples].tolist(),
        "test_accuracy": float(test_acc)
    }
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    print(f"  Model: {model_file}")
    print(f"  Test data: {test_file} ({max_samples} samples)")

    return test_acc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='+', help='Dataset name(s)')
    parser.add_argument('--all', action='store_true', help='Train all 3 datasets')
    args = parser.parse_args()

    datasets = DATASETS if args.all else (args.dataset or DATASETS)

    for ds in datasets:
        train_and_export(ds)
