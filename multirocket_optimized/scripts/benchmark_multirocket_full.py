#!/usr/bin/env python3
"""
MultiRocket84 Full Dataset Benchmark
Tests on complete GunPoint test set (150 samples) for comparison with MiniRocket
"""

import numpy as np
import json
import time
from pathlib import Path
from sktime.datasets import load_UCR_UEA_dataset
import warnings
warnings.filterwarnings('ignore')

# Load the custom MultiRocket84 implementation
from custom_multirocket84 import MultiRocket84

def benchmark_gunpoint_full():
    """Benchmark MultiRocket84 on full GunPoint dataset"""
    print("=" * 80)
    print("MultiRocket84 Full GunPoint Benchmark")
    print("=" * 80)
    print()

    # Load GunPoint dataset
    print("Loading GunPoint dataset...")
    X_train, y_train = load_UCR_UEA_dataset("GunPoint", split="train", return_type="numpy3d")
    X_test, y_test = load_UCR_UEA_dataset("GunPoint", split="test", return_type="numpy3d")

    # Convert to 2D if needed (num_samples, time_steps)
    if X_train.ndim == 3:
        X_train = X_train.squeeze(1)
    if X_test.ndim == 3:
        X_test = X_test.squeeze(1)

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Time series length: {X_train.shape[1]}")
    print(f"Classes: {np.unique(y_train)}")
    print()

    # Initialize and train MultiRocket84
    print("=" * 80)
    print("Training MultiRocket84")
    print("=" * 80)
    print()

    mr = MultiRocket84(max_dilations_per_kernel=8)

    # Extract features and train classifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifierCV

    train_start = time.time()
    X_train_features = mr.fit_transform(X_train)
    X_test_features = mr.transform(X_test)

    scaler = StandardScaler()
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=5)

    X_train_scaled = scaler.fit_transform(X_train_features)
    classifier.fit(X_train_scaled, y_train)
    train_time = time.time() - train_start

    print(f"Training time: {train_time:.2f} seconds")
    print(f"Features: {X_train_features.shape[1]}")
    print()

    # Evaluate on training set
    print("=" * 80)
    print("Training Set Evaluation")
    print("=" * 80)
    print()

    train_pred_start = time.time()
    y_train_pred = classifier.predict(X_train_scaled)
    train_pred_time = time.time() - train_pred_start

    train_accuracy = np.mean(y_train_pred == y_train)
    print(f"Training accuracy: {train_accuracy*100:.2f}% ({np.sum(y_train_pred == y_train)}/{len(y_train)})")
    print(f"Training prediction time: {train_pred_time:.2f} seconds")
    print(f"Average per-sample: {train_pred_time/len(y_train)*1000:.2f} ms")
    print()

    # Evaluate on test set
    print("=" * 80)
    print("Test Set Evaluation")
    print("=" * 80)
    print()

    X_test_scaled = scaler.transform(X_test_features)

    test_pred_start = time.time()
    y_test_pred = classifier.predict(X_test_scaled)
    test_pred_time = time.time() - test_pred_start

    test_accuracy = np.mean(y_test_pred == y_test)
    print(f"Test accuracy: {test_accuracy*100:.2f}% ({np.sum(y_test_pred == y_test)}/{len(y_test)})")
    print(f"Test prediction time: {test_pred_time:.2f} seconds")
    print(f"Average per-sample: {test_pred_time/len(y_test)*1000:.2f} ms")
    print()

    # Per-class analysis
    print("=" * 80)
    print("Per-Class Test Accuracy")
    print("=" * 80)
    print()

    for cls in np.unique(y_test):
        cls_mask = y_test == cls
        cls_acc = np.mean(y_test_pred[cls_mask] == y_test[cls_mask])
        cls_count = np.sum(cls_mask)
        cls_correct = np.sum((y_test_pred[cls_mask] == y_test[cls_mask]))
        print(f"Class {cls}: {cls_acc*100:.2f}% ({cls_correct}/{cls_count} samples)")
    print()

    # Confusion matrix
    print("=" * 80)
    print("Confusion Matrix")
    print("=" * 80)
    print()

    classes = np.unique(y_test)
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            confusion[i, j] = np.sum((y_test == true_cls) & (y_test_pred == pred_cls))

    print("        Predicted")
    print(f"        {' '.join([f'{c:>6}' for c in classes])}")
    for i, true_cls in enumerate(classes):
        print(f"True {true_cls:>2} {' '.join([f'{confusion[i, j]:>6}' for j in range(len(classes))])}")
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print(f"Model: MultiRocket84")
    print(f"Kernels: 84")
    print(f"Features: {mr.num_features}")
    print(f"Dilations (orig): {len(mr.dilations_orig)}")
    print(f"Training time: {train_time:.2f}s")
    print(f"Train accuracy: {train_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    print(f"Test inference time: {test_pred_time/len(y_test)*1000:.2f} ms/sample")
    print()

    # Save results
    results = {
        'model': 'MultiRocket84',
        'dataset': 'GunPoint',
        'num_kernels': 84,  # Fixed at 84
        'num_features': int(mr.num_features),
        'num_dilations': len(mr.dilations_orig),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'time_series_length': int(X_train.shape[1]),
        'train_time_sec': float(train_time),
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'test_prediction_time_sec': float(test_pred_time),
        'test_ms_per_sample': float(test_pred_time/len(y_test)*1000),
        'confusion_matrix': confusion.tolist(),
        'classes': classes.tolist()
    }

    with open('multirocket_gunpoint_full_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to: multirocket_gunpoint_full_results.json")
    print()

    return results

if __name__ == "__main__":
    results = benchmark_gunpoint_full()
