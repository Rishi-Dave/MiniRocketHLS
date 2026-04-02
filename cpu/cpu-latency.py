#!/usr/bin/env python3

import numpy as np
import json
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import os
from itertools import combinations
import time
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based import MiniRocket
from aeon.classification.convolution_based import HydraClassifier

# sktime datasets import will be done locally in functions

def load_real_dataset(dataset_name='ArrowHead'):
    """Load real time series datasets from sktime"""
    print(f"Attempting to load dataset: {dataset_name} from UCRArchive")
    
    from aeon.datasets import load_classification, load_from_arff_file
    from pathlib import Path
    
    if dataset_name == 'ArrowHead':
        X_train, y_train = load_classification(dataset_name, split="train")
        X_test, y_test = load_classification(dataset_name, split="test")
    else:
        X_train, y_train = load_from_arff_file(os.path.join(Path.home(), f".local/lib/python3.10/site-packages/aeon/datasets/local_data/{dataset_name}/{dataset_name}/{dataset_name}_TRAIN.arff"))
        X_test, y_test = load_from_arff_file(os.path.join(Path.home(), f".local/lib/python3.10/site-packages/aeon/datasets/local_data/{dataset_name}/{dataset_name}/{dataset_name}_TEST.arff"))

    X_train = np.squeeze(X_train)
    X_test = np.squeeze(X_test)
    unique_labels_train = sorted(set(y_train))
    unique_labels_test = sorted(set(y_test))
    label_to_int_train = {label: i for i, label in enumerate(unique_labels_train)}
    label_to_int_test = {label: i for i, label in enumerate(unique_labels_test)}
    y_numpy_train = np.array([label_to_int_train[label] for label in y_train])
    y_numpy_test = np.array([label_to_int_test[label] for label in y_test])


    print(f"Real dataset loaded: {dataset_name}")
    print(f"  Classes: {len(np.unique(y_numpy_train))} ({np.unique(y_numpy_train)})")
    print(f"  X-train Shape: {X_train.shape}")
    print(f"  y-train Shape: {y_numpy_train.shape}")
    print(f"  X-test Shape: {X_test.shape}")
    print(f"  y-test Shape: {y_numpy_test.shape}")
    print(f"  distribution: {np.bincount(y_numpy_train.astype(int))}")
    
    return X_train.astype(np.float32), y_numpy_train.astype(int), X_test.astype(np.float32), y_numpy_test.astype(int)


def main():
    parser = argparse.ArgumentParser(description='Train Rocket model')
    parser.add_argument('--dataset', type=str, default='ArrowHead', choices=['ArrowHead', 'InsectSound', 'FruitFlies', 'MosquitoSound', 'GunPoint'], help='Dataset to use from UCR Archive' )
    parser.add_argument('--model', type=str, default='MiniRocket', choices=['MiniRocket', 'MultiRocket', 'Hydra'], help='Rocket-based model to use' )
    parser.add_argument('--per-sample', action='store_true', help='Measure per-sample latency')
    parser.add_argument('--output-csv', type=str, default=None, help='Output CSV path for per-sample results')
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_real_dataset(args.dataset)

    print(f"Training {args.model}...")
    if (args.model == 'MultiRocket'):
        rocket = MultiRocket()
    elif (args.model == 'Hydra'):
        rocket = HydraClassifier()
    else:
        rocket = MiniRocket()

    X_train_features = rocket.fit_transform(X_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    # Train classifier
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_scaled, y_train)

    if args.per_sample:
        print("Evaluating model (per-sample timing)...")
        latencies_ms = []
        predictions = []
        n_test = len(X_test)
        for i in range(n_test):
            sample = X_test[i:i+1]
            t0 = time.perf_counter()
            feat = rocket.transform(sample)
            feat_scaled = scaler.transform(feat)
            pred = classifier.predict(feat_scaled)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000)
            predictions.append(int(pred[0]))
            if (i + 1) % 1000 == 0:
                print(f"  Sample {i+1}/{n_test}...")

        predictions = np.array(predictions)
        accuracy = accuracy_score(y_test, predictions)
        correct = np.sum(predictions == y_test)
        lat = np.array(latencies_ms)

        print(f"\nAccuracy: {accuracy:.6f} ({correct}/{n_test})")
        print(f"Throughput: {1000.0 / np.mean(lat):.1f} inferences/sec")
        print(f"\nLatency distribution (ms):")
        print(f"  Mean:  {np.mean(lat):.3f}")
        print(f"  P50:   {np.percentile(lat, 50):.3f}")
        print(f"  P95:   {np.percentile(lat, 95):.3f}")
        print(f"  P99:   {np.percentile(lat, 99):.3f}")
        print(f"  Min:   {np.min(lat):.3f}")
        print(f"  Max:   {np.max(lat):.3f}")
        print(f"  Std:   {np.std(lat):.3f}")

        # Write CSV
        csv_path = args.output_csv
        if csv_path is None:
            csv_path = f"../results/{args.model}_CPU_python_{args.dataset}_per_sample.csv"
        import csv
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "total_ms", "predicted", "actual", "correct"])
            for i in range(n_test):
                writer.writerow([i, f"{latencies_ms[i]:.3f}", predictions[i],
                                 int(y_test[i]), int(predictions[i] == y_test[i])])
        print(f"\nPer-sample CSV written to: {csv_path}")
    else:
        print("Evaluating model...")
        start_time = time.perf_counter()
        X_test_features = rocket.transform(X_test)
        X_test_scaled = scaler.transform(X_test_features)
        y_pred = classifier.predict(X_test_scaled)
        end_time = time.perf_counter()
        accuracy = accuracy_score(y_test, y_pred)
        total_time = end_time - start_time
        correct = np.sum(y_pred == y_test)
        n_test = len(y_test)

        print(f"Accuracy: {accuracy:.6f} ({correct}/{n_test})")
        print(f"Throughput: {n_test / total_time:.1f} inferences/sec")
        print(f"Latency per sample: {total_time * 1000 / n_test:.3f} ms")
        print(f"Test time: {total_time:.6f} seconds")


if __name__ == "__main__":
    main()