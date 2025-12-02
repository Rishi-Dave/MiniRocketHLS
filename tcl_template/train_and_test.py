#!/usr/bin/env python3
"""
Train MiniRocket model on UCR dataset and export for FPGA testing.
This enables meaningful accuracy comparison between CPU and FPGA.
"""

import numpy as np
import json
from aeon.datasets import load_classification
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_minirocket(dataset_name, max_dilations=5):
    """Train MiniRocket model on a UCR dataset."""
    print(f"Loading {dataset_name}...")
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")

    # Squeeze to remove channel dimension
    X_train = X_train.squeeze()
    X_test = X_test.squeeze()

    # Ensure 2D arrays
    if X_train.ndim == 1:
        X_train = X_train.reshape(1, -1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Normalize class labels to 0-indexed
    unique_classes = np.unique(np.concatenate([y_train, y_test]))
    class_map = {c: i for i, c in enumerate(unique_classes)}
    y_train_mapped = np.array([class_map[c] for c in y_train])
    y_test_mapped = np.array([class_map[c] for c in y_test])

    num_classes = len(unique_classes)
    ts_length = X_train.shape[1]

    print(f"  Classes: {num_classes}, Time series length: {ts_length}")

    # Create MiniRocket transform (limited features for FPGA)
    # MiniRocket typically generates 84 * num_dilations * 2 features
    # We'll use fewer dilations to keep features manageable
    minirocket = MiniRocket(n_jobs=-1, random_state=42)

    print("Training MiniRocket transform...")
    minirocket.fit(X_train.reshape(X_train.shape[0], 1, -1))

    print("Transforming data...")
    X_train_transformed = minirocket.transform(X_train.reshape(X_train.shape[0], 1, -1))
    X_test_transformed = minirocket.transform(X_test.reshape(X_test.shape[0], 1, -1))

    num_features = X_train_transformed.shape[1]
    print(f"  Generated {num_features} features")

    # Train classifier with scaling
    print("Training classifier...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_transformed)
    X_test_scaled = scaler.transform(X_test_transformed)

    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_scaled, y_train_mapped)

    # Evaluate
    train_acc = classifier.score(X_train_scaled, y_train_mapped)
    test_acc = classifier.score(X_test_scaled, y_test_mapped)

    print(f"\nResults for {dataset_name}:")
    print(f"  Train accuracy: {train_acc*100:.2f}%")
    print(f"  Test accuracy:  {test_acc*100:.2f}%")

    return {
        "dataset_name": dataset_name,
        "num_classes": num_classes,
        "num_features": num_features,
        "time_series_length": ts_length,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train_mapped,
        "y_test": y_test_mapped,
        "transform": minirocket,
        "scaler": scaler,
        "classifier": classifier,
        "class_map": class_map,
    }


def export_model_for_fpga(model_data, output_prefix):
    """Export trained model to JSON for FPGA."""
    # Get dilations and biases from MiniRocket
    transform = model_data["transform"]
    scaler = model_data["scaler"]
    classifier = model_data["classifier"]

    # For simplicity, export the key parameters
    # Note: Full MiniRocket export requires extracting internal parameters
    # which depends on the specific aeon version

    model_json = {
        "dataset_name": model_data["dataset_name"],
        "num_features": model_data["num_features"],
        "num_classes": model_data["num_classes"],
        "time_series_length": model_data["time_series_length"],
        "train_accuracy": model_data["train_acc"],
        "test_accuracy": model_data["test_acc"],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classifier_coef": classifier.coef_.flatten().tolist(),
        "classifier_intercept": classifier.intercept_.tolist() if hasattr(classifier.intercept_, 'tolist') else [classifier.intercept_],
    }

    # Export test data for FPGA verification
    test_data = {
        "dataset_name": model_data["dataset_name"],
        "num_samples": len(model_data["X_test"]),
        "time_series_length": model_data["time_series_length"],
        "num_classes": model_data["num_classes"],
        "samples": model_data["X_test"].tolist(),
        "labels": model_data["y_test"].tolist(),
    }

    with open(f"{output_prefix}_model_info.json", 'w') as f:
        json.dump(model_json, f, indent=2)

    with open(f"{output_prefix}_test_data.json", 'w') as f:
        json.dump(test_data, f)

    print(f"Exported model info to {output_prefix}_model_info.json")
    print(f"Exported test data to {output_prefix}_test_data.json")


if __name__ == "__main__":
    # Test on multiple datasets
    datasets = ["CBF", "ECG200", "GunPoint"]
    results = []

    print("=" * 60)
    print("MiniRocket Training and Accuracy Evaluation")
    print("=" * 60)

    for ds in datasets:
        try:
            model_data = train_minirocket(ds)
            results.append({
                "dataset": ds,
                "train_acc": model_data["train_acc"],
                "test_acc": model_data["test_acc"],
                "features": model_data["num_features"],
                "classes": model_data["num_classes"],
            })
        except Exception as e:
            print(f"Error with {ds}: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY - MiniRocket CPU Accuracy")
    print("=" * 60)
    print(f"{'Dataset':<20} {'Train':<10} {'Test':<10} {'Features':<10} {'Classes'}")
    print("-" * 60)
    for r in results:
        print(f"{r['dataset']:<20} {r['train_acc']*100:>6.2f}%   {r['test_acc']*100:>6.2f}%   {r['features']:<10} {r['classes']}")

    print("\nNote: FPGA accuracy should match CPU accuracy when using the same")
    print("trained model, demonstrating functional correctness of the HLS implementation.")
