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

from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket
from aeon.datasets import load_arrow_head

import mmap
import struct
import posix_ipc
import time

def load_real_dataset(dataset_name='InsectSound'):
    """Load real time series datasets from sktime"""
    try:
        print(f"Attempting to load dataset: {dataset_name} from UCRArchive")
        
        from aeon.datasets import load_classification, load_from_arff_file
        from pathlib import Path
        
        X_train, y_train = load_from_arff_file(os.path.join(Path.home(), f"/home/pyuva001/MiniRocketHLS/benchmarks/{dataset_name}/{dataset_name}_TRAIN.arff"))
        X_test, y_test = load_from_arff_file(os.path.join(Path.home(), f"/home/pyuva001/MiniRocketHLS/benchmarks/{dataset_name}/{dataset_name}_TEST.arff"))

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
        
    except ImportError as e:
        print(f"sktime import error: {e}")
        print("falling back to synthetic data")
        return generate_sample_data()
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}. Using local file instead.")
        X_train, y_train = load_classification("ArrowHead", split="train")  
        X_test, y_test = load_classification("ArrowHead", split="test")
        return X_train.astype(np.float32), y_train.astype(int), X_test.astype(np.float32), y_test.astype(int)


def save_model_parameters(minirocket, scaler, classifier, time_series_length, filename="minirocket_model.json"):
    """Save all model parameters needed for C++ implementation"""

    # Generate the 84 kernel indices (combinations of 3 from 0-8)
    kernel_indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)

    # Prepare data for JSON serialization

    model_data = {
        "num_kernels": minirocket.n_kernels,  # Always 84 in MiniRocket
        "num_dilations": len(minirocket.parameters[2]),
        "num_features": len(minirocket.parameters[4]),
        "time_series_length": time_series_length,
        "dilations": minirocket.parameters[2].tolist(),
        "num_features_per_dilation": minirocket.parameters[3].tolist(),
        "biases": minirocket.parameters[4].tolist(),

        "kernel_indices": kernel_indices.tolist(),

        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),

        "num_classes": len(classifier.classes_),
        "classifier_coef": classifier.coef_.tolist(),
        "classifier_intercept": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist()
    }

    with open(filename, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"Model parameters saved to {filename}")
    return model_data


shm = posix_ipc.SharedMemory("/dpdk_shm")
mapfile = mmap.mmap(shm.fd, shm.size)
shm.close_fd()

def send_to_dpdk(value):
    """Write float to DPDK, set ready flag."""
    mapfile.seek(0)
    # Wait if previous value hasn't been consumed
    while struct.unpack('i', mapfile.read(4))[0] == 1:
        time.sleep(0.001)
        mapfile.seek(0)
    # Write new value
    mapfile.seek(0)
    mapfile.write(struct.pack('f', value))
    mapfile.write(struct.pack('i', 1))  # ready_to_dpdk = 1

def wait_for_reply():
    """Block until DPDK writes a reply."""
    while True:
        mapfile.seek(8)  # offset of from_dpdk
        ready = struct.unpack('i', mapfile.read(4))[0]
        if ready == 1:
            mapfile.seek(8)
            value = struct.unpack('f', mapfile.read(4))[0]
            # Mark consumed
            mapfile.seek(12)
            mapfile.write(struct.pack('i', 0))
            return value
        time.sleep(0.001)  # small sleep to avoid 100% CPU



def main():
    parser = argparse.ArgumentParser(description='Train MiniRocket model')
    parser.add_argument('--dataset', type=str, default='ArrowHead', help='Dataset to use (arrow_head, gun_point, italy_power, or synthetic)')
    parser.add_argument('--model', type=str, default='MiniRocket', help='Model to use (MiniRocket or MultiRocket)')
    args = parser.parse_args()
    

    rocket = None

    if args.model == 'MiniRocket':
        rocket = MiniRocket(n_kernels=840, random_state=42, n_jobs=-1)
    elif args.model == 'MultiRocket':
        rocket = MultiRocket(n_kernels=840, random_state=42, n_jobs=-1)
    else:
        print(f"Unknown model: {args.model}. Defaulting to MiniRocket.")
        rocket = MiniRocket(n_kernels=840, random_state=42, n_jobs=-1)

    print(f"Loading and training on dataset: {args.dataset}")
    X_train,  y_train, X_test, y_test = load_real_dataset(args.dataset)
    X_train_features = rocket.fit_transform(X_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)

    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_scaled, y_train)
    
    window = np.zeros((128,), dtype=np.float32)  # sliding windo
    window = window.reshape(1, -1)

    while True:

        value = wait_for_reply()
        window = np.roll(window, -1, axis=1)
        window[0, -1] = value

        window_transform = rocket.transform(window)
        window_scaled = scaler.transform(window_transform)
        y_pred = classifier.predict(window_scaled)

        print(f"Predicted class: {y_pred[0]}")
        
        send_to_dpdk(float(y_pred[0]))  # Send predicted class back to DPDK

if __name__ == "__main__":
    main()