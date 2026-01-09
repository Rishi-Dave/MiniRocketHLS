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
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based import MiniRocket

def load_real_dataset(dataset_name='InsectSound'):
    """Load real time series datasets from sktime"""
    try:
        print(f"Attempting to load dataset: {dataset_name} from UCRArchive")
        
        from aeon.datasets import load_classification, load_from_arff_file
        from pathlib import Path
        
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
        
    except ImportError as e:
        print(f"sktime import error: {e}")
        print("falling back to synthetic data")
        return generate_sample_data()
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}. Using local file instead.")
        from pathlib import Path
        X, y = load_from_arff_file(os.path.join(Path.home(), f".local/lib/python3.10/site-packages/aeon/datasets/local_data//{dataset_name}/{dataset_name}/{dataset_name}_TRAIN.arff"))
        X = np.squeeze(X)
        unique_labels = sorted(set(y))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        y_numpy = np.array([label_to_int[label] for label in y])


        print(f"Real dataset loaded: {dataset_name}")
        print(f"  Classes: {len(np.unique(y))} ({np.unique(y)})")
        print(f"  X Shape: {X.shape}")
        print(f"  y Shape: {y.shape}")
        print(f"  distribution: {np.bincount(y_numpy.astype(int))}")
        return X.astype(np.float32), y_numpy.astype(int)

def save_model_parameters(rocket, scaler, classifier, time_series_length, filename="minirocket_model.json"):
    """Save all model parameters needed for C++ implementation"""

    # Generate the 84 kernel indices (combinations of 3 from 0-8)
    kernel_indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)

    # Prepare data for JSON serialization
    model_data = {
        "num_kernels": rocket.n_kernels,  # Always 84 in MultiRocket implementation
        "num_classes": len(classifier.classes_),
        "time_series_length": time_series_length,
        "n_feature_per_kernel": rocket.n_features_per_kernel,

        "dilations_0": rocket.parameter[0].tolist(),
        "num_features_per_dilation_0": rocket.parameter[1].tolist(),
        "biases_0": rocket.parameter[2].tolist(),
        "num_dilations_0": len(rocket.parameter[0]),
        "num_features_0": len(rocket.parameter[2]), 

        "dilations_1": rocket.parameter1[0].tolist(),
        "num_features_per_dilation_1": rocket.parameter1[1].tolist(),
        "biases_1": rocket.parameter1[2].tolist(),
        "num_dilations_1": len(rocket.parameter1[0]),
        "num_features_1": len(rocket.parameter1[2]),

        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),

        "classifier_coef": classifier.coef_.tolist(),
        "classifier_intercept": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist()
    }

    print(f"Mean and Scale length {len(scaler.mean_.tolist())}...")

    with open(filename, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"Model parameters saved to {filename}")
    return model_data

def main():
    parser = argparse.ArgumentParser(description='Train MiniRocket model')
    parser.add_argument('--dataset', type=str, default='InsectSound')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples (for synthetic data)')
    parser.add_argument('--length', type=int, default=128, help='Time series length (for synthetic data)')
    parser.add_argument('--classes', type=int, default=4, help='Number of classes (for synthetic data)')
    parser.add_argument('--output', type=str, default='multirocket_model.json', help='Output file')
    args = parser.parse_args()
    
    
    # Split data
    X_train, y_train, X_test, y_test = load_real_dataset(args.dataset)
    
    print("Training MiniRocket...")
    # Fit MiniRocket (num_kernels must be multiple of 84, using 840 to fit HLS limits)
    rocket = MultiRocket()
    X_train_features = rocket.fit_transform(X_train)
    X_test_features = rocket.transform(X_test)
    
    print(f"Feature shape: {X_train_features.shape}")
    

    print(X_test_features.shape)
    print(X_test_features[0][:20])  # Print first 20 features of first test sample
    print(np.min(X_test_features), np.max(X_test_features))

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)
    
    print(X_test_scaled.shape)
    print(X_test_scaled)
    print(np.min(X_test_scaled), np.max(X_test_scaled))

    # Train classifier
    print("Training classifier...")
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_scaled, y_train)
    
    # Test accuracy
    y_pred = classifier.predict(X_test_scaled)

    print(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Save model
    model_data = save_model_parameters(rocket, scaler, classifier, len(X_train[0]), args.output)
    
    # Save test data for C++ verification with dataset info
    test_data = {
        "dataset_name": args.dataset,
        "X_test": X_test.tolist(),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "test_accuracy": float(accuracy),
        "num_samples": len(X_test),
        "series_length": X_test.shape[1] if len(X_test.shape) > 1 else len(X_test[0]),
        "num_classes": len(np.unique(y_train))
    }
    
    test_filename = args.output.replace('.json', '_test_data.json')
    with open(test_filename, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Test data saved to {test_filename}")
    print(f"Python baseline accuracy: {accuracy:.4f}")
    print("Training complete!")

if __name__ == "__main__":
    main()