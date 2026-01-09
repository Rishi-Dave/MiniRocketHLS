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

def load_real_dataset(dataset_name='italy_power'):
    """Load real time series datasets from sktime"""
    try:
        print(f"Attempting to import sktime for dataset: {dataset_name}")
        from sktime.datasets import load_arrow_head, load_gunpoint, load_italy_power_demand
        print("sktime import successful")
        
        if dataset_name == 'arrow_head':
            X, y = load_arrow_head()
        elif dataset_name == 'gun_point':
            X, y = load_gunpoint()
        elif dataset_name == 'italy_power':
            X, y = load_italy_power_demand()
        else:
            print(f"Unknown dataset {dataset_name}, using arrow_head")
            X, y = load_arrow_head()
        
        # Convert from pandas to numpy
        X_numpy = np.array([series.values for series in X.iloc[:, 0]])
        
        # Convert string labels to integers
        unique_labels = sorted(set(y))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        y_numpy = np.array([label_to_int[label] for label in y])
        
        print(f"Real dataset loaded: {dataset_name}")
        print(f"  Shape: {X_numpy.shape}")
        print(f"  Classes: {len(unique_labels)} ({unique_labels})")
        print(f"  Class distribution: {np.bincount(y_numpy)}")
        
        return X_numpy.astype(np.float32), y_numpy
        
    except ImportError as e:
        print(f"sktime import error: {e}")
        print("falling back to synthetic data")
        return generate_sample_data()
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        print("Falling back to synthetic data")
        import traceback
        traceback.print_exc()
        return generate_sample_data()


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
    parser.add_argument('--dataset', type=str, default='arrow_head', 
                       choices=['arrow_head', 'gun_point', 'italy_power', 'synthetic'],
                       help='Dataset to use (arrow_head, gun_point, italy_power, or synthetic)')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples (for synthetic data)')
    parser.add_argument('--length', type=int, default=128, help='Time series length (for synthetic data)')
    parser.add_argument('--classes', type=int, default=4, help='Number of classes (for synthetic data)')
    parser.add_argument('--output', type=str, default='multirocket_model.json', help='Output file')
    args = parser.parse_args()
    
    # Load data based on dataset choice
    if args.dataset == 'synthetic':
        print("Generating synthetic data...")
        X, y = generate_sample_data(args.samples, args.length, args.classes)
    else:
        print(f"Loading real dataset: {args.dataset}")
        X, y = load_real_dataset(args.dataset)
    
    print(f"Data shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
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
        "num_classes": len(np.unique(y))
    }
    
    test_filename = args.output.replace('.json', '_test_data.json')
    with open(test_filename, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Test data saved to {test_filename}")
    print(f"Python baseline accuracy: {accuracy:.4f}")
    print("Training complete!")

if __name__ == "__main__":
    main()