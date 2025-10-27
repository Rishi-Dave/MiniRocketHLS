#!/usr/bin/env python3

import numpy as np
import json
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import os

# sktime datasets import will be done locally in functions

class MiniRocket:
    """MiniRocket implementation for time series classification"""
    
    def __init__(self, num_kernels=84, max_dilations_per_kernel=32):
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.kernels = None
        self.dilations = None
        self.num_features_per_dilation = None
        self.biases = None
        
    def _get_kernel_combinations(self, input_length):
        """Generate the 84 fixed kernel index combinations"""
        # MiniRocket uses fixed combinations of 3 indices from positions 0-8 in a kernel of length 9
        kernel_indices = []
        
        # Generate all combinations of 3 indices from 0 to 8
        from itertools import combinations
        for combo in combinations(range(9), 3):
            kernel_indices.append(list(combo))
            if len(kernel_indices) >= self.num_kernels:
                break
                
        return np.array(kernel_indices[:self.num_kernels])
    
    def _get_dilations(self, input_length):
        """Generate exponentially increasing dilations"""
        num_dilations = int(np.log2(input_length)) - 2
        num_dilations = max(1, min(num_dilations, 8))  # Limit to max 8 dilations
        
        dilations = np.array([2**i for i in range(num_dilations)], dtype=np.int32)
        return dilations
    
    def _generate_biases(self, X, dilations, kernel_indices):
        """Generate biases based on training data statistics"""
        biases = []
        
        for dilation in dilations:
            for kernel_idx in kernel_indices:
                # Compute convolution for this kernel/dilation combo
                convolutions = self._apply_kernel(X, kernel_idx, dilation)
                
                # Quantiles for PPV (positive proportion of values)
                if convolutions.size > 0:
                    bias = np.quantile(convolutions, 0.9)  # Use 90th percentile
                else:
                    bias = 0.0
                    
                biases.append(bias)
                
        return np.array(biases, dtype=np.float32)
    
    def _apply_kernel(self, X, kernel_indices, dilation):
        """Apply a single kernel with given dilation to all time series"""
        n_samples, length = X.shape
        kernel_length = 9
        
        # Calculate output length after convolution
        output_length = (length - (kernel_length - 1) * dilation)
        if output_length <= 0:
            return np.array([])
            
        results = []
        
        for i in range(n_samples):
            convolution_result = []
            for j in range(output_length):
                value = 0.0
                for k, idx in enumerate(kernel_indices):
                    pos = j + idx * dilation
                    if pos < length:
                        # Use simple weights: -1, 0, 1 pattern
                        weight = -1 if k == 0 else (1 if k == 2 else 0)
                        value += X[i, pos] * weight
                convolution_result.append(value)
            results.extend(convolution_result)
            
        return np.array(results, dtype=np.float32)
    
    def fit_transform(self, X):
        """Fit MiniRocket and transform the data"""
        n_samples, length = X.shape
        
        # Store time series length
        self.time_series_length = length
        
        # Generate fixed components
        self.kernel_indices = self._get_kernel_combinations(length)
        self.dilations = self._get_dilations(length)
        self.biases = self._generate_biases(X, self.dilations, self.kernel_indices)
        
        # Count features per dilation
        self.num_features_per_dilation = np.array([len(self.kernel_indices)] * len(self.dilations), dtype=np.int32)
        
        # Transform the data
        return self.transform(X)
    
    def transform(self, X):
        """Transform time series to MiniRocket features"""
        n_samples, length = X.shape
        total_features = len(self.dilations) * len(self.kernel_indices)
        
        features = np.zeros((n_samples, total_features), dtype=np.float32)
        
        feature_idx = 0
        for dil_idx, dilation in enumerate(self.dilations):
            for kernel_idx in self.kernel_indices:
                # Apply kernel
                convolutions = self._apply_kernel_single_series(X, kernel_idx, dilation)
                
                # Apply bias and get positive proportion
                bias = self.biases[feature_idx]
                ppv_features = np.mean(convolutions > bias, axis=1)  # Positive proportion per sample
                
                features[:, feature_idx] = ppv_features
                feature_idx += 1
                
        return features
    
    def _apply_kernel_single_series(self, X, kernel_indices, dilation):
        """Apply kernel to time series returning per-sample results"""
        n_samples, length = X.shape
        kernel_length = 9
        
        output_length = (length - (kernel_length - 1) * dilation)
        if output_length <= 0:
            return np.zeros((n_samples, 1))
            
        results = np.zeros((n_samples, output_length))
        
        for i in range(n_samples):
            for j in range(output_length):
                value = 0.0
                for k, idx in enumerate(kernel_indices):
                    pos = j + idx * dilation
                    if pos < length:
                        weight = -1 if k == 0 else (1 if k == 2 else 0)
                        value += X[i, pos] * weight
                results[i, j] = value
                
        return results

def load_real_dataset(dataset_name='arrow_head'):
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

def generate_sample_data(n_samples=1000, length=128, n_classes=4):
    """Generate sample time series data for testing"""
    np.random.seed(42)
    
    X = np.zeros((n_samples, length))
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        class_id = i % n_classes
        
        # Generate different patterns for different classes
        if class_id == 0:
            # Sine wave
            X[i] = np.sin(np.linspace(0, 4*np.pi, length)) + 0.1 * np.random.randn(length)
        elif class_id == 1:
            # Square wave
            X[i] = np.sign(np.sin(np.linspace(0, 4*np.pi, length))) + 0.1 * np.random.randn(length)
        elif class_id == 2:
            # Random walk
            X[i] = np.cumsum(0.1 * np.random.randn(length))
        else:
            # Exponential decay
            X[i] = np.exp(-np.linspace(0, 3, length)) + 0.1 * np.random.randn(length)
            
        y[i] = class_id
    
    return X.astype(np.float32), y

def save_model_parameters(minirocket, scaler, classifier, filename="minirocket_model.json"):
    """Save all model parameters needed for C++ implementation"""
    
    # Prepare data for JSON serialization
    model_data = {
        "num_kernels": int(minirocket.num_kernels),
        "num_dilations": len(minirocket.dilations),
        "num_features": len(minirocket.biases),
        "num_classes": len(classifier.classes_),
        "time_series_length": minirocket.time_series_length,
        "kernel_indices": minirocket.kernel_indices.tolist(),
        "dilations": minirocket.dilations.tolist(),
        "num_features_per_dilation": minirocket.num_features_per_dilation.tolist(),
        "biases": minirocket.biases.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classifier_coef": classifier.coef_.tolist(),
        "classifier_intercept": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist()
    }
    
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
    parser.add_argument('--output', type=str, default='minirocket_model.json', help='Output file')
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
    # Fit MiniRocket
    minirocket = MiniRocket(num_kernels=84)
    X_train_features = minirocket.fit_transform(X_train)
    X_test_features = minirocket.transform(X_test)
    
    print(f"Feature shape: {X_train_features.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)
    
    # Train classifier
    print("Training classifier...")
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_scaled, y_train)
    
    # Test accuracy
    y_pred = classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Save model
    model_data = save_model_parameters(minirocket, scaler, classifier, args.output)
    
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