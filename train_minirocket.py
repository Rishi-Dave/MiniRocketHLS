#!/usr/bin/env python3

import numpy as np
import json
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import os

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
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--length', type=int, default=128, help='Time series length')
    parser.add_argument('--classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--output', type=str, default='minirocket_model.json', help='Output file')
    args = parser.parse_args()
    
    print("Generating sample data...")
    X, y = generate_sample_data(args.samples, args.length, args.classes)
    
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
    
    # Save test data for C++ verification
    test_data = {
        "X_test": X_test.tolist(),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist()
    }
    
    test_filename = args.output.replace('.json', '_test_data.json')
    with open(test_filename, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Test data saved to {test_filename}")
    print("Training complete!")

if __name__ == "__main__":
    main()