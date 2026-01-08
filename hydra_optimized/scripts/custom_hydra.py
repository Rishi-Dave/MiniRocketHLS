#!/usr/bin/env python3
"""
Custom HYDRA Implementation for FPGA

HYDRA (HYbrid Dictionary Representation Algorithm) is a fast time series classification
algorithm that uses dictionary-based convolutional kernels.

This implementation is designed to match the FPGA constraints:
- 512 fixed dictionary kernels
- 8 kernel groups
- 2 pooling operators per kernel (max + mean)
- Total features: 512 × 2 = 1,024

Reference:
Dempster, A., Schmidt, D. F., & Webb, G. I. (2023).
HYDRA: Competing convolutional kernels for fast and accurate time series classification.
Data Mining and Knowledge Discovery.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
import json


class Hydra:
    """
    HYDRA time series classifier optimized for FPGA implementation

    Uses 512 dictionary kernels organized into 8 groups.
    Each kernel extracts 2 features (max pooling + global mean).
    """

    def __init__(self, num_kernels=512, num_groups=8, kernel_size=9,
                 max_dilations=8, random_state=None):
        """
        Initialize HYDRA

        Args:
            num_kernels: Total number of dictionary kernels (default: 512)
            num_groups: Number of kernel groups (default: 8)
            kernel_size: Size of each kernel (default: 9)
            max_dilations: Maximum dilation factor (default: 8)
            random_state: Random seed for reproducibility
        """
        self.num_kernels = num_kernels
        self.num_groups = num_groups
        self.kernels_per_group = num_kernels // num_groups
        self.kernel_size = kernel_size
        self.max_dilations = max_dilations
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # Will be set during fit
        self.dictionary_ = None
        self.biases_ = None
        self.dilations_ = None
        self.group_assignments_ = None
        self.scaler_ = None
        self.classifier_ = None
        self.num_features_ = None
        self.classes_ = None

    def _initialize_dictionary(self, X):
        """
        Initialize dictionary kernels

        Uses random initialization with normalized kernels.
        For production, this could be replaced with k-means clustering
        or learned dictionary methods.
        """
        # Random dictionary initialization
        dictionary = np.random.randn(self.num_kernels, self.kernel_size)

        # Normalize each kernel
        for i in range(self.num_kernels):
            kernel = dictionary[i]
            kernel = kernel - kernel.mean()
            std = kernel.std()
            if std > 1e-8:
                kernel = kernel / std
            dictionary[i] = kernel

        return dictionary

    def _initialize_dilations(self):
        """
        Initialize dilation values for each kernel

        Uses powers of 2 from 1 to max_dilations
        """
        dilations = np.zeros(self.num_kernels, dtype=np.int32)

        for i in range(self.num_kernels):
            # Cycle through dilations 1, 2, 4, 8, ...
            dilation_idx = i % self.max_dilations
            dilations[i] = 2 ** min(dilation_idx, int(np.log2(self.max_dilations)))

        return dilations

    def _initialize_biases(self):
        """Initialize bias values (all zeros)"""
        return np.zeros(self.num_kernels, dtype=np.float32)

    def _initialize_group_assignments(self):
        """Assign kernels to groups"""
        group_assignments = np.zeros(self.num_kernels, dtype=np.int32)

        for i in range(self.num_kernels):
            group_assignments[i] = i // self.kernels_per_group

        return group_assignments

    def _apply_kernel(self, X, kernel, bias, dilation):
        """
        Apply single convolutional kernel to time series (vectorized)

        Args:
            X: Time series [length]
            kernel: Kernel weights [kernel_size]
            bias: Bias value (scalar)
            dilation: Dilation factor

        Returns:
            Convolution output [output_length]
        """
        length = len(X)
        kernel_span = (self.kernel_size - 1) * dilation + 1

        if length < kernel_span:
            return np.array([])

        output_length = length - kernel_span + 1

        # Vectorized convolution using strided indexing
        indices = np.arange(kernel_span)[::dilation]
        indices = indices[np.newaxis, :] + np.arange(output_length)[:, np.newaxis]
        windows = X[indices]
        output = np.dot(windows, kernel) + bias

        return output

    def _extract_features_single(self, x):
        """
        Extract features from single time series (optimized)

        For each kernel:
            1. Apply convolution
            2. Extract max pooling
            3. Extract global mean

        Returns:
            Feature vector [num_kernels × 2]
        """
        features = np.zeros(self.num_kernels * 2)

        # Group kernels by dilation for batch processing
        unique_dilations = np.unique(self.dilations_)

        for dilation in unique_dilations:
            # Find all kernels with this dilation
            kernel_indices = np.where(self.dilations_ == dilation)[0]

            # Calculate output dimensions for this dilation
            kernel_span = (self.kernel_size - 1) * dilation + 1
            if len(x) < kernel_span:
                continue

            output_length = len(x) - kernel_span + 1

            # Create strided view for all positions at once
            indices = np.arange(kernel_span)[::dilation]
            indices = indices[np.newaxis, :] + np.arange(output_length)[:, np.newaxis]
            windows = x[indices]  # [output_length, kernel_size]

            # Apply all kernels with this dilation at once
            for k in kernel_indices:
                conv_output = np.dot(windows, self.dictionary_[k]) + self.biases_[k]
                # Max pooling and mean
                features[k * 2] = np.max(conv_output)
                features[k * 2 + 1] = np.mean(conv_output)

        return features

    def transform(self, X, verbose=False):
        """
        Transform time series to feature space

        Args:
            X: Time series data [n_samples, time_series_length] or [n_samples]
            verbose: Print progress updates

        Returns:
            Features [n_samples, num_features]
        """
        if self.dictionary_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Handle 2D input
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        features = np.zeros((n_samples, self.num_features_))

        if verbose and n_samples > 100:
            print(f"  Transforming {n_samples} samples...")
            progress_interval = max(1, n_samples // 10)
        else:
            progress_interval = None

        for i in range(n_samples):
            features[i] = self._extract_features_single(X[i])
            if progress_interval and (i + 1) % progress_interval == 0:
                print(f"    Progress: {i+1}/{n_samples} ({100*(i+1)//n_samples}%)")

        return features

    def fit(self, X, y, verbose=False):
        """
        Fit HYDRA model to training data

        Args:
            X: Training time series [n_samples, time_series_length]
            y: Training labels [n_samples]
            verbose: Print progress updates

        Returns:
            self
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.classes_ = np.unique(y)

        # Initialize dictionary and parameters
        if verbose:
            print("  Initializing dictionary...")
        self.dictionary_ = self._initialize_dictionary(X)
        self.biases_ = self._initialize_biases()
        self.dilations_ = self._initialize_dilations()
        self.group_assignments_ = self._initialize_group_assignments()

        self.num_features_ = self.num_kernels * 2

        # Extract features
        if verbose:
            print("  Extracting features from training data...")
        X_transformed = self.transform(X, verbose=verbose)

        # Normalize features
        if verbose:
            print("  Normalizing features...")
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_transformed)

        # Train classifier
        if verbose:
            print("  Training Ridge classifier...")
        self.classifier_ = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.classifier_.fit(X_scaled, y)

        return self

    def predict(self, X):
        """
        Predict class labels

        Args:
            X: Test time series [n_samples, time_series_length]

        Returns:
            Predictions [n_samples]
        """
        X_transformed = self.transform(X)
        X_scaled = self.scaler_.transform(X_transformed)
        return self.classifier_.predict(X_scaled)

    def predict_proba(self, X):
        """
        Predict class probabilities (decision function scores)

        Args:
            X: Test time series [n_samples, time_series_length]

        Returns:
            Scores [n_samples, n_classes]
        """
        X_transformed = self.transform(X)
        X_scaled = self.scaler_.transform(X_transformed)
        return self.classifier_.decision_function(X_scaled)

    def score(self, X, y):
        """
        Compute classification accuracy

        Args:
            X: Test time series [n_samples, time_series_length]
            y: True labels [n_samples]

        Returns:
            Accuracy score
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def export_to_json(self, filename):
        """
        Export model parameters to JSON for FPGA

        Args:
            filename: Output JSON file path
        """
        if self.dictionary_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare model data
        model_data = {
            'num_kernels': int(self.num_kernels),
            'num_groups': int(self.num_groups),
            'kernel_size': int(self.kernel_size),
            'num_features': int(self.num_features_),
            'num_classes': len(self.classes_),

            # Dictionary kernels (flattened)
            'kernel_weights': self.dictionary_.flatten().tolist(),

            # Kernel parameters
            'biases': self.biases_.tolist(),
            'dilations': self.dilations_.tolist(),
            'group_assignments': self.group_assignments_.tolist(),

            # Scaler parameters
            'scaler_mean': self.scaler_.mean_.tolist(),
            'scaler_scale': self.scaler_.scale_.tolist(),

            # Classifier parameters
            'coefficients': self.classifier_.coef_.flatten().tolist(),
            'intercept': self.classifier_.intercept_.tolist(),

            # Metadata
            'classes': self.classes_.tolist()
        }

        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)

        print(f"Model exported to {filename}")
        print(f"  Kernels: {self.num_kernels}")
        print(f"  Features: {self.num_features_}")
        print(f"  Classes: {len(self.classes_)}")


if __name__ == "__main__":
    print("HYDRA Custom Implementation Test")
    print("=" * 50)

    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 100
    length = 150

    # Two classes: sinusoidal with different frequencies
    X_train = []
    y_train = []

    for i in range(n_samples):
        if i < n_samples // 2:
            # Class 0: low frequency
            t = np.linspace(0, 2 * np.pi, length)
            x = np.sin(t) + 0.1 * np.random.randn(length)
            y = 0
        else:
            # Class 1: high frequency
            t = np.linspace(0, 4 * np.pi, length)
            x = np.sin(t) + 0.1 * np.random.randn(length)
            y = 1

        X_train.append(x)
        y_train.append(y)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"Training data: {X_train.shape}")
    print(f"Labels: {y_train.shape}")

    # Train HYDRA
    print("\nTraining HYDRA...")
    hydra = Hydra(num_kernels=512, num_groups=8, random_state=42)
    hydra.fit(X_train, y_train)

    # Evaluate
    print("\nEvaluating...")
    train_accuracy = hydra.score(X_train, y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")

    # Export model
    print("\nExporting model...")
    hydra.export_to_json("hydra_test_model.json")

    print("\n" + "=" * 50)
    print("Test completed successfully!")
