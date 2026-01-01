#!/usr/bin/env python3
"""
Custom MultiRocket Implementation with 84 Kernels

This is a simplified MultiRocket that uses the same 84 kernels as MiniRocket
(all combinations of 3 indices from 0-8), but applies 4 pooling operators
to both the original time series and its first-order difference.

Features generated: 84 kernels × num_dilations × 4 pooling × 2 representations
For 8 dilations: 84 × 8 × 4 × 2 = 5,376 features

This matches our FPGA implementation constraints while maintaining the
MultiRocket algorithmic improvements over MiniRocket.
"""

import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
import json


class MultiRocket84:
    """
    Custom MultiRocket with 84 kernels (matching MiniRocket)

    Implements the MultiRocket algorithm from:
    Tan et al., "MultiRocket: Multiple pooling operators and transformations
    for fast and effective time series classification" (2021)

    Key differences from standard MultiRocket:
    - Uses 84 kernels instead of 6,250 (FPGA constraint)
    - Fixed kernel structure: all C(9,3) combinations
    - Simplified weights: -1, 0, +2 pattern (from optimized MiniRocket)
    """

    def __init__(self, max_dilations_per_kernel=8, random_state=None):
        """
        Initialize MultiRocket84

        Args:
            max_dilations_per_kernel: Maximum number of dilations (default: 8)
            random_state: Random seed for reproducibility
        """
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_state = random_state
        self.num_kernels = 84  # Fixed: C(9,3)

        # Will be set during fit
        self.dilations_orig = None
        self.dilations_diff = None
        self.biases_orig = None
        self.biases_diff = None
        self.num_features = None

        # Generate fixed kernel indices
        self.kernel_indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)

        # Generate simplified kernel weights (matching optimized MiniRocket)
        self.weights = self._generate_simplified_weights()

    def _generate_simplified_weights(self):
        """
        Generate simplified kernel weights: -1, 0, +2 pattern

        This matches the optimized MiniRocket approach:
        - Position of indices get weight +2
        - Position 4 (center) gets weight 0 (implicit)
        - All other positions get weight -1
        """
        weights = np.zeros((self.num_kernels, 9), dtype=np.float32)

        for k, (i0, i1, i2) in enumerate(self.kernel_indices):
            # Initialize all to -1
            weights[k, :] = -1
            # Set selected positions to +2
            weights[k, i0] = 2
            weights[k, i1] = 2
            weights[k, i2] = 2

        return weights

    def _fit_dilations(self, n_timepoints):
        """
        Determine dilations based on time series length

        Uses exponential spacing from 1 to max_dilation
        where max_dilation = (n_timepoints - 1) / (kernel_size - 1)
        """
        kernel_size = 9
        max_exponent = np.log2((n_timepoints - 1) / (kernel_size - 1))

        dilations = np.unique(
            np.logspace(0, max_exponent, self.max_dilations_per_kernel,
                       base=2, dtype=np.float32).astype(np.int32)
        )

        return dilations

    def _apply_kernel(self, X, kernel_idx, dilation):
        """
        Apply a single kernel to time series with given dilation

        Args:
            X: Time series (n_timepoints,)
            kernel_idx: Index of kernel to apply (0-83)
            dilation: Dilation value

        Returns:
            Convolution output (n_timepoints,)
        """
        n_timepoints = len(X)
        weights = self.weights[kernel_idx]

        # Output array
        C = np.zeros(n_timepoints, dtype=np.float32)

        # Apply convolution with padding
        for j in range(n_timepoints):
            value = 0.0
            for k in range(-4, 5):  # Kernel positions -4 to +4
                idx = j + k * dilation
                if 0 <= idx < n_timepoints:
                    value += X[idx] * weights[k + 4]
            C[j] = value

        return C

    def _compute_ppv(self, C, bias):
        """Proportion of Positive Values"""
        return np.mean(C > bias)

    def _compute_mpv(self, C, bias):
        """Mean of Positive Values"""
        positive = C[C > bias]
        return np.mean(positive) if len(positive) > 0 else 0.0

    def _compute_mipv(self, C, bias):
        """Mean of Indices of Positive Values (normalized)"""
        indices = np.where(C > bias)[0]
        if len(indices) > 0:
            return np.mean(indices) / len(C)
        return 0.0

    def _compute_lspv(self, C, bias):
        """Longest Stretch of Positive Values (normalized)"""
        is_positive = C > bias
        stretches = []
        current_stretch = 0

        for val in is_positive:
            if val:
                current_stretch += 1
            else:
                if current_stretch > 0:
                    stretches.append(current_stretch)
                current_stretch = 0

        if current_stretch > 0:
            stretches.append(current_stretch)

        max_stretch = max(stretches) if stretches else 0
        return max_stretch / len(C)

    def _compute_four_pooling(self, C, bias):
        """Compute all four pooling operators"""
        ppv = self._compute_ppv(C, bias)
        mpv = self._compute_mpv(C, bias)
        mipv = self._compute_mipv(C, bias)
        lspv = self._compute_lspv(C, bias)

        return ppv, mpv, mipv, lspv

    def _fit_biases_for_representation(self, X, dilations, seed=None):
        """
        Fit bias thresholds for a representation

        For each dilation and kernel, compute convolution on a random sample
        and use the 0.5 quantile (median) as the bias threshold.
        """
        if seed is not None:
            np.random.seed(seed)

        n_samples, n_timepoints = X.shape
        num_dilations = len(dilations)

        # One bias per kernel per dilation
        biases = np.zeros(self.num_kernels * num_dilations, dtype=np.float32)

        for d_idx, dilation in enumerate(dilations):
            for k_idx in range(self.num_kernels):
                # Random sample
                sample_idx = np.random.randint(n_samples)
                sample = X[sample_idx]

                # Apply convolution
                C = self._apply_kernel(sample, k_idx, dilation)

                # Use median as bias
                bias = np.median(C)

                biases[d_idx * self.num_kernels + k_idx] = bias

        return biases

    def _extract_features_single_repr(self, X, dilations, biases):
        """
        Extract features from a single representation

        Args:
            X: Time series data (n_samples, n_timepoints)
            dilations: Array of dilation values
            biases: Bias thresholds

        Returns:
            Features (n_samples, num_kernels × num_dilations × 4)
        """
        n_samples, n_timepoints = X.shape
        num_dilations = len(dilations)
        num_features_per_sample = self.num_kernels * num_dilations * 4

        features = np.zeros((n_samples, num_features_per_sample), dtype=np.float32)

        for sample_idx in range(n_samples):
            sample = X[sample_idx]
            feature_idx = 0

            for d_idx, dilation in enumerate(dilations):
                for k_idx in range(self.num_kernels):
                    # Apply convolution
                    C = self._apply_kernel(sample, k_idx, dilation)

                    # Get bias
                    bias = biases[d_idx * self.num_kernels + k_idx]

                    # Compute 4 pooling features
                    ppv, mpv, mipv, lspv = self._compute_four_pooling(C, bias)

                    features[sample_idx, feature_idx] = ppv
                    features[sample_idx, feature_idx + 1] = mpv
                    features[sample_idx, feature_idx + 2] = mipv
                    features[sample_idx, feature_idx + 3] = lspv

                    feature_idx += 4

        return features

    def fit(self, X):
        """
        Fit MultiRocket84 to training data

        Args:
            X: Training time series (n_samples, n_timepoints)
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_timepoints)")

        n_samples, n_timepoints = X.shape

        # Fit dilations for original representation
        self.dilations_orig = self._fit_dilations(n_timepoints)

        # Fit dilations for diff representation (one less timepoint)
        self.dilations_diff = self._fit_dilations(n_timepoints - 1)

        # Fit biases
        seed = self.random_state
        self.biases_orig = self._fit_biases_for_representation(X, self.dilations_orig, seed)

        # Compute diff representation for bias fitting
        X_diff = np.diff(X, axis=1)
        self.biases_diff = self._fit_biases_for_representation(X_diff, self.dilations_diff, seed)

        # Calculate total number of features
        num_feat_orig = self.num_kernels * len(self.dilations_orig) * 4
        num_feat_diff = self.num_kernels * len(self.dilations_diff) * 4
        self.num_features = num_feat_orig + num_feat_diff

        return self

    def transform(self, X):
        """
        Transform time series to MultiRocket features

        Args:
            X: Time series (n_samples, n_timepoints)

        Returns:
            Features (n_samples, num_features)
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_timepoints)")

        # Extract features from original representation
        features_orig = self._extract_features_single_repr(
            X, self.dilations_orig, self.biases_orig
        )

        # Compute first-order difference
        X_diff = np.diff(X, axis=1)

        # Extract features from diff representation
        features_diff = self._extract_features_single_repr(
            X_diff, self.dilations_diff, self.biases_diff
        )

        # Concatenate features
        features = np.concatenate([features_orig, features_diff], axis=1)

        return features

    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)

    def export_for_fpga(self, scaler, classifier, output_file):
        """
        Export model parameters for FPGA implementation

        Args:
            scaler: Fitted StandardScaler
            classifier: Fitted RidgeClassifier
            output_file: Output JSON file path
        """
        export_data = {
            'model_type': 'MultiRocket84',
            'num_kernels': self.num_kernels,
            'num_features': self.num_features,
            'num_classes': len(classifier.classes_),
            'classes': classifier.classes_.tolist(),

            # Original representation parameters
            'num_dilations_orig': len(self.dilations_orig),
            'dilations_orig': self.dilations_orig.tolist(),
            'biases_orig': self.biases_orig.tolist(),

            # Diff representation parameters
            'num_dilations_diff': len(self.dilations_diff),
            'dilations_diff': self.dilations_diff.tolist(),
            'biases_diff': self.biases_diff.tolist(),

            # Scaler parameters
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),

            # Classifier parameters (handle binary classification case)
            # For binary classification, sklearn returns 1D coef_ which separates class 0 from class 1
            # Decision function computes: [-score, score] where score = X @ coef + intercept
            # So we need [-coef, coef] and [-intercept, intercept] for the two classes
            'coefficients': classifier.coef_.tolist() if len(classifier.coef_.shape) == 2 else [[-float(x) for x in classifier.coef_], classifier.coef_.tolist()],
            'intercept': classifier.intercept_.tolist() if len(classifier.intercept_.shape) > 0 and len(classifier.intercept_) > 1 else [-float(classifier.intercept_[0]), float(classifier.intercept_[0])],
            'alpha': float(classifier.alpha_),

            # Kernel weights (for reference)
            'kernel_indices': self.kernel_indices.tolist(),
            'weights': self.weights.tolist()
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"\n✓ Model exported to {output_file}")
        print(f"  Total features: {self.num_features}")
        print(f"  Features (orig): {self.num_kernels * len(self.dilations_orig) * 4}")
        print(f"  Features (diff): {self.num_kernels * len(self.dilations_diff) * 4}")
        print(f"  Num dilations (orig): {len(self.dilations_orig)}")
        print(f"  Num dilations (diff): {len(self.dilations_diff)}")

        return export_data


def train_and_export(dataset_name="GunPoint", output_prefix="multirocket84"):
    """
    Complete training and export pipeline

    Args:
        dataset_name: UCR dataset name
        output_prefix: Prefix for output files
    """
    from sktime.datasets import load_UCR_UEA_dataset

    print("="*80)
    print(f"Training MultiRocket84 on {dataset_name}")
    print("="*80)

    # Load dataset
    print(f"\n1. Loading {dataset_name} dataset...")
    X_train, y_train = load_UCR_UEA_dataset(dataset_name, split="train", return_type="numpy3d")
    X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_type="numpy3d")

    # Convert to 2D
    if X_train.ndim == 3:
        X_train = X_train.squeeze(1)
    if X_test.ndim == 3:
        X_test = X_test.squeeze(1)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {np.unique(y_train)}")

    # Initialize MultiRocket84
    print(f"\n2. Initializing MultiRocket84...")
    multirocket = MultiRocket84(max_dilations_per_kernel=8, random_state=42)

    # Extract features
    print(f"\n3. Extracting features...")
    X_train_features = multirocket.fit_transform(X_train)
    X_test_features = multirocket.transform(X_test)

    print(f"  Feature shape: {X_train_features.shape}")

    # Train classifier
    print(f"\n4. Training classifier...")
    scaler = StandardScaler()
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=5)

    X_train_scaled = scaler.fit_transform(X_train_features)
    classifier.fit(X_train_scaled, y_train)

    # Evaluate
    print(f"\n5. Evaluating...")
    y_train_pred = classifier.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, y_train_pred)

    X_test_scaled = scaler.transform(X_test_features)
    y_test_pred = classifier.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\n  Train accuracy: {train_acc:.4f} ({np.sum(y_train == y_train_pred)}/{len(y_train)})")
    print(f"  Test accuracy:  {test_acc:.4f} ({np.sum(y_test == y_test_pred)}/{len(y_test)})")

    # Export for FPGA
    print(f"\n6. Exporting for FPGA...")
    model_file = f"{output_prefix}_{dataset_name.lower()}_model.json"
    multirocket.export_for_fpga(scaler, classifier, model_file)

    # Export test data
    test_file = f"{output_prefix}_{dataset_name.lower()}_test.json"

    # Convert class labels to indices
    class_to_idx = {label: idx for idx, label in enumerate(classifier.classes_)}
    pred_indices = [class_to_idx[pred] for pred in y_test_pred[:10]]

    test_data = {
        'num_samples': min(10, len(X_test)),
        'time_series_length': X_test.shape[1],
        'time_series': X_test[:10].tolist(),
        'labels': y_test[:10].tolist(),
        'features_python': X_test_features[:10].tolist(),
        'predictions_python': pred_indices,  # Now integers
        'test_accuracy': float(test_acc)
    }

    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"\n✓ Test data exported to {test_file}")

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)

    return multirocket, scaler, classifier, test_acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train MultiRocket84 for FPGA')
    parser.add_argument('--dataset', type=str, default='GunPoint',
                       help='UCR dataset name (default: GunPoint)')
    parser.add_argument('--output', type=str, default='multirocket84',
                       help='Output file prefix (default: multirocket84)')

    args = parser.parse_args()

    train_and_export(args.dataset, args.output)
