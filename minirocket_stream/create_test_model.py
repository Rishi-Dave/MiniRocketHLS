#!/usr/bin/env python3
"""Create a test MiniRocket model with time_series_length=96 for HLS testing"""

import numpy as np
import json
from itertools import combinations
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def generate_sample_data(n_samples=1000, length=96, n_classes=2):
    """Generate sample time series data for testing"""
    np.random.seed(42)

    X = np.zeros((n_samples, length), dtype=np.float32)
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        class_id = i % n_classes

        if class_id == 0:
            # Sine wave
            X[i] = np.sin(np.linspace(0, 4*np.pi, length)) + 0.1 * np.random.randn(length)
        else:
            # Square wave
            X[i] = np.sign(np.sin(np.linspace(0, 4*np.pi, length))) + 0.1 * np.random.randn(length)

        y[i] = class_id

    return X, y

class MiniRocket:
    """MiniRocket implementation for HLS-compatible model generation"""

    def __init__(self, num_kernels=840, max_dilations_per_kernel=32, random_state=None):
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_state = random_state
        self.parameters = None

    def _quantiles(self, n):
        return np.array(
            [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
        )

    def _fit_dilations(self, n_timepoints, num_features, max_dilations_per_kernel):
        num_kernels = 84

        if num_features < 84:
            num_features = 84

        num_features_per_kernel = num_features // num_kernels
        true_max_dilations_per_kernel = min(
            num_features_per_kernel, max_dilations_per_kernel
        )
        multiplier = num_features_per_kernel / true_max_dilations_per_kernel

        max_exponent = np.log2((n_timepoints - 1) / (9 - 1))
        dilations, num_features_per_dilation = np.unique(
            np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
                np.int32
            ),
            return_counts=True,
        )
        num_features_per_dilation = (num_features_per_dilation * multiplier).astype(
            np.int32
        )

        remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
        i = 0
        while remainder > 0:
            num_features_per_dilation[i] += 1
            remainder -= 1
            i = (i + 1) % len(num_features_per_dilation)

        return dilations, num_features_per_dilation

    def _fit_biases(self, X, dilations, num_features_per_dilation, quantiles, seed):
        if seed is not None:
            np.random.seed(seed)

        n_instances, n_timepoints = X.shape
        indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)

        num_kernels = len(indices)
        num_dilations = len(dilations)
        num_features = num_kernels * np.sum(num_features_per_dilation)
        biases = np.zeros(num_features, dtype=np.float32)

        feature_index_start = 0

        for dilation_index in range(num_dilations):
            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2
            num_features_this_dilation = num_features_per_dilation[dilation_index]

            for kernel_index in range(num_kernels):
                feature_index_end = feature_index_start + num_features_this_dilation

                _X = X[np.random.randint(n_instances)]
                A = -_X
                G = _X + _X + _X

                C_alpha = np.zeros(n_timepoints, dtype=np.float32)
                C_alpha[:] = A

                C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
                C_gamma[9 // 2] = G

                start = dilation
                end = n_timepoints - padding

                for gamma_index in range(9 // 2):
                    C_alpha[-end:] = C_alpha[-end:] + A[:end]
                    C_gamma[gamma_index, -end:] = G[:end]
                    end += dilation

                for gamma_index in range(9 // 2 + 1, 9):
                    C_alpha[:-start] = C_alpha[:-start] + A[start:]
                    C_gamma[gamma_index, :-start] = G[start:]
                    start += dilation

                index_0, index_1, index_2 = indices[kernel_index]
                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                biases[feature_index_start:feature_index_end] = np.quantile(
                    C, quantiles[feature_index_start:feature_index_end]
                )

                feature_index_start = feature_index_end

        return biases

    def fit(self, X):
        X = X.astype(np.float32)
        _, n_timepoints = X.shape

        self.time_series_length = n_timepoints

        if self.num_kernels < 84:
            self.num_kernels_ = 84
        else:
            self.num_kernels_ = (self.num_kernels // 84) * 84

        seed = np.int32(self.random_state) if isinstance(self.random_state, int) else None

        dilations, num_features_per_dilation = self._fit_dilations(
            n_timepoints, self.num_kernels_, self.max_dilations_per_kernel
        )

        num_features_per_kernel = np.sum(num_features_per_dilation)
        quantiles = self._quantiles(84 * num_features_per_kernel)
        biases = self._fit_biases(X, dilations, num_features_per_dilation, quantiles, seed)

        self.parameters = (dilations, num_features_per_dilation, biases)
        self.dilations = dilations
        self.num_features_per_dilation = num_features_per_dilation
        self.biases = biases

        return self

    def transform(self, X):
        X = X.astype(np.float32)
        n_instances, n_timepoints = X.shape

        dilations, num_features_per_dilation, biases = self.parameters
        indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)

        num_kernels = len(indices)
        num_features = num_kernels * np.sum(num_features_per_dilation)
        features = np.zeros((n_instances, num_features), dtype=np.float32)

        for example_index in range(n_instances):
            _X = X[example_index]

            A = -_X
            G = _X + _X + _X

            feature_index_start = 0

            for dilation_index in range(len(dilations)):
                _padding0 = dilation_index % 2
                dilation = dilations[dilation_index]
                padding = ((9 - 1) * dilation) // 2
                num_features_this_dilation = num_features_per_dilation[dilation_index]

                C_alpha = np.zeros(n_timepoints, dtype=np.float32)
                C_alpha[:] = A

                C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
                C_gamma[9 // 2] = G

                start = dilation
                end = n_timepoints - padding

                for gamma_index in range(9 // 2):
                    C_alpha[-end:] = C_alpha[-end:] + A[:end]
                    C_gamma[gamma_index, -end:] = G[:end]
                    end += dilation

                for gamma_index in range(9 // 2 + 1, 9):
                    C_alpha[:-start] = C_alpha[:-start] + A[start:]
                    C_gamma[gamma_index, :-start] = G[start:]
                    start += dilation

                for kernel_index in range(num_kernels):
                    feature_index_end = feature_index_start + num_features_this_dilation
                    _padding1 = (_padding0 + kernel_index) % 2

                    index_0, index_1, index_2 = indices[kernel_index]
                    C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                    if _padding1 == 0:
                        for feature_count in range(num_features_this_dilation):
                            features[example_index, feature_index_start + feature_count] = (
                                np.mean(C > biases[feature_index_start + feature_count])
                            )
                    else:
                        for feature_count in range(num_features_this_dilation):
                            features[example_index, feature_index_start + feature_count] = (
                                np.mean(C[padding:-padding] > biases[feature_index_start + feature_count])
                            )

                    feature_index_start = feature_index_end

        return features

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def main():
    print("Generating synthetic data with time_series_length=96...")
    X, y = generate_sample_data(n_samples=1000, length=96, n_classes=2)

    # Split into train/test
    split_idx = 800
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    print("Fitting MiniRocket with 840 kernels...")
    minirocket = MiniRocket(num_kernels=840, random_state=42)
    X_train_features = minirocket.fit_transform(X_train)
    X_test_features = minirocket.transform(X_test)

    print(f"Feature shape: {X_train_features.shape}")
    print(f"Dilations: {minirocket.dilations}")
    print(f"Num dilations: {len(minirocket.dilations)}")
    print(f"Num features per dilation: {minirocket.num_features_per_dilation}")

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

    # Generate kernel indices
    kernel_indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)

    # Save model
    model_data = {
        "num_kernels": 84,
        "num_dilations": len(minirocket.dilations),
        "num_features": len(minirocket.biases),
        "num_classes": len(classifier.classes_),
        "time_series_length": minirocket.time_series_length,
        "kernel_indices": kernel_indices.tolist(),
        "dilations": minirocket.dilations.tolist(),
        "num_features_per_dilation": minirocket.num_features_per_dilation.tolist(),
        "biases": minirocket.biases.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classifier_coef": classifier.coef_.tolist(),
        "classifier_intercept": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist()
    }

    model_filename = "minirocket_model_96.json"
    with open(model_filename, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"Model saved to {model_filename}")

    # Save test data
    test_data = {
        "dataset_name": "synthetic_96",
        "X_test": X_test.tolist(),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "test_accuracy": float(accuracy),
        "num_samples": len(X_test),
        "series_length": 96,
        "num_classes": 2
    }

    test_filename = "minirocket_model_96_test_data.json"
    with open(test_filename, 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"Test data saved to {test_filename}")
    print("\nModel summary:")
    print(f"  time_series_length: {model_data['time_series_length']}")
    print(f"  num_features: {model_data['num_features']}")
    print(f"  num_dilations: {model_data['num_dilations']}")
    print(f"  num_classes: {model_data['num_classes']}")
    print(f"  dilations: {model_data['dilations']}")

if __name__ == "__main__":
    main()
