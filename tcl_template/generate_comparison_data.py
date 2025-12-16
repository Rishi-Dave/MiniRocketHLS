#!/usr/bin/env python3

"""
Generate detailed step-by-step comparison data for C++ validation.
This script runs a single test sample through the Python MiniRocket pipeline
and outputs intermediate results at each stage for comparison with C++.
"""

import numpy as np
import json
import sys

def load_model_and_data(model_file, test_file):
    """Load model parameters and test data"""
    with open(model_file, 'r') as f:
        model = json.load(f)

    with open(test_file, 'r') as f:
        test_data = json.load(f)

    return model, test_data

def transform_single_sample(X, dilations, num_features_per_dilation, biases):
    """Transform a single time series sample using MiniRocket algorithm"""
    from itertools import combinations

    X = X.astype(np.float32)
    n_timepoints = len(X)

    # All combinations of 3 indices from 0-8 (C(9,3) = 84)
    indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)

    num_kernels = len(indices)
    num_dilations = len(dilations)
    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros(num_features, dtype=np.float32)

    # Compute A = -X and G = 3X
    A = -X
    G = X + X + X

    print(f"\n=== PYTHON: Convolution Weights ===")
    print(f"Alpha (α) = -1, Gamma (γ) = +3")
    print(f"A = α * X = -X (first 5 values): {A[:5]}")
    print(f"G = γ * X = 3X (first 5 values): {G[:5]}")

    feature_index_start = 0

    for dilation_index in range(num_dilations):
        _padding0 = dilation_index % 2

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        # Initialize cumulative arrays
        C_alpha = np.zeros(n_timepoints, dtype=np.float32)
        C_alpha[:] = A

        C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
        C_gamma[9 // 2] = G

        # Build cumulative convolution arrays
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

        if dilation_index == 0:  # Show first dilation details
            print(f"\n=== PYTHON: Dilation {dilation} (first dilation) ===")
            print(f"Padding: {padding}")
            print(f"C_alpha (cumulative alpha, first 5): {C_alpha[:5]}")
            print(f"C_gamma[0] (first position, first 5): {C_gamma[0, :5]}")
            print(f"C_gamma[4] (center, first 5): {C_gamma[4, :5]}")

        for kernel_index in range(num_kernels):
            feature_index_end = feature_index_start + num_features_this_dilation

            _padding1 = (_padding0 + kernel_index) % 2

            index_0, index_1, index_2 = indices[kernel_index]

            # Combine cumulative arrays
            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

            if dilation_index == 0 and kernel_index == 0:  # Show first kernel
                print(f"\n=== PYTHON: Kernel {kernel_index} (indices {index_0}, {index_1}, {index_2}) ===")
                print(f"C = C_alpha + C_gamma[{index_0}] + C_gamma[{index_1}] + C_gamma[{index_2}]")
                print(f"C (convolution result, first 5): {C[:5]}")
                print(f"Bias values for this kernel: {biases[feature_index_start:feature_index_end]}")

            # Compute PPV (Proportion of Positive Values)
            if _padding1 == 0:
                for feature_count in range(num_features_this_dilation):
                    features[feature_index_start + feature_count] = (
                        np.mean(C > biases[feature_index_start + feature_count])
                    )
            else:
                for feature_count in range(num_features_this_dilation):
                    features[feature_index_start + feature_count] = (
                        np.mean(
                            C[padding:-padding] > biases[feature_index_start + feature_count]
                        )
                    )

            if dilation_index == 0 and kernel_index == 0:
                print(f"PPV features (first 3): {features[feature_index_start:feature_index_start+3]}")

            feature_index_start = feature_index_end

    return features

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 generate_comparison_data.py <model.json> <test_data.json>")
        sys.exit(1)

    model_file = sys.argv[1]
    test_file = sys.argv[2]

    print("=== PYTHON MiniRocket Step-by-Step Comparison ===")
    print(f"Loading model: {model_file}")
    print(f"Loading test data: {test_file}")

    model, test_data = load_model_and_data(model_file, test_file)

    # Extract model parameters
    dilations = np.array(model['dilations'], dtype=np.int32)
    num_features_per_dilation = np.array(model['num_features_per_dilation'], dtype=np.int32)
    biases = np.array(model['biases'], dtype=np.float32)
    scaler_mean = np.array(model['scaler_mean'], dtype=np.float32)
    scaler_scale = np.array(model['scaler_scale'], dtype=np.float32)

    # Handle binary vs multi-class coefficients
    num_classes = model['num_classes']
    if num_classes == 2 and not isinstance(model['classifier_coef'][0], list):
        # Binary: 1D array
        coef_1d = np.array(model['classifier_coef'], dtype=np.float32)
        coefficients = np.zeros((2, len(coef_1d)), dtype=np.float32)
        coefficients[0, :] = coef_1d
    else:
        # Multi-class: 2D array
        coefficients = np.array(model['classifier_coef'], dtype=np.float32)

    if num_classes == 2 and len(model['classifier_intercept']) == 1:
        intercept = np.array([model['classifier_intercept'][0], 0.0], dtype=np.float32)
    else:
        intercept = np.array(model['classifier_intercept'], dtype=np.float32)

    # Use first test sample
    X_test = np.array(test_data['X_test'][0], dtype=np.float32)
    y_test = test_data['y_test'][0]

    print(f"\n=== PYTHON: Input ===")
    print(f"Time series length: {len(X_test)}")
    print(f"First 10 values: {X_test[:10]}")
    print(f"True label: {y_test}")

    # Step 1: Feature extraction
    print(f"\n{'='*60}")
    print("STEP 1: FEATURE EXTRACTION (MiniRocket Transform)")
    print(f"{'='*60}")
    features = transform_single_sample(X_test, dilations, num_features_per_dilation, biases)
    print(f"\n=== PYTHON: Extracted Features ===")
    print(f"Total features: {len(features)}")
    print(f"First 10 features: {features[:10]}")
    print(f"Feature range: [{np.min(features):.6f}, {np.max(features):.6f}]")

    # Step 2: Scaling
    print(f"\n{'='*60}")
    print("STEP 2: FEATURE SCALING (StandardScaler)")
    print(f"{'='*60}")
    scaled_features = (features - scaler_mean) / scaler_scale
    print(f"=== PYTHON: Scaled Features ===")
    print(f"Scaler mean (first 5): {scaler_mean[:5]}")
    print(f"Scaler scale (first 5): {scaler_scale[:5]}")
    print(f"Scaled features (first 10): {scaled_features[:10]}")
    print(f"Scaled range: [{np.min(scaled_features):.6f}, {np.max(scaled_features):.6f}]")

    # Step 3: Classification
    print(f"\n{'='*60}")
    print("STEP 3: LINEAR CLASSIFICATION (Ridge Classifier)")
    print(f"{'='*60}")
    predictions = np.dot(coefficients, scaled_features) + intercept
    predicted_class = np.argmax(predictions)

    print(f"=== PYTHON: Classification ===")
    print(f"Coefficients shape: {coefficients.shape}")
    print(f"Coefficients[0] (first 5): {coefficients[0, :5]}")
    print(f"Intercept: {intercept}")
    print(f"Decision scores: {predictions}")
    print(f"Predicted class: {predicted_class}")
    print(f"True class: {y_test}")
    print(f"Correct: {predicted_class == y_test}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Feature extraction: {len(features)} features extracted")
    print(f"✓ Scaling: Applied StandardScaler normalization")
    print(f"✓ Classification: Ridge classifier decision function")
    print(f"✓ Prediction: Class {predicted_class} (expected: {y_test})")
    print(f"✓ Match: {'YES ✓' if predicted_class == y_test else 'NO ✗'}")

if __name__ == "__main__":
    main()
