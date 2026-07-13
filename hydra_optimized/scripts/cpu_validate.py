#!/usr/bin/env python3
"""CPU-side HYDRA validation: checks if host-side dilation sorting produces correct results."""
import json
import numpy as np
import sys

LABEL_MAP = {
    "aedes_female": 0, "aedes_male": 1, "fruit_flies": 2, "house_flies": 3,
    "quinx_female": 4, "quinx_male": 5, "stigma_female": 6, "stigma_male": 7,
    "tarsalis_female": 8, "tarsalis_male": 9
}

def load_model(model_path):
    with open(model_path) as f:
        model = json.load(f)
    weights = np.array(model['kernel_weights']).reshape(512, 9)
    biases = np.array(model['biases'])
    dilations = np.array(model['dilations'], dtype=int)
    scaler_mean = np.array(model['scaler_mean'])
    scaler_scale = np.array(model['scaler_scale'])
    num_classes = len(model['intercept'])
    num_features = len(scaler_mean)
    coefficients = np.array(model['coefficients']).reshape(num_classes, num_features)
    intercept = np.array(model['intercept'])
    return weights, biases, dilations, scaler_mean, scaler_scale, coefficients, intercept

def load_test(test_path):
    with open(test_path) as f:
        test = json.load(f)
    inputs = np.array(test['time_series'])
    raw_labels = test['labels']
    # Map string labels to integers
    if isinstance(raw_labels[0], str):
        labels = np.array([LABEL_MAP.get(l, int(l) if l.isdigit() else -1) for l in raw_labels])
    else:
        labels = np.array(raw_labels, dtype=int)
    return inputs, labels

def hydra_features_vectorized(ts, weights, biases, dilations):
    """Compute HYDRA features: max pool + mean pool per kernel (vectorized)."""
    num_kernels = len(biases)
    ts_len = len(ts)
    features = np.zeros(num_kernels * 2)

    for k in range(num_kernels):
        d = dilations[k]
        kernel_span = 8 * d + 1
        conv_len = ts_len - kernel_span + 1
        if conv_len <= 0:
            continue
        # Vectorized convolution
        conv_out = np.full(conv_len, biases[k])
        for w in range(9):
            conv_out += ts[w*d : w*d + conv_len] * weights[k, w]
        features[2*k] = np.max(conv_out)
        features[2*k + 1] = np.mean(conv_out)
    return features

def predict(features, scaler_mean, scaler_scale, coefficients, intercept):
    """Apply scaler + linear classifier."""
    scaled = np.where(scaler_scale != 0, (features - scaler_mean) / scaler_scale, 0.0)
    scores = coefficients @ scaled + intercept
    return np.argmax(scores)

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/hydra_insectsound_model.json"
    test_path = sys.argv[2] if len(sys.argv) > 2 else "models/hydra_insectsound_test_1000.json"
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    print(f"Model: {model_path}")
    print(f"Test: {test_path}")
    weights, biases, dilations, scaler_mean, scaler_scale, coefficients, intercept = load_model(model_path)
    inputs, labels = load_test(test_path)

    num_samples = len(inputs)
    num_kernels = len(biases)
    num_features = len(scaler_mean)
    print(f"Kernels: {num_kernels}, Features: {num_features}, Classes: {len(intercept)}, Samples: {num_samples}")
    print(f"Labels sample: {labels[:5]}")
    print(f"Dilation dist: ", {d: np.sum(dilations==d) for d in [1,2,4,8]})

    # Test 1: Original order
    print(f"\n--- Test 1: Original order (first {n_test}) ---")
    correct = 0
    for i in range(min(num_samples, n_test)):
        feats = hydra_features_vectorized(inputs[i], weights, biases, dilations)
        pred = predict(feats, scaler_mean, scaler_scale, coefficients, intercept)
        if pred == labels[i]:
            correct += 1
        elif i < 5:
            print(f"  Sample {i}: pred={pred}, actual={labels[i]}, scores_range=[{min(coefficients @ ((feats - scaler_mean) / np.where(scaler_scale!=0, scaler_scale, 1)) + intercept):.3f}, {max(coefficients @ ((feats - scaler_mean) / np.where(scaler_scale!=0, scaler_scale, 1)) + intercept):.3f}]")
    print(f"Accuracy: {correct}/{min(num_samples, n_test)} = {100*correct/min(num_samples, n_test):.1f}%")

    # Test 2: Sorted by dilation
    print(f"\n--- Test 2: Sorted by dilation (first {n_test}) ---")
    perm = np.argsort(dilations, kind='stable')
    s_weights = weights[perm]
    s_biases = biases[perm]
    s_dilations = dilations[perm]

    feat_perm = np.zeros(num_features, dtype=int)
    for i in range(num_kernels):
        feat_perm[2*i] = 2*perm[i]
        feat_perm[2*i+1] = 2*perm[i]+1
    s_scaler_mean = scaler_mean[feat_perm]
    s_scaler_scale = scaler_scale[feat_perm]
    s_coefficients = coefficients[:, feat_perm]

    correct = 0
    for i in range(min(num_samples, n_test)):
        feats = hydra_features_vectorized(inputs[i], s_weights, s_biases, s_dilations)
        pred = predict(feats, s_scaler_mean, s_scaler_scale, s_coefficients, intercept)
        if pred == labels[i]:
            correct += 1
    print(f"Accuracy: {correct}/{min(num_samples, n_test)} = {100*correct/min(num_samples, n_test):.1f}%")

    # Check batch uniformity
    print("\n--- Batch uniformity check ---")
    UNROLL = 16
    bad = 0
    for batch in range(num_kernels // UNROLL):
        batch_d = s_dilations[batch*UNROLL:(batch+1)*UNROLL]
        if len(set(batch_d)) > 1:
            print(f"  MISMATCH: Batch {batch} has dilations: {set(batch_d)}")
            bad += 1
    if bad == 0:
        print("  All batches have uniform dilation ✓")

if __name__ == "__main__":
    main()
