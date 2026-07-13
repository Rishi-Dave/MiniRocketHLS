#!/usr/bin/env python3
"""
Round 2 MultiRocket kernel accuracy check.
Mirrors the C++ kernel logic exactly (float arithmetic, same loop structure).
Uses numpy for convolution speed, but pooling ops match C++ logic bit-for-bit.
"""
import json
import numpy as np
import time

NUM_KERNELS = 84
KERNEL_SIZE = 9

# Weights from weights.txt (84 kernels x 9 weights) - verified against file
WEIGHTS_RAW = [
    [2,2,2,-1,-1,-1,-1,-1,-1], [2,2,-1,2,-1,-1,-1,-1,-1], [2,2,-1,-1,2,-1,-1,-1,-1],
    [2,2,-1,-1,-1,2,-1,-1,-1], [2,2,-1,-1,-1,-1,2,-1,-1], [2,2,-1,-1,-1,-1,-1,2,-1],
    [2,2,-1,-1,-1,-1,-1,-1,2], [2,-1,2,2,-1,-1,-1,-1,-1], [2,-1,2,-1,2,-1,-1,-1,-1],
    [2,-1,2,-1,-1,2,-1,-1,-1], [2,-1,2,-1,-1,-1,2,-1,-1], [2,-1,2,-1,-1,-1,-1,2,-1],
    [2,-1,2,-1,-1,-1,-1,-1,2], [2,-1,-1,2,2,-1,-1,-1,-1], [2,-1,-1,2,-1,2,-1,-1,-1],
    [2,-1,-1,2,-1,-1,2,-1,-1], [2,-1,-1,2,-1,-1,-1,2,-1], [2,-1,-1,2,-1,-1,-1,-1,2],
    [2,-1,-1,-1,2,2,-1,-1,-1], [2,-1,-1,-1,2,-1,2,-1,-1], [2,-1,-1,-1,2,-1,-1,2,-1],
    [2,-1,-1,-1,2,-1,-1,-1,2], [2,-1,-1,-1,-1,2,2,-1,-1], [2,-1,-1,-1,-1,2,-1,2,-1],
    [2,-1,-1,-1,-1,2,-1,-1,2], [2,-1,-1,-1,-1,-1,2,2,-1], [2,-1,-1,-1,-1,-1,2,-1,2],
    [2,-1,-1,-1,-1,-1,-1,2,2], [-1,2,2,2,-1,-1,-1,-1,-1], [-1,2,2,-1,2,-1,-1,-1,-1],
    [-1,2,2,-1,-1,2,-1,-1,-1], [-1,2,2,-1,-1,-1,2,-1,-1], [-1,2,2,-1,-1,-1,-1,2,-1],
    [-1,2,2,-1,-1,-1,-1,-1,2], [-1,2,-1,2,2,-1,-1,-1,-1], [-1,2,-1,2,-1,2,-1,-1,-1],
    [-1,2,-1,2,-1,-1,2,-1,-1], [-1,2,-1,2,-1,-1,-1,2,-1], [-1,2,-1,2,-1,-1,-1,-1,2],
    [-1,2,-1,-1,2,2,-1,-1,-1], [-1,2,-1,-1,2,-1,2,-1,-1], [-1,2,-1,-1,2,-1,-1,2,-1],
    [-1,2,-1,-1,2,-1,-1,-1,2], [-1,2,-1,-1,-1,2,2,-1,-1], [-1,2,-1,-1,-1,2,-1,2,-1],
    [-1,2,-1,-1,-1,2,-1,-1,2], [-1,2,-1,-1,-1,-1,2,2,-1], [-1,2,-1,-1,-1,-1,2,-1,2],
    [-1,2,-1,-1,-1,-1,-1,2,2], [-1,-1,2,2,2,-1,-1,-1,-1], [-1,-1,2,2,-1,2,-1,-1,-1],
    [-1,-1,2,2,-1,-1,2,-1,-1], [-1,-1,2,2,-1,-1,-1,2,-1], [-1,-1,2,2,-1,-1,-1,-1,2],
    [-1,-1,2,-1,2,2,-1,-1,-1], [-1,-1,2,-1,2,-1,2,-1,-1], [-1,-1,2,-1,2,-1,-1,2,-1],
    [-1,-1,2,-1,2,-1,-1,-1,2], [-1,-1,2,-1,-1,2,2,-1,-1], [-1,-1,2,-1,-1,2,-1,2,-1],
    [-1,-1,2,-1,-1,2,-1,-1,2], [-1,-1,2,-1,-1,-1,2,2,-1], [-1,-1,2,-1,-1,-1,2,-1,2],
    [-1,-1,2,-1,-1,-1,-1,2,2], [-1,-1,-1,2,2,2,-1,-1,-1], [-1,-1,-1,2,2,-1,2,-1,-1],
    [-1,-1,-1,2,2,-1,-1,2,-1], [-1,-1,-1,2,2,-1,-1,-1,2], [-1,-1,-1,2,-1,2,2,-1,-1],
    [-1,-1,-1,2,-1,2,-1,2,-1], [-1,-1,-1,2,-1,2,-1,-1,2], [-1,-1,-1,2,-1,-1,2,2,-1],
    [-1,-1,-1,2,-1,-1,2,-1,2], [-1,-1,-1,2,-1,-1,-1,2,2], [-1,-1,-1,-1,2,2,2,-1,-1],
    [-1,-1,-1,-1,2,2,-1,2,-1], [-1,-1,-1,-1,2,2,-1,-1,2], [-1,-1,-1,-1,2,-1,2,2,-1],
    [-1,-1,-1,-1,2,-1,2,-1,2], [-1,-1,-1,-1,2,-1,-1,2,2], [-1,-1,-1,-1,-1,2,2,2,-1],
    [-1,-1,-1,-1,-1,2,2,-1,2], [-1,-1,-1,-1,-1,2,-1,2,2], [-1,-1,-1,-1,-1,-1,2,2,2],
]
WEIGHTS = np.array(WEIGHTS_RAW, dtype=np.float32)  # (84, 9)


def apply_kernel_np(ts_arr, kernel_idx, dilation, ts_length):
    """Vectorized convolution matching apply_kernel_hls.
    Center-zero-padded with k in [-4..4], stride dilation.
    """
    w = WEIGHTS[kernel_idx]  # (9,)
    # Build dilated sliding window matrix: (ts_length, 9)
    indices = np.arange(ts_length)[:, None] + np.arange(-4, 5)[None, :] * dilation  # (L, 9)
    valid = (indices >= 0) & (indices < ts_length)
    clipped = np.clip(indices, 0, ts_length - 1)
    vals = np.where(valid, ts_arr[clipped], 0.0)  # (L, 9)
    conv = vals @ w  # (L,)
    return conv


def compute_four_pooling_np(conv, bias, ts_length):
    """Mirrors compute_four_pooling_operators (Round 2) exactly.
    start=0, length=ts_length, effective_length = ts_length.
    """
    effective_length = float(ts_length)
    positive = conv > bias
    ppv_count = np.sum(positive)
    inv_length = 1.0 / effective_length

    ppv = ppv_count * inv_length

    if ppv_count > 0:
        mpv = float(np.sum(conv[positive])) / ppv_count
        # mipv: mean of indices (0-based, since start=0), normalized by length
        pos_indices = np.where(positive)[0]  # i - start where start=0
        mipv = float(np.sum(pos_indices)) / ppv_count * inv_length
    else:
        mpv = 0.0
        mipv = 0.0

    # lspv: longest consecutive run — must loop (data dependency)
    lspv_current = 0
    lspv_max = 0
    for v in positive:
        if v:
            lspv_current += 1
            if lspv_current > lspv_max:
                lspv_max = lspv_current
        else:
            lspv_current = 0
    lspv = lspv_max * inv_length

    return float(ppv), mpv, mipv, lspv


def multirocket_feature_extraction_fast(ts_arr, dilations, nfpd, biases,
                                         ts_length, n_feature_per_kernel,
                                         starting_feature_idx, features):
    """Mirrors multirocket_feature_extraction_hls.
    CRITICAL: feature_idx increments by features_this_dilation for EACH kernel,
    not once per dilation. Matches the C++ KERNEL_LOOP structure exactly.
    """
    feature_idx = 0
    for dil_idx, (dilation, features_this_dilation) in enumerate(zip(dilations, nfpd)):
        for kernel_idx in range(NUM_KERNELS):
            conv = apply_kernel_np(ts_arr, kernel_idx, dilation, ts_length)
            for f in range(features_this_dilation):
                current_feature_idx = feature_idx + f
                bias = biases[current_feature_idx]
                ppv, mpv, mipv, lspv = compute_four_pooling_np(conv, bias, ts_length)
                base = (current_feature_idx * n_feature_per_kernel) + starting_feature_idx
                features[base + 0] = ppv
                features[base + 1] = mpv
                features[base + 2] = mipv
                features[base + 3] = lspv
            # Increment per kernel (not per dilation) — matches C++ KERNEL_LOOP
            feature_idx += features_this_dilation


def run_inference(ts, model):
    """Full inference pipeline mirroring multirocket_inference."""
    ts_arr = np.array(ts, dtype=np.float32)
    ts_length = len(ts_arr)
    n_feature_per_kernel = 4
    num_features = model['num_features']
    num_classes = model['num_classes']
    dilations_0 = model['dilations_orig']
    dilations_1 = model['dilations_diff']
    biases_0 = model['biases_orig']
    biases_1 = model['biases_diff']
    num_dilations_0 = model['num_dilations_orig']
    num_dilations_1 = model['num_dilations_diff']
    num_features_0 = len(biases_0)
    num_features_1 = len(biases_1)

    slots_orig = num_features_0 // (NUM_KERNELS * num_dilations_0)
    nfpd_0 = [slots_orig] * num_dilations_0
    slots_diff = num_features_1 // (NUM_KERNELS * num_dilations_1)
    nfpd_1 = [slots_diff] * num_dilations_1

    # First-order difference (mirrors: local_time_series_diff[i] = input[i+1] - input[i])
    ts_diff = ts_arr[1:] - ts_arr[:-1]

    features = [0.0] * num_features

    multirocket_feature_extraction_fast(ts_arr, dilations_0, nfpd_0, biases_0,
                                         ts_length, n_feature_per_kernel, 0, features)

    multirocket_feature_extraction_fast(ts_diff, dilations_1, nfpd_1, biases_1,
                                         ts_length - 1, n_feature_per_kernel,
                                         num_features_0 * n_feature_per_kernel, features)

    # Scale
    mean = np.array(model['scaler_mean'], dtype=np.float64)
    scale = np.array(model['scaler_scale'], dtype=np.float64)
    feat_arr = np.array(features, dtype=np.float64)
    scaled = (feat_arr - mean) / scale

    # Classify
    coef = np.array(model['coefficients'], dtype=np.float64)   # (C, F)
    intercept = np.array(model['intercept'], dtype=np.float64)  # (C,)

    if num_classes == 2:
        score = intercept[0] + float(coef[0] @ scaled)
        predictions = np.array([-score, score])
    else:
        predictions = intercept + coef @ scaled  # (C,)

    return predictions


def evaluate_dataset(model_file, test_file, dataset_name):
    print(f"\n=== {dataset_name} ===")
    with open(model_file) as f:
        model = json.load(f)
    with open(test_file) as f:
        test_data = json.load(f)

    classes = model['classes']
    class_to_idx = {c: i for i, c in enumerate(classes)}

    samples = test_data['time_series']
    raw_labels = test_data['labels']
    # Convert string labels to indices if needed
    if isinstance(raw_labels[0], str):
        labels = [class_to_idx[l] for l in raw_labels]
    else:
        labels = raw_labels

    n = len(samples)
    print(f"  Samples: {n}, Classes: {model['num_classes']}, TS length: {model['time_series_length']}")

    correct = 0
    t0 = time.time()
    for idx, (ts, label) in enumerate(zip(samples, labels)):
        if idx % 50 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (idx + 1)) * (n - idx) if idx > 0 else 0
            print(f"  [{idx}/{n}] elapsed={elapsed:.1f}s eta={eta:.0f}s", end='\r', flush=True)
        preds = run_inference(ts, model)
        pred_class = int(np.argmax(preds))
        if pred_class == label:
            correct += 1

    elapsed = time.time() - t0
    acc = correct / n * 100.0
    print(f"  [{n}/{n}] done in {elapsed:.1f}s                    ")
    print(f"  Accuracy: {correct}/{n} = {acc:.2f}%")
    return acc


if __name__ == '__main__':
    base = '/home/rdave009/minirocket-hls/MiniRocketHLS/multirocket_optimized/models'
    datasets = [
        ('multirocket84_insectsound_model.json',   'multirocket84_insectsound_test_shuf.json',   'InsectSound',   74.0),
        ('multirocket84_mosquitosound_model.json',  'multirocket84_mosquitosound_test_shuf.json', 'MosquitoSound', 88.0),
        ('multirocket84_fruitflies_model.json',     'multirocket84_fruitflies_test_shuf.json',    'FruitFlies',    96.0),
    ]

    results = {}
    for mf, tf, name, exp in datasets:
        acc = evaluate_dataset(f'{base}/{mf}', f'{base}/{tf}', name)
        results[name] = (acc, exp)

    print("\n=== SUMMARY ===")
    all_pass = True
    for name, (acc, exp) in results.items():
        diff = abs(acc - exp)
        status = "PASS" if diff < 5.0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {name}: {acc:.2f}% (expected ~{exp:.0f}%) [{status}]")

    print()
    if all_pass:
        print("VERDICT: ROUND 2 FIXES WORK -- proceed with xclbin rebuild")
    else:
        print("VERDICT: STILL BUGGY -- accuracy mismatch detected")
