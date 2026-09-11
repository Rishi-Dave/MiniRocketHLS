#!/usr/bin/env python3
"""
Train MiniRocket + Ridge on a REAL, FULL-SIZE UCR/UEA archive dataset,
and export model.json + test_data.json in the exact schema that
minirocket_cpu_inference.cpp / minirocket_gpu_inference.cu expect.

This exists because the "_compact" test files you've been testing with
(e.g. InsectSound_test_data_compact.json) only have 5 samples — fine for
a correctness sanity check, but too small for meaningful throughput,
latency percentile, or accuracy numbers. This script pulls the FULL
train/test split for a named dataset directly from the UCR archive
(via aeon's built-in downloader/cache) and trains a real model on it.

Usage:
    pip install aeon scikit-learn numpy --break-system-packages   # if needed
    python export_minirocket_model.py InsectSound
    python export_minirocket_model.py FruitFlies
    python export_minirocket_model.py MosquitoSound

    # Or any other UCR/UEA archive dataset name, e.g.:
    python export_minirocket_model.py ECG5000
    python export_minirocket_model.py Wafer

Produces, in the current directory:
    <name>_minirocket_model.json
    <name>_test_data_full.json      (note: "_full", not "_compact" —
                                      won't overwrite your existing files)

Then run your existing binaries against these, e.g.:
    ./minirocket_cpu <name>_minirocket_model.json <name>_test_data_full.json ...
    ./minirocket_gpu <name>_minirocket_model.json <name>_test_data_full.json ...
"""

import sys
import json
from itertools import combinations

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import RidgeClassifierCV
from aeon.datasets import load_classification
from aeon.transformations.collection.convolution_based import MiniRocket


def convert(obj):
    """Make numpy types JSON-serializable."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <UCR_dataset_name>")
        print("Example: python export_minirocket_model.py InsectSound")
        sys.exit(1)

    dataset_name = sys.argv[1]

    # ------------------------------------------------------------
    # 1. Load the REAL, full train/test split from the UCR/UEA archive.
    #    aeon downloads and caches these automatically the first time.
    # ------------------------------------------------------------
    print(f"Loading '{dataset_name}' (full train/test split) ...")
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")

    # aeon returns 3D arrays (n_cases, n_channels, n_timepoints) even for
    # univariate data. This pipeline (and your C++/CUDA code) only
    # handles univariate series, so squeeze out the channel dimension.
    if X_train.ndim == 3:
        assert X_train.shape[1] == 1, "This exporter only supports univariate series"
        X_train = X_train.squeeze(1)
        X_test = X_test.squeeze(1)

    n_train, L = X_train.shape
    n_test = X_test.shape[0]
    print(f"  Train: {n_train} samples, Test: {n_test} samples, length: {L}")

    # ------------------------------------------------------------
    # 2. Class labels: UCR archive labels are often strings (e.g. "1",
    #    "2", ...). Your C++ model.classes is an int array, so encode
    #    them to a dense 0..n_classes-1 integer range. This encoding
    #    MUST be used consistently for both training labels and the
    #    y_test values written to test_data.json, so predictions can
    #    be compared correctly downstream.
    # ------------------------------------------------------------
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    classes = list(range(len(le.classes_)))  # 0..n_classes-1
    print(f"  Classes: {len(classes)} (original labels: {list(le.classes_)})")

    # ------------------------------------------------------------
    # 3. Fit MiniRocket feature extractor.
    #    aeon expects 3D input (n_cases, n_channels, n_timepoints), so
    #    add back the channel dimension of size 1.
    # ------------------------------------------------------------
    print("Fitting MiniRocket transform ...")
    mr = MiniRocket(random_state=42)  # default num_kernels rounds to a multiple of 84
    mr.fit(X_train[:, np.newaxis, :])

    X_train_feat = mr.transform(X_train[:, np.newaxis, :])
    X_test_feat = mr.transform(X_test[:, np.newaxis, :])
    print(f"  Extracted {X_train_feat.shape[1]} features per sample")

    # ------------------------------------------------------------
    # 4. Pull out MiniRocket's internal fitted parameters. The exact
    #    tuple layout of `parameters_` has varied slightly across aeon
    #    versions (univariate vs. multivariate-capable builds), so
    #    handle both a 3-tuple and a 5-tuple defensively.
    # ------------------------------------------------------------
    # NOTE: the attribute is `parameters` (no trailing underscore) — this
    # breaks the usual sklearn "fitted attributes end in _" convention,
    # easy to trip over. Always a 5-tuple, even for univariate data:
    # (n_channels_per_comb, channel_indices, dilations,
    #  num_features_per_dilation, biases) — channel info is unused here.
    params = mr.parameters
    if len(params) == 5:
        _, _, dilations, num_features_per_dilation, biases = params
    elif len(params) == 3:
        # Fallback in case an older/different aeon version returns the
        # bare 3-tuple without channel info.
        dilations, num_features_per_dilation, biases = params
    else:
        raise RuntimeError(
            f"Unexpected MiniRocket.parameters layout (len={len(params)}); "
            "inspect mr.parameters manually and adjust this script."
        )

    dilations = np.asarray(dilations).astype(int).tolist()
    num_features_per_dilation = np.asarray(num_features_per_dilation).astype(int).tolist()
    biases = np.asarray(biases).astype(float).tolist()

    # The 84 kernels are FIXED and deterministic — not learned — so this
    # is the same for every dataset. Matches MiniRocket._indices internally.
    kernel_indices = [list(c) for c in combinations(range(9), 3)]
    assert len(kernel_indices) == 84

    num_dilations = len(dilations)
    num_features = X_train_feat.shape[1]
    # Sanity check: our understanding of the feature layout should match
    # the transformer's actual output width.
    assert num_features == sum(84 * n for n in num_features_per_dilation), (
        "Feature count mismatch — MiniRocket's internal layout may differ "
        "from what minirocket_cpu_inference.cpp assumes. Investigate before trusting results."
    )

    # ------------------------------------------------------------
    # 5. Scale features, then fit RidgeClassifierCV — same pipeline
    #    aeon's own MiniRocketClassifier uses internally.
    # ------------------------------------------------------------
    print("Fitting scaler + RidgeClassifierCV ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)  # not used for export, just for a quick sanity accuracy check below

    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(X_train_scaled, y_train_enc)

    # Quick Python-side accuracy check, so you have a reference number to
    # compare your C++/CUDA accuracy against.
    py_accuracy = clf.score(X_test_scaled, y_test_enc)
    print(f"  Python (sklearn) test accuracy: {py_accuracy * 100:.2f}%")

    # RidgeClassifierCV gives coef_ shape (1, n_features) for binary,
    # (n_classes, n_features) for multiclass — matches how
    # minirocket_cpu_inference.cpp's classify() branches on coef size.
    classifier_coef = clf.coef_.tolist()
    classifier_intercept = np.atleast_1d(clf.intercept_).tolist()

    # ------------------------------------------------------------
    # 6. Write model.json
    # ------------------------------------------------------------
    model = {
        "num_kernels": 84,
        "num_dilations": num_dilations,
        "num_features": num_features,
        "num_classes": len(classes),
        "time_series_length": L,
        "kernel_indices": kernel_indices,
        "dilations": dilations,
        "num_features_per_dilation": num_features_per_dilation,
        "biases": biases,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classifier_coef": classifier_coef,
        "classifier_intercept": classifier_intercept,
        "classes": classes,
    }
    model_path = f"{dataset_name}_minirocket_model.json"
    with open(model_path, "w") as f:
        json.dump(model, f, default=convert)
    print(f"Wrote {model_path}")

    # ------------------------------------------------------------
    # 7. Write test_data.json — the FULL test set, not a 5-sample
    #    compact subset, so throughput/latency stats are meaningful.
    # ------------------------------------------------------------
    test_data = {
        "dataset_name": dataset_name,
        "num_samples": n_test,
        "time_series_length": L,
        "X_test": X_test.tolist(),
        "y_test": [int(v) for v in y_test_enc],
    }
    test_path = f"{dataset_name}_test_data_full.json"
    with open(test_path, "w") as f:
        json.dump(test_data, f, default=convert)
    print(f"Wrote {test_path}")

    print("\nDone. Run your C++/CUDA binaries against these files, e.g.:")
    print(f"  ./minirocket_cpu {model_path} {test_path} ../results/{dataset_name}_cpu_full.csv")
    print(f"  ./minirocket_gpu {model_path} {test_path} ../results/{dataset_name}_gpu_full.csv 256")
    print(f"\nCompare the reported accuracy against the Python reference above ({py_accuracy*100:.2f}%) —")
    print("they should match closely (small floating-point differences aside).")


if __name__ == "__main__":
    main()
