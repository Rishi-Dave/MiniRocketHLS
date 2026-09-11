#!/usr/bin/env python3
"""
Time aeon's ACTUAL Python MiniRocket + Ridge inference pipeline, to get
a real comparison number for the paper's CPU baseline — instead of
reading it off a bar chart in Figure 8.

Matches the paper's methodology (Section IV-B) as closely as possible:
  - "multi-threaded Python 3.10" + Aeon-toolkit
  - regression classifier: Ridge Regression (footnote 8)
  - timing via Python's `time` module (Section IV-A)
  - full train/test split from the UCR archive (Table I), not a
    synthetic/compact subset

IMPORTANT SCOPING NOTE: this times INFERENCE only (MiniRocket transform +
scaler + classifier predict on the test set) — NOT model fitting/training
— to stay directly comparable to your C++/CUDA inference benchmarks and
to the paper's single-node execution-time experiments (Section IV-C),
which measure "starting with the time series loaded into memory," i.e.
inference on an already-trained model.

aeon's MiniRocket transform parallelizes internally via `n_jobs` (joblib
under the hood) rather than raw Python `multiprocessing`, which is a
minor implementation detail vs. the paper's own description, but the
effect — using multiple CPU threads/cores for the transform — is the
same thing being measured.

Usage:
    python time_aeon_baseline.py InsectSound
    python time_aeon_baseline.py InsectSound 1     # force single-threaded, for comparison
    python time_aeon_baseline.py FruitFlies
"""

import sys
import time
import os

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import RidgeClassifierCV
from aeon.datasets import load_classification
from aeon.transformations.collection.convolution_based import MiniRocket


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <UCR_dataset_name> [n_jobs]")
        print("Example: python time_aeon_baseline.py InsectSound")
        print("         python time_aeon_baseline.py InsectSound 1   (single-threaded)")
        sys.exit(1)

    dataset_name = sys.argv[1]
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else -1  # -1 = use all cores

    print(f"CPUs visible to this process: {os.cpu_count()}")
    print(f"n_jobs for MiniRocket transform: {n_jobs} "
          f"({'all cores' if n_jobs == -1 else n_jobs})")

    # ------------------------------------------------------------
    # Load the REAL full train/test split (same as export_minirocket_model.py)
    # ------------------------------------------------------------
    print(f"\nLoading '{dataset_name}' (full train/test split) ...")
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")

    if X_train.ndim == 3:
        assert X_train.shape[1] == 1, "This script only supports univariate series"
        X_train = X_train.squeeze(1)
        X_test = X_test.squeeze(1)

    n_train, L = X_train.shape
    n_test = X_test.shape[0]
    print(f"  Train: {n_train} samples, Test: {n_test} samples, length: {L}")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # ------------------------------------------------------------
    # TRAINING (not timed — matches the paper's single-node scoping,
    # which measures execution time starting from an in-memory,
    # already-trained model, same as your C++/CUDA benchmarks)
    # ------------------------------------------------------------
    print("\nFitting MiniRocket + scaler + Ridge (NOT timed — this is training) ...")
    mr = MiniRocket(random_state=42, n_jobs=n_jobs)  # default n_kernels=10_000, matching the paper's ~10,000 features
    mr.fit(X_train[:, np.newaxis, :])
    X_train_feat = mr.transform(X_train[:, np.newaxis, :])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)

    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(X_train_scaled, y_train_enc)
    print(f"  Extracted {X_train_feat.shape[1]} features per sample (matches num_features in your model.json)")

    # ------------------------------------------------------------
    # INFERENCE — THIS is what gets timed, matching your C++/CUDA
    # per-run timing and the paper's single-node execution-time figures.
    # Uses Python's `time` module per the paper's own methodology
    # (Section IV-A cites Python's time library).
    # ------------------------------------------------------------
    print(f"\nTiming inference on {n_test} test samples ...")
    t0 = time.perf_counter()

    X_test_feat = mr.transform(X_test[:, np.newaxis, :])   # MiniRocket transform (Convolution Engine + PPV Pooling)
    X_test_scaled = scaler.transform(X_test_feat)           # z-score normalization
    predictions = clf.predict(X_test_scaled)                # Regression Classifier

    t1 = time.perf_counter()
    total_s = t1 - t0
    throughput = n_test / total_s

    accuracy = (predictions == y_test_enc).mean()

    print("\n========== AEON PYTHON BASELINE RESULTS ==========")
    print(f"Dataset:     {dataset_name}")
    print(f"Samples:     {n_test}")
    print(f"n_jobs:      {n_jobs}")
    print(f"Accuracy:    {accuracy * 100:.4f}% ({int((predictions == y_test_enc).sum())}/{n_test})")
    print(f"Total time:  {total_s:.3f} s")
    print(f"Throughput:  {throughput:.1f} inferences/sec")
    print(f"Mean latency: {total_s / n_test * 1000:.3f} ms/sample (batched, not single-sample — "
          f"see note below)")
    print("===================================================")
    print("\nNOTE: this times the WHOLE test set transformed/classified together")
    print("(aeon's transform is vectorized/batched internally), analogous to your")
    print("CUDA file's 'batched throughput' mode — NOT your single-sample batch=1")
    print("latency loop. For a fairer comparison to your C++/CUDA per-sample")
    print("latency numbers, this Python throughput number is the one to set")
    print("against your MULTITHREADED C++ throughput and your GPU BATCHED")
    print("throughput numbers, not your single-sample latency numbers.")


if __name__ == "__main__":
    main()
