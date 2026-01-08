#!/usr/bin/env python3
"""Test dataset loading speed"""

import time
from aeon.datasets import load_classification

print("Testing InsectSound dataset loading...")
print("="*50)

start = time.time()
X_train, y_train = load_classification("InsectSound", split="train", extract_path=None)
load_time = time.time() - start

print(f"Loaded training data in {load_time:.2f}s")
print(f"  Shape: {X_train.shape}")
print(f"  Type: {type(X_train)}")

if hasattr(X_train, 'values'):
    X_train = X_train.values
print(f"  After conversion: {X_train.shape}")

start = time.time()
X_test, y_test = load_classification("InsectSound", split="test", extract_path=None)
load_time = time.time() - start

print(f"\nLoaded test data in {load_time:.2f}s")
print(f"  Shape: {X_test.shape}")
