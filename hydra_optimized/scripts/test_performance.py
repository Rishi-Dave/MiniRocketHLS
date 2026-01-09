#!/usr/bin/env python3
"""Quick performance test for HYDRA"""

import time
import numpy as np
from custom_hydra import Hydra

# Generate synthetic test data similar to InsectSound
np.random.seed(42)
n_train = 220
n_test = 42
length = 600

print("Generating synthetic test data...")
print(f"  Train: {n_train} samples x {length} timesteps")
print(f"  Test: {n_test} samples x {length} timesteps")

X_train = np.random.randn(n_train, length)
y_train = np.random.randint(0, 10, n_train)
X_test = np.random.randn(n_test, length)
y_test = np.random.randint(0, 10, n_test)

print("\nTraining HYDRA with 512 kernels...")
start = time.time()
hydra = Hydra(num_kernels=512, random_state=42)
hydra.fit(X_train, y_train, verbose=True)
train_time = time.time() - start
print(f"Training completed in {train_time:.2f}s")

print("\nTesting...")
start = time.time()
y_pred = hydra.predict(X_test)
test_time = time.time() - start
print(f"Testing completed in {test_time:.2f}s")

acc = np.mean(y_pred == y_test)
print(f"\nAccuracy: {acc*100:.2f}%")
print(f"Total time: {train_time + test_time:.2f}s")
