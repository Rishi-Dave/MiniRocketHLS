#!/usr/bin/env python3
"""
Investigate the structure of MiniRocket and MultiRocket parameters in more detail
"""

import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket

# Create a small test dataset
np.random.seed(42)
X_train = np.random.randn(10, 100)

print("="*70)
print("INVESTIGATING MINIROCKET PARAMETERS IN DETAIL")
print("="*70)

# Train MiniRocket with specific number of kernels
n_kernels = 84
minirocket = MiniRocket(n_kernels=n_kernels, random_state=42)
minirocket.fit(X_train)

print(f"\nRequested n_kernels: {n_kernels}")
print(f"Actual n_kernels_: {minirocket.n_kernels_}")
print(f"\nParameters tuple length: {len(minirocket.parameters)}")

for i, param in enumerate(minirocket.parameters):
    print(f"\nparameters[{i}]:")
    print(f"  Type: {type(param)}")
    print(f"  Shape: {param.shape if hasattr(param, 'shape') else 'N/A'}")
    print(f"  Dtype: {param.dtype if hasattr(param, 'dtype') else 'N/A'}")
    print(f"  Unique values: {np.unique(param)[:10] if hasattr(param, '__len__') and len(param) > 0 else param}")
    if hasattr(param, '__len__') and len(param) <= 10:
        print(f"  All values: {param}")

# Let's also check with a larger kernel count to understand the pattern
print("\n" + "="*70)
print("TESTING WITH 100 KERNELS")
print("="*70)

n_kernels = 100
minirocket2 = MiniRocket(n_kernels=n_kernels, random_state=42)
minirocket2.fit(X_train)

print(f"\nRequested n_kernels: {n_kernels}")
print(f"Actual n_kernels_: {minirocket2.n_kernels_}")
print(f"\nParameters tuple structure:")

for i, param in enumerate(minirocket2.parameters):
    print(f"  parameters[{i}]: shape={param.shape if hasattr(param, 'shape') else 'N/A'}")

# Based on MiniRocket paper and implementation:
# parameters[0] should be dilations
# parameters[1] should be num_features_per_dilation (or similar)
# parameters[2] and [3] might be bias-related
# parameters[4] should be biases

print("\n" + "="*70)
print("INVESTIGATING MULTIROCKET PARAMETERS IN DETAIL")
print("="*70)

# Train MultiRocket
n_kernels = 100
multirocket = MultiRocket(n_kernels=n_kernels, random_state=42)
multirocket.fit(X_train)

print(f"\nRequested n_kernels: {n_kernels}")
print(f"\nMultiRocket attributes:")

# Check for parameter and parameter1
if hasattr(multirocket, 'parameter'):
    print(f"\n  Has 'parameter' attribute: True")
    print(f"  Type: {type(multirocket.parameter)}")
    if isinstance(multirocket.parameter, tuple):
        print(f"  Length: {len(multirocket.parameter)}")
        for i, param in enumerate(multirocket.parameter):
            print(f"    parameter[{i}]: shape={param.shape if hasattr(param, 'shape') else 'N/A'}, dtype={param.dtype if hasattr(param, 'dtype') else type(param)}")

if hasattr(multirocket, 'parameter1'):
    print(f"\n  Has 'parameter1' attribute: True")
    print(f"  Type: {type(multirocket.parameter1)}")
    if isinstance(multirocket.parameter1, tuple):
        print(f"  Length: {len(multirocket.parameter1)}")
        for i, param in enumerate(multirocket.parameter1):
            print(f"    parameter1[{i}]: shape={param.shape if hasattr(param, 'shape') else 'N/A'}, dtype={param.dtype if hasattr(param, 'dtype') else type(param)}")

print("\n" + "="*70)
