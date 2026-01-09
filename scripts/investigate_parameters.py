#!/usr/bin/env python3
"""
Investigate the structure of MiniRocket and MultiRocket parameters attribute
"""

import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket

# Create a small test dataset
np.random.seed(42)
X_train = np.random.randn(10, 100)

print("="*70)
print("INVESTIGATING MINIROCKET PARAMETERS")
print("="*70)

# Train MiniRocket
minirocket = MiniRocket(n_kernels=100, random_state=42)
minirocket.fit(X_train)

print("\nMiniRocket attributes after fit:")
print(f"  Available attributes: {[attr for attr in dir(minirocket) if not attr.startswith('_')]}")

# Check for different attribute names
if hasattr(minirocket, 'parameters'):
    print(f"\n  Has 'parameters' attribute: True")
    print(f"  Type: {type(minirocket.parameters)}")
    print(f"  Length: {len(minirocket.parameters)}")
    print(f"  Structure:")
    for i, param in enumerate(minirocket.parameters):
        print(f"    parameters[{i}]: shape={param.shape if hasattr(param, 'shape') else 'N/A'}, dtype={param.dtype if hasattr(param, 'dtype') else type(param)}")
        if hasattr(param, 'shape') and len(param.shape) > 0 and param.shape[0] <= 5:
            print(f"      Sample values: {param[:min(3, len(param))]}")
else:
    print(f"\n  Has 'parameters' attribute: False")

if hasattr(minirocket, 'kernels_'):
    print(f"\n  Has 'kernels_' attribute: True")
    print(f"  Shape: {minirocket.kernels_.shape}")
else:
    print(f"\n  Has 'kernels_' attribute: False")

if hasattr(minirocket, 'biases_'):
    print(f"\n  Has 'biases_' attribute: True")
    print(f"  Shape: {minirocket.biases_.shape}")
else:
    print(f"\n  Has 'biases_' attribute: False")

if hasattr(minirocket, 'dilations_'):
    print(f"\n  Has 'dilations_' attribute: True")
    print(f"  Shape: {minirocket.dilations_.shape}")
else:
    print(f"\n  Has 'dilations_' attribute: False")

print("\n" + "="*70)
print("INVESTIGATING MULTIROCKET PARAMETERS")
print("="*70)

# Train MultiRocket
multirocket = MultiRocket(n_kernels=100, random_state=42)
multirocket.fit(X_train)

print("\nMultiRocket attributes after fit:")
print(f"  Available attributes: {[attr for attr in dir(multirocket) if not attr.startswith('_')]}")

# Check for different attribute names
if hasattr(multirocket, 'parameters'):
    print(f"\n  Has 'parameters' attribute: True")
    print(f"  Type: {type(multirocket.parameters)}")
    print(f"  Length: {len(multirocket.parameters)}")
    print(f"  Structure:")
    for i, param in enumerate(multirocket.parameters):
        print(f"    parameters[{i}]: shape={param.shape if hasattr(param, 'shape') else 'N/A'}, dtype={param.dtype if hasattr(param, 'dtype') else type(param)}")
        if hasattr(param, 'shape') and len(param.shape) > 0 and param.shape[0] <= 5:
            print(f"      Sample values: {param[:min(3, len(param))]}")
else:
    print(f"\n  Has 'parameters' attribute: False")

if hasattr(multirocket, 'kernels_'):
    print(f"\n  Has 'kernels_' attribute: True")
    print(f"  Shape: {multirocket.kernels_.shape}")
else:
    print(f"\n  Has 'kernels_' attribute: False")

if hasattr(multirocket, 'biases_'):
    print(f"\n  Has 'biases_' attribute: True")
    print(f"  Shape: {multirocket.biases_.shape}")
else:
    print(f"\n  Has 'biases_' attribute: False")

if hasattr(multirocket, 'dilations_'):
    print(f"\n  Has 'dilations_' attribute: True")
    print(f"  Shape: {multirocket.dilations_.shape}")
else:
    print(f"\n  Has 'dilations_' attribute: False")

print("\n" + "="*70)
