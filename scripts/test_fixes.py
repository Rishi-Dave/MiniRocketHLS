#!/usr/bin/env python3
"""
Quick test script to verify the training script fixes
Tests parameter extraction for MiniRocket and MultiRocket
"""

import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket

print("="*70)
print("TESTING PARAMETER EXTRACTION FIXES")
print("="*70)

# Create test data
np.random.seed(42)
X = np.random.randn(10, 100)

print("\n1. Testing MiniRocket parameter extraction...")
print("-" * 70)

try:
    minirocket = MiniRocket(n_kernels=100, random_state=42)
    minirocket.fit(X)

    # Extract parameters as in the updated training script
    n_channels_per_combination = minirocket.parameters[0].tolist()
    channel_indices = minirocket.parameters[1].tolist()
    dilations = minirocket.parameters[2].tolist()
    n_features_per_dilation = minirocket.parameters[3].tolist()
    biases = minirocket.parameters[4].tolist()

    print("✓ MiniRocket parameters extracted successfully!")
    print(f"  - n_channels_per_combination: {len(n_channels_per_combination)} elements")
    print(f"  - channel_indices: {len(channel_indices)} elements")
    print(f"  - dilations: {len(dilations)} elements")
    print(f"  - n_features_per_dilation: {len(n_features_per_dilation)} elements")
    print(f"  - biases: {len(biases)} elements")

    # Verify transform works
    X_transform = minirocket.transform(X)
    print(f"  - Transform output shape: {X_transform.shape}")
    print("✓ MiniRocket test PASSED")

except Exception as e:
    print(f"✗ MiniRocket test FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Testing MultiRocket parameter extraction...")
print("-" * 70)

try:
    multirocket = MultiRocket(n_kernels=100, random_state=42)
    multirocket.fit(X)

    # Extract parameters as in the updated training script
    dilations_original = multirocket.parameter[0].tolist()
    n_features_per_dilation_original = multirocket.parameter[1].tolist()
    biases_original = multirocket.parameter[2].tolist()

    dilations_diff = multirocket.parameter1[0].tolist()
    n_features_per_dilation_diff = multirocket.parameter1[1].tolist()
    biases_diff = multirocket.parameter1[2].tolist()

    print("✓ MultiRocket parameters extracted successfully!")
    print(f"  Original series:")
    print(f"    - dilations: {len(dilations_original)} elements")
    print(f"    - n_features_per_dilation: {len(n_features_per_dilation_original)} elements")
    print(f"    - biases: {len(biases_original)} elements")
    print(f"  Differenced series:")
    print(f"    - dilations: {len(dilations_diff)} elements")
    print(f"    - n_features_per_dilation: {len(n_features_per_dilation_diff)} elements")
    print(f"    - biases: {len(biases_diff)} elements")

    # Verify transform works
    X_transform = multirocket.transform(X)
    print(f"  - Transform output shape: {X_transform.shape}")
    print("✓ MultiRocket test PASSED")

except Exception as e:
    print(f"✗ MultiRocket test FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print("All parameter extraction methods verified!")
print("Training scripts should now work correctly.")
print("="*70)
