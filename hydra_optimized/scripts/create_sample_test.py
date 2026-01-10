#!/usr/bin/env python3
"""
Create a smaller test dataset by sampling from the full test set.
This allows quick validation without waiting for 25,000 samples.
"""

import json
import sys
import numpy as np

def sample_test_data(input_json, output_json, num_samples=1000):
    """Sample a subset of test data from the full test file"""

    print(f"Loading full test data from: {input_json}")
    with open(input_json, 'r') as f:
        data = json.load(f)

    total_samples = data['num_samples']
    print(f"Total samples available: {total_samples}")

    if num_samples > total_samples:
        print(f"Warning: Requested {num_samples} samples but only {total_samples} available")
        num_samples = total_samples

    # Sample evenly across the dataset
    indices = np.linspace(0, total_samples - 1, num_samples, dtype=int)

    print(f"Sampling {num_samples} indices...")
    sampled_data = {
        "dataset": data['dataset'],
        "num_samples": num_samples,
        "time_series_length": data['time_series_length'],
        "num_classes": data['num_classes'],
        "time_series": [data['time_series'][i] for i in indices],
        "labels": [data['labels'][i] for i in indices]
    }

    print(f"Saving sampled data to: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(sampled_data, f, indent=2)

    print(f"Done! Created test file with {num_samples} samples")
    print(f"Estimated test time: {num_samples * 5.2 / 1000:.1f} seconds")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: create_sample_test.py <input.json> <output.json> [num_samples]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    sample_test_data(input_file, output_file, num_samples)
