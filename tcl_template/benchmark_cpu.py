#!/usr/bin/env python3
"""
CPU Baseline Benchmark for MiniRocket FPGA Comparison
Measures inference latency and throughput on CPU for comparison with FPGA results.
"""

import numpy as np
import json
import time
import argparse
from pathlib import Path
import sys

class MiniRocketCPU:
    """Simplified MiniRocket inference matching FPGA implementation."""

    def __init__(self, model_path):
        with open(model_path, 'r') as f:
            model = json.load(f)

        self.num_features = model.get('num_features', [420])[0] if isinstance(model.get('num_features'), list) else model.get('num_features', 420)
        self.num_classes = model.get('num_classes', [4])[0] if isinstance(model.get('num_classes'), list) else model.get('num_classes', 4)
        self.num_dilations = model.get('num_dilations', [5])[0] if isinstance(model.get('num_dilations'), list) else model.get('num_dilations', 5)

        self.dilations = np.array(model['dilations'], dtype=np.int32)
        self.num_features_per_dilation = np.array(model['num_features_per_dilation'], dtype=np.int32)
        self.biases = np.array(model['biases'], dtype=np.float32)
        self.scaler_mean = np.array(model['scaler_mean'], dtype=np.float32)
        self.scaler_scale = np.array(model['scaler_scale'], dtype=np.float32)
        self.classifier_coef = np.array(model['classifier_coef'], dtype=np.float32).reshape(self.num_classes, -1)
        self.classifier_intercept = np.array(model['classifier_intercept'], dtype=np.float32)

        print(f"Model loaded: {self.num_features} features, {self.num_classes} classes, {self.num_dilations} dilations")

    def compute_ppv(self, time_series):
        """Compute proportion of positive values (PPV) features."""
        ts_length = len(time_series)
        features = np.zeros(self.num_features, dtype=np.float32)

        feature_idx = 0
        weights = np.array([1.0, 2.0, 1.0], dtype=np.float32)  # Simplified kernel

        for d_idx in range(self.num_dilations):
            dilation = self.dilations[d_idx]
            num_feat = self.num_features_per_dilation[d_idx]

            for f in range(num_feat):
                if feature_idx >= self.num_features:
                    break

                bias = self.biases[feature_idx] if feature_idx < len(self.biases) else 0.0

                # Convolve with dilated kernel
                positive_count = 0
                total_count = 0

                for t in range(ts_length - 2 * dilation):
                    conv_result = (
                        weights[0] * time_series[t] +
                        weights[1] * time_series[t + dilation] +
                        weights[2] * time_series[t + 2 * dilation]
                    )

                    if conv_result > bias:
                        positive_count += 1
                    total_count += 1

                features[feature_idx] = positive_count / max(total_count, 1)
                feature_idx += 1

        return features

    def scale_features(self, features):
        """Apply standard scaling."""
        return (features - self.scaler_mean[:len(features)]) / (self.scaler_scale[:len(features)] + 1e-8)

    def classify(self, scaled_features):
        """Linear classification."""
        scores = np.dot(self.classifier_coef[:, :len(scaled_features)], scaled_features) + self.classifier_intercept
        return scores

    def predict(self, time_series):
        """Full inference pipeline."""
        features = self.compute_ppv(time_series)
        scaled = self.scale_features(features)
        scores = self.classify(scaled)
        return np.argmax(scores), scores


def generate_synthetic_dataset(num_samples, ts_length, num_classes):
    """Generate synthetic time series dataset."""
    samples = []
    labels = []

    for i in range(num_samples):
        label = i % num_classes
        freq = 1.0 + label * 0.5
        phase = label * np.pi / num_classes

        t = np.linspace(0, 1, ts_length)
        sample = np.sin(2 * np.pi * freq * t + phase) + 0.1 * np.random.randn(ts_length)

        samples.append(sample.astype(np.float32))
        labels.append(label)

    return samples, labels


def load_csv_dataset(filepath):
    """Load dataset from CSV (UCR format: label, value1, value2, ...)."""
    samples = []
    labels = []

    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            values = [float(x) for x in line.strip().split(',')]
            labels.append(int(values[0]))
            samples.append(np.array(values[1:], dtype=np.float32))

    print(f"Dataset loaded: {len(samples)} samples, {len(samples[0])} time points")
    return samples, labels


def benchmark_cpu(model, samples, labels, warmup=3, verbose=False):
    """Run CPU benchmark and collect timing statistics."""
    num_samples = len(samples)

    # Warmup
    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        model.predict(samples[0])

    # Benchmark
    print(f"Processing {num_samples} samples...")

    predictions = []
    times_per_sample = []

    total_start = time.perf_counter()

    for i, sample in enumerate(samples):
        sample_start = time.perf_counter()
        pred_class, scores = model.predict(sample)
        sample_end = time.perf_counter()

        predictions.append(pred_class)
        times_per_sample.append((sample_end - sample_start) * 1000)  # ms

        if verbose and i < 10:
            print(f"Sample {i}: label={labels[i]}, predicted={pred_class}, score={scores[pred_class]:.4f}")

    total_end = time.perf_counter()
    total_time_ms = (total_end - total_start) * 1000

    # Calculate accuracy
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = 100.0 * correct / num_samples

    # Calculate statistics
    avg_time = np.mean(times_per_sample)
    std_time = np.std(times_per_sample)
    min_time = np.min(times_per_sample)
    max_time = np.max(times_per_sample)
    throughput = num_samples * 1000 / total_time_ms

    return {
        'num_samples': num_samples,
        'total_time_ms': total_time_ms,
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'throughput': throughput,
        'accuracy': accuracy,
        'correct': correct
    }


def print_results(results, csv_output=False):
    """Print benchmark results."""
    if csv_output:
        print("num_samples,total_ms,avg_ms,std_ms,min_ms,max_ms,throughput,accuracy")
        print(f"{results['num_samples']},{results['total_time_ms']:.4f},{results['avg_time_ms']:.4f},"
              f"{results['std_time_ms']:.4f},{results['min_time_ms']:.4f},{results['max_time_ms']:.4f},"
              f"{results['throughput']:.2f},{results['accuracy']:.2f}")
    else:
        print("\n========== CPU TIMING RESULTS ==========")
        print(f"Samples processed: {results['num_samples']}")
        print(f"Total time:        {results['total_time_ms']:.3f} ms")
        print(f"Avg per sample:    {results['avg_time_ms']:.4f} ms (std: {results['std_time_ms']:.4f})")
        print(f"Min/Max:           {results['min_time_ms']:.4f} / {results['max_time_ms']:.4f} ms")
        print(f"Throughput:        {results['throughput']:.1f} inferences/sec")
        print("=========================================")

        print("\n========== ACCURACY RESULTS ==========")
        print(f"Correct predictions: {results['correct']} / {results['num_samples']}")
        print(f"Accuracy: {results['accuracy']:.2f}%")
        print("=======================================")


def main():
    parser = argparse.ArgumentParser(description='CPU Baseline Benchmark for MiniRocket')
    parser.add_argument('model', help='Model JSON file')
    parser.add_argument('--dataset', help='Dataset CSV file (UCR format)')
    parser.add_argument('--synthetic', type=int, default=100, help='Number of synthetic samples')
    parser.add_argument('--warmup', type=int, default=3, help='Warmup iterations')
    parser.add_argument('--csv', action='store_true', help='Output in CSV format')
    parser.add_argument('--verbose', action='store_true', help='Show per-sample results')

    args = parser.parse_args()

    # Load model
    model = MiniRocketCPU(args.model)

    # Load or generate dataset
    if args.dataset:
        samples, labels = load_csv_dataset(args.dataset)
    else:
        samples, labels = generate_synthetic_dataset(args.synthetic, 128, model.num_classes)
        print(f"Generated {len(samples)} synthetic samples")

    # Run benchmark
    results = benchmark_cpu(model, samples, labels, warmup=args.warmup, verbose=args.verbose)

    # Print results
    print_results(results, csv_output=args.csv)

    return 0


if __name__ == '__main__':
    sys.exit(main())
