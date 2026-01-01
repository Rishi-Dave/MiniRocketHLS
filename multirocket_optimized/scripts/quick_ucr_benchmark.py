#!/usr/bin/env python3
"""
MultiRocket84 Quick UCR Benchmark
Tests on 5 diverse UCR datasets for realistic performance assessment
"""

import numpy as np
import json
import time
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

from custom_multirocket84 import MultiRocket84

# Selected diverse UCR datasets (varying difficulty and characteristics)
UCR_DATASETS = [
    'GunPoint',          # Easy - 100% expected
    'ItalyPowerDemand',  # Medium - ~95% expected
    'Coffee',            # Easy-Medium - ~96% expected
    'ECG200',            # Medium-Hard - ~88% expected
    'Beef',              # Hard - ~70% expected (small training set)
]

def benchmark_dataset(dataset_name, max_dilations_per_kernel=8):
    """Benchmark MultiRocket84 on a single UCR dataset"""

    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")

    try:
        # Load dataset
        print("Loading data...")
        X_train, y_train = load_UCR_UEA_dataset(dataset_name, split="train", return_type="numpy3d")
        X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_type="numpy3d")

        # Convert to 2D if needed
        if X_train.ndim == 3:
            X_train = X_train.squeeze(1)
        if X_test.ndim == 3:
            X_test = X_test.squeeze(1)

        n_train = len(X_train)
        n_test = len(X_test)
        length = X_train.shape[1]
        n_classes = len(np.unique(y_train))

        print(f"  Train: {n_train}, Test: {n_test}, Length: {length}, Classes: {n_classes}")

        # Feature extraction
        print("Extracting features...")
        mr = MultiRocket84(max_dilations_per_kernel=max_dilations_per_kernel)

        feat_start = time.time()
        X_train_features = mr.fit_transform(X_train)
        X_test_features = mr.transform(X_test)
        feat_time = time.time() - feat_start

        n_features = X_train_features.shape[1]
        print(f"  Features: {n_features}")
        print(f"  Feature extraction time: {feat_time:.2f}s")

        # Train classifier
        print("Training classifier...")
        scaler = StandardScaler()
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=min(5, n_train))

        train_start = time.time()
        X_train_scaled = scaler.fit_transform(X_train_features)
        classifier.fit(X_train_scaled, y_train)
        train_time = time.time() - train_start

        # Evaluate
        print("Evaluating...")
        y_train_pred = classifier.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_train_pred)

        X_test_scaled = scaler.transform(X_test_features)

        test_start = time.time()
        y_test_pred = classifier.predict(X_test_scaled)
        test_time = time.time() - test_start

        test_acc = accuracy_score(y_test, y_test_pred)

        # Results
        print(f"\nResults:")
        print(f"  Train accuracy: {train_acc*100:.2f}% ({np.sum(y_train_pred == y_train)}/{n_train})")
        print(f"  Test accuracy:  {test_acc*100:.2f}% ({np.sum(y_test_pred == y_test)}/{n_test})")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Test inference: {test_time/n_test*1000:.2f} ms/sample")

        return {
            'dataset': dataset_name,
            'n_train': int(n_train),
            'n_test': int(n_test),
            'length': int(length),
            'n_classes': int(n_classes),
            'n_features': int(n_features),
            'train_acc': float(train_acc),
            'test_acc': float(test_acc),
            'feat_time_sec': float(feat_time),
            'train_time_sec': float(train_time),
            'test_time_sec': float(test_time),
            'test_ms_per_sample': float(test_time/n_test*1000),
            'status': 'success'
        }

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'dataset': dataset_name,
            'status': 'failed',
            'error': str(e)
        }

def run_quick_ucr_benchmark():
    """Run quick UCR benchmark on 5 diverse datasets"""

    print("="*80)
    print("MultiRocket84 Quick UCR Benchmark")
    print("="*80)
    print(f"\nTesting on {len(UCR_DATASETS)} diverse UCR datasets")
    print()

    results = []
    successful = 0
    failed = 0

    start_time = time.time()

    for dataset in UCR_DATASETS:
        result = benchmark_dataset(dataset)
        results.append(result)

        if result['status'] == 'success':
            successful += 1
        else:
            failed += 1

    total_time = time.time() - start_time

    # Summary statistics
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"\nDatasets tested: {len(UCR_DATASETS)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print()

    # Calculate statistics on successful runs
    successful_results = [r for r in results if r['status'] == 'success']

    if successful_results:
        train_accs = [r['train_acc'] for r in successful_results]
        test_accs = [r['test_acc'] for r in successful_results]

        print("Accuracy Statistics:")
        print(f"  Train: {np.mean(train_accs)*100:.2f}% ± {np.std(train_accs)*100:.2f}%")
        print(f"         Min: {np.min(train_accs)*100:.2f}%, Max: {np.max(train_accs)*100:.2f}%")
        print(f"  Test:  {np.mean(test_accs)*100:.2f}% ± {np.std(test_accs)*100:.2f}%")
        print(f"         Min: {np.min(test_accs)*100:.2f}%, Max: {np.max(test_accs)*100:.2f}%")
        print()

        # Per-dataset results table
        print("Per-Dataset Results:")
        print("-"*80)
        print(f"{'Dataset':<25} {'Train':<10} {'Test':<10} {'Samples':<12} {'Length':<8}")
        print("-"*80)

        for r in successful_results:
            print(f"{r['dataset']:<25} {r['train_acc']*100:>6.2f}%   {r['test_acc']*100:>6.2f}%   "
                  f"{r['n_train']:>4}/{r['n_test']:<4}  {r['length']:>6}")
        print("-"*80)
        print()

    # Save results
    output_file = 'multirocket_quick_ucr_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'benchmark_info': {
                'model': 'MultiRocket84',
                'datasets_tested': len(UCR_DATASETS),
                'successful': successful,
                'failed': failed,
                'total_time_sec': total_time,
            },
            'summary_stats': {
                'train_acc_mean': float(np.mean(train_accs)) if successful_results else None,
                'train_acc_std': float(np.std(train_accs)) if successful_results else None,
                'test_acc_mean': float(np.mean(test_accs)) if successful_results else None,
                'test_acc_std': float(np.std(test_accs)) if successful_results else None,
                'test_acc_min': float(np.min(test_accs)) if successful_results else None,
                'test_acc_max': float(np.max(test_accs)) if successful_results else None,
            },
            'results': results
        }, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    return results

if __name__ == "__main__":
    results = run_quick_ucr_benchmark()
