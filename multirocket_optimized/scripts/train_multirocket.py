#!/usr/bin/env python3
"""
MultiRocket Training Script for FPGA Implementation
Trains a MultiRocket model on UCR time series datasets and exports for FPGA

MultiRocket uses:
- Two representations: original + first-order difference
- Four pooling operators: PPV, MPV, MIPV, LSPV
- Generates ~2,688 features per representation = ~5,376 total features
"""

import numpy as np
import json
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sktime.datasets import load_UCR_UEA_dataset
import warnings
warnings.filterwarnings('ignore')

# Try to import MultiRocket from aeon (newer) or sktime (older)
try:
    from aeon.transformations.collection.convolution_based import MultiRocket
    print("Using aeon.MultiRocket")
except ImportError:
    try:
        from sktime.transformations.panel.rocket import MultiRocket
        print("Using sktime.MultiRocket")
    except ImportError:
        print("ERROR: Neither aeon nor sktime MultiRocket found!")
        print("Install with: pip install aeon")
        exit(1)


def train_multirocket_model(dataset_name="GunPoint", num_features=2688, random_state=42):
    """
    Train MultiRocket model on a UCR dataset

    Args:
        dataset_name: Name of UCR dataset
        num_features: Target number of features (will be adjusted for MultiRocket)
        random_state: Random seed

    Returns:
        Trained model components and test data
    """

    print(f"\n{'='*60}")
    print(f"Training MultiRocket on {dataset_name}")
    print(f"{'='*60}\n")

    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    X_train, y_train = load_UCR_UEA_dataset(dataset_name, split="train", return_type="numpy3d")
    X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_type="numpy3d")

    # Convert to 2D if needed (num_samples, time_steps)
    if X_train.ndim == 3:
        X_train = X_train.squeeze(1)
    if X_test.ndim == 3:
        X_test = X_test.squeeze(1)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Classes: {np.unique(y_train)}")
    print(f"Time series length: {X_train.shape[1]}")

    # Initialize MultiRocket
    # MultiRocket parameters:
    # - num_kernels: 84 (default, same as MiniRocket)
    # - max_dilations_per_kernel: controls number of dilations
    # - reference_length: used to determine dilations

    # Calculate dilations to control feature count
    # MultiRocket generates: num_kernels × num_dilations × 4 pooling × 2 representations
    # For 2,688 features: 84 × 8 × 4 × 2 = 5,376 (too many, use less dilations)
    # For 1,344 features: 84 × 4 × 4 × 2 = 2,688 ✓

    print(f"\nInitializing MultiRocket...")
    print(f"  num_kernels=84 (default)")
    print(f"  max_dilations_per_kernel=4 (targeting ~2,700 features)")
    print(f"  4 pooling operators × 2 representations = 8 features per kernel-dilation")

    multirocket = MultiRocket(
        num_kernels=84,
        max_dilations_per_kernel=4,
        random_state=random_state
    )

    # Fit and transform training data
    print(f"\nExtracting training features...")
    X_train_transform = multirocket.fit_transform(X_train)
    print(f"Training features shape: {X_train_transform.shape}")

    # Transform test data
    print(f"Extracting test features...")
    X_test_transform = multirocket.transform(X_test)
    print(f"Test features shape: {X_test_transform.shape}")

    # Train classifier with StandardScaler
    print(f"\nTraining Ridge classifier with StandardScaler...")
    scaler = StandardScaler()
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=5)

    X_train_scaled = scaler.fit_transform(X_train_transform)
    classifier.fit(X_train_scaled, y_train)

    # Evaluate
    print(f"\nEvaluating...")
    y_train_pred = classifier.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    X_test_scaled = scaler.transform(X_test_transform)
    y_test_pred = classifier.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\nResults:")
    print(f"  Train accuracy: {train_accuracy:.4f} ({np.sum(y_train == y_train_pred)}/{len(y_train)})")
    print(f"  Test accuracy:  {test_accuracy:.4f} ({np.sum(y_test == y_test_pred)}/{len(y_test)})")

    return {
        'multirocket': multirocket,
        'scaler': scaler,
        'classifier': classifier,
        'X_test': X_test,
        'y_test': y_test,
        'X_test_transform': X_test_transform,
        'test_accuracy': test_accuracy
    }


def export_multirocket_model_for_fpga(model_data, output_file="multirocket_model.json"):
    """
    Export MultiRocket model to JSON format for FPGA implementation

    IMPORTANT NOTE: MultiRocket's internal structure is more complex than MiniRocket
    This export extracts the necessary parameters for FPGA implementation
    """

    multirocket = model_data['multirocket']
    scaler = model_data['scaler']
    classifier = model_data['classifier']

    print(f"\n{'='*60}")
    print(f"Exporting model for FPGA: {output_file}")
    print(f"{'='*60}\n")

    # MultiRocket internal parameters (these are tricky to extract)
    # We'll need to inspect the MultiRocket object structure

    print("WARNING: MultiRocket parameter extraction is complex!")
    print("MultiRocket stores parameters differently than MiniRocket.")
    print("\nMultiRocket object attributes:")
    for attr in dir(multirocket):
        if not attr.startswith('_'):
            print(f"  {attr}")

    # Try to extract parameters
    # Note: This may vary depending on aeon/sktime version
    try:
        # Get fitted parameters
        if hasattr(multirocket, 'parameters_'):
            params = multirocket.parameters_
            print(f"\nFound parameters_: {type(params)}")
        elif hasattr(multirocket, '_parameters'):
            params = multirocket._parameters
            print(f"\nFound _parameters: {type(params)}")
        else:
            print("\nERROR: Cannot find MultiRocket parameters!")
            print("This requires manual extraction based on your aeon/sktime version.")
            print("Proceeding with placeholder structure...")
            params = None

        # MultiRocket structure (from paper):
        # - 84 kernels (same as MiniRocket)
        # - Dilations for original representation
        # - Dilations for diff representation
        # - Biases for each kernel-dilation combination
        # - 4 pooling operators applied to each convolution

        # For now, create a simplified export structure
        # This will need to be adapted based on actual MultiRocket internals

        export_data = {
            'model_type': 'MultiRocket',
            'num_features': scaler.mean_.shape[0],
            'num_classes': len(classifier.classes_),
            'time_series_length': model_data['X_test'].shape[1],
            'classes': classifier.classes_.tolist(),

            # MultiRocket parameters (simplified)
            'num_kernels': 84,
            'num_dilations': 4,  # Adjust based on actual model
            'num_pooling_operators': 4,  # PPV, MPV, MIPV, LSPV
            'num_representations': 2,  # Original + diff

            # Placeholder for actual parameters
            # These need to be extracted from multirocket object
            'dilations_orig': [1, 2, 4, 8],  # Placeholder
            'dilations_diff': [1, 2, 4, 8],  # Placeholder
            'biases_orig': np.zeros(84 * 4).tolist(),  # Placeholder
            'biases_diff': np.zeros(84 * 4).tolist(),  # Placeholder

            # StandardScaler parameters
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),

            # Ridge classifier parameters
            'coefficients': classifier.coef_.tolist(),
            'intercept': classifier.intercept_.tolist(),

            # Metadata
            'alpha': float(classifier.alpha_),
            'test_accuracy': float(model_data['test_accuracy'])
        }

        print(f"\nExport summary:")
        print(f"  Total features: {export_data['num_features']}")
        print(f"  Expected: 84 kernels × 4 dilations × 4 pooling × 2 repr = {84*4*4*2}")
        print(f"  Num classes: {export_data['num_classes']}")
        print(f"  Time series length: {export_data['time_series_length']}")
        print(f"  Test accuracy: {export_data['test_accuracy']:.4f}")

        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"\nModel exported to {output_file}")
        print(f"File size: {len(json.dumps(export_data)) / 1024:.1f} KB")

        return export_data

    except Exception as e:
        print(f"\nERROR during export: {e}")
        print("\nMultiRocket parameter extraction requires manual implementation")
        print("based on the specific aeon/sktime version you're using.")
        raise


def export_test_data(model_data, output_file="multirocket_test_data.json", num_samples=10):
    """Export test data for FPGA validation"""

    X_test = model_data['X_test']
    y_test = model_data['y_test']
    X_test_transform = model_data['X_test_transform']

    # Select first num_samples for testing
    test_data = {
        'num_samples': min(num_samples, len(X_test)),
        'time_series_length': X_test.shape[1],
        'time_series': X_test[:num_samples].tolist(),
        'labels': y_test[:num_samples].tolist(),
        'features_python': X_test_transform[:num_samples].tolist(),
        'python_accuracy': float(model_data['test_accuracy'])
    }

    with open(output_file, 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"\nTest data exported to {output_file}")
    print(f"  Num samples: {test_data['num_samples']}")
    print(f"  Time series length: {test_data['time_series_length']}")

    return test_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train MultiRocket model for FPGA')
    parser.add_argument('--dataset', type=str, default='GunPoint',
                       help='UCR dataset name (default: GunPoint)')
    parser.add_argument('--features', type=int, default=2688,
                       help='Target number of features (default: 2688)')
    parser.add_argument('--output', type=str, default='multirocket_model.json',
                       help='Output model file (default: multirocket_model.json)')
    parser.add_argument('--test-output', type=str, default='multirocket_test_data.json',
                       help='Output test data file')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Train model
    model_data = train_multirocket_model(
        dataset_name=args.dataset,
        num_features=args.features,
        random_state=args.random_state
    )

    # Export for FPGA
    try:
        export_multirocket_model_for_fpga(model_data, args.output)
        export_test_data(model_data, args.test_output)

        print(f"\n{'='*60}")
        print("Training and export complete!")
        print(f"{'='*60}\n")
        print("Next steps:")
        print("1. Review the exported model JSON file")
        print("2. Extract MultiRocket internal parameters (biases, dilations)")
        print("3. Implement C++ testbench to validate against Python features")
        print("4. Synthesize HLS code for FPGA")

    except Exception as e:
        print(f"\n{'='*60}")
        print("EXPORT FAILED - MANUAL INTERVENTION REQUIRED")
        print(f"{'='*60}\n")
        print("The MultiRocket parameter extraction needs custom implementation")
        print("for your specific aeon/sktime version.")
        print("\nYou'll need to:")
        print("1. Inspect multirocket.parameters_ or multirocket._parameters")
        print("2. Extract dilations for original and diff representations")
        print("3. Extract biases for each kernel-dilation combination")
        print("4. Export kernel weights (same 84 kernels as MiniRocket)")
        print("\nSee MiniRocket train_minirocket.py for reference implementation.")
