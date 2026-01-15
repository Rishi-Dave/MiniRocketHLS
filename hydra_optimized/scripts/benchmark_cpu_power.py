#!/usr/bin/env python3
"""
CPU Power Benchmarking for HYDRA
Measures actual CPU power consumption using Intel RAPL interface
"""

import os
import sys
import json
import time
import threading
import numpy as np
from pathlib import Path
from numba import njit, prange

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from custom_hydra import Hydra
except ImportError:
    print("ERROR: Could not import Hydra from custom_hydra.py")
    sys.exit(1)


class RAPLPowerMonitor:
    """Monitor CPU power using Intel RAPL interface"""

    def __init__(self):
        self.rapl_base = "/sys/class/powercap/intel-rapl"
        self.packages = []
        self.monitoring = False
        self.power_samples = []
        self.sample_interval = 0.01  # 10ms sampling (100Hz like FPGA)

        # Find all CPU packages
        for entry in os.listdir(self.rapl_base):
            if entry.startswith("intel-rapl:"):
                path = os.path.join(self.rapl_base, entry)
                name_file = os.path.join(path, "name")
                energy_file = os.path.join(path, "energy_uj")

                if os.path.exists(name_file) and os.path.exists(energy_file):
                    with open(name_file, 'r') as f:
                        name = f.read().strip()

                    if "package" in name.lower():
                        self.packages.append({
                            'name': name,
                            'energy_file': energy_file,
                            'last_energy': None,
                            'last_time': None
                        })

        if not self.packages:
            raise RuntimeError("No RAPL package interfaces found")

        print(f"Found {len(self.packages)} CPU package(s) for power monitoring")

    def read_energy_uj(self, package):
        """Read energy counter in microjoules"""
        try:
            with open(package['energy_file'], 'r') as f:
                return int(f.read().strip())
        except (IOError, PermissionError):
            return None

    def _monitor_thread(self):
        """Background thread to sample power at regular intervals"""
        # Initialize energy counters
        for pkg in self.packages:
            pkg['last_energy'] = self.read_energy_uj(pkg)
            pkg['last_time'] = time.time()

        while self.monitoring:
            time.sleep(self.sample_interval)

            current_time = time.time()
            total_power = 0

            for pkg in self.packages:
                current_energy = self.read_energy_uj(pkg)
                if current_energy is None or pkg['last_energy'] is None:
                    continue

                # Handle counter wraparound (48-bit counter)
                max_energy = 2**48
                energy_diff = current_energy - pkg['last_energy']
                if energy_diff < 0:
                    energy_diff += max_energy

                time_diff = current_time - pkg['last_time']

                # Power = Energy / Time
                power_w = (energy_diff / 1e6) / time_diff if time_diff > 0 else 0

                pkg['last_energy'] = current_energy
                pkg['last_time'] = current_time

                total_power += power_w

            if total_power > 0:
                self.power_samples.append({
                    'timestamp': current_time,
                    'power_w': total_power
                })

    def start(self):
        """Start power monitoring"""
        self.monitoring = True
        self.power_samples = []
        self.thread = threading.Thread(target=self._monitor_thread, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop power monitoring and return statistics"""
        self.monitoring = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)

        if not self.power_samples:
            return None

        powers = [s['power_w'] for s in self.power_samples]
        return {
            'mean_w': np.mean(powers),
            'min_w': np.min(powers),
            'max_w': np.max(powers),
            'std_w': np.std(powers),
            'samples': len(powers),
            'duration_s': self.power_samples[-1]['timestamp'] - self.power_samples[0]['timestamp']
        }


def benchmark_dataset(model_file, test_file, dataset_name):
    """Run benchmark for a single dataset with power monitoring"""

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"Model: {model_file}")
    print(f"Test:  {test_file}")
    print()

    # Load model
    print("Loading HYDRA model...")
    with open(model_file, 'r') as f:
        model_data = json.load(f)

    num_kernels = model_data['num_kernels']
    num_groups = model_data['num_groups']
    num_classes = model_data['num_classes']

    print(f"  Kernels: {num_kernels}")
    print(f"  Groups: {num_groups}")
    print(f"  Classes: {num_classes}")
    print()

    # Load test data
    print("Loading test data...")
    with open(test_file, 'r') as f:
        test_data = json.load(f)

    X_test = np.array(test_data['time_series'])
    y_test = np.array(test_data['labels'])

    n_samples = X_test.shape[0]
    n_timesteps = X_test.shape[1]
    print(f"  Samples: {n_samples}")
    print(f"  Timesteps: {n_timesteps}")
    print()

    # Create and train HYDRA classifier
    print("Initializing HYDRA classifier...")
    hydra = Hydra(
        num_kernels=num_kernels,
        num_groups=num_groups,
        random_state=42
    )

    # For CPU benchmark, we need training data - use test data subset as proxy
    print("NOTE: Using subset of test data for CPU classifier initialization...")
    X_train_proxy = X_test[:100]  # Use first 100 samples
    y_train_proxy = y_test[:100]

    print("Fitting classifier (this may take a moment)...")
    hydra.fit(X_train_proxy, y_train_proxy)
    print()

    # Initialize power monitor
    try:
        power_monitor = RAPLPowerMonitor()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("Try running with: sudo python3 benchmark_cpu_power.py")
        return None

    # Start power monitoring
    print("Starting power monitoring...")
    power_monitor.start()
    time.sleep(0.1)  # Let monitor stabilize

    # Run inference
    print("Running CPU inference...")
    start_time = time.time()

    predictions = hydra.predict(X_test)

    end_time = time.time()

    # Stop power monitoring
    power_stats = power_monitor.stop()

    # Calculate metrics
    total_time = end_time - start_time
    throughput = n_samples / total_time
    latency_ms = (total_time / n_samples) * 1000
    accuracy = np.mean(predictions == y_test) * 100

    if power_stats:
        energy_per_inference_mj = (power_stats['mean_w'] * total_time * 1000) / n_samples
    else:
        energy_per_inference_mj = None

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print()

    if power_stats:
        print("=== POWER STATISTICS ===")
        print(f"Mean Power:     {power_stats['mean_w']:6.2f} W")
        print(f"Min Power:      {power_stats['min_w']:6.2f} W")
        print(f"Max Power:      {power_stats['max_w']:6.2f} W")
        print(f"Std Dev:        {power_stats['std_w']:6.2f} W")
        print(f"Samples:        {power_stats['samples']}")
        print(f"Duration:       {power_stats['duration_s']:.2f} s")
        print()

    print("=== PERFORMANCE STATISTICS ===")
    print(f"Total Time:     {total_time:.2f} s")
    print(f"Throughput:     {throughput:.1f} inferences/sec")
    print(f"Latency:        {latency_ms:.2f} ms/inference")
    print(f"Accuracy:       {accuracy:.2f}%")

    if energy_per_inference_mj:
        print(f"Energy/Inf:     {energy_per_inference_mj:.1f} mJ")
    print()

    return {
        'dataset': dataset_name,
        'n_samples': n_samples,
        'n_timesteps': n_timesteps,
        'power': power_stats,
        'total_time_s': total_time,
        'throughput_inf_per_s': throughput,
        'latency_ms': latency_ms,
        'energy_per_inference_mj': energy_per_inference_mj,
        'accuracy': accuracy
    }


def main():
    # Check for root access
    if os.geteuid() != 0:
        print("="*60)
        print("WARNING: Not running as root")
        print("RAPL interface requires root access for accurate readings")
        print("Please run: sudo python3 benchmark_cpu_power.py")
        print("="*60)
        print()

    # Dataset configurations
    datasets = {
        'InsectSound': {
            'model': 'models/hydra_insectsound_model.json',
            'test': 'models/hydra_insectsound_test_1000.json'
        },
        'FruitFlies': {
            'model': 'models/hydra_fruitflies_model.json',
            'test': 'models/hydra_fruitflies_test_1000.json'
        },
        'MosquitoSound': {
            'model': 'models/hydra_mosquitosound_model.json',
            'test': 'models/hydra_mosquitosound_test_1000.json'
        }
    }

    # Change to hydra_optimized directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)

    print("="*60)
    print("HYDRA CPU Power Benchmarking")
    print("="*60)
    print(f"Working directory: {os.getcwd()}")
    print()

    # Run benchmarks
    results = []
    for name, config in datasets.items():
        model_file = config['model']
        test_file = config['test']

        if not os.path.exists(model_file):
            print(f"WARNING: Model file not found: {model_file}")
            continue

        if not os.path.exists(test_file):
            print(f"WARNING: Test file not found: {test_file}")
            continue

        result = benchmark_dataset(model_file, test_file, name)
        if result:
            results.append(result)

        time.sleep(1)  # Brief pause between datasets

    # Save results
    if results:
        output_dir = Path('power_results')
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / 'cpu_power_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*60}")

        # Print summary
        print("\n=== SUMMARY ===")
        print(f"\n{'Dataset':<15} {'Power (W)':<12} {'Throughput':<15} {'Latency (ms)':<15} {'Energy/Inf (mJ)'}")
        print("-" * 80)

        for r in results:
            if r['power']:
                power_str = f"{r['power']['mean_w']:.1f} ± {r['power']['std_w']:.1f}"
                energy_str = f"{r['energy_per_inference_mj']:.1f}"
            else:
                power_str = "N/A"
                energy_str = "N/A"

            print(f"{r['dataset']:<15} {power_str:<12} {r['throughput_inf_per_s']:>6.1f} inf/s   "
                  f"{r['latency_ms']:>8.2f} ms     {energy_str:>10}")


if __name__ == '__main__':
    main()
