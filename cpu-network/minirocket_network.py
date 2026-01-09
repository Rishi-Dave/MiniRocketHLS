#!/usr/bin/env python3

import numpy as np
import json
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import os
from itertools import combinations
import socket
import time
import struct


BUFFER_SIZE = 4 # float size in bytes

# sktime datasets import will be done locally in functions

class MiniRocket:
    """MiniRocket implementation matching sktime reference implementation"""

    def __init__(self, num_kernels=10000, max_dilations_per_kernel=32, random_state=None):
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_state = random_state
        self.kernels = None
        self.dilations = None
        self.num_features_per_dilation = None
        self.biases = None
        self.parameters = None
        
    def _quantiles(self, n):
        """Generate quantiles using golden ratio as per sktime implementation"""
        return np.array(
            [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
        )

    def _fit_dilations(self, n_timepoints, num_features, max_dilations_per_kernel):
        """Fit dilations as per sktime implementation"""
        num_kernels = 84

        if num_features < 84:
            num_features = 84

        num_features_per_kernel = num_features // num_kernels
        true_max_dilations_per_kernel = min(
            num_features_per_kernel, max_dilations_per_kernel
        )
        multiplier = num_features_per_kernel / true_max_dilations_per_kernel

        max_exponent = np.log2((n_timepoints - 1) / (9 - 1))
        dilations, num_features_per_dilation = np.unique(
            np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
                np.int32
            ),
            return_counts=True,
        )
        num_features_per_dilation = (num_features_per_dilation * multiplier).astype(
            np.int32
        )

        remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
        i = 0
        while remainder > 0:
            num_features_per_dilation[i] += 1
            remainder -= 1
            i = (i + 1) % len(num_features_per_dilation)

        return dilations, num_features_per_dilation

    def _fit_biases(self, X, dilations, num_features_per_dilation, quantiles, seed):
        """Fit biases using cumulative convolution as per sktime implementation"""
        if seed is not None:
            np.random.seed(seed)

        n_instances, n_timepoints = X.shape

        # All combinations of 3 indices from 0-8 (C(9,3) = 84)
        indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)

        num_kernels = len(indices)
        num_dilations = len(dilations)

        num_features = num_kernels * np.sum(num_features_per_dilation)

        biases = np.zeros(num_features, dtype=np.float32)

        feature_index_start = 0

        for dilation_index in range(num_dilations):
            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            for kernel_index in range(num_kernels):
                feature_index_end = feature_index_start + num_features_this_dilation

                _X = X[np.random.randint(n_instances)]

                A = -_X  # A = alpha * X = -X
                G = _X + _X + _X  # G = gamma * X = 3X

                C_alpha = np.zeros(n_timepoints, dtype=np.float32)
                C_alpha[:] = A

                C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
                C_gamma[9 // 2] = G

                start = dilation
                end = n_timepoints - padding

                for gamma_index in range(9 // 2):
                    C_alpha[-end:] = C_alpha[-end:] + A[:end]
                    C_gamma[gamma_index, -end:] = G[:end]

                    end += dilation

                for gamma_index in range(9 // 2 + 1, 9):
                    C_alpha[:-start] = C_alpha[:-start] + A[start:]
                    C_gamma[gamma_index, :-start] = G[start:]

                    start += dilation

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                biases[feature_index_start:feature_index_end] = np.quantile(
                    C, quantiles[feature_index_start:feature_index_end]
                )

                feature_index_start = feature_index_end

        return biases

    def fit(self, X):
        """Fit MiniRocket parameters"""
        X = X.astype(np.float32)
        _, n_timepoints = X.shape

        self.time_series_length = n_timepoints

        if n_timepoints < 9:
            raise ValueError(
                f"n_timepoints must be >= 9, but found {n_timepoints};"
                " zero pad shorter series so that n_timepoints == 9"
            )

        # Determine actual number of kernels (must be multiple of 84)
        if self.num_kernels < 84:
            self.num_kernels_ = 84
        else:
            self.num_kernels_ = (self.num_kernels // 84) * 84

        seed = np.int32(self.random_state) if isinstance(self.random_state, int) else None

        dilations, num_features_per_dilation = self._fit_dilations(
            n_timepoints, self.num_kernels_, self.max_dilations_per_kernel
        )

        num_features_per_kernel = np.sum(num_features_per_dilation)

        quantiles = self._quantiles(84 * num_features_per_kernel)

        biases = self._fit_biases(X, dilations, num_features_per_dilation, quantiles, seed)

        self.parameters = (dilations, num_features_per_dilation, biases)
        self.dilations = dilations
        self.num_features_per_dilation = num_features_per_dilation
        self.biases = biases

        return self

    def transform(self, X):
        """Transform time series using fitted parameters"""
        X = X.astype(np.float32)
        n_instances, n_timepoints = X.shape

        dilations, num_features_per_dilation, biases = self.parameters

        # All combinations of 3 indices from 0-8
        indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)

        num_kernels = len(indices)
        num_dilations = len(dilations)

        num_features = num_kernels * np.sum(num_features_per_dilation)
        # print(f"num_features: {num_features}")
        # print(f"num_features_per_dilation: {num_features_per_dilation}")
        # print(f"sum: {np.sum(num_features_per_dilation)}")

        features = np.zeros((n_instances, num_features), dtype=np.float32)

        for example_index in range(n_instances):
            _X = X[example_index]

            A = -_X  # A = alpha * X = -X
            G = _X + _X + _X  # G = gamma * X = 3X

            feature_index_start = 0

            for dilation_index in range(num_dilations):
                _padding0 = dilation_index % 2

                dilation = dilations[dilation_index]
                padding = ((9 - 1) * dilation) // 2

                num_features_this_dilation = num_features_per_dilation[dilation_index]

                C_alpha = np.zeros(n_timepoints, dtype=np.float32)
                C_alpha[:] = A

                C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
                C_gamma[9 // 2] = G

                start = dilation
                end = n_timepoints - padding

                for gamma_index in range(9 // 2):
                    C_alpha[-end:] = C_alpha[-end:] + A[:end]
                    C_gamma[gamma_index, -end:] = G[:end]

                    end += dilation

                for gamma_index in range(9 // 2 + 1, 9):
                    C_alpha[:-start] = C_alpha[:-start] + A[start:]
                    C_gamma[gamma_index, :-start] = G[start:]

                    start += dilation

                for kernel_index in range(num_kernels):
                    feature_index_end = feature_index_start + num_features_this_dilation

                    _padding1 = (_padding0 + kernel_index) % 2

                    index_0, index_1, index_2 = indices[kernel_index]

                    C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                    # PPV (Proportion of Positive Values)
                    if _padding1 == 0:
                        for feature_count in range(num_features_this_dilation):
                            features[example_index, feature_index_start + feature_count] = (
                                np.mean(C > biases[feature_index_start + feature_count])
                            )
                    else:
                        for feature_count in range(num_features_this_dilation):
                            features[example_index, feature_index_start + feature_count] = (
                                np.mean(
                                    C[padding:-padding] > biases[feature_index_start + feature_count]
                                )
                            )

                    feature_index_start = feature_index_end

        return features

    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)

def load_real_dataset(dataset_name='arrow_head'):
    """Load real time series datasets from sktime"""
    try:
        print(f"Attempting to import sktime for dataset: {dataset_name}")
        from sktime.datasets import load_arrow_head, load_gunpoint, load_italy_power_demand
        print("sktime import successful")
        
        if dataset_name == 'arrow_head':
            X, y = load_arrow_head()
        elif dataset_name == 'gun_point':
            X, y = load_gunpoint()
        elif dataset_name == 'italy_power':
            X, y = load_italy_power_demand()
        else:
            print(f"Unknown dataset {dataset_name}, using arrow_head")
            X, y = load_arrow_head()
        
        # Convert from pandas to numpy
        X_numpy = np.array([series.values for series in X.iloc[:, 0]])
        
        # Convert string labels to integers
        unique_labels = sorted(set(y))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        y_numpy = np.array([label_to_int[label] for label in y])
        
        print(f"Real dataset loaded: {dataset_name}")
        print(f"  Shape: {X_numpy.shape}")
        print(f"  Classes: {len(unique_labels)} ({unique_labels})")
        print(f"  Class distribution: {np.bincount(y_numpy)}")
        
        return X_numpy.astype(np.float32), y_numpy
        
    except ImportError as e:
        print(f"sktime import error: {e}")
        print("falling back to synthetic data")
        return generate_sample_data()
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        print("Falling back to synthetic data")
        import traceback
        traceback.print_exc()
        return generate_sample_data()



def main():
    parser = argparse.ArgumentParser(description='Train MiniRocket model')
    parser.add_argument('--dataset', type=str, default='arrow_head', 
                       choices=['arrow_head', 'gun_point', 'italy_power', 'synthetic'],
                       help='Dataset to use (arrow_head, gun_point, italy_power, or synthetic)')
    parser.add_argument('--client_ip', type=str, default='127.0.0.1', help='Client IP address')
    parser.add_argument('--server_ip', type=str, default='127.0.0.1', help='Server IP address')
    parser.add_argument('--client_port', type=int, default=8080, help='Client port (for synthetic data)')
    parser.add_argument('--server_port', type=int, default=9090, help='Server port (for synthetic data)')
    parser.add_argument('--client', action='store_true', help='Run as client')
    parser.add_argument('--server', action='store_true', help='Run as server')
    args = parser.parse_args()
    
    X, y = load_real_dataset(args.dataset)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    pred = []

    if args.client:
        print("Starting UDP client to send/receive data...")
        local_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        local_sock.bind((args.client_ip, args.client_port))  # Bind to client IP and port
        # remote_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # remote_sock.bind((args.server_ip, args.server_port))  # Bind to server IP and port

        print("Client generate data to send of UDP...")
        data = X_test[0]
        for value in data:
            data = struct.pack("f", value)  # float32 → 4 bytes
            local_sock.sendto(data, (args.server_ip, args.server_port))
            data, addr = local_sock.recvfrom(4)  # Wait for ACK
            value = struct.unpack("f", data)[0]
            pred.append(value)

        print("Predictions received from server:")
        print(pred)


    elif args.server:
        print("Training MiniRocket...")
        # Fit MiniRocket (num_kernels must be multiple of 84, using 840 to fit HLS limits)
        minirocket = MiniRocket(num_kernels=840, random_state=42)
        X_train_features = minirocket.fit_transform(X_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)

        # Train classifier
        print("Training classifier...")
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        classifier.fit(X_train_scaled, y_train)

        window = np.zeros((minirocket.time_series_length,), dtype=np.float32)  # sliding windo
        window = window.reshape(1, -1)       # shape (1, time_series_length)
        print("Starting UDP client to send/receive data...")
        # remote_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # remote_sock.bind((args.client_ip, args.client_port))  # Bind to client IP and port
        local_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        local_sock.bind((args.server_ip, args.server_port))  # Bind to server IP and port

        print("Server online for data over UDP...")

        while True:
            
            data, addr = local_sock.recvfrom(4)  # Wait for ACK
            value = struct.unpack("f", data)[0]
            
            window = np.roll(window, -1, axis=1)
            window[0, -1] = value

            print(window.flatten())

            window_features = minirocket.transform(window)
            window_scaled = scaler.transform(window_features)
            window_pred = classifier.predict(window_scaled)

            data = struct.pack("f", float(window_pred[0]))  # float32 → 4 bytes
            local_sock.sendto(data, (args.client_ip, args.client_port))
            



if __name__ == "__main__":
    main()