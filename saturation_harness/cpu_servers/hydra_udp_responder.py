#!/usr/bin/env python3
"""CPU streaming-inference UDP responder for HYDRA -- the software baseline
("red curve") for the network-saturation bandwidth chart.

Wire contract: identical to minirocket_udp_responder.py / _udp_common.py --
one UDP packet = one sample, 64-byte payload, float32 in bytes[0:4). This
responder buffers L samples into a non-overlapping window, runs ONE HYDRA
inference, optionally emits response packet(s), then refills -- so per-
packet socket/syscall overhead (not periodic feature recompute) is what
caps CPU throughput, matching the FPGA-vs-CPU saturation comparison this
harness is building.

Inference path -- ported from the actual FPGA kernel math, NOT textbook
Dempster et al. HYDRA. Reading MiniRocketHLS/hydra_optimized/hydra_axis/
src/hydra_axis.cpp end to end shows there is no group-competition / hard-
counting / argmax-across-kernels anywhere in the compute path -- despite
`num_groups` / `group_assignments` appearing in the header and model JSON,
they are unused/vestigial metadata (confirmed also in
MiniRocketHLS/saturation_harness/load_hydra_hbm.cpp, which never loads
`group_assignments`, and in MiniRocketHLS/hydra_optimized/scripts/
custom_hydra.py, the from-scratch numpy implementation used to train these
models). Each of `num_kernels` kernels is convolved and pooled completely
independently:

    for k in range(num_kernels):                       # 512 typically
        conv[t] = bias[k] + sum_w kernel_weights[k][w] * x[t + w*dilation[k]]
        features[2k]   = max(conv)
        features[2k+1] = mean(conv)
    scaled = (features - scaler_mean) / scaler_scale        # guard |scale|<=1e-8 -> 0
    score[c] = intercept[c] + sum_f scaled[f] * coefficients[f*num_classes + c]
    predicted_class = argmax(score)

(hydra_axis.cpp:25-60 convolution, :62-121 pooling, :123-141 scaler,
:143-163 ridge classifier; coefficients are stored **transposed** --
`hydra.classifier_.coef_.T.flatten()` in
hydra_optimized/scripts/train_and_save_models.py -- so they must be indexed
as coefficients[f*num_classes+c], not sklearn's native [c*num_features+f].)

Model JSON schema (see e.g. MiniRocketHLS/hydra_optimized/models/
hydra_insectsound_model.json):
    dataset, num_kernels, num_groups, kernel_size (9), num_features
        (= 2*num_kernels), num_classes, time_series_length
    kernel_weights   : [num_kernels*kernel_size] float -- learned (not
                        fixed -1/0/+1 like MiniRocket)
    biases           : [num_kernels] float (always 0 in this implementation)
    dilations        : [num_kernels] int, drawn from {1,2,4,8} in practice
    scaler_mean / scaler_scale : [num_features] float
    coefficients     : [num_features*num_classes] float, transposed layout
                        (see above)
    intercept        : [num_classes] float
    train_accuracy / test_accuracy : float (informational only)

Usage:
    python3 hydra_udp_responder.py \\
        --listen-ip 0.0.0.0 --listen-port 15001 \\
        --model-json ../../hydra_optimized/models/hydra_insectsound_model.json
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _udp_common import StatsReporter, pack_beat, unpack_sample  # noqa: E402


class HydraModel:
    """Loads a HYDRA model JSON and runs single-sample feature-extraction +
    ridge-classifier inference, vectorized (grouped by dilation) with numpy.
    """

    def __init__(self, path: str):
        with open(path) as f:
            m = json.load(f)

        self.dataset = m.get("dataset")
        self.num_kernels = int(m["num_kernels"])
        self.kernel_size = int(m.get("kernel_size", 9))
        self.num_features = int(m["num_features"])
        self.num_classes = int(m["num_classes"])
        self.time_series_length = int(m["time_series_length"])

        self.kernel_weights = np.asarray(m["kernel_weights"], dtype=np.float64).reshape(
            self.num_kernels, self.kernel_size
        )
        self.biases = np.asarray(m["biases"], dtype=np.float64).reshape(self.num_kernels)
        self.dilations = np.asarray(m["dilations"], dtype=np.int64).reshape(self.num_kernels)

        self.scaler_mean = np.asarray(m["scaler_mean"], dtype=np.float64)
        self.scaler_scale = np.asarray(m["scaler_scale"], dtype=np.float64)
        # Guard tiny scales exactly like hydra_axis.cpp:123-141.
        safe_scale = np.where(np.abs(self.scaler_scale) <= 1e-8, 1.0, self.scaler_scale)
        self._scale_is_safe = np.abs(self.scaler_scale) > 1e-8
        self._safe_scale = safe_scale

        coef = np.asarray(m["coefficients"], dtype=np.float64)
        self.intercept = np.asarray(m["intercept"], dtype=np.float64).reshape(self.num_classes)
        # coefficients[f*num_classes+c] (transposed sklearn layout, see
        # docstring) -> reshape (num_features, num_classes) row-major.
        self.coef = coef.reshape(self.num_features, self.num_classes)

        # Precompute per-dilation kernel groupings for vectorized conv.
        self._dilation_groups = {}
        for d in np.unique(self.dilations):
            k_idx = np.where(self.dilations == d)[0]
            self._dilation_groups[int(d)] = k_idx

    def transform_single(self, x: np.ndarray) -> np.ndarray:
        """Per-kernel independent dilated conv -> [max, mean] pooling.
        features layout: [k0_max, k0_mean, k1_max, k1_mean, ...]."""
        n = self.time_series_length
        ks = self.kernel_size
        features = np.empty(self.num_features, dtype=np.float64)

        for dilation, k_idx in self._dilation_groups.items():
            out_len = n - (ks - 1) * dilation
            if out_len <= 0:
                # Series shorter than this kernel's receptive field -- match
                # hydra_axis.cpp behavior of a degenerate (length-1) window.
                out_len = 1
                idx = np.zeros((1, ks), dtype=np.int64)
            else:
                base = np.arange(out_len)[:, None]
                taps = np.arange(ks)[None, :] * dilation
                idx = base + taps  # (out_len, ks)
                idx = np.clip(idx, 0, n - 1)

            windows = x[idx]  # (out_len, ks)
            w = self.kernel_weights[k_idx]  # (Kd, ks)
            b = self.biases[k_idx]  # (Kd,)

            conv = windows @ w.T + b[None, :]  # (out_len, Kd)
            max_vals = conv.max(axis=0)  # (Kd,)
            mean_vals = conv.mean(axis=0)  # (Kd,)

            features[2 * k_idx] = max_vals
            features[2 * k_idx + 1] = mean_vals

        return features

    def predict(self, x: np.ndarray):
        feats = self.transform_single(x)
        scaled = np.where(self._scale_is_safe, (feats - self.scaler_mean) / self._safe_scale, 0.0)
        scores = self.intercept + scaled @ self.coef  # (num_classes,)
        cls = int(np.argmax(scores))
        return cls, scores.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen-ip", default="0.0.0.0")
    ap.add_argument("--listen-port", type=int, required=True)
    ap.add_argument("--model-json", required=True,
                     help="HYDRA model JSON (see schema in module docstring)")
    ap.add_argument("--series-length", type=int, default=None,
                     help="Override model's time_series_length (window size L)")
    ap.add_argument("--respond", dest="respond", action="store_true", default=True)
    ap.add_argument("--no-respond", dest="respond", action="store_false")
    ap.add_argument("--variant", default="cpu-hydra",
                     help="Label used in the stats file name / log lines")
    ap.add_argument("--dataset", default=None, help="Dataset label (for CSV bookkeeping)")
    ap.add_argument("--stats-file", default=None,
                     help="Path to write live JSON stats (default: /tmp/<variant>_stats.json)")
    ap.add_argument("--stats-interval", type=float, default=0.2)
    ap.add_argument("--rcvbuf-bytes", type=int, default=64 << 20)
    args = ap.parse_args()

    model = HydraModel(args.model_json)
    L = args.series_length or model.time_series_length

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.rcvbuf_bytes)
    sock.bind((args.listen_ip, args.listen_port))

    buf = np.zeros(L, dtype=np.float64)
    idx = 0

    counters = {
        "variant": args.variant,
        "dataset": args.dataset,
        "packets_received": 0,
        "windows_processed": 0,
        "last_prediction": None,
        "last_infer_ms": 0.0,
        "start_unix": time.time(),
    }
    lock = threading.Lock()

    stats_path = args.stats_file or f"/tmp/{args.variant}_stats.json"
    reporter = StatsReporter(stats_path, counters, lock, args.stats_interval)
    reporter.start()

    print(
        f"[hydra-responder] listening on {args.listen_ip}:{args.listen_port} "
        f"L={L} num_classes={model.num_classes} num_kernels={model.num_kernels} "
        f"num_features={model.num_features} model={args.model_json} stats={stats_path}",
        file=sys.stderr,
    )

    import signal

    def handle_sigterm(signum, frame):
        reporter.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    try:
        while True:
            try:
                data, addr = sock.recvfrom(2048)
            except OSError:
                continue
            if len(data) < 4:
                continue
            buf[idx] = unpack_sample(data)
            idx += 1
            with lock:
                counters["packets_received"] += 1

            if idx >= L:
                t0 = time.perf_counter()
                pred_cls, scores = model.predict(buf)
                infer_ms = (time.perf_counter() - t0) * 1000.0
                idx = 0

                with lock:
                    counters["windows_processed"] += 1
                    counters["last_prediction"] = pred_cls
                    counters["last_infer_ms"] = infer_ms

                if args.respond:
                    for s in scores:
                        try:
                            sock.sendto(pack_beat(s), addr)
                        except OSError:
                            break
    except KeyboardInterrupt:
        pass
    finally:
        reporter.stop()


if __name__ == "__main__":
    main()
