#!/usr/bin/env python3
"""CPU streaming-inference UDP responder for MiniRocket -- the software
baseline ("red curve") for the network-saturation bandwidth chart.

Wire contract (see _udp_common.py and MiniRocketHLS/cpu-network/
fpga_network_host.py header comment): one UDP packet = one sample; 64-byte
payload; float32 sample in bytes[0:4), zero-padded. This responder mirrors
the FPGA's "shift-register of length time_series_length" model in spirit
(MiniRocketHLS/fpga-network/minirocket_dpr/src/minirocket.cpp:283-297), but
per this task's explicit CPU baseline design it buffers L samples, runs ONE
MiniRocket inference over the full window, optionally emits response
packet(s), then starts refilling the next (non-overlapping) window -- so the
network+syscall overhead per packet is what caps CPU throughput, not a
per-packet feature recompute. The point of this responder is to show the
CPU curve FLATTEN at the host/NIC packet-rate ceiling while the FPGA (which
truly reclassifies every incoming beat) keeps climbing.

Inference path (exact port of the reference implementation in
MiniRocketHLS/fpga-network/train_minirocket.py's MiniRocket.transform(),
which itself matches the sktime/aeon MiniRocket algorithm and the HLS
kernel's math): 84 fixed kernels (length 9, C(9,3) index combinations),
per-dilation running-sum trick (A = -x, G = 3x) to build the 9 "gamma rows"
once per dilation, PPV (proportion of positive values) against per-feature
quantile biases, StandardScaler normalization, then a Ridge classifier
(linear decision function / argmax over classes).

Model JSON schema (see MiniRocketHLS/fpga-network/minirocket_model.json --
this is the model actually consumed by the minirocket_dpr HLS kernel, with
time_series_length=150; no file literally named
"dpr_minirocket_model_len150_fixed.json" was found in the repo, so this is
the closest -- and functionally exact -- match for that description):

    num_kernels                 : 84 (always, MiniRocket fixes this)
    num_dilations                : int, len(dilations)
    num_features                 : int, len(biases) == len(scaler_mean) ==
                                    len(scaler_scale) == num_kernels * sum(
                                    num_features_per_dilation)
    num_classes                  : int
    time_series_length           : int (150 for this model)
    kernel_indices                : [num_kernels][3] int -- which 3 of the 9
                                    kernel taps get weight +2 (rest -1);
                                    C(9,3) = 84 combinations
    dilations                    : [num_dilations] int
    num_features_per_dilation    : [num_dilations] int (sums to
                                    num_features / num_kernels)
    biases                       : [num_features] float32 -- per-feature PPV
                                    quantile thresholds
    scaler_mean / scaler_scale   : [num_features] float32 -- StandardScaler
    classifier_coef              : flat [num_features] (binary,
                                    RidgeClassifierCV collapses 2-class coef_
                                    to a single row) OR flat
                                    [num_classes*num_features] (multiclass,
                                    row-major over classes then features,
                                    i.e. sklearn coef_.tolist())
    classifier_intercept         : [1] (binary) or [num_classes]
    classes                      : [num_classes] original class labels

Usage:
    python3 minirocket_udp_responder.py \\
        --listen-ip 0.0.0.0 --listen-port 15000 \\
        --model-json ../../fpga-network/minirocket_model.json
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _udp_common import BEAT_BYTES, StatsReporter, pack_beat, unpack_sample  # noqa: E402

KERNEL_SIZE = 9


class MiniRocketModel:
    """Loads a MiniRocket model JSON and runs single-sample transform+predict.

    This is a direct, faithful port of MiniRocket.transform() /
    save_model_parameters() in MiniRocketHLS/fpga-network/train_minirocket.py,
    specialized to a single instance (no batch dimension) since the
    responder always transforms exactly one length-L window at a time.
    """

    def __init__(self, path: str):
        with open(path) as f:
            m = json.load(f)

        self.time_series_length = int(m["time_series_length"])
        self.num_classes = int(m["num_classes"])
        self.num_features = int(m["num_features"])
        self.dilations = np.asarray(m["dilations"], dtype=np.int64)
        self.num_features_per_dilation = np.asarray(
            m["num_features_per_dilation"], dtype=np.int64
        )
        self.biases = np.asarray(m["biases"], dtype=np.float32)
        self.scaler_mean = np.asarray(m["scaler_mean"], dtype=np.float32)
        self.scaler_scale = np.asarray(m["scaler_scale"], dtype=np.float32)
        self.classes = np.asarray(m["classes"])

        if "kernel_indices" in m:
            self.kernel_indices = np.asarray(m["kernel_indices"], dtype=np.int64)
        else:
            # Fixed C(9,3)=84 scheme -- regenerate if the JSON omits it.
            self.kernel_indices = np.asarray(
                list(combinations(range(KERNEL_SIZE), 3)), dtype=np.int64
            )
        self.num_kernels = self.kernel_indices.shape[0]

        coef = np.asarray(m["classifier_coef"], dtype=np.float64).reshape(-1)
        self.intercept = np.asarray(m["classifier_intercept"], dtype=np.float64).reshape(-1)
        if coef.shape[0] == self.num_features:
            # Binary RidgeClassifierCV: single decision-function row.
            self.coef = coef.reshape(1, self.num_features)
            self.binary = True
        else:
            self.coef = coef.reshape(len(self.classes), self.num_features)
            self.binary = False

        if len(self.scaler_scale) != self.num_features or len(self.biases) != self.num_features:
            raise ValueError(
                f"model JSON shape mismatch: num_features={self.num_features} "
                f"but biases/scaler arrays have different length"
            )

    def transform_single(self, x: np.ndarray) -> np.ndarray:
        """PPV feature transform for one length-L series. Faithful port of
        MiniRocket.transform() in train_minirocket.py (single instance)."""
        n = self.time_series_length
        A = -x
        G = x + x + x

        features = np.empty(self.num_features, dtype=np.float32)
        feature_index_start = 0
        num_dilations = len(self.dilations)

        for d_idx in range(num_dilations):
            _padding0 = d_idx % 2
            dilation = int(self.dilations[d_idx])
            padding = ((KERNEL_SIZE - 1) * dilation) // 2
            nfd = int(self.num_features_per_dilation[d_idx])

            C_alpha = A.copy()
            C_gamma = np.zeros((KERNEL_SIZE, n), dtype=np.float32)
            C_gamma[KERNEL_SIZE // 2] = G

            start = dilation
            end = n - padding
            for g in range(KERNEL_SIZE // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[g, -end:] = G[:end]
                end += dilation

            for g in range(KERNEL_SIZE // 2 + 1, KERNEL_SIZE):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[g, :-start] = G[start:]
                start += dilation

            for k_idx in range(self.num_kernels):
                feature_index_end = feature_index_start + nfd
                _padding1 = (_padding0 + k_idx) % 2
                i0, i1, i2 = self.kernel_indices[k_idx]
                C = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2]
                seg = C if _padding1 == 0 else C[padding:-padding]

                for fc in range(nfd):
                    features[feature_index_start + fc] = np.mean(
                        seg > self.biases[feature_index_start + fc]
                    )
                feature_index_start = feature_index_end

        return features

    def predict(self, x: np.ndarray):
        """Return (predicted_class, per_class_scores[num_classes])."""
        feats = self.transform_single(x)
        scaled = (feats - self.scaler_mean) / self.scaler_scale

        if self.binary:
            decision = float(np.dot(self.coef[0], scaled) + self.intercept[0])
            cls = self.classes[1] if decision >= 0 else self.classes[0]
            # Emit a 2-wide logit pair so the response packet count matches
            # num_classes, same spirit as the FPGA emitting num_classes
            # response beats per classification (minirocket.cpp:321-334).
            scores = np.array([-decision, decision], dtype=np.float32)
        else:
            raw = self.coef @ scaled + self.intercept
            cls = self.classes[int(np.argmax(raw))]
            scores = raw.astype(np.float32)

        return int(cls), scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen-ip", default="0.0.0.0")
    ap.add_argument("--listen-port", type=int, required=True)
    ap.add_argument("--model-json", required=True,
                     help="MiniRocket model JSON (see schema in module docstring)")
    ap.add_argument("--series-length", type=int, default=None,
                     help="Override model's time_series_length (window size L)")
    ap.add_argument("--respond", dest="respond", action="store_true", default=True)
    ap.add_argument("--no-respond", dest="respond", action="store_false")
    ap.add_argument("--variant", default="cpu-minirocket",
                     help="Label used in the stats file name / log lines")
    ap.add_argument("--dataset", default=None, help="Dataset label (for CSV bookkeeping)")
    ap.add_argument("--stats-file", default=None,
                     help="Path to write live JSON stats (default: /tmp/<variant>_stats.json)")
    ap.add_argument("--stats-interval", type=float, default=0.2)
    ap.add_argument("--rcvbuf-bytes", type=int, default=64 << 20)
    args = ap.parse_args()

    model = MiniRocketModel(args.model_json)
    L = args.series_length or model.time_series_length

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.rcvbuf_bytes)
    sock.bind((args.listen_ip, args.listen_port))

    buf = np.zeros(L, dtype=np.float32)
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
        f"[minirocket-responder] listening on {args.listen_ip}:{args.listen_port} "
        f"L={L} num_classes={model.num_classes} num_features={model.num_features} "
        f"model={args.model_json} stats={stats_path}",
        file=sys.stderr,
    )

    def handle_sigterm(signum, frame):
        reporter.stop()
        sys.exit(0)

    import signal
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
