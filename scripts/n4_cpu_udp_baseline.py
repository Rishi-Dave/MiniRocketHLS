#!/usr/bin/env python3
"""
N4 — CPU UDP-server baseline RTT for the [NET] axis.

Spawns cpu-network/minirocket_network.py in --server mode, sends a timed UDP
stream of float32 samples to it, records per-sample round-trip latency, and
writes P50/P95/P99 to results/n4_cpu_udp_baseline.json.

This is the *CPU-side* baseline that the FPGA-on-OCT [NET] numbers (N2/N3)
will be compared against. Per the plan: "co-located with OCT for same-network
fairness". This wrapper defaults to loopback (127.0.0.1); set --server-ip /
--client-ip for cross-host runs.

Notes / honest caveats baked into the JSON:
- The server runs a *full* MiniRocket transform on the sliding window per
  sample (`window.shape == (1, time_series_length)`), which is the dominant
  cost — this is the apples-to-apples baseline for the FPGA-NET pipeline,
  which also does per-sample inference.
- Loopback RTT is a *floor*, not realistic — real OCT runs must be over a
  physical link.
- A short warmup is discarded before measurement.
"""

import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = REPO_ROOT / "cpu-network" / "minirocket_network.py"
RESULTS_DIR = REPO_ROOT / "results"


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def wait_for_server(host, port, timeout=120):
    """Probe by sending a single dummy float and expecting any reply."""
    deadline = time.time() + timeout
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)
    try:
        while time.time() < deadline:
            try:
                sock.sendto(struct.pack("f", 0.0), (host, port))
                sock.recvfrom(4)
                return True
            except (socket.timeout, ConnectionRefusedError, OSError):
                time.sleep(1.0)
        return False
    finally:
        sock.close()


def run_client(server_ip, server_port, client_ip, client_port,
               n_warmup, n_measure):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((client_ip, client_port))
    sock.settimeout(10.0)
    try:
        # Warmup (response time ignored)
        for i in range(n_warmup):
            payload = struct.pack("f", float(i) * 0.01)
            sock.sendto(payload, (server_ip, server_port))
            sock.recvfrom(4)

        rtts_us = []
        for i in range(n_measure):
            payload = struct.pack("f", float(i) * 0.01 + 1.0)
            t0 = time.perf_counter_ns()
            sock.sendto(payload, (server_ip, server_port))
            sock.recvfrom(4)
            t1 = time.perf_counter_ns()
            rtts_us.append((t1 - t0) / 1000.0)
        return rtts_us
    finally:
        sock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gun_point",
                    choices=["arrow_head", "gun_point", "italy_power"])
    ap.add_argument("--server-ip", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=9090)
    ap.add_argument("--client-ip", default="127.0.0.1")
    ap.add_argument("--client-port", type=int, default=8080)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=200)
    ap.add_argument("--out", default=str(RESULTS_DIR / "n4_cpu_udp_baseline.json"))
    args = ap.parse_args()

    if not SERVER_SCRIPT.exists():
        print(f"FATAL: server script not found at {SERVER_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    server_log = open(RESULTS_DIR / "n4_server.log", "w")
    server_cmd = [
        sys.executable, "-u", str(SERVER_SCRIPT),
        "--server",
        "--dataset", args.dataset,
        "--server_ip", args.server_ip,
        "--server_port", str(args.server_port),
        "--client_ip", args.client_ip,
        "--client_port", str(args.client_port),
    ]
    print(f"[n4] launching server: {' '.join(server_cmd)}")
    server = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT)

    try:
        # Wait for server training + UDP bind
        # The server prints "Server online for data over UDP..." once ready;
        # but cleaner to probe by sending and waiting for a real reply.
        # First sleep a bit to let RidgeClassifierCV training finish.
        print("[n4] waiting for server training + UDP bind (up to 180s)...")
        time.sleep(5.0)
        ready_deadline = time.time() + 180
        while time.time() < ready_deadline:
            if server.poll() is not None:
                print("[n4] server exited prematurely; see results/n4_server.log",
                      file=sys.stderr)
                sys.exit(2)
            # Check log for the "Server online" line
            with open(RESULTS_DIR / "n4_server.log") as f:
                if "Server online for data over UDP" in f.read():
                    break
            time.sleep(2.0)
        else:
            print("[n4] timed out waiting for server ready line", file=sys.stderr)
            sys.exit(3)

        print(f"[n4] running {args.warmup} warmup + {args.measure} measured samples")
        rtts_us = run_client(args.server_ip, args.server_port,
                             args.client_ip, args.client_port,
                             args.warmup, args.measure)
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()

    rtts_sorted = sorted(rtts_us)
    summary = {
        "experiment": "N4 — CPU UDP-server baseline RTT (MiniRocket)",
        "dataset": args.dataset,
        "server_ip": args.server_ip,
        "server_port": args.server_port,
        "client_ip": args.client_ip,
        "client_port": args.client_port,
        "n_warmup": args.warmup,
        "n_measure": args.measure,
        "rtt_us": {
            "min": min(rtts_sorted),
            "p50": percentile(rtts_sorted, 0.50),
            "p95": percentile(rtts_sorted, 0.95),
            "p99": percentile(rtts_sorted, 0.99),
            "max": max(rtts_sorted),
            "mean": sum(rtts_sorted) / len(rtts_sorted),
        },
        "caveats": [
            "Loopback RTT is a floor (no NIC, no wire); real OCT runs must use a physical link.",
            "Server runs a full MiniRocket transform on a (1, ts_length) sliding window per sample — this dominates RTT and is the relevant CPU baseline for FPGA-NET single-sample inference.",
            "Server is single-threaded Python; this is the CPU UDP-server baseline, not a tuned C++ server.",
        ],
        "raw_rtt_us_first50": rtts_us[:50],
    }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[n4] wrote {args.out}")
    r = summary["rtt_us"]
    print(f"[n4] RTT us: p50={r['p50']:.1f} p95={r['p95']:.1f} "
          f"p99={r['p99']:.1f} mean={r['mean']:.1f} (n={args.measure})")


if __name__ == "__main__":
    main()
