#!/usr/bin/env python3
"""RTT measurement: send a value, wait for ACK, log per-packet timing.

Sender-side timestamps. One packet at a time (no pipelining) -- this is RTT,
not throughput. Writes raw.csv (per-packet) and summary.json on exit.

Adapted from pyuvaraj37/nlp:fpga-network/xrt_host_api/network_experiments/RTT/rtt.py
"""
import argparse
import csv
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpga-ip", default=os.getenv("FPGA_IP", "192.168.40.10"))
    ap.add_argument("--fpga-port", type=int,
                    default=int(os.getenv("FPGA_PORT", "62177")))
    ap.add_argument("--listen-port", type=int,
                    default=int(os.getenv("LISTEN_PORT", "62178")),
                    help="Local bind port; MUST match SocketTable[0].theirPort.")
    ap.add_argument("--packets", type=int, default=1000)
    ap.add_argument("--packet-bytes", type=int, default=64,
                    help="UDP payload size. 64 = one AXI-S beat (one float sample).")
    ap.add_argument("--timeout-s", type=float, default=1.0,
                    help="Per-packet recv timeout; counted as loss if exceeded.")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", required=True, help="Output dir.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Single bound socket -- src_port MUST equal listen_port or NetLayer
    # drops the packet at SocketTable lookup (Bug A from saturation_pivot 2026-05-05).
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 << 20)
    sock.bind(("0.0.0.0", args.listen_port))
    sock.settimeout(args.timeout_s)

    payload_template = b"\x00" * args.packet_bytes
    fpga = (args.fpga_ip, args.fpga_port)

    rows = []
    losses = 0
    print(f"[rtt] fpga={args.fpga_ip}:{args.fpga_port} "
          f"listen=:{args.listen_port} packets={args.packets} "
          f"bytes={args.packet_bytes}", file=sys.stderr)

    for seq in range(args.packets + args.warmup):
        payload = struct.pack("<I", seq) + payload_template[4:]
        t_send = time.perf_counter_ns()
        try:
            sock.sendto(payload, fpga)
            _, _ = sock.recvfrom(2048)
            t_recv = time.perf_counter_ns()
            rtt_us = (t_recv - t_send) / 1000.0
        except socket.timeout:
            t_recv = -1
            rtt_us = -1.0
            losses += 1
        if seq >= args.warmup:
            rows.append((seq - args.warmup, t_send, t_recv, rtt_us))

    with open(out / "raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "t_send_ns", "t_recv_ns", "rtt_us"])
        w.writerows(rows)

    rtts = sorted(r[3] for r in rows if r[3] >= 0)
    n_sent = len(rows)
    n_recv = len(rtts)

    def pct(p):
        if not rtts:
            return None
        i = min(len(rtts) - 1, int(round(p / 100.0 * (len(rtts) - 1))))
        return rtts[i]

    summary = {
        "fpga_ip": args.fpga_ip,
        "fpga_port": args.fpga_port,
        "listen_port": args.listen_port,
        "packets": n_sent,
        "received": n_recv,
        "drop_count": n_sent - n_recv,
        "drop_pct": 100.0 * (n_sent - n_recv) / max(n_sent, 1),
        "rtt_us_min": rtts[0] if rtts else None,
        "rtt_us_p50": pct(50),
        "rtt_us_p95": pct(95),
        "rtt_us_p99": pct(99),
        "rtt_us_max": rtts[-1] if rtts else None,
        "packet_bytes": args.packet_bytes,
        "warmup_discarded": args.warmup,
        "ts_unix": time.time(),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
