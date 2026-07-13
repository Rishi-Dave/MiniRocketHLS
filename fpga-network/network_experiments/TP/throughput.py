#!/usr/bin/env python3
"""Throughput sweep: fire packets at target rate without waiting for ACK.

Two threads: sender paces with time.perf_counter deadlines, receiver drains
any ACKs that come back (counts them, doesn't block sender). Sweeps rates
geometrically, auto-stops if loss > --abort-loss-pct for two consecutive rates.

Adapted from pyuvaraj37/nlp:fpga-network/xrt_host_api/network_experiments/TP/throughput.py
"""
import argparse
import csv
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path


def run_step(sock, fpga, target_pps, duration_s, abort_evt):
    interval_ns = int(1e9 / target_pps) if target_pps > 0 else 0
    payload = b"\x00" * 64
    sent = 0
    recvd = 0
    stop = threading.Event()

    def rx():
        nonlocal recvd
        sock.settimeout(0.05)
        while not stop.is_set():
            try:
                sock.recvfrom(2048)
                recvd += 1
            except socket.timeout:
                pass
            except OSError:
                break

    t = threading.Thread(target=rx, daemon=True)
    t.start()

    t_end = time.perf_counter_ns() + int(duration_s * 1e9)
    next_ns = time.perf_counter_ns()
    while time.perf_counter_ns() < t_end and not abort_evt.is_set():
        while True:
            now = time.perf_counter_ns()
            if now >= next_ns:
                break
            slack = next_ns - now
            if slack > 50_000:
                time.sleep((slack - 50_000) / 1e9)
        try:
            sock.sendto(payload, fpga)
            sent += 1
        except BlockingIOError:
            pass
        next_ns += interval_ns

    stop.set()
    t.join(timeout=1.0)
    return sent, recvd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpga-ip", default=os.getenv("FPGA_IP", "192.168.40.10"))
    ap.add_argument("--fpga-port", type=int,
                    default=int(os.getenv("FPGA_PORT", "62177")))
    ap.add_argument("--listen-port", type=int,
                    default=int(os.getenv("LISTEN_PORT", "62178")))
    ap.add_argument("--rate-min", type=float, default=1024)
    ap.add_argument("--rate-max", type=float, default=262144)
    ap.add_argument("--rate-steps", type=int, default=8)
    ap.add_argument("--duration-s", type=float, default=2.0)
    ap.add_argument("--abort-loss-pct", type=float, default=80.0,
                    help="Stop sweep if 2 consecutive steps exceed this loss %.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 << 20)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 << 20)
    sock.bind(("0.0.0.0", args.listen_port))
    fpga = (args.fpga_ip, args.fpga_port)

    if args.rate_steps < 2:
        rates = [args.rate_min]
    else:
        ratio = args.rate_max / args.rate_min
        rates = [args.rate_min * ratio ** (i / (args.rate_steps - 1))
                 for i in range(args.rate_steps)]

    rows = []
    high_loss_streak = 0
    abort_evt = threading.Event()

    for tgt in rates:
        sent, recvd = run_step(sock, fpga, tgt, args.duration_s, abort_evt)
        loss_pct = 100.0 * (sent - recvd) / max(sent, 1)
        achieved = sent / args.duration_s
        bps = achieved * 64 * 8
        rows.append({
            "target_pps": tgt,
            "sent": sent,
            "recv_acks": recvd,
            "achieved_pps": achieved,
            "loss_pct": loss_pct,
            "bps": bps,
        })
        print(f"[tp] tgt={tgt:>10.0f}pps sent={sent:>8d} ack={recvd:>8d} "
              f"loss={loss_pct:5.1f}% achieved={achieved:>10.0f}pps",
              file=sys.stderr)
        if loss_pct > args.abort_loss_pct:
            high_loss_streak += 1
            if high_loss_streak >= 2:
                print("[tp] high loss two steps in a row -- stopping sweep.",
                      file=sys.stderr)
                break
        else:
            high_loss_streak = 0

    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(out / "summary.json", "w") as f:
        json.dump({
            "fpga_ip": args.fpga_ip,
            "fpga_port": args.fpga_port,
            "rates": rows,
            "ts_unix": time.time(),
        }, f, indent=2)

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
