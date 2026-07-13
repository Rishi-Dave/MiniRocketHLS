#!/usr/bin/env python3
"""Simple UDP echo server. Stand-in for the FPGA NetLayer for CPU baseline.

Binds to <listen-ip>:<listen-port>, echoes every received datagram back to
the sender. Used as the echo peer for rtt.py / throughput.py when no FPGA
is in the loop.
"""
import argparse
import socket
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-ip", default="0.0.0.0")
    ap.add_argument("--listen-port", type=int, default=62177)
    ap.add_argument("--rcvbuf", type=int, default=16 << 20)
    ap.add_argument("--sndbuf", type=int, default=16 << 20)
    args = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.rcvbuf)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, args.sndbuf)
    s.bind((args.listen_ip, args.listen_port))
    print(f"[echo] listening on {args.listen_ip}:{args.listen_port}", flush=True)

    n = 0
    while True:
        try:
            data, addr = s.recvfrom(2048)
        except KeyboardInterrupt:
            break
        s.sendto(data, addr)
        n += 1
        if n % 10000 == 0:
            print(f"[echo] n={n}", flush=True)


if __name__ == "__main__":
    main()
