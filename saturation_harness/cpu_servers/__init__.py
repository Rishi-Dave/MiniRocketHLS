"""Standalone CPU inference servers used by the CPU baseline adapter.

Each script is a UDP server that accepts one whole time-series sample per
datagram and replies with one prediction. The harness can run them locally
(loopback) for smoke tests or deploy them on a separate host for a true
two-machine CPU vs FPGA comparison.
"""
