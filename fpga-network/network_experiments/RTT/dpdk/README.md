# DPDK RTT (sketch -- DO NOT BUILD)

The upstream pyuvaraj37/nlp `RTT/dpdk/rtt_client.cpp` is a sketch:

- Sends raw zero-byte frames; never constructs Ethernet / IPv4 / UDP headers.
- Polls RX in an unbounded blocking loop; no timeout.
- No DPDK port init, no MAC/ARP, no mempool sizing.
- Upstream CMakeLists referenced `rtt_dpdk.c`; the actual source is
  `rtt_client.cpp`. The file is preserved here for reference only.

A working DPDK client needs the full Xilinx VNX `Ethernet/` example as a
template (port init, mempool, header construction, ARP). Estimated 1-2 days
of work. Out of scope before the June 6 ICCAD deadline.

For RTT measurements before then, use `../rtt.py` or `../sockets/rtt_client`.
