# fpga-network/docs — stale AND misplaced copy, see canonical docs

This file was a verbatim copy of the Dec-2025 root `ALGORITHM.md` — it is
not just stale, it is **misplaced**: `fpga-network/` is the NetLayer / F2F
network-saturation stack (`minirocket_fused`, `pktDropper`, `udp_generator`),
and algorithm documentation for the kernels it runs belongs with the
host-dispatched thread's docs, not here. Do not cite it.

Canonical documentation: [../../README.md](../../README.md) ·
[../../docs/ALGORITHM.md](../../docs/ALGORITHM.md) ·
[../../docs/NETWORK_RESULTS.md](../../docs/NETWORK_RESULTS.md) (this
directory's actual thread) ·
[../../saturation_harness/FCCM_network_result_SUMMARY.md](../../saturation_harness/FCCM_network_result_SUMMARY.md)
