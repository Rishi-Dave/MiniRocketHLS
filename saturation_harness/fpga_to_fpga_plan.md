# FPGA-to-FPGA (pc161-FPGA → pc160-FPGA) — plan & blocker

## Goal
Drive the DUT (pc160 fused v4 kernel) at true line rate from an FPGA packet
generator (pc161's U280), removing the host/NIC sender bottleneck entirely —
the strongest "no host anywhere" narrative and the way to find the DUT's true
compute ceiling (est. millions of pps, well past the ~700k the host-NIC C
sender reaches).

## BLOCKER: physical cabling (needs OCT admin / datacenter hands)
Verified 2026-07-06:
- Current data link = **pc161 host NIC `enp135s0f0` (40G Intel i40e) ↔
  pc160 FPGA CMAC**. (Proven: Python/C senders on pc161's host NIC reach
  pc160's FPGA CMAC; app_in increments.)
- **pc161's FPGA CMAC QSFP is dark** — programmed pc161's U280 with the
  cmac_0-containing fused xclbin and read `cmac_status`:
  `stat_rx_status=0, stat_rx_block_lock=0` → "NO RX signal, no peer/cable."
- Both hosts have only ONE high-speed NIC (the 40G i40e). The U280 CMAC QSFP
  is a separate port, currently with no live peer on pc161.

### The re-cabling request (for OCT admin)
Move the cable end at **pc161** from the host NIC QSFP (`enp135s0f0`) to the
**pc161 U280 QSFP0** cage, so the link becomes:
  **pc161-U280-CMAC-QSFP0  ↔  pc160-U280-CMAC-QSFP0**  (direct FPGA-to-FPGA).
(Alternatively: patch both U280 CMAC QSFP ports into a common 100G switch.)
Note this repurposes pc161: it can no longer be the host-NIC sender while
cabled FPGA-to-FPGA — but that's the point (FPGA generator replaces it).
Confirm the CMAC line rate both cmac_0 IPs are built for (our cmac_0 links at
40G to the i40e today; a 100G FPGA-to-FPGA link may need a 100G cmac_0 build).

## After re-cabling: generator kernel (build effort on Wolverine)
pc161's FPGA needs a CMAC packet-blaster to emit UDP frames to pc160's CMAC:
- Reuse the 100G-fpga-network-stack (cmac_0 + networklayer) + a small
  "traffic generator" kernel that emits UDP packets (dst .10:62177, src port
  62178 to match pc160's socket table) at a configurable/line rate. Check the
  network stack repo for an existing benchmark/generator (e.g. a UDP TX test
  kernel) before writing one.
- Program pc161's FPGA with generator xclbin; program pc160 with fused v4;
  measure pc160 app_in delta = FPGA-to-FPGA processed rate (ground truth).

## Remotely-achievable alternative (no re-cabling)
DPDK (kernel-bypass) or multi-thread/multi-queue AF_PACKET sender on pc161's
40G host NIC → pushes past the current single-thread C-sender ~700k toward
~40G line rate (~47M pps for 106B frames), finding the DUT ceiling without
admin. Substantial DPDK setup (hugepages, VFIO/NIC binding).

## Current standing (already decisive, host-NIC path)
Fused v4 free-run: 0% loss to ~700k pps / ~358 Mbps (C sender), STILL
unsaturated; ~50x CPU-MiniRocket (14k/7.2 Mbps). Python path: 0% loss to
~312k/160 Mbps, ~22x. FPGA compute ceiling not yet reached.
