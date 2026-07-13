# OCT / CloudLab re-cabling request — FPGA-to-FPGA 100G link (pc160 ↔ pc161)

**Project:** octfpga-PG0   **User:** rdave009   **Date:** 2026-07-06

## Request (one cable move)
On **pc161**, move the QSFP cable currently in the **host NIC (`enp135s0f0`,
Intel i40e, 40G)** to the node's **Alveo U280 CMAC QSFP0 cage**, so the 100G
data link becomes a **direct FPGA-to-FPGA connection**:

    pc161  U280 CMAC QSFP0   <───────>   pc160  U280 CMAC QSFP0

(Equivalently: patch **both** nodes' U280 CMAC QSFP0 ports into a common
100G switch, if a switch is available and preferred.)

## Current state (why)
- Today the cable runs **pc161 host-NIC ↔ pc160 U280 CMAC**, so our streaming-
  inference network experiment is driven by a CPU/NIC sender on pc161 (capped
  ~700k pps), while pc161's own U280 CMAC QSFP sits **dark/unused**
  (verified: `cmac_status` on pc161's programmed U280 shows
  `stat_rx_block_lock=0`, no RX signal).
- We want to use **pc161's U280 as a hardware packet generator** driving
  **pc160's U280** (fused MiniRocket inference kernel) at 100G line rate —
  a true FPGA-to-FPGA path with no host/NIC in the loop, to characterize the
  DUT's real throughput ceiling.

## After re-cabling — please confirm
1. Both U280 CMAC QSFP0 ports show link/GT lock once each is programmed with a
   CMAC design (we'll program them).
2. Whether the link negotiates 100G or 40G (our current `cmac_0` IP links at
   40G to the i40e today; we may rebuild cmac_0 for 100G — let us know the
   physical link capability).

## Impact / rollback
- pc161 will no longer be reachable as a host-NIC UDP sender on the
  192.168.40.0/24 data net while cabled FPGA-to-FPGA (that is intended; the
  FPGA generator replaces it). Management/ssh on the control net is unaffected.
- Rollback = move the cable back to pc161's host NIC QSFP (restores today's
  setup).

## Contact
rdave009 (experiment owner). Nodes pc160 (BDF 0000:3b:00.1) and pc161
(BDF 0000:3b:00.1), both U280 gen3x16_xdma, shelled/HEALTHY.
