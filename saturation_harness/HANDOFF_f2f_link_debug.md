# HANDOFF — Debug the FPGA-to-FPGA CMAC link (pc161 → pc160 not forwarding)

> **SUPERSEDED — incident resolved.** Root cause was stale socket-table
> offsets; the recovery recipe is in `FCCM_network_result_SUMMARY.md` (see
> also `docs/NETWORK_RESULTS.md`). Kept for historical debugging context only.

You are a fresh Claude Code session (Fable). Read this whole prompt, then solve ONE problem:
**restore the FPGA-to-FPGA 100G link so pc161's in-fabric UDP generator packets reach
pc160's fused-MiniRocket DUT.** It worked this morning (2026-07-09) and stopped working this
evening after a 4h build gap, several reprograms, and two cold power-cycles. Once the link is
stable, run the already-prepared saturation-knee sweep. This is the MiniRocketHLS FCCM project
(`/home/rdave009/minirocket-hls/MiniRocketHLS`, CLAUDE.md at repo root — follow its safety rules,
esp. Rule 2: NO `xbutil program`/hw-run without stating variant/freq/CUs/HBM; you already have
standing authorization for THIS deploy+sweep task).

## The one symptom to fix
- pc161 (generator) transmits: its NetLayer `eth_out_packets` climbs at the offered rate
  (e.g. 700k–2.7M pps).
- pc160 (DUT) receives NOTHING: `eth_in_packets` delta = 0, even for broadcast ARP frames
  (pc161 sent 2.7M ARP broadcasts, pc160 got 0).
- **Reverse direction WORKS**: pc160 → pc161 delivers fine (pc160 sent 255 ARP, pc161
  `eth_in` +260). So the switch bridges pc160→pc161 but not pc161→pc160.
- Both CMACs flap `stat_rx_block_lock` between `0x000fffff` (locked) and `0x0` (unlocked).
- Last reads: pc161 `conf_tx=1, conf_rx=1, stat_tx_status=0x1, block_lock=0x0` (its RX unlocked);
  pc160 `block_lock` reaches `0x0fffff` (RX locks to the switch idle) but no frames arrive.
- **Leading hypothesis:** pc161's CMAC TX is not physically driving the fiber (NetLayer counts
  the handoff but the GT/PMA TX isn't up), and/or the two GT links won't hold bidirectional lock.
  It is asymmetric and TX-side (pc161), so focus there.

## Environment / topology
- **Wolverine** = the build+orchestration host (where you run). It can `ssh` BOTH nodes.
  IMPORTANT: pc160 CANNOT ssh pc161 (publickey denied) — orchestrate everything FROM Wolverine.
  Vitis/Vitis_HLS 2023.2 at `/home/Xilinx/`. XRT on nodes: `source /opt/xilinx/xrt/setup.sh`.
- **pc160** = `rdave009@pc160.cloudlab.umass.edu`, U280 BDF `0000:3b:00.1`. The fused DUT.
  FPGA CMAC IP `192.168.40.10`, CMAC MAC `02:00:00:00:0a:0a`, UDP myPort 62177.
- **pc161** = `rdave009@pc161.cloudlab.umass.edu`, U280 BDF `0000:3b:00.1`. The packet generator.
  FPGA CMAC IP `192.168.40.11`, CMAC MAC `02:00:00:00:0b:0b`, UDP myPort 62178.
  (pc161 host NIC MAC is `f8:f2:1e:d3:26:b0` — not relevant to the FPGA-to-FPGA path.)
- CloudLab manifest: ONE switched LAN (`link-1`, VLAN 2501) connects all 6 interfaces incl. both
  FPGA CMAC ports. So pc161→pc160 goes pc161 CMAC TX → switch → pc160 CMAC RX. NOT point-to-point.
- SSH prefix that works: `ssh -o BatchMode=yes -o ConnectTimeout=30`. Filter XRT banner noise with
  `grep -viE "Autocomplete|XILINX_XRT|^PATH=|LD_LIBRARY|PYTHONPATH"`.

## xclbins & tools already on the nodes (persist across reboots — home dirs)
- pc160: `~/krnl.xclbin.fused_dropfull` (NEW drop-on-full DUT, timing-clean, USE THIS),
  `~/krnl.xclbin.fused` (old v4), `~/deploy_dropfull_f2f.sh`, `~/load_minirocket_hbm_fused`,
  `~/InsectSound_minirocket_model.json`, plus `cmac_conf setup_netlayer tweak2 read_netlayer_counters_ext`.
- pc161: `~/krnl.xclbin.gen` (generator), `~/start_gen`, plus `cmac_conf setup_netlayer tweak2
  read_netlayer_counters_ext`.
- The drop-on-full DUT build is DONE (BUILD OK, timing MET WNS +0.016ns @250MHz). Not your problem.

## Hard-won facts you MUST reuse (do not re-derive)
1. **NetLayer socket-table offsets are 8-byte stride, base off by +0x10 from the stale
   `kernel.xml` numbers `setup_netlayer` still uses.** Correct offsets (write via `tweak2 <xclbin>
   <bdf> <hexoff> <hexval> ...`, all host-order hex; `udpTxEngine` byteSwaps internally):
   `theirIP(i)=0x810+8*i`, `theirPort(i)=0x890+8*i`, `myPort(i)=0x910+8*i`, `valid(i)=0x990+8*i`.
   Ports: 62177=0xF2E1, 62178=0xF2E2. IPs: .10=0xC0A8280A, .11=0xC0A8280B. Fill all 16 slots for
   TDEST robustness. `udpTxEngine` SILENTLY DROPS any packet whose `SocketTable[TDEST].valid!=1`.
   `deploy_dropfull_f2f.sh` on pc160 already uses these correct offsets.
2. **`start_gen` uses `xrt::ip`** (NOT `xrt::kernel`/`xrt::run`, which SEGFAULT on these nodes):
   `start_gen <gen.xclbin> <bdf> <rate_divider>` → writes rate_divider@0x10, pulses ap_start@0x00.
   Generator is ap_ctrl_hs BUILD=1 free-run: pulse once, it runs forever. Emit rate ≈
   66.5M / rate_divider pps (div=100 ≈ 665k, div=30 ≈ 2.2M, div=0 ≈ max ~6M+).
3. **`cmac_conf <xclbin> <bdf> <mode>`** reads/writes cmac_0 with CORRECT offsets (the old
   `cmac_status` tool has WRONG offsets — ignore it). Modes: `read` (status, pm_tick-latched),
   `txon` (conf_tx=0x1, conf_rx=0x1 — NO test pattern), `bringup` (writes conf_tx=0x10 = TEST
   PATTERN, WRONG for real traffic — do not use), `gtreset` (toggles gt_reset 1→0 then reset 1→0
   with settle delays, re-enables tx/rx — a GT retrain). CMAC offsets: gt_reset@0x00, reset@0x04,
   conf_tx@0x0C, conf_rx@0x14, gt_loopback@0x90, stat_tx_status@0x200, stat_rx_status@0x204,
   stat_rx_block_lock@0x20C (0x0fffff=all 20 lanes locked), stat_tx_total@0x500,
   stat_rx_total@0x608, pm_tick@0x2B0 (write 1 to latch stats before reading).
   WARNING: every tool call does `load_xclbin` (same-xclbin = no-op for programming, but repeated
   cmac_conf calls seem to disturb GT lock — minimize them; read once, act, don't hammer).
4. **NetLayer counters** via `read_netlayer_counters_ext <xclbin> <bdf>` (grep the line you want):
   `eth_in_packets` (frames CMAC→NetLayer), `eth_out_packets` (NetLayer→CMAC TX),
   `app_in_packets` (NetLayer→kernel), `udp_out_packets` (kernel predictions→NetLayer TX),
   `arp_in/arp_out_packets`. These are stable/monotonic (unlike the pm_tick cmac counters).
5. **Switch MAC-learning gotcha:** pc160 only TXes (predictions) when it's receiving; when it stops
   receiving it stops TXing, the switch ages out pc160's MAC (~300s), and pc161 unicast to pc160
   stops being forwarded. Keep pc160 TXing to hold the entry: `tweak2 <fd> <bdf> 1010 C0A8280B`
   triggers an ARP (arp_discovery reg @0x1010 = target IP). BUT note: this evening even pc161→pc160
   BROADCAST failed, so MAC-learning is not the whole story right now — the TX-side/GT is suspect.

## What "working this morning" looked like (target state)
FPGA-to-FPGA, no host anywhere: pc161 gen → CMAC → switch → pc160 CMAC → fused DUT processed
**≥2.24M pps at 0% loss** (see `runs/tp_2026-07-09_f2f/ceiling.csv`; div=100→664k, div=30→2.24M, all
0% loss). That result is ALREADY BANKED — you do NOT need to re-earn the number. The bring-up that
worked (afternoon): fresh `xbutil reset --user` + `xbutil program krnl.xclbin.gen` on pc161 →
`start_gen` → `setup_netlayer` (iface+ARP) → `tweak2` corrected socket → .10; pc160 deployed with
corrected socket to accept from .11. It "just worked" — no explicit gtreset was needed. Both CMACs
showed STABLE `block_lock=0x0fffff`.

## What was already tried this evening (don't just repeat)
- Rebuilt/reprogrammed the generator on pc161 (clean reset+program) — it transmits (eth_out climbs).
- Deployed drop-on-full DUT on pc160 with corrected socket (`deploy_dropfull_f2f.sh`) — 0 mismatch.
- `cmac_conf txon` on both. Cold power-cycled pc160 (portal), then cold power-cycled pc161 (portal).
- ARP-triggered both directions. Settled 40s for STP.
- Result after ALL of that: pc161→pc160 still dead (0 frames, incl. broadcast); pc160→pc161 works;
  CMAC block_lock flaps on both.

## Your job — restore the link, then measure
Think about WHY pc161→pc160 is one-way-dead. Candidate root causes to investigate/rule out:
- pc161 CMAC **TX GT not physically up** despite conf_tx=1/tx_status=1 → try `cmac_conf gtreset` on
  pc161 (forces GT retrain), then verify pc160 starts locking to real frames. Watch pc161
  stat_tx_status and pc160 stat_rx_total/eth_in deltas.
- **GT link won't hold bidirectional lock** → a specific coordinated bring-up ORDER may be needed:
  e.g. gtreset both, then conf_tx/conf_rx=1 both, then let them train ~10–30s BEFORE any NetLayer
  config, and confirm `block_lock=0x0fffff` is STABLE (read 3× spaced, all 0xfffff) on BOTH before
  proceeding. Avoid hammering cmac_conf (each load_xclbin perturbs the GT).
- **Accidental gt_loopback** (0x90 nonzero) on either CMAC → would RX-lock to own TX, no peer
  traffic. Read 0x90 on both; if nonzero, write 0.
- **QSFP cage / lane-polarity / RS-FEC mismatch** (cmac_0 built INCLUDE_RS_FEC=1, CAUI4) — if a
  gtreset won't lock, this is physical/infra; escalate (the user set up the working F2F with "Prith"
  earlier and has power-cycled via CloudLab portal — you can ASK the user to power-cycle, but only
  after software bring-up attempts, and give exact portal steps).
- Since it worked this morning and only degraded after churn, a clean coordinated GT retrain on both
  is the most likely fix. The user notes "it was working this morning" — so the hardware/cabling is
  fine; this is a bring-up/state problem.

Verify success with: pc161 `start_gen <gen> <bdf> 100` (≈665k pps) → pc160 `eth_in_packets` delta
≈ 665k/s AND `app_in_packets` climbing at 0% loss. Keep pc160's socket pointed at .11 (corrected
offsets) so app_in counts, and pc160 TXing predictions (which also keeps the switch MAC fresh).

## Then: run the saturation-knee sweep (the actual deliverable)
Once the link is stable, run `saturation_harness/f2f_knee_sweep.sh` (already written; edit only if
the deploy path/model differs). It sweeps rate_divider 500→0 and records, per point:
`offered_pps` (pc160 eth_in/dur), `processed_pps = udp_out_delta × L / dur` (L=600 for InsectSound;
udp_out = completed prediction windows), `loss = 1 − processed/offered`, `bandwidth_kbps =
processed × 64 × 8 / 1000`. Because the DUT ingest is now DROP-ON-FULL, it will NOT wedge — you get
a clean saturation KNEE (processed plateaus at the ~2.24–2.9M-pps compute ceiling while offered
keeps rising and loss climbs smoothly), unlike the v4 sticky wedge. Output CSV →
`runs/tp_2026-07-09_f2f_knee/`. Then plot with `plot_bandwidth.py` (locked Kbps metric,
PAYLOAD_BYTES=64) for the FCCM figure: throughput plateau + rising loss.

## Ordering gotcha for the sweep loop
`start_gen` (load_xclbin) does NOT wipe the NetLayer socket for same-xclbin (verified), but if you
see zeros mid-sweep, re-assert pc161's socket (setup_netlayer + tweak2 corrected offsets) AFTER
start_gen. And re-trigger pc160 arp_discovery periodically to keep the switch MAC entry alive.

## Memory / references
Full project state: `~/.claude/projects/-home-rdave009-minirocket-hls/memory/` — esp.
`network_bandwidth_2026_07_04.md` (this whole saga, corrected offsets, resume steps) and
`MEMORY.md` index. Deliverable summary: `saturation_harness/FCCM_network_result_SUMMARY.md`.
Corrected-socket deploy: pc160 `~/deploy_dropfull_f2f.sh`. Generator source:
`fpga-network/udp_generator/`. DUT source: `fpga-network/minirocket_fused/` (ingest drop-on-full at
src/minirocket_fused.cpp:370). NetLayer HLS: `fpga-network/NetLayers/100G-fpga-network-stack-core/`.
