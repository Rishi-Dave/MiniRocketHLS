#!/usr/bin/env python3
"""
setup_netlayer.py — Bring up the networklayer kernel from
xupsh/100G-fpga-network-stack-core after `xbutil program <xclbin>`.

This is a one-shot bring-up step that programs the NetLayer's AXI-Lite
registers so that incoming UDP packets on a chosen port are dispatched to
the compute kernel via the AXIS streams wired in fpga-network/config.cfg.

Register map (from fpga-network/NetLayers/kernel.xml, axi-lite slave
S_AXIL_nl, base 0x0, range 0x2000):

  Interface config (writable, written once):
    mac_address    @0x0010 8B  (low 48b of u64; FPGA's own MAC)
    ip_address     @0x0018 4B  (FPGA's own IPv4, host byte order u32)
    gateway        @0x001C 4B  (default gateway IP)
    ip_mask        @0x0020 4B  (subnet mask)

  UDP socket table (16 slots):
    udp_number_sockets   @0x0810 4B
    theirIP base         @0x0820 (+4 per slot)
    theirPort base       @0x08A0 (+4 per slot)
    myPort base          @0x0920 (+4 per slot)
    valid base           @0x09A0 (+4 per slot, write 1 to activate)

  ARP table (256 entries, indexed by hash):
    arp_discovery        @0x1010 4B (write target IP to trigger ARP)
    arp_valid base       @0x1100
    arp_ip_addr base     @0x1400
    arp_mac_addr base    @0x1800

  Debug counters: 0x0400..0x05F0 (read-only, plus 0x05F0 reset).

Usage example (pc158, FPGA NetLayer = 192.168.40.10, host = 192.168.40.30):
    python3 setup_netlayer.py \
        --xclbin /users/rdave009/MiniRocketHLS/fpga-network/build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/krnl.xclbin \
        --device 0000:3b:00.1 \
        --fpga-mac 00:0a:35:02:9d:e5 \
        --fpga-ip 192.168.40.10 \
        --fpga-mask 255.255.255.0 \
        --fpga-gw 192.168.40.1 \
        --fpga-port 62177 \
        --host-ip 192.168.40.30 \
        --host-mac auto \
        --host-port 62178

After successful bring-up, packets sent to the FPGA on 62177 will be
dispatched as AXIS to minirocket_inference.input_timeseries, and packets
emitted by minirocket_inference.output_predictions will be sent back to
host:62178.

Note: the NetLayer holds these registers as long as the FPGA is programmed.
A subsequent `xbutil program` resets them — re-run setup_netlayer.py after
each program.
"""
import argparse
import ipaddress
import struct
import sys
import time

try:
    import pyxrt as xrt
except ImportError:
    print("[FATAL] pyxrt not importable. Source /opt/xilinx/xrt/setup.sh first.",
          file=sys.stderr)
    sys.exit(1)


# Register offsets
REG_MAC          = 0x0010
REG_IP           = 0x0018
REG_GATEWAY      = 0x001C
REG_MASK         = 0x0020
REG_NUM_SOCKETS  = 0x0810
REG_THEIR_IP_BASE   = 0x0820
REG_THEIR_PORT_BASE = 0x08A0
REG_MY_PORT_BASE    = 0x0920
REG_VALID_BASE      = 0x09A0
REG_ARP_DISCOVERY   = 0x1010
REG_ARP_VALID_BASE  = 0x1100
REG_ARP_IP_BASE     = 0x1400
REG_ARP_MAC_BASE    = 0x1800
REG_DEBUG_RESET     = 0x05F0


def parse_mac(s: str) -> int:
    """Convert 'aa:bb:cc:dd:ee:ff' to 48-bit int (high byte = first byte)."""
    parts = s.split(":")
    if len(parts) != 6:
        raise ValueError(f"bad MAC {s!r}")
    val = 0
    for p in parts:
        val = (val << 8) | int(p, 16)
    return val


def ip_to_u32(s: str) -> int:
    """IPv4 string → host-order u32 (most significant octet at MSB)."""
    return int(ipaddress.IPv4Address(s))


def auto_host_mac(host_ip: str) -> str:
    """Try to learn host's MAC from /sys/class/net or `ip neigh`/arp.
    Returns 'aa:bb:cc:dd:ee:ff' or raises."""
    import os, subprocess, re
    # First try: which iface has this IP
    try:
        for iface in os.listdir("/sys/class/net"):
            p = f"/sys/class/net/{iface}/address"
            if not os.path.exists(p):
                continue
            with open(f"/proc/net/fib_trie") as f:
                pass  # just to ensure /proc is mounted
            # check iface IP
            try:
                out = subprocess.check_output(
                    ["ip", "-4", "addr", "show", "dev", iface],
                    text=True, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                continue
            if host_ip in out:
                with open(p) as fmac:
                    return fmac.read().strip()
    except Exception as e:
        pass
    raise RuntimeError(
        f"could not auto-detect MAC for host IP {host_ip}; "
        f"pass --host-mac explicitly")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--device", default="0",
                    help="XRT device BDF (e.g. 0000:3b:00.1) or index")
    ap.add_argument("--fpga-mac", required=True,
                    help="FPGA's own MAC, e.g. 00:0a:35:02:9d:e5")
    ap.add_argument("--fpga-ip", required=True)
    ap.add_argument("--fpga-mask", default="255.255.255.0")
    ap.add_argument("--fpga-gw", default="192.168.40.1")
    ap.add_argument("--fpga-port", type=int, required=True,
                    help="UDP port the NetLayer listens on (= harness --fpga-port)")
    ap.add_argument("--host-ip", required=True,
                    help="Host's IP on the data-plane subnet")
    ap.add_argument("--host-mac", default="auto",
                    help="Host's MAC, or 'auto' to detect from /sys/class/net")
    ap.add_argument("--host-port", type=int, required=True,
                    help="Host UDP port for replies (= harness --listen-port)")
    ap.add_argument("--socket-slot", type=int, default=0,
                    help="Which of the 16 socket slots to program")
    ap.add_argument("--reset-counters", action="store_true",
                    help="Reset debug counters after setup")
    ap.add_argument("--readback", action="store_true",
                    help="Read back all programmed registers and print")
    ap.add_argument("--skip-program", action="store_true",
                    help="Assume xclbin already loaded; skip device.load_xclbin")
    args = ap.parse_args()

    if args.host_mac == "auto":
        args.host_mac = auto_host_mac(args.host_ip)
        print(f"[info] auto-detected host MAC: {args.host_mac}")

    # Open device + load xclbin
    print(f"[info] opening XRT device '{args.device}'")
    try:
        if args.device.isdigit():
            d = xrt.device(int(args.device))
        else:
            d = xrt.device(args.device)
    except Exception as e:
        print(f"[fatal] xrt.device() failed: {e}", file=sys.stderr)
        sys.exit(2)

    if not args.skip_program:
        print(f"[info] loading xclbin {args.xclbin}")
        uuid = d.load_xclbin(args.xclbin)
    else:
        # We need a UUID for xrt::ip; fetch from already-loaded xclbin
        xb = xrt.xclbin(args.xclbin)
        uuid = d.register_xclbin(xb)

    # networklayer is ap_ctrl_none → must use xrt::ip
    ip = xrt.ip(d, uuid, "networklayer:{networklayer_1}")
    print(f"[info] opened ip handle for networklayer:{{networklayer_1}}")

    def w(off, val):
        ip.write_register(off, val & 0xFFFFFFFF)

    def w64(off, val64):
        # AXI-Lite is 32b; write low then high.
        w(off,     val64 & 0xFFFFFFFF)
        w(off + 4, (val64 >> 32) & 0xFFFFFFFF)

    def r(off):
        return ip.read_register(off)

    # 1) Interface config
    fpga_mac_int  = parse_mac(args.fpga_mac)
    fpga_ip_u32   = ip_to_u32(args.fpga_ip)
    fpga_mask_u32 = ip_to_u32(args.fpga_mask)
    fpga_gw_u32   = ip_to_u32(args.fpga_gw)
    print(f"[cfg ] FPGA MAC={args.fpga_mac} IP={args.fpga_ip} mask={args.fpga_mask} gw={args.fpga_gw}")
    w64(REG_MAC, fpga_mac_int)
    w(REG_IP,      fpga_ip_u32)
    w(REG_GATEWAY, fpga_gw_u32)
    w(REG_MASK,    fpga_mask_u32)

    # 2) UDP socket slot
    slot = args.socket_slot
    if not (0 <= slot < 16):
        print("[fatal] --socket-slot must be 0..15", file=sys.stderr)
        sys.exit(3)
    host_ip_u32 = ip_to_u32(args.host_ip)
    print(f"[cfg ] socket[{slot}]: their {args.host_ip}:{args.host_port} my {args.fpga_port}")
    w(REG_THEIR_IP_BASE   + slot * 4, host_ip_u32)
    w(REG_THEIR_PORT_BASE + slot * 4, args.host_port)
    w(REG_MY_PORT_BASE    + slot * 4, args.fpga_port)
    w(REG_VALID_BASE      + slot * 4, 1)
    # number of active sockets = max(slot+1, current). We just set to slot+1.
    w(REG_NUM_SOCKETS, slot + 1)

    # 3) ARP: install host MAC manually for fastest convergence (avoid
    # ARP discovery race). We pick a deterministic ARP table index by hashing
    # the host IP into 8 bits — but the simpler approach: write host_ip+mac
    # into entry 0 (or any free slot). The networklayer hashes lookups, so
    # we mirror the same scheme: low byte of IP is the index.
    arp_idx = host_ip_u32 & 0xFF
    host_mac_int = parse_mac(args.host_mac)
    print(f"[cfg ] ARP[{arp_idx}]: ip={args.host_ip} mac={args.host_mac}")
    w(REG_ARP_IP_BASE  + arp_idx * 4, host_ip_u32)
    w64(REG_ARP_MAC_BASE + arp_idx * 8, host_mac_int)
    w(REG_ARP_VALID_BASE + arp_idx * 4, 1)

    # 4) Optionally reset debug counters
    if args.reset_counters:
        print("[info] resetting debug counters")
        w(REG_DEBUG_RESET, 1)

    # 5) Settle
    time.sleep(0.1)

    # 6) Readback (sanity)
    if args.readback:
        print("[chk ] readback:")
        print(f"  IP@0x18         = 0x{r(REG_IP):08x} (expect 0x{fpga_ip_u32:08x})")
        print(f"  MASK@0x20       = 0x{r(REG_MASK):08x}")
        print(f"  GW@0x1C         = 0x{r(REG_GATEWAY):08x}")
        print(f"  num_sockets     = {r(REG_NUM_SOCKETS)}")
        print(f"  slot{slot}.theirIP   = 0x{r(REG_THEIR_IP_BASE + slot*4):08x}")
        print(f"  slot{slot}.theirPort = {r(REG_THEIR_PORT_BASE + slot*4)}")
        print(f"  slot{slot}.myPort    = {r(REG_MY_PORT_BASE + slot*4)}")
        print(f"  slot{slot}.valid     = {r(REG_VALID_BASE + slot*4)}")
        print(f"  arp[{arp_idx}].valid  = {r(REG_ARP_VALID_BASE + arp_idx*4)}")
        print(f"  arp[{arp_idx}].ip     = 0x{r(REG_ARP_IP_BASE + arp_idx*4):08x}")

    print("[done] NetLayer bring-up complete. Send UDP traffic now.")


if __name__ == "__main__":
    main()
