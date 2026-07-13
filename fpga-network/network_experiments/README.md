# Network experiments (RTT & TP)

Sender-side latency + throughput measurement against the FPGA's NetLayer/UDP path.

Adapted from pyuvaraj37/nlp@master:fpga-network/xrt_host_api/network_experiments/.
Upstream defaults targeted a host-to-host TCP/MSS sweep (1408-byte packets,
.31<->.29 subnet). This version targets the OCT MiniRocketHLS setup:
64-byte AXI-S beats, OCT pc158 NIC, current SocketTable conventions.

## Default targets (1-node, pc158 sender + pc158 FPGA)

- FPGA NetLayer:        192.168.40.10:62177  (configured by setup_netlayer)
- Host bound socket:    0.0.0.0:62178        (MUST match SocketTable[0].theirPort)
- Host data-plane NIC:  enp134s0f0 (192.168.40.30)

For 2-node (sender on pc159, FPGA on pc158):

    export FPGA_IP=192.168.40.10 FPGA_PORT=62177 LISTEN_PORT=62178
    # Run setup_netlayer on pc158 with HOST_IP=192.168.40.31 (pc159 NIC)
    # and HOST_MAC=<pc159 NIC mac>.

## Prerequisites

1. xclbin programmed (e.g. v5 HYDRA-axis or Prith's MiniRocket reference).
2. setup_netlayer run (programs SocketTable[0] and ARP[host_ip & 0xFF]).
3. load_hydra_hbm (or load_minirocket_hbm) running in tmux -- kernel is in
   the `while(true){ run.start(); run.wait(); }` spin.
4. Static ARP on host:
       sudo ip neigh replace 192.168.40.10 lladdr 02:00:00:00:0a:0a dev enp134s0f0

## RTT -- Python

    python3 RTT/rtt.py --fpga-ip 192.168.40.10 --fpga-port 62177 \
        --listen-port 62178 --packets 1000 --packet-bytes 64 \
        --out runs/rtt_$(date +%s)

## RTT -- C++ (sockets)

The Xilinx Vitis env prepends cmake 3.3.2 to PATH, which is too old for
`cmake -B`. Use system cmake explicitly (`/usr/bin/cmake`, v3.22):

    cd RTT/sockets
    mkdir -p build && cd build
    /usr/bin/cmake .. && make -j
    ./rtt_client --fpga-ip 192.168.40.10 --fpga-port 62177 \
        --listen-port 62178 --packets 1000 \
        --out ../../../../../runs/rtt_cpp_$(date +%s)

## TP -- Python

    python3 TP/throughput.py --fpga-ip 192.168.40.10 --fpga-port 62177 \
        --listen-port 62178 --rate-min 1024 --rate-max 262144 \
        --rate-steps 8 --duration-s 2 --out runs/tp_$(date +%s)

## TP sweep with FPGA-side counters

    bash TP/run_sweep.sh runs/tp_$(date +%s) localhost
    # Reads udp_in_packets via read_netlayer_counters before/after the sweep.

## Known limitation (May 2026)

Current MiniRocketHLS xclbins (v3/v4/v5) have a NetLayer UDP TX bug: the
kernel emits responses (app_out increments) but NetLayer's udp_out_packets
stays 0. RTT scripts will see 100% loss on these xclbins. Mitigations:

- Use Prith's older fpga-network/minirocket reference xclbin (not yet
  re-verified post-src-port-fix; see memory `saturation_pivot_2026_04_28.md`).
- Use a pure-echo xclbin (CMAC + NetLayer + passthrough kernel) for the
  pure-network RTT baseline; subtract from FPGA-compute RTT.
- TP works fine without ACKs -- measure achieved rate via FPGA-side
  udp_in_packets delta (see run_sweep.sh).

## DPDK

`RTT/dpdk/` contains a partial sketch by pyuvaraj37. **Not usable as-is** --
no Ethernet/IPv4/UDP header construction, no MAC/ARP, no DPDK port init.
Salvage post-June-6 ICCAD deadline. Reference for a real implementation:
Xilinx VNX `Ethernet/` directory has DPDK examples.
