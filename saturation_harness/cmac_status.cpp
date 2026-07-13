// cmac_status.cpp — read CMAC link/PHY status registers via XRT.
//
// Build:
//   g++ -std=c++17 -O2 -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
//       -o cmac_status cmac_status.cpp -lxrt_coreutil -luuid
//
// Usage: ./cmac_status <xclbin> <bdf>
//
// Reads stat_* registers from cmac_0 to determine if the 100G CMAC link is up
// (any peer connected) and counts. Register offsets from the cmac_0 kernel arg
// list in the xclbin (each arg is sequential 32-bit AXI-Lite).
//
// We just probe a handful that matter:
//   stat_tx_status (arg 9)        offset 0x0058
//   stat_rx_status (arg 10)       offset 0x0060
//   stat_rx_block_lock (arg 12)   offset 0x0070
//   stat_rx_lane_sync (arg 13)    offset 0x0078
//   stat_tx_total_packets (24)    offset 0x00d8
//   stat_rx_total_packets (40)    offset 0x0158
//
// (These offsets are computed: kernel args start at 0x10, and the xclbin's
// kernel.xml has size=4 + 4-byte stride per arg. Args 0..N-1 → offsets
// 0x10, 0x18, 0x20, ... = 0x10 + 8*i. So arg i offset = 0x10 + i*8.)

#include <experimental/xrt_ip.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <xclbin> <bdf>\n";
        return 2;
    }
    std::string xclbin = argv[1];
    std::string bdf    = argv[2];

    auto dev  = xrt::device(bdf);
    auto uuid = dev.load_xclbin(xclbin);
    auto cmac = xrt::ip(dev, uuid, "cmac_0:{cmac_0_1}");

    // Helpers
    auto rd = [&](uint32_t off) -> uint32_t { return cmac.read_register(off); };

    // Arg i is at offset 0x10 + 8*i for a kernel control_reg layout
    auto arg_off = [](int i) -> uint32_t { return 0x10u + 8u * static_cast<uint32_t>(i); };

    // Trigger a stat snapshot — arg 17 is stat_pm_tick: writing it latches stats
    cmac.write_register(arg_off(17), 1);

    uint32_t tx_status        = rd(arg_off(9));
    uint32_t rx_status        = rd(arg_off(10));
    uint32_t stat_status      = rd(arg_off(11));
    uint32_t rx_block_lock    = rd(arg_off(12));
    uint32_t rx_lane_sync     = rd(arg_off(13));
    uint32_t rx_lane_sync_err = rd(arg_off(14));
    uint32_t stat_an_link_ctl = rd(arg_off(15));
    uint32_t lt_status        = rd(arg_off(16));
    uint32_t tx_tot_pkts      = rd(arg_off(20));
    uint32_t rx_tot_pkts      = rd(arg_off(40));
    uint32_t rx_bad_fcs       = rd(arg_off(63));

    std::printf("cmac_0 status snapshot (bdf=%s):\n", bdf.c_str());
    std::printf("  stat_tx_status        = 0x%08x\n", tx_status);
    std::printf("  stat_rx_status        = 0x%08x  <- link OK if bit 0 = 1\n", rx_status);
    std::printf("  stat_status           = 0x%08x\n", stat_status);
    std::printf("  stat_rx_block_lock    = 0x%08x  <- signal locked if non-zero\n", rx_block_lock);
    std::printf("  stat_rx_lane_sync     = 0x%08x\n", rx_lane_sync);
    std::printf("  stat_rx_lane_sync_err = 0x%08x\n", rx_lane_sync_err);
    std::printf("  stat_an_link_ctl      = 0x%08x\n", stat_an_link_ctl);
    std::printf("  stat_lt_status        = 0x%08x\n", lt_status);
    std::printf("  stat_tx_total_packets = %u\n", tx_tot_pkts);
    std::printf("  stat_rx_total_packets = %u\n", rx_tot_pkts);
    std::printf("  stat_rx_bad_fcs       = %u\n", rx_bad_fcs);

    bool link_up   = (rx_status & 1) && (rx_block_lock != 0);
    bool signal_on = (rx_block_lock != 0);
    std::printf("\nVerdict: signal_present=%s  link_up=%s\n",
                signal_on ? "YES" : "NO",
                link_up   ? "YES" : "NO");

    if (link_up)        std::puts("  -> a 100G peer IS connected and aligned.");
    else if (signal_on) std::puts("  -> RX signal detected but PHY not aligned (link partially up).");
    else                std::puts("  -> NO RX signal. CMAC port has no peer or no cable.");
    return 0;
}
