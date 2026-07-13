// read_netlayer_counters.cpp — dump NetLayer packet counters via xrt::ip.
//
// Counters are 64-bit, exposed via S_AXIL_nl at known offsets (per
// fpga-network/NetLayers/kernel.xml). Use to diagnose where packets vanish:
//   eth_in   -> arp_in / icmp_in / udp_in   (parse stage)
//   udp_in   -> app_out                      (UDP -> compute-kernel stream)
//   app_in   -> udp_out -> eth_out           (compute-kernel stream -> wire)
//
// Build:
//   g++ -std=c++17 -O2 -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
//       -o read_netlayer_counters read_netlayer_counters.cpp \
//       -lxrt_coreutil -luuid

#include <xrt/xrt_device.h>
#include <experimental/xrt_ip.h>
#include <experimental/xrt_xclbin.h>

#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct Counter {
    const char* name;
    uint32_t off;
};

// All 64-bit, only the *_packets ones are listed (drop bytes/cycles).
static const std::vector<Counter> COUNTERS = {
    {"eth_in_packets",   0x0410},
    {"pkth_in_packets",  0x0428},
    {"arp_in_packets",   0x0440},
    {"arp_out_packets",  0x0458},
    {"icmp_in_packets",  0x0470},
    {"icmp_out_packets", 0x0488},
    {"ethhi_out_packets",0x04A0},
    {"eth_out_packets",  0x04B8},
    {"udp_in_packets",   0x04D0},
    {"app_out_packets",  0x04E8},
    {"udp_out_packets",  0x0500},
    {"app_in_packets",   0x0518},
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <xclbin> <bdf>\n", argv[0]);
        return 2;
    }
    const std::string xclbin_path = argv[1];
    const std::string bdf         = argv[2];

    auto dev  = xrt::device(bdf);
    auto xb   = xrt::xclbin(xclbin_path);
    auto uuid = dev.register_xclbin(xb);
    auto ip   = xrt::ip(dev, uuid, "networklayer:{networklayer_1}");

    std::printf("%-22s %s\n", "counter", "value");
    std::printf("%-22s %s\n", "-------", "-----");
    for (const auto& c : COUNTERS) {
        uint32_t lo = ip.read_register(c.off);
        uint32_t hi = ip.read_register(c.off + 4);
        uint64_t v = (uint64_t(hi) << 32) | uint64_t(lo);
        std::printf("%-22s %llu\n", c.name, (unsigned long long)v);
    }
    return 0;
}
