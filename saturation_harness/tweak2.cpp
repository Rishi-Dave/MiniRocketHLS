// tweak2.cpp — write 32-bit value(s) to NetLayer AXI-Lite register(s).
// BATCH: opens the device / registers the xclbin ONCE, then writes all
// <offset value> pairs (the old one-write-per-invocation form re-registered
// the xclbin every call — far too slow for 64 SocketTable slot writes).
//
// Build: g++ -std=c++17 -O2 -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
//        -o tweak2 tweak2.cpp -lxrt_coreutil -luuid -pthread
//
// Usage: tweak2 <xclbin> <bdf> <hexoff> <hexval> [<hexoff> <hexval> ...]
//   e.g. tweak2 k.xclbin 0000:3b:00.1 1C C0A8281F 820 C0A8281F 8A0 F2E2 ...

#include <xrt/xrt_device.h>
#include <experimental/xrt_ip.h>
#include <experimental/xrt_xclbin.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc < 5 || ((argc - 3) % 2) != 0) {
        std::fprintf(stderr,
            "usage: %s <xclbin> <bdf> <hexoff> <hexval> [<hexoff> <hexval> ...]\n", argv[0]);
        return 2;
    }
    const std::string xclbin_path = argv[1];
    const std::string bdf         = argv[2];

    auto dev  = xrt::device(bdf);
    auto xb   = xrt::xclbin(xclbin_path);
    auto uuid = dev.register_xclbin(xb);
    auto ip   = xrt::ip(dev, uuid, "networklayer:{networklayer_1}");

    int n = 0, bad = 0;
    for (int i = 3; i + 1 < argc; i += 2) {
        uint32_t off = (uint32_t)strtoul(argv[i],   nullptr, 16);
        uint32_t val = (uint32_t)strtoul(argv[i+1], nullptr, 16);
        ip.write_register(off, val);
        uint32_t rb = ip.read_register(off);
        if (rb != val) { bad++; std::printf("  MISMATCH 0x%X: wrote 0x%X got 0x%X\n", off, val, rb); }
        n++;
    }
    std::printf("tweak2: %d writes, %d mismatch\n", n, bad);
    return bad ? 1 : 0;
}
