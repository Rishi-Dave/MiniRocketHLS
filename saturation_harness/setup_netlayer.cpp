// setup_netlayer.cpp — minimal NetLayer bring-up via XRT C++ API.
//
// Programs the AXI-Lite registers of the xupsh/100G-fpga-network-stack-core
// networklayer kernel after `xbutil program <xclbin>`. pyxrt 2.16 does not
// expose xrt::ip, but the C++ API does, so we use this binary as a one-shot.
//
// Build:
//   g++ -std=c++17 -O2 -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
//       -o setup_netlayer setup_netlayer.cpp -lxrt_coreutil -luuid
//
// Usage:
//   ./setup_netlayer <xclbin> <bdf> <fpga_mac> <fpga_ip> <fpga_port> \
//                    <host_ip> <host_mac> <host_port>
//
// Example:
//   ./setup_netlayer krnl.xclbin 0000:3b:00.1 \
//                    02:00:00:00:0a:0a 192.168.40.10 62177 \
//                    192.168.40.30 f8:f2:1e:bc:e5:b0 62178

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <experimental/xrt_ip.h>
#include <experimental/xrt_xclbin.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <thread>
#include <arpa/inet.h>

namespace { // register offsets
constexpr uint32_t REG_MAC          = 0x0010;
constexpr uint32_t REG_IP           = 0x0018;
constexpr uint32_t REG_GATEWAY      = 0x001C;
constexpr uint32_t REG_MASK         = 0x0020;
constexpr uint32_t REG_NUM_SOCKETS  = 0x0810;
constexpr uint32_t REG_THEIR_IP_BASE   = 0x0820;
constexpr uint32_t REG_THEIR_PORT_BASE = 0x08A0;
constexpr uint32_t REG_MY_PORT_BASE    = 0x0920;
constexpr uint32_t REG_VALID_BASE      = 0x09A0;
constexpr uint32_t REG_ARP_VALID_BASE  = 0x1100;
constexpr uint32_t REG_ARP_IP_BASE     = 0x1400;
constexpr uint32_t REG_ARP_MAC_BASE    = 0x1800;
}

static uint64_t parse_mac(const std::string& s) {
    uint64_t v = 0;
    int byte = 0;
    for (char c : s) {
        if (c == ':') continue;
        v <<= 4;
        if (c >= '0' && c <= '9') v |= c - '0';
        else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
        else throw std::runtime_error("bad mac char: " + std::string(1, c));
        byte++;
    }
    if (byte != 12) throw std::runtime_error("bad mac length: " + s);
    return v;
}

static uint32_t parse_ip(const std::string& s) {
    in_addr a;
    if (inet_pton(AF_INET, s.c_str(), &a) != 1) {
        throw std::runtime_error("bad ip: " + s);
    }
    // ntohl gives host byte order = MSB first byte
    return ntohl(a.s_addr);
}

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "usage: " << argv[0]
                  << " <xclbin> <bdf> <fpga_mac> <fpga_ip> <fpga_port>"
                  << " <host_ip> <host_mac> <host_port>\n";
        return 2;
    }
    const std::string xclbin_path = argv[1];
    const std::string bdf         = argv[2];
    const uint64_t   fpga_mac     = parse_mac(argv[3]);
    const uint32_t   fpga_ip      = parse_ip(argv[4]);
    const uint32_t   fpga_port    = std::stoul(argv[5]);
    const uint32_t   host_ip      = parse_ip(argv[6]);
    const uint64_t   host_mac     = parse_mac(argv[7]);
    const uint32_t   host_port    = std::stoul(argv[8]);

    std::cout << "[info] opening device " << bdf << "\n";
    auto d = xrt::device(bdf);

    std::cout << "[info] registering xclbin " << xclbin_path << "\n";
    auto xb = xrt::xclbin(xclbin_path);
    auto uuid = d.register_xclbin(xb);

    // Open networklayer:{networklayer_1} as ap_ctrl_none ip
    std::cout << "[info] opening ip handle networklayer:{networklayer_1}\n";
    auto ip = xrt::ip(d, uuid, "networklayer:{networklayer_1}");

    auto w = [&](uint32_t off, uint32_t val) {
        ip.write_register(off, val);
    };
    auto w64 = [&](uint32_t off, uint64_t val) {
        w(off,     static_cast<uint32_t>(val & 0xFFFFFFFFu));
        w(off + 4, static_cast<uint32_t>((val >> 32) & 0xFFFFFFFFu));
    };
    auto r = [&](uint32_t off) -> uint32_t {
        return ip.read_register(off);
    };

    std::cout << std::hex << std::setfill('0');
    std::cout << "[cfg ] FPGA mac=0x" << std::setw(12) << fpga_mac
              << " ip=0x" << std::setw(8) << fpga_ip
              << " port=" << std::dec << fpga_port << std::hex << "\n";
    w64(REG_MAC, fpga_mac);
    w(REG_IP,      fpga_ip);
    w(REG_GATEWAY, fpga_ip & 0xFFFFFF00u);  // assume .x.1 isn't required for ARP'd peer
    w(REG_MASK,    0xFFFFFF00u);            // /24

    std::cout << std::dec << "[cfg ] socket[0] their="
              << ((host_ip>>24)&0xff) << "." << ((host_ip>>16)&0xff) << "."
              << ((host_ip>>8)&0xff) << "." << (host_ip&0xff)
              << ":" << host_port << " my=" << fpga_port << "\n";
    w(REG_THEIR_IP_BASE   + 0 * 4, host_ip);
    w(REG_THEIR_PORT_BASE + 0 * 4, host_port);
    w(REG_MY_PORT_BASE    + 0 * 4, fpga_port);
    w(REG_VALID_BASE      + 0 * 4, 1);

    // Bug B workaround: MiniRocket kernel doesn't set out_pkt.dest, so
    // TDEST on the kernel->NetLayer stream is undefined and can land on
    // ANY socket slot 0..15. NetLayer drops the response if
    // SocketTable[TDEST].valid == 0. Mirror slot[0] to slots 1..15 so any
    // TDEST routes back to the host.
    std::cout << std::dec
              << "[cfg ] Bug B workaround: mirroring slot[0] to slots 1..15\n";
    for (int slot = 1; slot < 16; ++slot) {
        w(REG_THEIR_IP_BASE   + slot * 4, host_ip);
        w(REG_THEIR_PORT_BASE + slot * 4, host_port);
        w(REG_MY_PORT_BASE    + slot * 4, fpga_port);
        w(REG_VALID_BASE      + slot * 4, 1);
    }
    w(REG_NUM_SOCKETS, 16);

    // ARP entry: index = host_ip & 0xFF (low octet hash)
    const uint32_t arp_idx = host_ip & 0xFFu;
    std::cout << "[cfg ] ARP[" << arp_idx << "] ip=" << ((host_ip>>24)&0xff)
              << "." << ((host_ip>>16)&0xff) << "." << ((host_ip>>8)&0xff)
              << "." << (host_ip&0xff) << " mac=0x" << std::hex
              << std::setw(12) << host_mac << std::dec << "\n";
    w(REG_ARP_IP_BASE  + arp_idx * 4, host_ip);
    w64(REG_ARP_MAC_BASE + arp_idx * 8, host_mac);
    w(REG_ARP_VALID_BASE + arp_idx * 4, 1);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << std::hex << std::setfill('0');
    std::cout << "[chk ] readback:\n";
    std::cout << "  IP@0x18         = 0x" << std::setw(8) << r(REG_IP) << "\n";
    std::cout << "  MASK@0x20       = 0x" << std::setw(8) << r(REG_MASK) << "\n";
    std::cout << "  num_sockets     = " << std::dec << r(REG_NUM_SOCKETS) << "\n";
    std::cout << "  slot0.theirIP   = 0x" << std::hex << std::setw(8) << r(REG_THEIR_IP_BASE) << "\n";
    std::cout << "  slot0.theirPort = " << std::dec << r(REG_THEIR_PORT_BASE) << "\n";
    std::cout << "  slot0.myPort    = " << std::dec << r(REG_MY_PORT_BASE) << "\n";
    std::cout << "  slot0.valid     = " << std::dec << r(REG_VALID_BASE) << "\n";
    std::cout << "  arp[" << arp_idx << "].valid  = " << r(REG_ARP_VALID_BASE + arp_idx*4) << "\n";
    std::cout << "  arp[" << arp_idx << "].ip     = 0x" << std::hex << std::setw(8) << r(REG_ARP_IP_BASE + arp_idx*4) << "\n";
    std::cout << std::dec
              << "  slot15.theirIP   = 0x" << std::hex << std::setw(8) << r(REG_THEIR_IP_BASE + 15*4) << "\n"
              << std::dec
              << "  slot15.theirPort = " << r(REG_THEIR_PORT_BASE + 15*4) << "\n"
              << "  slot15.myPort    = " << r(REG_MY_PORT_BASE + 15*4) << "\n"
              << "  slot15.valid     = " << r(REG_VALID_BASE + 15*4) << "\n";

    std::cout << "[done] netlayer bring-up complete\n";
    return 0;
}
