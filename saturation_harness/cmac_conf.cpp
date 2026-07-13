// cmac_conf.cpp — read/bring-up CMAC (cmac_0) using the CORRECT kernel.xml
// register offsets (cmac_status.cpp used wrong 0x10+8*i offsets and always
// read 0 — do NOT trust it). Offsets from
//   fpga-network/_x.*/link/int/xo/cmac_0/cmac_0/kernel.xml
//
// Build:
//   g++ -std=c++17 -O2 -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
//       -o cmac_conf cmac_conf.cpp -lxrt_coreutil -luuid
// Usage:
//   ./cmac_conf <xclbin> <bdf> read      # read config + status (correct offsets)
//   ./cmac_conf <xclbin> <bdf> bringup   # write gt_reset=0,reset=0,conf_tx=0x10,conf_rx=0x1 then read
#include <experimental/xrt_ip.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <cstdio>
#include <cstdint>
#include <string>

// absolute AXI-Lite offsets within cmac_0 (from kernel.xml)
enum {
  GT_RESET=0x00, RESET=0x04, MODE=0x08, CONF_TX=0x0C, CONF_RX=0x14,
  CORE_MODE=0x20, VERSION=0x24, GT_LOOPBACK=0x90,
  STAT_TX_STATUS=0x200, STAT_RX_STATUS=0x204, STAT_STATUS=0x208, STAT_RX_BLOCK_LOCK=0x20C,
  STAT_RX_LANE_SYNC=0x210, STAT_PM_TICK=0x2B0,
  STAT_TX_TOTAL_PKTS=0x500, STAT_RX_TOTAL_PKTS=0x608
};

int main(int argc, char** argv){
  if (argc < 3){ std::fprintf(stderr,"usage: %s <xclbin> <bdf> [read|bringup]\n", argv[0]); return 2; }
  std::string xclbin=argv[1], bdf=argv[2];
  std::string mode = (argc>3)? argv[3] : "read";
  auto dev = xrt::device(bdf);
  auto uuid = dev.load_xclbin(xclbin);
  auto cmac = xrt::ip(dev, uuid, "cmac_0:{cmac_0_1}");
  auto rd=[&](uint32_t o){ return cmac.read_register(o); };
  auto wr=[&](uint32_t o, uint32_t v){ cmac.write_register(o,v); };

  if (mode=="txon"){
    std::printf("[txon] conf_tx=0x1 (tx_enable, NO test pattern), conf_rx=0x1\n");
    wr(GT_RESET,0); wr(RESET,0); wr(CONF_TX,0x1); wr(CONF_RX,0x1);
  }
  else if (mode=="bringup"){
    std::printf("[bringup] gt_reset_reg=0, reset_reg=0, conf_tx=0x10, conf_rx=0x1\n");
    wr(GT_RESET,0); wr(RESET,0); wr(CONF_TX,0x10); wr(CONF_RX,0x1);
  }
  else if (mode=="gtreset"){
    // Force a GT / CMAC reset cycle to re-train the transceiver link.
    std::printf("[gtreset] toggling gt_reset then reset, re-enabling tx/rx...\n");
    wr(CONF_TX,0); wr(CONF_RX,0);
    wr(GT_RESET,1); for(volatile long i=0;i<200000000;i++){}   // assert ~
    wr(GT_RESET,0); for(volatile long i=0;i<200000000;i++){}   // deassert, let GT retrain
    wr(RESET,1);    for(volatile long i=0;i<100000000;i++){}
    wr(RESET,0);    for(volatile long i=0;i<200000000;i++){}
    wr(CONF_TX,0x1); wr(CONF_RX,0x1);
    for(volatile long i=0;i<400000000;i++){}                    // settle before status read
  }
  // Latch a fresh stat snapshot (stat_* regs are stale without this pm_tick write).
  wr(STAT_PM_TICK, 1);
  std::printf("cmac_0 (bdf=%s, mode=%s) CORRECT-offset registers (pm_tick latched):\n", bdf.c_str(), mode.c_str());
  std::printf("  version            @0x24 = 0x%08x\n", rd(VERSION));
  std::printf("  conf_tx            @0x0C = 0x%08x   conf_rx @0x14 = 0x%08x\n", rd(CONF_TX), rd(CONF_RX));
  std::printf("  gt_reset @0x00 = 0x%08x   reset @0x04 = 0x%08x\n", rd(GT_RESET), rd(RESET));
  std::printf("  stat_tx_status     @0x200= 0x%08x\n", rd(STAT_TX_STATUS));
  std::printf("  stat_rx_status     @0x204= 0x%08x  <- aligned if bit0=1\n", rd(STAT_RX_STATUS));
  std::printf("  stat_rx_block_lock @0x20C= 0x%08x  <- PCS locked if nonzero\n", rd(STAT_RX_BLOCK_LOCK));
  std::printf("  stat_rx_lane_sync  @0x210= 0x%08x\n", rd(STAT_RX_LANE_SYNC));
  std::printf("  *** stat_tx_total_packets @0x500 = %u\n", rd(STAT_TX_TOTAL_PKTS));
  std::printf("  *** stat_rx_total_packets @0x608 = %u   (PHY-level frame counts = ground truth)\n", rd(STAT_RX_TOTAL_PKTS));
  return 0;
}
