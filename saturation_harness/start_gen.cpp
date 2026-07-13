// start_gen.cpp — start the udp_generator free-run kernel via xrt::ip (the
// PROVEN path; xrt::kernel/xrt::run segfault on this XRT). udp_generator is
// ap_ctrl_hs + BUILD=1 while(true): write rate_divider @0x10, pulse ap_start @0x00.
#include <experimental/xrt_ip.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
enum { AP_CTRL=0x00, RATE_DIVIDER=0x10 };
int main(int argc, char** argv){
  if(argc<3){ std::fprintf(stderr,"usage: %s <xclbin> <bdf> [rate_divider=0]\n",argv[0]); return 2; }
  std::string xclbin=argv[1], bdf=argv[2];
  uint32_t rate = argc>3 ? (uint32_t)strtoul(argv[3],0,0) : 0;
  std::printf("[1] device\n"); std::fflush(stdout);
  auto dev  = xrt::device(bdf);
  std::printf("[2] load_xclbin\n"); std::fflush(stdout);
  auto uuid = dev.load_xclbin(xclbin);
  std::printf("[3] ip open udp_generator\n"); std::fflush(stdout);
  auto ip = xrt::ip(dev, uuid, "udp_generator:{udp_generator_1}");
  std::printf("[4] write rate_divider@0x10 = %u\n", rate); std::fflush(stdout);
  ip.write_register(RATE_DIVIDER, rate);
  uint32_t ctrl0 = ip.read_register(AP_CTRL);
  std::printf("[5] ap_ctrl before = 0x%08x; pulsing ap_start\n", ctrl0); std::fflush(stdout);
  ip.write_register(AP_CTRL, 0x1);          // ap_start
  uint32_t ctrl1 = ip.read_register(AP_CTRL);
  uint32_t rb    = ip.read_register(RATE_DIVIDER);
  std::printf("[6] STARTED. ap_ctrl after=0x%08x rate_readback=%u\n", ctrl1, rb); std::fflush(stdout);
  return 0;
}
