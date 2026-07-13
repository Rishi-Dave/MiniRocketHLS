#include <iostream>
#include <cstring>
#include "../include/udp_generator.hpp"

// Single-shot csim harness for udp_generator.
//
// BUILD is intentionally left undefined for csim (same convention as
// minirocket_fused's test_minirocket_fused.cpp / make.tcl.in): the
// `#if BUILD == 1 while(true)` gate in udp_generator.cpp compiles away,
// so udp_generator_drain/udp_generator_emit each execute their body
// exactly ONCE per top-level call -- deterministic, no risk of csim
// spinning forever on an unbounded loop with a finite testbench stream.
//
// This checks:
//   1. One call with rate_divider=0 emits exactly one packet, with the
//      expected sideband (keep=all-ones, strb=all-ones, last=1, dest=0,
//      user=0, id=0) and a float32 sample (0.0f, the initial counter
//      value) in the low 4 bytes of `data`.
//   2. A second call increments the sample counter (data low 4 bytes ==
//      1.0f).
//   3. A garbage packet fed into input_drain is silently consumed (no
//      crash, no interaction with output_stream) -- exercises the drain
//      path that terminates NetLayer's M_AXIS_nl2sk port.
//
// This is NOT a rate-control (rate_divider>0) test or a free-run
// (BUILD=1 / COSIM_MAX_WINDOWS) liveness test -- run those via
// `vitis_hls -f make.tcl cosim` with COSIM_MAX_WINDOWS set (mirrors
// minirocket_fused's test_minirocket_fused_freerun_cosim.cpp pattern)
// before any hw build.

extern "C" void udp_generator(
    hls::stream<pkt> &input_drain,
    hls::stream<pkt> &output_stream,
    ap_uint<32> rate_divider
);

static int fails = 0;

#define CHECK(cond, msg) \
    do { if (!(cond)) { std::cerr << "[FAIL] " << msg << std::endl; fails++; } \
         else { std::cout << "[ OK ] " << msg << std::endl; } } while (0)

int main() {
    hls::stream<pkt> input_drain("input_drain");
    hls::stream<pkt> output_stream("output_stream");

    // --- Call 1: expect exactly one packet, sample == 0.0f ---
    udp_generator(input_drain, output_stream, /*rate_divider=*/0);

    CHECK(output_stream.size() == 1, "call 1 emits exactly one packet");
    if (output_stream.size() == 1) {
        pkt p = output_stream.read();
        ap_uint<32> low32 = p.data.range(31, 0);
        float sample;
        std::memcpy(&sample, &low32, sizeof(float));

        CHECK(sample == 0.0f, "call 1 sample == 0.0f");
        CHECK(p.keep == ap_uint<GEN_DWIDTH/8>(-1), "call 1 keep == all-ones (64B payload)");
        CHECK(p.strb == ap_uint<GEN_DWIDTH/8>(-1), "call 1 strb == all-ones");
        CHECK(p.last == 1, "call 1 last == 1 (single-beat packet)");
        CHECK(p.dest == 0, "call 1 dest == 0 (NetLayer socket slot 0)");
        CHECK(p.user == 0, "call 1 user == 0");
        CHECK(p.id == 0, "call 1 id == 0");
    }

    // --- Call 2: sample counter increments ---
    udp_generator(input_drain, output_stream, /*rate_divider=*/0);
    CHECK(output_stream.size() == 1, "call 2 emits exactly one packet");
    if (output_stream.size() == 1) {
        pkt p = output_stream.read();
        ap_uint<32> low32 = p.data.range(31, 0);
        float sample;
        std::memcpy(&sample, &low32, sizeof(float));
        CHECK(sample == 1.0f, "call 2 sample == 1.0f (counter incremented)");
    }

    // --- Drain path: feed garbage, make sure it's silently consumed and
    //     doesn't perturb the emit path ---
    pkt garbage;
    garbage.data = 0xDEADBEEF;
    garbage.keep = -1;
    garbage.strb = -1;
    garbage.last = 1;
    garbage.dest = 0;
    garbage.user = 0;
    garbage.id   = 0;
    input_drain.write(garbage);

    udp_generator(input_drain, output_stream, /*rate_divider=*/0);
    CHECK(input_drain.empty(), "drain path consumed the garbage packet");
    CHECK(output_stream.size() == 1, "call 3 (after drain) still emits exactly one packet");
    if (output_stream.size() == 1) {
        output_stream.read();
    }

    if (fails == 0) {
        std::cout << "csim: ALL CHECKS PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "csim: " << fails << " CHECK(S) FAILED" << std::endl;
        return 1;
    }
}
