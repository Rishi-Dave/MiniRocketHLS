#ifndef UDP_GENERATOR_HPP
#define UDP_GENERATOR_HPP

// UDP line-rate packet generator kernel — scoped/prepared 2026-07-07 for
// pc161's U280, to blast fixed-size UDP packets at the fused MiniRocket DUT
// on pc160 once an FPGA-to-FPGA CMAC cable link is in place (see
// PROMPT scoping report / memory network_bandwidth_2026_07_04.md).
//
// Structural sibling of fpga-network/minirocket_fused's AXIS wrapper: same
// `pkt` (ap_axiu<512,1,1,16>) type, same S_AXIS_sk2nl / M_AXIS_nl2sk stream
// shape, same `#if BUILD == 1 while(true)` on-chip free-run gate (the ONLY
// pattern proven on hardware to sustain line-rate UDP TX without a host-spin
// cap — see minirocket_fused.cpp v4 "DECISIVE WIN" history: single-shot
// caps ~35-40k pps via host ap_start spin; BUILD=1 free-run reached 700k+
// pps, sender-limited, FPGA ceiling still unfound).
//
// Unlike minirocket_fused, this kernel does NO compute: it is a pure
// generator. It free-runs writing one 64B (512-bit, single-beat) UDP
// payload packet per available cycle (or paced by `rate_divider`), and
// separately drains-and-discards anything NetLayer forwards back to it on
// M_AXIS_nl2sk (that port must be wired to *something* per NetLayers/
// kernel.xml, so we terminate it ourselves instead of pulling in
// pktDropper).

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"

#define GEN_DWIDTH  512   // AXIS beat width in bits == 64B == the fixed UDP payload size
#define GEN_TDWIDTH 16    // tdest width, matches NetLayers/kernel.xml pkt type

typedef ap_axiu<GEN_DWIDTH, 1, 1, GEN_TDWIDTH> pkt;

// float32 sample value carried in the low 4 bytes of each packet's `data`
// field — matches minirocket_fused_ingest's
// `data_t new_val = *((data_t*)&tmp)` decode (low-order reinterpret of the
// 512-bit beat), so a DUT expecting minirocket_fused-style single-sample-
// per-packet framing decodes this generator's packets the same way.
typedef float data_t;

#endif // UDP_GENERATOR_HPP
