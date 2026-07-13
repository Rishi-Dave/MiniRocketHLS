// pktDropper interface — matches NetLayer's M_AXIS_nl2sk pkt format.
// ap_axiu<512, 1, 1, 16> must match what minirocket_inference's
// input_timeseries stream expects (see fpga-network/minirocket/include/minirocket.hpp).

#ifndef PKTDROPPER_HPP
#define PKTDROPPER_HPP

#include "hls_stream.h"
#include "ap_axi_sdata.h"

#define DWIDTH  512
#define TDWIDTH 16
typedef ap_axiu<DWIDTH, 1, 1, TDWIDTH> pkt;


extern "C" void pktDropper(
    hls::stream<pkt> &input,
    hls::stream<pkt> &output
);

#endif
