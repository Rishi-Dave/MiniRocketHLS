#ifndef MINIROCKET_INFERENCE_HLS_H
#define MINIROCKET_INFERENCE_HLS_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include <stdint.h>
#include "ap_axi_sdata.h"

#define BUILD 1

// Network packet definitions for AXI stream interface
#define DWIDTH 512
#define TDWIDTH 16
typedef ap_axiu<DWIDTH, 1, 1, TDWIDTH> pkt;




extern "C" void pass_through(
    hls::stream<pkt> &input_timeseries,      // Input time series
    hls::stream<pkt> &output_predictions,      // Output predictions
    ap_uint<8> dest
);

#endif // MINIROCKET_INFERENCE_HLS_H
