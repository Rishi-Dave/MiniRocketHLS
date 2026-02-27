#include "../include/pass_through.hpp"
#include <cstring>

// HLS-optimized top-level function for FPGA
extern "C" void pass_through(
    hls::stream<pkt> &input_timeseries,      // Input time series
    hls::stream<pkt> &output_predictions,      // Output predictions
    ap_uint<8>  dest
) {

    #pragma HLS INTERFACE axis port = input_timeseries
    #pragma HLS INTERFACE axis port = output_predictions
    #pragma HLS INTERFACE s_axilite port = dest
    #pragma HLS INTERFACE s_axilite port = return
    
#if BUILD == 1
    while (true) {
#endif 
        if (!input_timeseries.empty()) {

            pkt v = input_timeseries.read();
            ap_uint< DWIDTH > tmp = v.data;

            pkt out_pkt;
            ap_uint< DWIDTH > out_data = 0xDEADBEEFDEADBEEF; // Dummy prediction data
            out_pkt.data = out_data;
            out_pkt.keep = -1;
            out_pkt.dest = dest; 
            out_pkt.last = 1; 
            output_predictions.write(out_pkt);

        }

#if BUILD == 1
    }
#endif 


}