#include "../include/load.hpp"
#include <cstring>



// HLS-optimized top-level function for FPGA
extern "C" void load(
    data_t* time_series_input,      // Input time series
    hls::stream<data_t>& output,
    int_t time_series_length
) {
    #pragma HLS INTERFACE m_axi port=time_series_input bundle=gmem0 depth=512
    #pragma HLS INTERFACE axis port=output 
    #pragma HLS INTERFACE s_axilite port=time_series_input bundle=control
    #pragma HLS INTERFACE s_axilite port=time_series_length bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    for (int i = 0; i < time_series_length; i++) {
        #pragma HLS PIPELINE
        output.write(time_series_input[i]);
    }
}