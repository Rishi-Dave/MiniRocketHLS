#include "../include/store.hpp"
#include <cstring>

// HLS-optimized top-level function for FPGA
extern "C" void store(
    hls::stream<data_t>& input,
    data_t* prediction_output,      // Input time series
    int_t time_series_length,
    int_t num_classes
) {
    #pragma HLS INTERFACE m_axi port=prediction_output bundle=gmem0 depth=512
    #pragma HLS INTERFACE axis port=input 
    #pragma HLS INTERFACE s_axilite port=prediction_output bundle=control
    #pragma HLS INTERFACE s_axilite port=time_series_length bundle=control
    #pragma HLS INTERFACE s_axilite port=num_classes bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    for (int i = 0; i < time_series_length * num_classes; i++) {
        #pragma HLS PIPELINE
        prediction_output[i] = input.read();
    }
}