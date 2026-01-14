#include "../include/store.hpp"
#include <cstring>

// HLS-optimized top-level function for FPGA
extern "C" void store(
    hls::stream<data_t>& prediction_input,
    data_t* prediction_output,
    int_t num_values,
    int_t num_classes
) {
    #pragma HLS INTERFACE m_axi port=prediction_output bundle=gmem0 depth=(MAX_TIME_SERIES_LENGTH*MAX_CLASSES)
    #pragma HLS INTERFACE axis port=prediction_input 
    #pragma HLS INTERFACE s_axilite port=prediction_output bundle=control
    #pragma HLS INTERFACE s_axilite port=num_values bundle=control
    #pragma HLS INTERFACE s_axilite port=num_classes bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    for (int i = 0; i < num_values; i++) {
        for (int j = 0; j < num_classes; j++) {
            prediction_output[i * num_classes + j] = prediction_input.read();
        }
    }            

}