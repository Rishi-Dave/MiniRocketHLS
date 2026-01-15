#include "../include/load.hpp"

// Load kernel: reads time series from HBM and streams it to minirocket_inference
extern "C" void load_kernel(
    data_t* time_series_input,      // Input time series from HBM
    hls::stream<data_t>& output,    // Output stream to minirocket_inference
    int_t num_values                // Number of values to stream
) {
    #pragma HLS INTERFACE m_axi port=time_series_input bundle=gmem0 depth=8192 max_read_burst_length=256
    #pragma HLS INTERFACE axis port=output
    #pragma HLS INTERFACE s_axilite port=num_values bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Stream time series values from HBM to output
    LOAD_LOOP: for (int_t i = 0; i < num_values; i++) {
        #pragma HLS PIPELINE II=1
        data_t value = time_series_input[i];
        output.write(value);
    }
}
