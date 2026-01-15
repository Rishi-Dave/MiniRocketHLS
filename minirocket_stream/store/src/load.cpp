#include "../include/load.hpp"

// Store kernel: reads predictions from stream and writes them to HBM
extern "C" void store_kernel(
    hls::stream<data_t>& input,      // Input stream from minirocket_inference
    data_t* predictions_output,      // Output predictions to HBM
    int_t num_predictions            // Number of predictions to store
) {
    #pragma HLS INTERFACE axis port=input
    #pragma HLS INTERFACE m_axi port=predictions_output bundle=gmem1 depth=1024 max_write_burst_length=256
    #pragma HLS INTERFACE s_axilite port=num_predictions bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Store predictions from stream to memory
    STORE_LOOP: for (int_t i = 0; i < num_predictions; i++) {
        #pragma HLS PIPELINE II=1
        data_t prediction = input.read();
        predictions_output[i] = prediction;
    }
}
