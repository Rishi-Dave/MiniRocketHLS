#include "scaler.hpp"

// Top-level K2 kernel: Scaler with AXI-Stream I/O
// Uses precomputed inv_scale = 1/scale to avoid division on FPGA
extern "C" void scaler(
    data_t* scaler_mean,
    data_t* inv_scale,
    hls::stream<data_t>& features_in,
    hls::stream<data_t>& scaled_features_out,
    int_t num_features
) {
    #pragma HLS INTERFACE m_axi port=scaler_mean bundle=gmem4 depth=1024
    #pragma HLS INTERFACE m_axi port=inv_scale bundle=gmem5 depth=1024

    #pragma HLS INTERFACE axis port=features_in
    #pragma HLS INTERFACE axis port=scaled_features_out

    #pragma HLS INTERFACE s_axilite port=num_features bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS DATAFLOW

    // Local arrays for model parameters
    data_t local_mean[MAX_FEATURES];
    data_t local_inv_scale[MAX_FEATURES];

    #pragma HLS ARRAY_PARTITION variable=local_mean type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=local_inv_scale type=cyclic factor=16

    // Copy scaler parameters from HBM to local BRAM
    COPY_PARAMS: for (int_t i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
        local_mean[i] = scaler_mean[i];
        local_inv_scale[i] = inv_scale[i];
    }

    // Read features from stream, scale, and write to output stream
    // Multiply by inv_scale instead of dividing by scale (eliminates division)
    SCALE_LOOP: for (int_t i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=100 max=1024

        data_t feature = features_in.read();
        data_t scaled = (feature - local_mean[i]) * local_inv_scale[i];
        scaled_features_out.write(scaled);
    }
}
