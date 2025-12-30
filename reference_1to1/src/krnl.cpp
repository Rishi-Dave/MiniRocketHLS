#include "krnl.hpp"
#include "minirocket_inference_hls.h"

// Top-level kernel wrapper for Vitis HLS synthesis
// This function will be synthesized as the hardware kernel
void krnl_top(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t predictions[MAX_CLASSES],
    int_t dilations[MAX_DILATIONS],
    int_t num_features_per_dilation[MAX_DILATIONS],
    data_t biases[MAX_FEATURES],
    data_t scaler_mean[MAX_FEATURES],
    data_t scaler_scale[MAX_FEATURES],
    data_t coefficients[MAX_CLASSES][MAX_FEATURES],
    data_t intercept[MAX_CLASSES],
    int_t time_series_length,
    int_t num_dilations,
    int_t num_features,
    int_t num_classes
) {
    #pragma HLS INTERFACE m_axi port=time_series offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=predictions offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=coefficients offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=intercept offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi port=scaler_mean offset=slave bundle=gmem4
    #pragma HLS INTERFACE m_axi port=scaler_scale offset=slave bundle=gmem5
    #pragma HLS INTERFACE m_axi port=dilations offset=slave bundle=gmem6
    #pragma HLS INTERFACE m_axi port=num_features_per_dilation offset=slave bundle=gmem7
    #pragma HLS INTERFACE m_axi port=biases offset=slave bundle=gmem8
    #pragma HLS INTERFACE s_axilite port=time_series_length bundle=control
    #pragma HLS INTERFACE s_axilite port=num_dilations bundle=control
    #pragma HLS INTERFACE s_axilite port=num_features bundle=control
    #pragma HLS INTERFACE s_axilite port=num_classes bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Call the MiniRocket inference pipeline
    // Note: Flatten coefficients from 2D to 1D pointer
    minirocket_inference_hls_top(
        time_series,
        predictions,
        (data_t*)coefficients,  // Flatten 2D array to 1D pointer
        intercept,
        scaler_mean,
        scaler_scale,
        dilations,
        num_features_per_dilation,
        biases,
        time_series_length,
        num_features,
        num_classes,
        num_dilations
    );
}
