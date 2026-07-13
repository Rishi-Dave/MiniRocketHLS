#ifndef MINIROCKET_INFERENCE_HLS_H
#define MINIROCKET_INFERENCE_HLS_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include <stdint.h>

// HLS-optimized data types
//typedef ap_fixed<32,16> data_t;     // 32-bit fixed point: 16 integer, 16 fractional bits
typedef float data_t;
typedef ap_int<32> int_t;           // 32-bit signed integer
typedef ap_uint<8> idx_t;           // 8-bit unsigned for small indices

// Constants for MiniRocket (compile-time known)
#define MAX_TIME_SERIES_LENGTH 128  // Optimized for UCR dataset compatibility
#define MAX_FEATURES 1024
#define NUM_KERNELS 84
#define KERNEL_SIZE 9
#define MAX_DILATIONS 8
#define MAX_CLASSES 8

#define BUILD 1  // Streaming mode: while(true) loop with event-driven processing

typedef union {
    data_t fp_num;
    uint32_t raw_bits;
    struct {
        uint32_t mant : 23;
        uint32_t bexp : 8;
        uint32_t sign : 1;
    };
} float_num_t;

// HLS-optimized structure for model parameters (arrays instead of vectors)
struct MiniRocketModelParams_HLS {
    data_t coefficients[MAX_CLASSES][MAX_FEATURES];
    data_t intercept[MAX_CLASSES];
    data_t scaler_mean[MAX_FEATURES];
    data_t scaler_scale[MAX_FEATURES];
    
    int_t dilations[MAX_DILATIONS];
    int_t num_features_per_dilation[MAX_DILATIONS];
    data_t biases[MAX_FEATURES];
    
    int_t num_dilations;
    int_t num_features;
    int_t num_classes;
    int_t time_series_length;
};

// HLS-optimized top-level function for FPGA
extern "C" void minirocket_inference(
    hls::stream<data_t> &input_timeseries,      // Input time series
    hls::stream<data_t> &output_predictions,      // Output predictions
    data_t* coefficients,           // Model coefficients (flattened)
    data_t* intercept,              // Model intercept
    data_t* scaler_mean,            // Scaler mean values
    data_t* scaler_scale,           // Scaler scale values
    int_t* dilations,               // Dilation values
    int_t* num_features_per_dilation, // Features per dilation
    data_t* biases,                 // Bias values
    int_t time_series_length,
    int_t num_features,
    int_t num_classes,
    int_t num_dilations
);

// HLS-optimized MiniRocket feature extraction
void minirocket_feature_extraction_hls(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    int_t dilations[MAX_DILATIONS],
    int_t num_features_per_dilation[MAX_DILATIONS],
    data_t biases[MAX_FEATURES],
    int_t time_series_length,
    int_t num_dilations,
    int_t num_features
);

// HLS-optimized scaling function
void apply_scaler_hls(
    data_t features[MAX_FEATURES],
    data_t scaled_features[MAX_FEATURES],
    data_t scaler_mean[MAX_FEATURES],
    data_t scaler_scale[MAX_FEATURES],
    int_t num_features
);

// HLS-optimized linear classifier
void linear_classifier_predict_hls(
    data_t scaled_features[MAX_FEATURES],
    data_t predictions[MAX_CLASSES],
    data_t coefficients[MAX_CLASSES][MAX_FEATURES],
    data_t intercept[MAX_CLASSES],
    int_t num_features,
    int_t num_classes
);

#endif // MINIROCKET_INFERENCE_HLS_H
