#ifndef MINIROCKET_HLS_HPP
#define MINIROCKET_HLS_HPP

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"

// HLS-optimized data types
typedef ap_fixed<32,16> data_t;     // 32-bit fixed point: 16 integer, 16 fractional bits
typedef ap_int<32> int_t;           // 32-bit signed integer
typedef ap_uint<8> idx_t;           // 8-bit unsigned for small indices

// Constants for MiniRocket (compile-time known)
#define MAX_TIME_SERIES_LENGTH 512
#define MAX_FEATURES 10000
#define NUM_KERNELS 84
#define KERNEL_SIZE 9
#define MAX_DILATIONS 8
#define MAX_CLASSES 4

// Fixed kernel indices (84 combinations of 3 indices from 0-8)
extern const int_t kernel_indices[NUM_KERNELS][3];

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
extern "C" void krnl_top(
    data_t* time_series_input,      // Input time series
    data_t* prediction_output,      // Output predictions
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

#endif // MINIROCKET_HLS_HPP