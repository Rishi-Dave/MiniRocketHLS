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

// Constants for MultiRocket (compile-time known)
#define MAX_TIME_SERIES_LENGTH 128  // Updated for UCR datasets (matches HYDRA)
#define MAX_FEATURES 50000  // Increased for 4 pooling operators × 2 representations
#define NUM_KERNELS 84
#define KERNEL_SIZE 9
#define MAX_DILATIONS 8
#define MAX_CLASSES 8  // Increased to match HYDRA
#define NUM_POOLING_OPERATORS 4  // PPV, MPV, MIPV, LSPV
#define NUM_REPRESENTATIONS 2  // Original + First-order difference

// Fixed kernel indices (84 combinations of 3 indices from 0-8)
// extern const int_t kernel_indices[NUM_KERNELS][3];

#define BUILD 1

typedef union {
    data_t fp_num;
    uint32_t raw_bits;
    struct {
        uint32_t mant : 23;
        uint32_t bexp : 8;
        uint32_t sign : 1;
    };
} float_num_t;

// Structure to hold all four pooling operator results
struct PoolingStats {
    data_t ppv;   // Proportion of Positive Values
    data_t mpv;   // Mean of Positive Values
    data_t mipv;  // Mean of Indices of Positive Values
    data_t lspv;  // Longest Stretch of Positive Values
};

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
extern "C" void multirocket_inference(
    hls::stream<data_t> &time_series_input,      // Input time series
    hls::stream<data_t> &prediction_output,      // Output predictions
    data_t* coefficients,           // Model coefficients (flattened)
    data_t* intercept,              // Model intercept
    data_t* scaler_mean,            // Scaler mean values
    data_t* scaler_scale,           // Scaler scale values
    int_t* dilations_0,               // Dilation values
    int_t* num_features_per_dilation_0, // Features per dilation
    data_t* biases_0,                 // Bias values
    int_t num_dilations_0,
    int_t num_features_0,
    int_t* dilations_1,               // Dilation values
    int_t* num_features_per_dilation_1, // Features per dilation
    data_t* biases_1,                 // Bias values
    int_t num_dilations_1,
    int_t num_features_1,
    int_t time_series_length,
    int_t num_features,
    int_t num_classes,
    int_t n_feature_per_kernel
);//

// HLS-optimized MultiRocket feature extraction
void multirocket_feature_extraction_hls(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    int_t dilations[MAX_DILATIONS],
    int_t num_features_per_dilation[MAX_DILATIONS],
    data_t biases[MAX_FEATURES],
    int_t time_series_length,
    int_t num_dilations,
    int_t num_features,
    int_t n_feature_per_kernel,
    int_t starting_feature_idx
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

// Pooling operator functions
void compute_ppv(
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    data_t bias,
    int_t length,
    data_t* ppv_out
);

void compute_mpv(
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    data_t bias,
    int_t length,
    data_t* mpv_out
);

void compute_mipv(
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    data_t bias,
    int_t length,
    data_t* mipv_out
);

void compute_lspv(
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    data_t bias,
    int_t length,
    data_t* lspv_out
);

void compute_four_pooling_operators(
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    data_t bias,
    int_t start,
    int_t length,
    PoolingStats* stats
);

// First-order difference computation
void compute_first_order_difference(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t diff_series[MAX_TIME_SERIES_LENGTH],
    int_t length
);

#endif // MINIROCKET_INFERENCE_HLS_H
