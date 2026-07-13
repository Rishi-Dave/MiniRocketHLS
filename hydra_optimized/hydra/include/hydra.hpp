#ifndef HYDRA_INFERENCE_HLS_H
#define HYDRA_INFERENCE_HLS_H

#include "ap_fixed.h"
#include "ap_int.h"
#include <cstring>

// Data types for HLS synthesis
typedef ap_fixed<32,16> data_t;    // All weights, features, time series
typedef ap_fixed<48,32> acc_t;     // Running sum accumulator (overflow-safe for 5000+ steps)
typedef ap_int<32> int_t;

// HYDRA Algorithm Constants
#define MAX_TIME_SERIES_LENGTH 5120  // UCR datasets (InsectSound:600, MosquitoSound:3750, FruitFlies:5000)
#define NUM_KERNELS 512
#define NUM_GROUPS 8
#define KERNELS_PER_GROUP 64  // NUM_KERNELS / NUM_GROUPS
#define KERNEL_SIZE 9
#define MAX_FEATURES 2048
#define MAX_DILATIONS 8
#define MAX_CLASSES 10
#define POOLING_OPERATORS 2  // Max + Mean

// Optimization constants
#define UNROLL_FACTOR       16    // Kernels processed in parallel per batch
#define NUM_BATCHES         (NUM_KERNELS / UNROLL_FACTOR)  // 32 total batches
#define NUM_TS_COPIES       5     // ceil(9 reads / 2 BRAM ports) = 5 copies

// Function prototypes

extern "C" void hydra_inference(
    float* time_series_input,
    float* prediction_output,
    float* coefficients,
    float* intercept,
    float* scaler_mean,
    float* scaler_scale,
    float* kernel_weights,
    float* biases,
    int* dilations,
    int time_series_length,
    int num_features,
    int num_classes,
    int num_groups
);

/**
 * Batch-parallel feature extraction with UNROLL=16.
 * Host must sort kernels by dilation so each batch has uniform dilation.
 */
void hydra_feature_extraction_hls(
    data_t ts0[MAX_TIME_SERIES_LENGTH],
    data_t ts1[MAX_TIME_SERIES_LENGTH],
    data_t ts2[MAX_TIME_SERIES_LENGTH],
    data_t ts3[MAX_TIME_SERIES_LENGTH],
    data_t ts4[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    data_t local_weights[NUM_KERNELS][KERNEL_SIZE],
    data_t biases[NUM_KERNELS],
    int_t local_dilations[NUM_KERNELS],
    int_t time_series_length
);

void apply_scaler_hls(
    data_t features[MAX_FEATURES],
    data_t scaler_mean[MAX_FEATURES],
    data_t scaler_scale[MAX_FEATURES],
    int_t num_features
);

void linear_classifier_predict_hls(
    data_t features[MAX_FEATURES],
    data_t coefficients[MAX_FEATURES * MAX_CLASSES],
    data_t intercept[MAX_CLASSES],
    int_t num_features,
    int_t num_classes,
    data_t prediction[MAX_CLASSES]
);

#endif // HYDRA_INFERENCE_HLS_H
