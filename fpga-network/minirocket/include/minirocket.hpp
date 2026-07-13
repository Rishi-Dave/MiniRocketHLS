#ifndef MINIROCKET_INFERENCE_HLS_H
#define MINIROCKET_INFERENCE_HLS_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include <stdint.h>
#include "ap_axi_sdata.h"

// HLS-optimized data types
typedef float data_t;
typedef ap_int<32> int_t;           // 32-bit signed integer
typedef ap_uint<8> idx_t;           // 8-bit unsigned for small indices

// Constants for MiniRocket (compile-time known)
#define MAX_TIME_SERIES_LENGTH 8192
#define MAX_FEATURES 10000
#define NUM_KERNELS 84
#define KERNEL_SIZE 9
#define MAX_DILATIONS 8
#define MAX_CLASSES 4

// Fused CONV+PPV parameters
#define UNROLL_FACTOR 28
#define NUM_KERNEL_GROUPS (NUM_KERNELS / UNROLL_FACTOR)  // 84/28 = 3
#define MAX_FEATURES_PER_DILATION 3
#define NUM_TS_COPIES 5  // ceil(9 reads / 2 ports per BRAM) = 5

// Network packet definitions for AXI stream interface
#define DWIDTH 512
#define TDWIDTH 16
typedef ap_axiu<DWIDTH, 1, 1, TDWIDTH> pkt;

typedef union {
    data_t fp_num;
    uint32_t raw_bits;
    struct {
        uint32_t mant : 23;
        uint32_t bexp : 8;
        uint32_t sign : 1;
    };
} float_num_t;

// HLS-optimized top-level function for FPGA
extern "C" void minirocket_inference(
    hls::stream<pkt> &input_timeseries,
    hls::stream<pkt> &output_predictions,
    data_t* coefficients,
    data_t* intercept,
    data_t* scaler_mean,
    data_t* scaler_scale,
    int_t* dilations,
    int_t* num_features_per_dilation,
    data_t* biases,
    int_t time_series_length,
    int_t num_features,
    int_t num_classes,
    int_t num_dilations
);

#endif // MINIROCKET_INFERENCE_HLS_H
