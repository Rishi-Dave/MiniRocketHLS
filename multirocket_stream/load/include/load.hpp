#ifndef LOAD_H
#define LOAD_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"

// HLS-optimized data types
//typedef ap_fixed<32,16> data_t;     // 32-bit fixed point: 16 integer, 16 fractional bits
typedef float data_t;
typedef ap_int<32> int_t;           // 32-bit signed integer
typedef ap_uint<8> idx_t;           // 8-bit unsigned for small indices

// Constants for MiniRocket (compile-time known)
#define MAX_TIME_SERIES_LENGTH 512
#define MAX_FEATURES 1024
#define NUM_KERNELS 84
#define KERNEL_SIZE 9
#define MAX_DILATIONS 16
#define MAX_CLASSES 16

extern "C" void load(
    data_t* time_series_input,      // Input time series
    hls::stream<data_t>& output,
    int_t num_values
);


#endif // LOAD_H