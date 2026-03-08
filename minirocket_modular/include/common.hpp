#ifndef MINIROCKET_MODULAR_COMMON_HPP
#define MINIROCKET_MODULAR_COMMON_HPP

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"

// Data types
#ifdef USE_AP_FIXED
typedef ap_fixed<32,16> data_t;
#else
typedef float data_t;
#endif
typedef ap_int<32> int_t;
typedef ap_uint<8> idx_t;

// Constants (compile-time known, matching optimized_version)
#define MAX_TIME_SERIES_LENGTH 8192
#define MAX_FEATURES 1024
#define NUM_KERNELS 84
#define KERNEL_SIZE 9
#define MAX_DILATIONS 16
#define MAX_CLASSES 16

#endif // MINIROCKET_MODULAR_COMMON_HPP
