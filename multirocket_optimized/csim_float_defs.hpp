/*
 * csim_float_defs.hpp — stub header for csim compilation with plain g++
 * Replaces ap_int.h / ap_fixed.h / hls_stream.h with float equivalents.
 * This allows algorithm verification without Vitis HLS compiler.
 */
#pragma once
#include <cstdint>
#include <cstdio>
// Satisfy ap_int.h and ap_fixed.h guards
#define _AP_INT_H_  1
#define _AP_FIXED_H_ 1

typedef float      data_t;
typedef int32_t    int_t;
typedef uint8_t    idx_t;

namespace hls {
template<typename T>
struct stream {
    void write(T){}
    T read(){ return T(); }
    bool empty(){ return true; }
};
}

struct PoolingStats {
    data_t ppv;
    data_t mpv;
    data_t mipv;
    data_t lspv;
};

#define MAX_TIME_SERIES_LENGTH 8192
#define MAX_FEATURES 50000
#define NUM_KERNELS 84
#define KERNEL_SIZE 9
#define MAX_DILATIONS 32
#define MAX_CLASSES 16
#define NUM_POOLING_OPERATORS 4
#define NUM_REPRESENTATIONS 2
