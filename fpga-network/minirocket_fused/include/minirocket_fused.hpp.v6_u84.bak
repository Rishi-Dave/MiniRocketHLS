#ifndef MINIROCKET_FUSED_INFERENCE_HLS_H
#define MINIROCKET_FUSED_INFERENCE_HLS_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include <stdint.h>
#include "ap_axi_sdata.h"

// HLS-optimized data types
typedef float data_t;
typedef ap_int<32> int_t;           // 32-bit signed integer
typedef ap_uint<8> idx_t;           // 8-bit unsigned for small indices

// v5 (2026-07-11): classifier accumulator type. A float loop-carried add
// schedules at II=4 (multi-cycle FP add latency) — measured 33.6k of the
// ~55.8k cycles/window on InsectSound. A 48-bit fixed-point accumulator has
// a 1-cycle add => FEATURE_LOOP II=1. Products stay float32 (full multiply
// precision); only the accumulation quantizes: 24 frac bits = 6e-8
// resolution, ~5e-5 worst-case absolute error over 1024 terms — orders of
// magnitude below inter-class logit gaps. Same treatment as hydra
// v2_fixed/v6 ("1-cycle accumulation vs 7-cycle float").
typedef ap_fixed<48, 24> facc_t;

// Constants for MiniRocket (compile-time known)
// Fused CONV+PPV keeps only 5 duplicate raw time-series copies in BRAM
// (MAX_TIME_SERIES_LENGTH * 4B * 5 = 160KB @ 8192), NOT a per-dilation
// per-kernel convolution map (that is what forced minirocket_dpr down to
// MAX_TIME_SERIES_LENGTH=192). Full-recompute has no incremental algorithm,
// so we can afford the full 8192-sample buffer matching minirocket_modular's
// common.hpp, which covers InsectSound(600)/MosquitoSound(3750)/FruitFlies(5000).
#define MAX_TIME_SERIES_LENGTH 8192
#define MAX_FEATURES 1024
#define NUM_KERNELS 84
#define KERNEL_SIZE 9
#define MAX_DILATIONS 16
#define MAX_CLASSES 16

// Fused CONV+PPV parameters (from minirocket_modular feature_extraction_v16_fixed.cpp)
// v6 (2026-07-11): UNROLL 28 -> 84 — ALL kernels evaluated per cycle, so the
// series is swept once per dilation (9 sweeps/window) instead of 27. Conv is
// multiplier-free (±1/0/2 weights), so the cost is adder/LUT area, which was
// at a few % utilization. The 9-tap sliding window fans out to 84 trees;
// timing verified by the csynth + routed-WNS gates.
#define UNROLL_FACTOR 84
#define NUM_KERNEL_GROUPS (NUM_KERNELS / UNROLL_FACTOR)  // 84/84 = 1
#define MAX_FEATURES_PER_DILATION 3
#define NUM_TS_COPIES 5  // ceil(9 reads / 2 ports per BRAM) = 5

// Network packet definitions for AXI stream interface (same wire contract as
// minirocket_dpr: 1 UDP packet = 1 sample, float32 in low 4 bytes of the beat)
#define DWIDTH 512
#define TDWIDTH 16
typedef ap_axiu<DWIDTH, 1, 1, TDWIDTH> pkt;

// HLS-optimized top-level function for FPGA.
// Named distinctly from minirocket_dpr's `minirocket_inference` so the two
// kernels can coexist in the repo / be linked side by side without a name
// clash; the network link config (config.cfg in this directory) references
// this exact name. See README note in the top-level report about why this
// kernel uses buffer-then-classify semantics instead of minirocket_dpr's
// classify-every-packet DP-Reuse semantics.
extern "C" void minirocket_fused_inference(
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

#endif // MINIROCKET_FUSED_INFERENCE_HLS_H
