/*
 * Feature Extraction v16 Fixed - Fused CONV+PPV for single-pass operation
 *
 * Key optimizations:
 * 1. Invert loop nest: CONV_LOOP outer, KERNEL_LOOP inner (share sliding window reads)
 * 2. Duplicate time_series into 5 copies (each 2-port BRAM) for 10 read ports
 * 3. Dynamic CONV_LOOP bound: use actual time_series_length, not MAX
 * 4. Fused CONV+PPV: accumulate PPV counts inline during convolution, eliminating
 *    the large convolutions[16][8192] intermediate array and the separate PPV phase.
 *    Single pass through time series does both convolution and PPV counting.
 */

#include "../../include/common.hpp"
#include "hls_stream.h"
#include <cstring>

static data_t weights[NUM_KERNELS][KERNEL_SIZE] = {
    #include "../../include/weights.txt"
};

#define UNROLL_FACTOR 28
#define NUM_KERNEL_GROUPS (NUM_KERNELS / UNROLL_FACTOR)  // 84/28 = 3 groups, no remainder
#define REMAINDER_KERNELS (NUM_KERNELS % UNROLL_FACTOR)  // 84%28 = 0
#define NUM_TS_COPIES 5  // ceil(9 reads / 2 ports per BRAM) = 5 copies
#define MAX_FEATURES_PER_DILATION 3  // MiniRocket uses 1-3 features per dilation

void minirocket_feature_extraction_hls(
    data_t ts0[MAX_TIME_SERIES_LENGTH],
    data_t ts1[MAX_TIME_SERIES_LENGTH],
    data_t ts2[MAX_TIME_SERIES_LENGTH],
    data_t ts3[MAX_TIME_SERIES_LENGTH],
    data_t ts4[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    int_t dilations[MAX_DILATIONS],
    int_t num_features_per_dilation[MAX_DILATIONS],
    data_t biases[MAX_FEATURES],
    int_t time_series_length,
    int_t num_dilations,
    int_t num_features
) {
    #pragma HLS INLINE off

    #pragma HLS BIND_STORAGE variable=weights type=rom_1p impl=bram
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=28 dim=1

    int_t feature_idx = 0;

    DILATION_LOOP: for (int_t dil_idx = 0; dil_idx < num_dilations; dil_idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8

        int_t dilation = dilations[dil_idx];
        int_t features_this_dilation = num_features_per_dilation[dil_idx];
        int_t _padding0 = dil_idx % 2;
        int_t padding = ((9 - 1) * dilation) / 2;

        // Process kernels in groups of UNROLL_FACTOR
        KERNEL_GROUP_LOOP: for (int_t kg = 0; kg < NUM_KERNELS; kg += UNROLL_FACTOR) {
            #pragma HLS LOOP_TRIPCOUNT min=5 max=6

            // PPV counters and biases for fused operation (registers, not BRAM)
            // Replaces convolutions[16][8192] array — massive BRAM savings
            data_t group_biases[UNROLL_FACTOR][MAX_FEATURES_PER_DILATION];
            #pragma HLS ARRAY_PARTITION variable=group_biases complete dim=0
            int_t ppv_counts[UNROLL_FACTOR][MAX_FEATURES_PER_DILATION];
            #pragma HLS ARRAY_PARTITION variable=ppv_counts complete dim=0
            int_t padding_flag[UNROLL_FACTOR];
            #pragma HLS ARRAY_PARTITION variable=padding_flag complete

            // Pre-load biases and initialize counters (48 cycles, negligible)
            LOAD_BIASES: for (int_t bi = 0; bi < UNROLL_FACTOR * MAX_FEATURES_PER_DILATION; bi++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=48 max=48
                int_t ki = bi / MAX_FEATURES_PER_DILATION;
                int_t f = bi % MAX_FEATURES_PER_DILATION;
                int_t kernel_idx = kg + ki;
                int_t base = feature_idx + kernel_idx * features_this_dilation;
                group_biases[ki][f] = (f < features_this_dilation && kernel_idx < NUM_KERNELS && base + f < num_features)
                                      ? biases[base + f] : (data_t)0.0;
                ppv_counts[ki][f] = 0;
                if (f == 0) {
                    padding_flag[ki] = (_padding0 + kernel_idx) % 2;
                }
            }

            // Fused Convolution + PPV: single pass through time series
            // Computes dot products AND accumulates PPV counts simultaneously
            CONV_LOOP: for (int_t j = 0; j < time_series_length; j++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=150 max=8192

                // Read sliding window from 5 separate BRAMs (2 ports each = 10 ports)
                data_t sw[KERNEL_SIZE];
                #pragma HLS ARRAY_PARTITION variable=sw complete

                int_t idx0 = j + (-4) * dilation;
                int_t idx1 = j + (-3) * dilation;
                int_t idx2 = j + (-2) * dilation;
                int_t idx3 = j + (-1) * dilation;
                int_t idx4 = j;
                int_t idx5 = j + (1) * dilation;
                int_t idx6 = j + (2) * dilation;
                int_t idx7 = j + (3) * dilation;
                int_t idx8 = j + (4) * dilation;

                sw[0] = (idx0 < 0 || idx0 >= time_series_length) ? (data_t)0.0 : ts0[idx0];
                sw[1] = (idx1 < 0 || idx1 >= time_series_length) ? (data_t)0.0 : ts0[idx1];
                sw[2] = (idx2 < 0 || idx2 >= time_series_length) ? (data_t)0.0 : ts1[idx2];
                sw[3] = (idx3 < 0 || idx3 >= time_series_length) ? (data_t)0.0 : ts1[idx3];
                sw[4] = (idx4 < 0 || idx4 >= time_series_length) ? (data_t)0.0 : ts2[idx4];
                sw[5] = (idx5 < 0 || idx5 >= time_series_length) ? (data_t)0.0 : ts2[idx5];
                sw[6] = (idx6 < 0 || idx6 >= time_series_length) ? (data_t)0.0 : ts3[idx6];
                sw[7] = (idx7 < 0 || idx7 >= time_series_length) ? (data_t)0.0 : ts3[idx7];
                sw[8] = (idx8 < 0 || idx8 >= time_series_length) ? (data_t)0.0 : ts4[idx8];

                // Pre-compute padding range check (shared across all kernels)
                bool in_padded_range = (j >= padding && j < time_series_length - padding);

                // Compute 16 dot products + accumulate PPV counts in parallel
                PARALLEL_KERNELS: for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
                    #pragma HLS UNROLL
                    data_t value = 0.0;
                    DOT_PRODUCT: for (int_t k = 0; k < KERNEL_SIZE; k++) {
                        #pragma HLS UNROLL
                        value += sw[k] * weights[kg + ki][k];
                    }

                    // Inline PPV: compare conv output against biases
                    bool use_position = (padding_flag[ki] == 0) || in_padded_range;
                    PPV_FEATURES: for (int_t f = 0; f < MAX_FEATURES_PER_DILATION; f++) {
                        #pragma HLS UNROLL
                        if (f < features_this_dilation && use_position && value > group_biases[ki][f]) {
                            ppv_counts[ki][f]++;
                        }
                    }
                }
            }

            // Precompute reciprocals for PPV division (one per padding type)
            int_t padded_len = time_series_length - (int_t)(2 * padding);
            data_t inv_denom_full = (data_t)1.0 / (data_t)time_series_length;
            data_t inv_denom_padded = (padded_len > 0) ? (data_t)1.0 / (data_t)padded_len : (data_t)0.0;

            // Write PPV results to features array
            WRITE_PPV: for (int_t wi = 0; wi < UNROLL_FACTOR * MAX_FEATURES_PER_DILATION; wi++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=28 max=84
                int_t ki = wi / MAX_FEATURES_PER_DILATION;
                int_t f = wi % MAX_FEATURES_PER_DILATION;
                int_t kernel_idx = kg + ki;
                if (kernel_idx < NUM_KERNELS && f < features_this_dilation) {
                    int_t base = feature_idx + kernel_idx * features_this_dilation;
                    if (base + f < num_features) {
                        data_t inv_denom = (padding_flag[ki] == 0) ? inv_denom_full : inv_denom_padded;
                        features[base + f] = (data_t)ppv_counts[ki][f] * inv_denom;
                    }
                }
            }
        }
        feature_idx += features_this_dilation * NUM_KERNELS;
    }
}

// Top-level K1 kernel: Feature Extraction with AXI-Stream output
extern "C" void feature_extraction(
    data_t* time_series_input,
    int_t*  dilations,
    int_t*  num_features_per_dilation,
    data_t* biases,
    hls::stream<data_t>& features_out,
    int_t time_series_length,
    int_t num_features,
    int_t num_dilations
) {
    #pragma HLS INTERFACE m_axi port=time_series_input bundle=gmem0 depth=8192
    #pragma HLS INTERFACE m_axi port=dilations bundle=gmem1 depth=16
    #pragma HLS INTERFACE m_axi port=num_features_per_dilation bundle=gmem2 depth=16
    #pragma HLS INTERFACE m_axi port=biases bundle=gmem3 depth=1024

    #pragma HLS INTERFACE axis port=features_out

    #pragma HLS INTERFACE s_axilite port=time_series_length bundle=control
    #pragma HLS INTERFACE s_axilite port=num_features bundle=control
    #pragma HLS INTERFACE s_axilite port=num_dilations bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS DATAFLOW

    // 5 duplicate copies of time series for conflict-free 9-port read access
    data_t local_ts0[MAX_TIME_SERIES_LENGTH];
    data_t local_ts1[MAX_TIME_SERIES_LENGTH];
    data_t local_ts2[MAX_TIME_SERIES_LENGTH];
    data_t local_ts3[MAX_TIME_SERIES_LENGTH];
    data_t local_ts4[MAX_TIME_SERIES_LENGTH];

    data_t local_features[MAX_FEATURES];
    data_t local_biases[MAX_FEATURES];
    int_t local_dilations[MAX_DILATIONS];
    int_t local_num_features_per_dilation[MAX_DILATIONS];

    // Copy inputs from HBM to 5 local BRAM copies
    COPY_INPUT: for (int_t i = 0; i < time_series_length; i++) {
        #pragma HLS PIPELINE II=1
        data_t val = time_series_input[i];
        local_ts0[i] = val;
        local_ts1[i] = val;
        local_ts2[i] = val;
        local_ts3[i] = val;
        local_ts4[i] = val;
    }

    COPY_DILATIONS: for (int_t i = 0; i < num_dilations; i++) {
        #pragma HLS PIPELINE II=1
        local_dilations[i] = dilations[i];
        local_num_features_per_dilation[i] = num_features_per_dilation[i];
    }

    COPY_BIASES: for (int_t i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
        local_biases[i] = biases[i];
    }

    // Run feature extraction with 5 time series copies
    minirocket_feature_extraction_hls(
        local_ts0,
        local_ts1,
        local_ts2,
        local_ts3,
        local_ts4,
        local_features,
        local_dilations,
        local_num_features_per_dilation,
        local_biases,
        time_series_length,
        num_dilations,
        num_features
    );

    // Write features to AXI-Stream for downstream scaler kernel
    STREAM_OUT: for (int_t i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
        features_out.write(local_features[i]);
    }
}
