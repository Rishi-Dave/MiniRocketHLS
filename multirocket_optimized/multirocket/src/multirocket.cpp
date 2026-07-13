/*
 * MultiRocket Optimized v3 - Fused CONV + 4 Pooling Ops
 *
 * Key optimizations vs Round 2:
 * 1. ap_fixed<32,16> instead of float (eliminates FP latency)
 * 2. 5 duplicate TS copies for conflict-free 9-port BRAM read (same as v16_fixed)
 * 3. Fused conv+4pooling: accumulate PPV/MPV/MIPV/LSPV inline, eliminates
 *    convolutions[MAX_TIME_SERIES_LENGTH] intermediate array
 * 4. UNROLL_FACTOR=16 on PARALLEL_KERNELS inner loop inside CONV_LOOP
 * 5. Dynamic loop bound (time_series_length not MAX) — preserves Round 2 accuracy
 *
 * Round 2 semantics preserved:
 * - start=0 (no half-dilation padding) matching Python csim_round2_check.py
 * - base = current_feature_idx * n_feature_per_kernel + starting_feature_idx
 * - feature_idx increments by features_this_dilation per kernel (not per dilation)
 * - mipv index: (i - start) = i since start=0
 */

#include "../include/multirocket.hpp"
#include <cstring>

// Weights from weights.txt - 84 kernels × 9 weights
static data_t weights[NUM_KERNELS][KERNEL_SIZE] = {
    #include "../include/weights.txt"
};

// Tuning: UNROLL_FACTOR controls parallelism. 16 is safe for synthesis
// (84 kernels * 8192 max length * 16 = ~11M ops — well within limits since
// the unroll only covers the inner kernel tile, not the time loop).
// Each group of 16 kernels runs in parallel during each CONV_LOOP time step.
#define UNROLL_FACTOR 16
// Number of full groups: floor(84/16) = 5, remainder = 4
#define NUM_FULL_GROUPS (NUM_KERNELS / UNROLL_FACTOR)  // 5
#define REMAINDER_KERNELS (NUM_KERNELS % UNROLL_FACTOR) // 4
// Max features_this_dilation observed across all model configs
#define MAX_FEATURES_PER_DILATION 4

/*
 * Fused feature extraction for one representation (original or diff).
 * Takes 5 copies of time_series for conflict-free 9-read access per cycle.
 * Kernel group structure: process UNROLL_FACTOR kernels simultaneously
 * over the CONV_LOOP. Accumulates 4 pooling stats per kernel inline.
 *
 * CONV_LOOP is pipelined II=1. PARALLEL_KERNELS is UNROLL'd.
 * No convolutions[] intermediate array needed.
 */
static void multirocket_feature_extraction_opt(
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
    int_t num_features,
    int_t n_feature_per_kernel,
    int_t starting_feature_idx
) {
    #pragma HLS INLINE off

    #pragma HLS BIND_STORAGE variable=weights type=rom_1p impl=bram
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=16 dim=1

    int_t feature_idx = 0;

    DILATION_LOOP: for (int_t dil_idx = 0; dil_idx < num_dilations; dil_idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8

        int_t dilation = dilations[dil_idx];
        int_t features_this_dilation = num_features_per_dilation[dil_idx];

        // Process kernels in groups of UNROLL_FACTOR (handles 5 full groups of 16 + 1 partial of 4)
        KERNEL_GROUP_LOOP: for (int_t kg = 0; kg < NUM_KERNELS; kg += UNROLL_FACTOR) {
            #pragma HLS LOOP_TRIPCOUNT min=5 max=6

            // Per-kernel accumulators for 4 pooling ops × features_this_dilation biases
            // These live entirely in registers due to complete partition
            data_t group_biases[UNROLL_FACTOR][MAX_FEATURES_PER_DILATION];
            #pragma HLS ARRAY_PARTITION variable=group_biases complete dim=0

            // PPV: proportion of positive values
            int_t ppv_counts[UNROLL_FACTOR][MAX_FEATURES_PER_DILATION];
            #pragma HLS ARRAY_PARTITION variable=ppv_counts complete dim=0

            // MPV: mean of positive values (need sum and count)
            data_t mpv_sum[UNROLL_FACTOR][MAX_FEATURES_PER_DILATION];
            #pragma HLS ARRAY_PARTITION variable=mpv_sum complete dim=0
            int_t mpv_count[UNROLL_FACTOR][MAX_FEATURES_PER_DILATION];
            #pragma HLS ARRAY_PARTITION variable=mpv_count complete dim=0

            // MIPV: mean of indices of positive values (need index_sum and count)
            int_t mipv_isum[UNROLL_FACTOR][MAX_FEATURES_PER_DILATION];
            #pragma HLS ARRAY_PARTITION variable=mipv_isum complete dim=0

            // LSPV: longest stretch of positive values
            int_t lspv_cur[UNROLL_FACTOR][MAX_FEATURES_PER_DILATION];
            #pragma HLS ARRAY_PARTITION variable=lspv_cur complete dim=0
            int_t lspv_max[UNROLL_FACTOR][MAX_FEATURES_PER_DILATION];
            #pragma HLS ARRAY_PARTITION variable=lspv_max complete dim=0

            // Pre-load biases and zero-initialize accumulators
            INIT_LOOP: for (int_t init_i = 0; init_i < UNROLL_FACTOR * MAX_FEATURES_PER_DILATION; init_i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=64 max=64
                int_t ki = init_i / MAX_FEATURES_PER_DILATION;
                int_t f  = init_i % MAX_FEATURES_PER_DILATION;
                int_t kernel_idx = kg + ki;
                // current_feature_idx for this kernel in this dilation step
                int_t cfi = feature_idx + kernel_idx * features_this_dilation;
                group_biases[ki][f] = (f < features_this_dilation && kernel_idx < NUM_KERNELS && cfi + f < num_features)
                                      ? biases[cfi + f] : (data_t)0.0;
                ppv_counts[ki][f] = 0;
                mpv_sum[ki][f]    = (data_t)0.0;
                mpv_count[ki][f]  = 0;
                mipv_isum[ki][f]  = 0;
                lspv_cur[ki][f]   = 0;
                lspv_max[ki][f]   = 0;
            }

            // Fused Convolution + 4 Pooling Ops: single pass through time series
            // CONV_LOOP pipelined II=1; PARALLEL_KERNELS fully unrolled
            CONV_LOOP: for (int_t j = 0; j < time_series_length; j++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=150 max=8192

                // Read 9 sliding-window positions from 5 separate BRAMs
                // (2 read ports each = 10 ports, enough for 9 reads conflict-free)
                data_t sw[KERNEL_SIZE];
                #pragma HLS ARRAY_PARTITION variable=sw complete

                int_t idx0 = j + (-4) * dilation;
                int_t idx1 = j + (-3) * dilation;
                int_t idx2 = j + (-2) * dilation;
                int_t idx3 = j + (-1) * dilation;
                int_t idx4 = j;
                int_t idx5 = j + ( 1) * dilation;
                int_t idx6 = j + ( 2) * dilation;
                int_t idx7 = j + ( 3) * dilation;
                int_t idx8 = j + ( 4) * dilation;

                sw[0] = (idx0 < 0 || idx0 >= time_series_length) ? (data_t)0.0 : ts0[idx0];
                sw[1] = (idx1 < 0 || idx1 >= time_series_length) ? (data_t)0.0 : ts0[idx1];
                sw[2] = (idx2 < 0 || idx2 >= time_series_length) ? (data_t)0.0 : ts1[idx2];
                sw[3] = (idx3 < 0 || idx3 >= time_series_length) ? (data_t)0.0 : ts1[idx3];
                sw[4] = (idx4 < 0 || idx4 >= time_series_length) ? (data_t)0.0 : ts2[idx4];
                sw[5] = (idx5 < 0 || idx5 >= time_series_length) ? (data_t)0.0 : ts2[idx5];
                sw[6] = (idx6 < 0 || idx6 >= time_series_length) ? (data_t)0.0 : ts3[idx6];
                sw[7] = (idx7 < 0 || idx7 >= time_series_length) ? (data_t)0.0 : ts3[idx7];
                sw[8] = (idx8 < 0 || idx8 >= time_series_length) ? (data_t)0.0 : ts4[idx8];

                // Compute up to UNROLL_FACTOR dot products in parallel
                // and accumulate all 4 pooling stats inline
                PARALLEL_KERNELS: for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
                    #pragma HLS UNROLL
                    if (kg + ki < NUM_KERNELS) {
                        // Dot product
                        data_t value = (data_t)0.0;
                        DOT_PRODUCT: for (int_t k = 0; k < KERNEL_SIZE; k++) {
                            #pragma HLS UNROLL
                            value += sw[k] * weights[kg + ki][k];
                        }

                        // Update pooling accumulators for each bias threshold
                        POOL_BIAS: for (int_t f = 0; f < MAX_FEATURES_PER_DILATION; f++) {
                            #pragma HLS UNROLL
                            if (f < features_this_dilation) {
                                data_t bias = group_biases[ki][f];
                                if (value > bias) {
                                    // PPV
                                    ppv_counts[ki][f]++;
                                    // MPV
                                    mpv_sum[ki][f]  += value;
                                    mpv_count[ki][f]++;
                                    // MIPV (index is j, since start=0 per Round 2)
                                    mipv_isum[ki][f] += j;
                                    // LSPV
                                    lspv_cur[ki][f]++;
                                    if (lspv_cur[ki][f] > lspv_max[ki][f]) {
                                        lspv_max[ki][f] = lspv_cur[ki][f];
                                    }
                                } else {
                                    lspv_cur[ki][f] = 0;
                                }
                            }
                        }
                    }
                }
            } // end CONV_LOOP

            // Compute final pooling values and write to features[]
            // effective_length = time_series_length (start=0, per Round 2)
            data_t inv_length = (data_t)1.0 / (data_t)time_series_length;

            WRITE_FEATS: for (int_t wi = 0; wi < UNROLL_FACTOR * MAX_FEATURES_PER_DILATION; wi++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=16 max=64
                int_t ki = wi / MAX_FEATURES_PER_DILATION;
                int_t f  = wi % MAX_FEATURES_PER_DILATION;
                int_t kernel_idx = kg + ki;

                if (kernel_idx < NUM_KERNELS && f < features_this_dilation) {
                    // current_feature_idx = feature_idx + kernel_idx * features_this_dilation + f
                    // But note: feature_idx is the base for THIS dilation, and inside DILATION_LOOP
                    // feature_idx has already been accumulated from previous dilations.
                    // However we don't update feature_idx inside KERNEL_GROUP_LOOP.
                    // The actual current_feature_idx for features_per_dilation indexing is:
                    //   feature_idx + kernel_idx * features_this_dilation
                    // (feature_idx here is the start of this dilation group)
                    int_t cfi = feature_idx + kernel_idx * features_this_dilation;
                    int_t base = (cfi + f) * n_feature_per_kernel + starting_feature_idx;

                    if (base + 3 < num_features * n_feature_per_kernel + starting_feature_idx) {
                        int_t pc = ppv_counts[ki][f];
                        data_t ppv  = (data_t)pc * inv_length;
                        data_t mpv  = (pc > 0) ? (data_t)(mpv_sum[ki][f] / (data_t)pc) : (data_t)0.0;
                        data_t mipv = (pc > 0) ? (data_t)((data_t)mipv_isum[ki][f] / (data_t)pc * inv_length) : (data_t)0.0;
                        data_t lspv = (data_t)lspv_max[ki][f] * inv_length;

                        features[base + 0] = ppv;
                        features[base + 1] = mpv;
                        features[base + 2] = mipv;
                        features[base + 3] = lspv;
                    }
                }
            }
        } // end KERNEL_GROUP_LOOP

        // Advance feature_idx by all NUM_KERNELS * features_this_dilation for this dilation
        feature_idx += features_this_dilation * NUM_KERNELS;
    } // end DILATION_LOOP
}

// HLS-optimized scaling function
void apply_scaler_hls(
    data_t features[MAX_FEATURES],
    data_t scaled_features[MAX_FEATURES],
    data_t scaler_mean[MAX_FEATURES],
    data_t scaler_scale[MAX_FEATURES],
    int_t num_features
) {
    #pragma HLS INLINE off

    SCALE_LOOP: for (int_t i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=100 max=50000

        scaled_features[i] = (features[i] - scaler_mean[i]) / scaler_scale[i];
    }
}

// HLS-optimized linear classifier
void linear_classifier_predict_hls(
    data_t scaled_features[MAX_FEATURES],
    data_t predictions[MAX_CLASSES],
    data_t coefficients[MAX_CLASSES][MAX_FEATURES],
    data_t intercept[MAX_CLASSES],
    int_t num_features,
    int_t num_classes
) {
    #pragma HLS INLINE off

    if (num_classes == 2) {
        data_t score = intercept[0];

        BINARY_FEATURE_LOOP: for (int_t j = 0; j < num_features; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=100 max=50000
            score += coefficients[0][j] * scaled_features[j];
        }

        predictions[0] = (data_t)0.0 - score;
        predictions[1] = score;
    } else {
        CLASS_LOOP: for (int_t i = 0; i < num_classes; i++) {
            #pragma HLS PIPELINE off
            #pragma HLS LOOP_TRIPCOUNT min=2 max=16

            data_t score = intercept[i];

            FEATURE_LOOP: for (int_t j = 0; j < num_features; j++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=100 max=50000
                score += coefficients[i][j] * scaled_features[j];
            }

            predictions[i] = score;
        }
    }
}

// HLS-optimized top-level function for FPGA
// SIGNATURE MUST NOT CHANGE — host binary depends on this exact ABI.
extern "C" void multirocket_inference(
    data_t* time_series_input,      // Input time series
    data_t* prediction_output,      // Output predictions
    data_t* coefficients,           // Model coefficients (flattened)
    data_t* intercept,              // Model intercept
    data_t* scaler_mean,            // Scaler mean values
    data_t* scaler_scale,           // Scaler scale values
    int_t* dilations_0,               // Dilation values (orig)
    int_t* num_features_per_dilation_0, // Features per dilation (orig)
    data_t* biases_0,                 // Bias values (orig)
    int_t num_dilations_0,
    int_t num_features_0,
    int_t* dilations_1,               // Dilation values (diff)
    int_t* num_features_per_dilation_1, // Features per dilation (diff)
    data_t* biases_1,                 // Bias values (diff)
    int_t num_dilations_1,
    int_t num_features_1,
    int_t time_series_length,
    int_t num_features,
    int_t num_classes,
    int_t n_feature_per_kernel
) {
    #pragma HLS INTERFACE m_axi port=time_series_input bundle=gmem0 depth=8192
    #pragma HLS INTERFACE m_axi port=prediction_output bundle=gmem1 depth=16
    #pragma HLS INTERFACE m_axi port=coefficients bundle=gmem2 depth=200000
    #pragma HLS INTERFACE m_axi port=intercept bundle=gmem3 depth=16
    #pragma HLS INTERFACE m_axi port=scaler_mean bundle=gmem4 depth=50000
    #pragma HLS INTERFACE m_axi port=scaler_scale bundle=gmem5 depth=50000
    #pragma HLS INTERFACE m_axi port=dilations_0 bundle=gmem6 depth=32
    #pragma HLS INTERFACE m_axi port=num_features_per_dilation_0 bundle=gmem7 depth=32
    #pragma HLS INTERFACE m_axi port=biases_0 bundle=gmem8 depth=50000
    #pragma HLS INTERFACE m_axi port=dilations_1 bundle=gmem9 depth=32
    #pragma HLS INTERFACE m_axi port=num_features_per_dilation_1 bundle=gmem10 depth=32
    #pragma HLS INTERFACE m_axi port=biases_1 bundle=gmem11 depth=50000

    #pragma HLS INTERFACE s_axilite port=time_series_length bundle=control
    #pragma HLS INTERFACE s_axilite port=num_features bundle=control
    #pragma HLS INTERFACE s_axilite port=num_classes bundle=control
    #pragma HLS INTERFACE s_axilite port=num_dilations_0 bundle=control
    #pragma HLS INTERFACE s_axilite port=num_features_0 bundle=control
    #pragma HLS INTERFACE s_axilite port=num_dilations_1 bundle=control
    #pragma HLS INTERFACE s_axilite port=num_features_1 bundle=control
    #pragma HLS INTERFACE s_axilite port=n_feature_per_kernel bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // 5 duplicate copies of time series for conflict-free 9-port BRAM read
    // NOTE: static here avoids stack overflow in C++ csim (4MB+ of local arrays).
    // In HLS synthesis, static local arrays map to internal BRAM — identical
    // behaviour to non-static since multirocket_inference is single-threaded.
    static data_t local_ts0[MAX_TIME_SERIES_LENGTH];
    static data_t local_ts1[MAX_TIME_SERIES_LENGTH];
    static data_t local_ts2[MAX_TIME_SERIES_LENGTH];
    static data_t local_ts3[MAX_TIME_SERIES_LENGTH];
    static data_t local_ts4[MAX_TIME_SERIES_LENGTH];

    // 5 copies of diff series
    static data_t local_diff0[MAX_TIME_SERIES_LENGTH];
    static data_t local_diff1[MAX_TIME_SERIES_LENGTH];
    static data_t local_diff2[MAX_TIME_SERIES_LENGTH];
    static data_t local_diff3[MAX_TIME_SERIES_LENGTH];
    static data_t local_diff4[MAX_TIME_SERIES_LENGTH];

    static data_t local_features[MAX_FEATURES];
    static data_t local_scaled_features[MAX_FEATURES];
    static data_t local_predictions[MAX_CLASSES];
    static data_t local_coefficients[MAX_CLASSES][MAX_FEATURES];
    static data_t local_scaler_mean[MAX_FEATURES];
    static data_t local_scaler_scale[MAX_FEATURES];
    static data_t local_intercept[MAX_CLASSES];
    static data_t local_biases_0[MAX_FEATURES];
    static data_t local_biases_1[MAX_FEATURES];
    static int_t local_dilations_0[MAX_DILATIONS];
    static int_t local_num_features_per_dilation_0[MAX_DILATIONS];
    static int_t local_dilations_1[MAX_DILATIONS];
    static int_t local_num_features_per_dilation_1[MAX_DILATIONS];

    #pragma HLS ARRAY_PARTITION variable=local_predictions type=complete
    #pragma HLS ARRAY_PARTITION variable=local_intercept type=complete

    // Copy time series to 5 BRAM copies simultaneously
    COPY_TS: for (int_t i = 0; i < time_series_length; i++) {
        #pragma HLS PIPELINE II=1
        data_t val = time_series_input[i];
        local_ts0[i] = val;
        local_ts1[i] = val;
        local_ts2[i] = val;
        local_ts3[i] = val;
        local_ts4[i] = val;
    }

    // Compute diff and copy to 5 BRAM copies
    COPY_DIFF: for (int_t i = 0; i < time_series_length - 1; i++) {
        #pragma HLS PIPELINE II=1
        data_t dval = time_series_input[i + 1] - time_series_input[i];
        local_diff0[i] = dval;
        local_diff1[i] = dval;
        local_diff2[i] = dval;
        local_diff3[i] = dval;
        local_diff4[i] = dval;
    }

    COPY_DILATIONS_0: for (int_t i = 0; i < num_dilations_0; i++) {
        #pragma HLS PIPELINE II=1
        local_dilations_0[i] = dilations_0[i];
        local_num_features_per_dilation_0[i] = num_features_per_dilation_0[i];
    }

    COPY_DILATIONS_1: for (int_t i = 0; i < num_dilations_1; i++) {
        #pragma HLS PIPELINE II=1
        local_dilations_1[i] = dilations_1[i];
        local_num_features_per_dilation_1[i] = num_features_per_dilation_1[i];
    }

    COPY_BIASES_0: for (int_t i = 0; i < num_features_0; i++) {
        #pragma HLS PIPELINE II=1
        local_biases_0[i] = biases_0[i];
    }

    COPY_BIASES_1: for (int_t i = 0; i < num_features_1; i++) {
        #pragma HLS PIPELINE II=1
        local_biases_1[i] = biases_1[i];
    }

    COPY_SCALER: for (int_t i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
        local_scaler_mean[i]  = scaler_mean[i];
        local_scaler_scale[i] = scaler_scale[i];
    }

    COPY_INTERCEPT: for (int_t i = 0; i < num_classes; i++) {
        #pragma HLS PIPELINE II=1
        local_intercept[i] = intercept[i];
    }

    COPY_COEF: for (int_t i = 0; i < num_classes; i++) {
        for (int_t j = 0; j < num_features; j++) {
            #pragma HLS PIPELINE II=1
            local_coefficients[i][j] = coefficients[i * num_features + j];
        }
    }

    // Zero-initialize features array so any unwritten slots are 0 (not garbage)
    INIT_FEATURES: for (int_t i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
        local_features[i] = (data_t)0.0;
    }

    // Feature extraction — original representation
    multirocket_feature_extraction_opt(
        local_ts0, local_ts1, local_ts2, local_ts3, local_ts4,
        local_features,
        local_dilations_0,
        local_num_features_per_dilation_0,
        local_biases_0,
        time_series_length,
        num_dilations_0,
        num_features_0,
        n_feature_per_kernel,
        0  // starting_feature_idx for orig
    );

    // Feature extraction — first-order difference representation
    multirocket_feature_extraction_opt(
        local_diff0, local_diff1, local_diff2, local_diff3, local_diff4,
        local_features,
        local_dilations_1,
        local_num_features_per_dilation_1,
        local_biases_1,
        time_series_length - 1,
        num_dilations_1,
        num_features_1,
        n_feature_per_kernel,
        num_features_0 * n_feature_per_kernel  // starting_feature_idx for diff
    );

    // Apply scaling
    apply_scaler_hls(
        local_features,
        local_scaled_features,
        local_scaler_mean,
        local_scaler_scale,
        num_features
    );

    // Linear classification
    linear_classifier_predict_hls(
        local_scaled_features,
        local_predictions,
        local_coefficients,
        local_intercept,
        num_features,
        num_classes
    );

    // Copy output
    COPY_OUTPUT: for (int_t i = 0; i < num_classes; i++) {
        #pragma HLS PIPELINE II=1
        prediction_output[i] = local_predictions[i];
    }
}
