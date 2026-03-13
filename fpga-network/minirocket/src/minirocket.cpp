/*
 * MiniRocket Streaming Inference - Fused CONV+PPV with Binary Weights
 *
 * Ported from minirocket_modular/feature_extraction_v16_fixed.cpp into the
 * fpga-network streaming architecture. Key optimizations:
 *
 * 1. Fused CONV+PPV: single pass through time series computes convolutions
 *    AND accumulates PPV counts simultaneously, eliminating the large
 *    convolutions[] intermediate array.
 * 2. 5 BRAM copies of time series for conflict-free 9-port reads (II=1)
 * 3. Binary weights with optimized_fp_multiply: DSP=0, uses XNOR+ADD
 * 4. Dynamic loop bound: CONV_LOOP uses time_series_length, not MAX
 * 5. UNROLL=28: processes 28 kernels in parallel per CONV_LOOP iteration
 * 6. Monolithic kernel: feature extraction + scaling + classification in one
 *    function, no AXI-Stream overhead between stages
 *
 * Interface: AXI-Stream from NetLayer (100G Ethernet) for streaming inference
 */

#include "../include/minirocket.hpp"
#include <cstring>

// Binary weights: ap_uint<1> where 0 maps to weight -1, 1 maps to weight +2
static ap_uint<1> weights[NUM_KERNELS][KERNEL_SIZE] = {
    #include "../include/weights01.txt"
};

// Binary floating-point multiply for {-1, +2} weights
// weight=0 → multiply by -1 (flip sign)
// weight=1 → multiply by +2 (flip sign + increment exponent)
// Uses ZERO DSPs: XNOR on sign bit + conditional exponent increment
data_t optimized_fp_multiply(ap_uint<1> x, data_t y) {
    #pragma HLS INLINE
    #pragma HLS PIPELINE

    float_num_t yb;
    yb.fp_num = y;

    yb.sign = ~(yb.sign ^ x);
    yb.bexp = yb.bexp + x;

    return yb.fp_num;
}

// Fused feature extraction: CONV+PPV in single pass with binary weights
void minirocket_feature_extraction_fused(
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

        KERNEL_GROUP_LOOP: for (int_t kg = 0; kg < NUM_KERNELS; kg += UNROLL_FACTOR) {
            #pragma HLS LOOP_TRIPCOUNT min=3 max=3

            // PPV counters and biases in registers (not BRAM)
            // Eliminates the convolutions[MAX_TIME_SERIES_LENGTH] intermediate array
            data_t group_biases[UNROLL_FACTOR][MAX_FEATURES_PER_DILATION];
            #pragma HLS ARRAY_PARTITION variable=group_biases complete dim=0
            int_t ppv_counts[UNROLL_FACTOR][MAX_FEATURES_PER_DILATION];
            #pragma HLS ARRAY_PARTITION variable=ppv_counts complete dim=0
            int_t padding_flag[UNROLL_FACTOR];
            #pragma HLS ARRAY_PARTITION variable=padding_flag complete

            // Pre-load biases and initialize counters
            LOAD_BIASES: for (int_t bi = 0; bi < UNROLL_FACTOR * MAX_FEATURES_PER_DILATION; bi++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=84 max=84
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
            // Dynamic loop bound: uses actual length, avoids wasted cycles on short series
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

                bool in_padded_range = (j >= padding && j < time_series_length - padding);

                // Compute 28 dot products + accumulate PPV counts in parallel
                // Uses optimized_fp_multiply for DSP=0 binary weight multiplication
                PARALLEL_KERNELS: for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
                    #pragma HLS UNROLL
                    data_t value = 0.0;
                    DOT_PRODUCT: for (int_t k = 0; k < KERNEL_SIZE; k++) {
                        #pragma HLS UNROLL
                        value += optimized_fp_multiply(weights[kg + ki][k], sw[k]);
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

            // Precompute reciprocals for PPV division
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
        #pragma HLS LOOP_TRIPCOUNT min=100 max=10000

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
            #pragma HLS LOOP_TRIPCOUNT min=100 max=10000

            score += coefficients[0][j] * scaled_features[j];
        }

        predictions[0] = (data_t)0.0 - score;
        predictions[1] = score;
    } else {
        CLASS_LOOP: for (int_t i = 0; i < num_classes; i++) {
            #pragma HLS PIPELINE off
            #pragma HLS LOOP_TRIPCOUNT min=2 max=4

            data_t score = intercept[i];

            FEATURE_LOOP: for (int_t j = 0; j < num_features; j++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=100 max=10000

                score += coefficients[i][j] * scaled_features[j];
            }

            predictions[i] = score;
        }
    }
}

// Top-level streaming inference kernel
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
) {

    #pragma HLS INTERFACE axis port = input_timeseries
    #pragma HLS INTERFACE axis port = output_predictions
    #pragma HLS INTERFACE m_axi port=coefficients bundle=gmem2 depth=40000
    #pragma HLS INTERFACE m_axi port=intercept bundle=gmem3 depth=4
    #pragma HLS INTERFACE m_axi port=scaler_mean bundle=gmem4 depth=10000
    #pragma HLS INTERFACE m_axi port=scaler_scale bundle=gmem5 depth=10000
    #pragma HLS INTERFACE m_axi port=dilations bundle=gmem6 depth=8
    #pragma HLS INTERFACE m_axi port=num_features_per_dilation bundle=gmem7 depth=8
    #pragma HLS INTERFACE m_axi port=biases bundle=gmem8 depth=10000

    #pragma HLS INTERFACE s_axilite port=time_series_length bundle=control
    #pragma HLS INTERFACE s_axilite port=num_features bundle=control
    #pragma HLS INTERFACE s_axilite port=num_classes bundle=control
    #pragma HLS INTERFACE s_axilite port=num_dilations bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // 5 duplicate copies of time series for conflict-free 9-port reads (DP-Reuse)
    // Each BRAM provides 2 read ports; 9 sliding window positions need 5 BRAMs
    static data_t local_ts0[MAX_TIME_SERIES_LENGTH] = {0};
    static data_t local_ts1[MAX_TIME_SERIES_LENGTH] = {0};
    static data_t local_ts2[MAX_TIME_SERIES_LENGTH] = {0};
    static data_t local_ts3[MAX_TIME_SERIES_LENGTH] = {0};
    static data_t local_ts4[MAX_TIME_SERIES_LENGTH] = {0};

    data_t local_features[MAX_FEATURES];
    data_t local_scaled_features[MAX_FEATURES];
    data_t local_predictions[MAX_CLASSES];
    data_t local_coefficients[MAX_CLASSES][MAX_FEATURES];
    data_t local_scaler_mean[MAX_FEATURES];
    data_t local_scaler_scale[MAX_FEATURES];
    data_t local_intercept[MAX_CLASSES];
    data_t local_biases[MAX_FEATURES];
    int_t local_dilations[MAX_DILATIONS];
    int_t local_num_features_per_dilation[MAX_DILATIONS];

    #pragma HLS ARRAY_PARTITION variable=local_features type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=local_scaled_features type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=local_coefficients type=block factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_intercept type=complete

    // One-time copy of model parameters from global memory
    COPY_DILATIONS: for (int_t i = 0; i < num_dilations; i++) {
        #pragma HLS PIPELINE II=1
        local_dilations[i] = dilations[i];
        local_num_features_per_dilation[i] = num_features_per_dilation[i];
    }

    COPY_BIASES: for (int_t i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
        local_biases[i] = biases[i];
        local_scaler_mean[i] = scaler_mean[i];
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

#if BUILD == 1
    while (true) {
#endif

        if (!input_timeseries.empty()) {

            pkt v = input_timeseries.read();
            ap_uint< DWIDTH > tmp = v.data;
            data_t new_val = *((data_t*) &tmp);

            // Shift-register ingestion: shift all 5 BRAM copies simultaneously
            // Each new data point slides the window left by one position
            SHIFT_LOOP: for (int_t i = 0; i < time_series_length - 1; i++) {
                #pragma HLS PIPELINE II=1
                local_ts0[i] = local_ts0[i+1];
                local_ts1[i] = local_ts1[i+1];
                local_ts2[i] = local_ts2[i+1];
                local_ts3[i] = local_ts3[i+1];
                local_ts4[i] = local_ts4[i+1];
            }
            local_ts0[time_series_length-1] = new_val;
            local_ts1[time_series_length-1] = new_val;
            local_ts2[time_series_length-1] = new_val;
            local_ts3[time_series_length-1] = new_val;
            local_ts4[time_series_length-1] = new_val;

            // Fused feature extraction (CONV+PPV in single pass, DSP=0)
            minirocket_feature_extraction_fused(
                local_ts0, local_ts1, local_ts2, local_ts3, local_ts4,
                local_features,
                local_dilations,
                local_num_features_per_dilation,
                local_biases,
                time_series_length,
                num_dilations,
                num_features
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

            // Write predictions to AXI stream
            for (int_t i = 0; i < num_classes; i++) {
                #pragma HLS PIPELINE II=1
                pkt out_pkt;
                ap_uint< DWIDTH > out_data = *((ap_uint< 32 >*)&local_predictions[i]);
                out_pkt.data = out_data;
                out_pkt.keep = -1;
                out_pkt.last = 1;
                output_predictions.write(out_pkt);
            }

        }

#if BUILD == 1
    }
#endif

}
