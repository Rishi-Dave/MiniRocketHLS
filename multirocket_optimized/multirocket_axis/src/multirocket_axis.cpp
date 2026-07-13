/*
 * MultiRocket AXI-Stream Inference Kernel
 *
 * Streaming variant of multirocket_inference. Compute path mirrors
 * multirocket_optimized/multirocket/src/multirocket.cpp +
 * multirocket_pooling.cpp (84 fixed kernels, 4 pooling operators × 2
 * representations, StandardScaler, ridge classifier); only the I/O path
 * differs:
 *
 *   m_axi variant: pull `time_series_length` floats from HBM once.
 *   axis variant:  pull one float per AXI-Stream packet, maintain a shift
 *                  register of length `time_series_length`, run inference
 *                  once per incoming packet (same pattern as fpga-network
 *                  MiniRocket and hydra_axis).
 *
 * Helpers are file-local (`static`) to avoid linker conflicts.
 *
 * Kernel weights are static (compile-time) — same as the m_axi variant
 * — and pulled from ../../multirocket/include/weights.txt.
 */

#include "../include/multirocket_axis.hpp"
#include <cstring>

// Static fixed kernel weights (84 kernels × 9 weights, hardcoded -1/+2 from
// the MiniRocket paper). Single source of truth in the m_axi variant.
static data_t weights[NUM_KERNELS][KERNEL_SIZE] = {
    #include "../../multirocket/include/weights.txt"
};

// ---- File-local helpers (mirror multirocket.cpp / multirocket_pooling.cpp) ----

static void apply_kernel_axis(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    int_t kernel_idx,
    int_t dilation,
    int_t time_series_length,
    int_t* output_length
) {
    #pragma HLS INLINE off
    *output_length = time_series_length;
    if (*output_length <= 0) {
        *output_length = 0;
        return;
    }

    static data_t sliding_window[KERNEL_SIZE] = {0};
    #pragma HLS ARRAY_PARTITION variable=sliding_window complete
    #pragma HLS ARRAY_PARTITION variable=weights complete

    CONV_LOOP: for (int_t j = 0; j < time_series_length; j++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=100 max=8192

        int i = 0;
        for (int k = -4; k <= 4; k++) {
            int idx = j + k * dilation;
            sliding_window[i] = (idx < 0 || idx >= time_series_length) ? (data_t)0.0 : time_series[idx];
            i++;
        }
        data_t value = 0.0;
        KERNEL_LOOP: for (int_t k = 0; k < KERNEL_SIZE; k++) {
            #pragma HLS UNROLL
            value += sliding_window[k] * weights[kernel_idx][k];
        }
        convolutions[j] = value;
    }
}

// Combined PPV/MPV/MIPV/LSPV (single-pass). Mirrors
// multirocket_pooling.cpp's `compute_four_pooling_operators` (NOT the
// commented-out version above it).
static void compute_four_pooling_axis(
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    data_t bias,
    int_t start,
    int_t length,
    PoolingStats_axis* stats
) {
    #pragma HLS INLINE off

    int_t ppv = 0;
    int_t last_val = 0;
    data_t max_stretch = 0.0f;
    int_t mean_index = 0;
    data_t mean = 0.0f;
    data_t stretch = 0.0f;

    POOLING_LOOP: for (int_t i = start; i < length; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=100 max=8192

        if (convolutions[i] > bias) {
            ppv += 1;
            mean_index += (i - start);
            mean += convolutions[i] + bias;
        } else if (convolutions[i] < bias) {
            stretch = (i - start) - last_val;
            if (stretch > max_stretch) max_stretch = stretch;
            last_val = (i - start);
        }
    }
    stretch = (length - start) - 1 - last_val;
    if (stretch > max_stretch) max_stretch = stretch;

    int_t denom = length - start;
    stats->ppv  = (denom > 0) ? (data_t) ppv / (data_t) denom : 0.0f;
    stats->mpv  = max_stretch;
    stats->mipv = (ppv > 0) ? (data_t) mean / (data_t) ppv : 0.0f;
    stats->lspv = (ppv > 0) ? (data_t) mean_index / (data_t) ppv : -1.0f;
}

static void multirocket_feature_extraction_axis(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    int_t  dilations[MAX_DILATIONS],
    int_t  num_features_per_dilation[MAX_DILATIONS],
    data_t biases[MAX_FEATURES],
    int_t  time_series_length,
    int_t  num_dilations,
    int_t  num_features,
    int_t  n_feature_per_kernel,
    int_t  starting_feature_idx
) {
    #pragma HLS INLINE off

    static data_t convolutions[MAX_TIME_SERIES_LENGTH];
    #pragma HLS ARRAY_PARTITION variable=convolutions type=cyclic factor=8

    int_t feature_idx = 0;
    DILATION_LOOP: for (int_t dil_idx = 0; dil_idx < num_dilations; dil_idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32

        int_t dilation = dilations[dil_idx];
        int_t features_this_dilation = num_features_per_dilation[dil_idx];
        int_t _padding0 = dil_idx % 2;
        int_t padding = ((9 - 1) * dilation) / 2;

        KERNEL_LOOP: for (int_t kernel_idx = 0; kernel_idx < NUM_KERNELS; kernel_idx++) {
            #pragma HLS LOOP_TRIPCOUNT min=84 max=84
            #pragma HLS PIPELINE off

            int_t _padding1 = (_padding0 + kernel_idx) % 2;

            int_t conv_length;
            apply_kernel_axis(time_series, convolutions, kernel_idx, dilation,
                              time_series_length, &conv_length);

            for (int_t f = 0; f < features_this_dilation; f++) {
                int_t current_feature_idx = feature_idx + f;
                data_t bias = biases[current_feature_idx];
                PoolingStats_axis stats;
                if (_padding1 == 0) {
                    compute_four_pooling_axis(convolutions, bias, 0, conv_length, &stats);
                } else {
                    compute_four_pooling_axis(convolutions, bias, padding,
                                              conv_length - padding, &stats);
                }
                int_t end = current_feature_idx + starting_feature_idx;
                features[end + 0 * num_features] = stats.ppv;
                features[end + 1 * num_features] = stats.mpv;
                features[end + 2 * num_features] = stats.mipv;
                features[end + 3 * num_features] = stats.lspv;
            }
            feature_idx += features_this_dilation;
        }
    }
}

static void apply_scaler_axis(
    data_t features[MAX_FEATURES],
    data_t scaled_features[MAX_FEATURES],
    data_t scaler_mean[MAX_FEATURES],
    data_t scaler_scale[MAX_FEATURES],
    int_t  num_features
) {
    #pragma HLS INLINE off
    SCALE_LOOP: for (int_t i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=100 max=50000
        scaled_features[i] = (features[i] - scaler_mean[i]) / scaler_scale[i];
    }
}

static void linear_classifier_axis(
    data_t scaled_features[MAX_FEATURES],
    data_t predictions[MAX_CLASSES],
    data_t coefficients[MAX_CLASSES][MAX_FEATURES],
    data_t intercept[MAX_CLASSES],
    int_t  num_features,
    int_t  num_classes
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

// ---- Top-level kernel -----------------------------------------------------

extern "C" void multirocket_axis_inference(
    hls::stream<pkt> &input_timeseries,
    hls::stream<pkt> &output_predictions,
    data_t* coefficients,
    data_t* intercept,
    data_t* scaler_mean,
    data_t* scaler_scale,
    int_t*  dilations_0,
    int_t*  num_features_per_dilation_0,
    data_t* biases_0,
    int_t   num_dilations_0,
    int_t   num_features_0,
    int_t*  dilations_1,
    int_t*  num_features_per_dilation_1,
    data_t* biases_1,
    int_t   num_dilations_1,
    int_t   num_features_1,
    int_t   time_series_length,
    int_t   num_features,
    int_t   num_classes,
    int_t   n_feature_per_kernel
) {
    #pragma HLS INTERFACE axis port=input_timeseries
    #pragma HLS INTERFACE axis port=output_predictions
    // NOTE: do NOT specify offset=slave on m_axi for streaming axis kernels — it
    // causes HLS to drop the m_axi pointer slave registers from the control bundle
    // (or split them into a separate _R bundle that XRT can't reach), leaving the
    // xclbin unprogrammable. Use the bare bundle/depth pragma + paired explicit
    // s_axilite to put the address regs in the main `control` bundle. See
    // saturation_pivot_2026_04_28.md / hls-patterns.md (2026-05-04 entry).
    #pragma HLS INTERFACE m_axi port=coefficients                bundle=gmem2  depth=200000
    #pragma HLS INTERFACE m_axi port=intercept                   bundle=gmem3  depth=16
    #pragma HLS INTERFACE m_axi port=scaler_mean                 bundle=gmem4  depth=50000
    #pragma HLS INTERFACE m_axi port=scaler_scale                bundle=gmem5  depth=50000
    #pragma HLS INTERFACE m_axi port=dilations_0                 bundle=gmem6  depth=32
    #pragma HLS INTERFACE m_axi port=num_features_per_dilation_0 bundle=gmem7  depth=32
    #pragma HLS INTERFACE m_axi port=biases_0                    bundle=gmem8  depth=50000
    #pragma HLS INTERFACE m_axi port=dilations_1                 bundle=gmem9  depth=32
    #pragma HLS INTERFACE m_axi port=num_features_per_dilation_1 bundle=gmem10 depth=32
    #pragma HLS INTERFACE m_axi port=biases_1                    bundle=gmem11 depth=50000
    #pragma HLS INTERFACE s_axilite port=coefficients                bundle=control
    #pragma HLS INTERFACE s_axilite port=intercept                   bundle=control
    #pragma HLS INTERFACE s_axilite port=scaler_mean                 bundle=control
    #pragma HLS INTERFACE s_axilite port=scaler_scale                bundle=control
    #pragma HLS INTERFACE s_axilite port=dilations_0                 bundle=control
    #pragma HLS INTERFACE s_axilite port=num_features_per_dilation_0 bundle=control
    #pragma HLS INTERFACE s_axilite port=biases_0                    bundle=control
    #pragma HLS INTERFACE s_axilite port=dilations_1                 bundle=control
    #pragma HLS INTERFACE s_axilite port=num_features_per_dilation_1 bundle=control
    #pragma HLS INTERFACE s_axilite port=biases_1                    bundle=control

    #pragma HLS INTERFACE s_axilite port=time_series_length bundle=control
    #pragma HLS INTERFACE s_axilite port=num_features       bundle=control
    #pragma HLS INTERFACE s_axilite port=num_classes        bundle=control
    #pragma HLS INTERFACE s_axilite port=num_dilations_0    bundle=control
    #pragma HLS INTERFACE s_axilite port=num_features_0     bundle=control
    #pragma HLS INTERFACE s_axilite port=num_dilations_1    bundle=control
    #pragma HLS INTERFACE s_axilite port=num_features_1     bundle=control
    #pragma HLS INTERFACE s_axilite port=n_feature_per_kernel bundle=control
    #pragma HLS INTERFACE s_axilite port=return             bundle=control

    // Persistent shift register across kernel invocations (csim mode).
    // In BUILD=1 hw mode the while(true) loop scope makes naturalness; the
    // static qualifier here is harmless. Model buffers are NOT static —
    // matches the HYDRA axis variant pattern that csynths cleanly. They get
    // re-loaded per call from m_axi, which HLS pipelines into the same DMA
    // overhead the HBM-bandwidth-bound m_axi MultiRocket variant pays once
    // per inference.
    static data_t local_time_series[MAX_TIME_SERIES_LENGTH] = {0};
    static int_t  shift_filled = 0;

    data_t local_time_series_diff[MAX_TIME_SERIES_LENGTH - 1];
    data_t local_features[MAX_FEATURES];
    data_t local_scaled_features[MAX_FEATURES];
    data_t local_predictions[MAX_CLASSES];
    data_t local_coefficients[MAX_CLASSES][MAX_FEATURES];
    data_t local_scaler_mean[MAX_FEATURES];
    data_t local_scaler_scale[MAX_FEATURES];
    data_t local_intercept[MAX_CLASSES];
    int_t  local_dilations_0[MAX_DILATIONS];
    int_t  local_num_features_per_dilation_0[MAX_DILATIONS];
    data_t local_biases_0[MAX_FEATURES];
    int_t  local_dilations_1[MAX_DILATIONS];
    int_t  local_num_features_per_dilation_1[MAX_DILATIONS];
    data_t local_biases_1[MAX_FEATURES];

    #pragma HLS ARRAY_PARTITION variable=local_features type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=local_scaled_features type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=local_predictions type=complete
    #pragma HLS ARRAY_PARTITION variable=local_coefficients type=block factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_intercept type=complete

    // Load model parameters from global memory each call (HLS can pipeline).
    {
        LOAD_DIL0: for (int_t i = 0; i < num_dilations_0; i++) {
            #pragma HLS PIPELINE II=1
            local_dilations_0[i] = dilations_0[i];
            local_num_features_per_dilation_0[i] = num_features_per_dilation_0[i];
        }
        LOAD_DIL1: for (int_t i = 0; i < num_dilations_1; i++) {
            #pragma HLS PIPELINE II=1
            local_dilations_1[i] = dilations_1[i];
            local_num_features_per_dilation_1[i] = num_features_per_dilation_1[i];
        }
        LOAD_BIAS0: for (int_t i = 0; i < num_features_0; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=100 max=50000
            local_biases_0[i] = biases_0[i];
        }
        LOAD_BIAS1: for (int_t i = 0; i < num_features_1; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=100 max=50000
            local_biases_1[i] = biases_1[i];
        }
        LOAD_SCALE: for (int_t i = 0; i < num_features; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=100 max=50000
            local_scaler_mean[i]  = scaler_mean[i];
            local_scaler_scale[i] = scaler_scale[i];
        }
        LOAD_INTC: for (int_t c = 0; c < num_classes; c++) {
            #pragma HLS PIPELINE II=1
            local_intercept[c] = intercept[c];
        }
        LOAD_COEF: for (int_t c = 0; c < num_classes; c++) {
            for (int_t j = 0; j < num_features; j++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=100 max=50000
                local_coefficients[c][j] = coefficients[c * num_features + j];
            }
        }
    }

#if BUILD == 1
    while (true) {
#endif

        if (!input_timeseries.empty()) {
            pkt v = input_timeseries.read();
            ap_uint<DWIDTH> tmp = v.data;
            data_t new_val = *((data_t*) &tmp);

            // Shift-register ingestion.
            SHIFT_LOOP: for (int_t i = 0; i < time_series_length - 1; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=100 max=8192
                local_time_series[i] = local_time_series[i+1];
            }
            local_time_series[time_series_length - 1] = new_val;
            if (shift_filled < time_series_length) shift_filled++;

            // First-order difference — every beat (matches m_axi variant
            // structure; rebuilds it each call since shift register changed).
            DIFF_LOOP: for (int_t i = 0; i < time_series_length - 1; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=100 max=8192
                local_time_series_diff[i] = local_time_series[i+1] - local_time_series[i];
            }

            // Feature extraction over original representation.
            multirocket_feature_extraction_axis(
                local_time_series, local_features,
                local_dilations_0, local_num_features_per_dilation_0,
                local_biases_0,
                time_series_length, num_dilations_0, num_features_0,
                n_feature_per_kernel, 0
            );

            // Feature extraction over first-order-difference representation.
            int_t starting_idx_1 = ((num_features_1 + num_features_0) * n_feature_per_kernel) / 2;
            multirocket_feature_extraction_axis(
                local_time_series_diff, local_features,
                local_dilations_1, local_num_features_per_dilation_1,
                local_biases_1,
                time_series_length - 1, num_dilations_1, num_features_1,
                n_feature_per_kernel, starting_idx_1
            );

            apply_scaler_axis(
                local_features, local_scaled_features,
                local_scaler_mean, local_scaler_scale,
                num_features
            );

            linear_classifier_axis(
                local_scaled_features, local_predictions,
                local_coefficients, local_intercept,
                num_features, num_classes
            );

            // Emit num_classes prediction pkts. Use a value-copy + union for
            // float->u32 reinterpret instead of `*(ap_uint<32>*)&array[c]`,
            // because `complete`-partitioned arrays don't support taking
            // address of an element under HLS (works for HYDRA's smaller
            // MAX_CLASSES=10 partition but trips a "full array load/store"
            // pre-synth check at MAX_CLASSES=16).
            // Defensive sideband init (saturation_pivot 2026-05-05 finding):
            // explicit user/dest/id zeroing matches hydra_axis pattern.
            EMIT_PRED: for (int_t c = 0; c < num_classes; c++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=2 max=16
                pkt out_pkt;
                ap_uint<DWIDTH> out_data = 0;
                data_t pred_val = local_predictions[c];
                union { float f; uint32_t u; } cvt;
                cvt.f = pred_val;
                out_data.range(31, 0) = ap_uint<32>(cvt.u);
                out_pkt.data = out_data;
                out_pkt.keep = -1;
                out_pkt.strb = -1;
                out_pkt.last = 1;
                out_pkt.user = 0;
                out_pkt.id   = 0;
                out_pkt.dest = 0;
                output_predictions.write(out_pkt);
            }
        }

#if BUILD == 1
    }
#endif
}
