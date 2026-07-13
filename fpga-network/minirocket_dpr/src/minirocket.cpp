/*
 * MiniRocket Streaming Inference - DP-Reuse + bFP  (FCCM 2026 paper §4, variant 3)
 *
 * Temporal dot-product reuse (paper §3): the convolution feature map is a
 * circular buffer persisted across inferences. When a new datapoint arrives,
 * only the padding-boundary dot products are recomputed; the interior is
 * carried forward by shifting the prior feature map by one position.
 *
 * Based on Prith's DP-Reuse kernel (origin/optimized_version, author pyuvaraj).
 * Corrections applied 2026-06-08 (validated bit-exact vs brute-force golden,
 * see _prith_dpreuse/validate_dpreuse.cpp):
 *   1. Time-series ingestion shift: ascending in-place -> DESCENDING (the old
 *      ascending shift collapsed the history to a plateau).
 *   2. Conv carry-forward: shift the whole interior FIRST (descending), THEN
 *      recompute the boundary windows. Prith's ascending in-place center shift
 *      after the boundary recompute plateaued the interior (worst err 125).
 *   3. Output sideband: Bug-B fix (dest=0; user=0; id=0; strb=-1), dropped the
 *      per-call `dest` arg so the interface matches the deployed host loader.
 *   4. Removed stray s_axilite INTERFACE pragmas from optimized_fp_multiply and
 *      completed the convolutions ARRAY_PARTITION/BIND_STORAGE.
 *
 * bFP multiply: ap_uint<1> weights (0 -> x-1, 1 -> x+2) via XNOR + exponent
 * increment, DSP=0.  Compile -DBUILD=1 for the streaming while(true) loop.
 */

#include "../include/minirocket.hpp"
#include <cstring>

static ap_uint<1> weights[NUM_KERNELS][KERNEL_SIZE] = {
    #include "../include/weights01.txt"
};

// Convolution feature map, persisted across inferences (DP-Reuse circular buffer)
static data_t convolutions[MAX_DILATIONS][NUM_KERNELS][MAX_TIME_SERIES_LENGTH];

// Binary floating-point multiply: weight 0 -> -1*y, weight 1 -> +2*y. DSP=0.
data_t optimized_fp_multiply(ap_uint<1> x, data_t y) {
    #pragma HLS INLINE
    float_num_t yb;
    yb.fp_num = y;
    yb.sign = ~(yb.sign ^ x);
    yb.bexp = yb.bexp + x;
    return yb.fp_num;
}

// Single dilated convolution centered at center_idx (one boundary dot product).
data_t single_convolution(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    int_t dilation,
    int_t time_series_length,
    int_t kernel_idx,
    int_t center_idx
) {
    #pragma HLS INLINE off

    data_t sliding_window[KERNEL_SIZE];
    #pragma HLS ARRAY_PARTITION variable=sliding_window complete
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=2

    int i = 0;
    for (int k = -4; k <= 4; k++) {
        int_t idx = center_idx + k * dilation;
        sliding_window[i] = (idx < 0 || idx >= time_series_length) ? (data_t)0.0 : time_series[idx];
        i++;
    }

    data_t value = 0.0;
    KERNEL_LOOP: for (int_t k = 0; k < KERNEL_SIZE; k++) {
        #pragma HLS UNROLL
        value += optimized_fp_multiply(weights[kernel_idx][k], sliding_window[k]);
    }
    return value;
}

// DP-Reuse feature extraction: carry the feature map forward, recompute only
// the boundary dot products affected by the newly-arrived / dropped sample.
void minirocket_feature_extraction_hls(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    int_t dilations[MAX_DILATIONS],
    int_t num_features_per_dilation[MAX_DILATIONS],
    data_t biases[MAX_FEATURES],
    int_t time_series_length,
    int_t num_dilations,
    int_t num_features
) {
    #pragma HLS INLINE off
    #pragma HLS BIND_STORAGE variable=convolutions type=ram_2p impl=bram

    int_t feature_idx = 0;
    int_t hk = KERNEL_SIZE / 2;   // 4

    DILATION_LOOP: for (int_t dil_idx = 0; dil_idx < num_dilations; dil_idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8

        int_t dilation = dilations[dil_idx];
        int_t features_this_dilation = num_features_per_dilation[dil_idx];
        int_t bnd = hk * dilation;   // boundary half-width (= alpha = beta region)

        KERNEL_LOOP: for (int_t kernel_idx = 0; kernel_idx < NUM_KERNELS; kernel_idx++) {
            #pragma HLS LOOP_TRIPCOUNT min=84 max=84
            #pragma HLS PIPELINE off

            // (1) Carry forward: shift the prior feature map right by one.
            // DESCENDING so each cell reads its un-overwritten left neighbour.
            SHIFT_CONV: for (int_t j = time_series_length - 1; j >= 1; j--) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=150 max=192
                convolutions[dil_idx][kernel_idx][j] = convolutions[dil_idx][kernel_idx][j - 1];
            }

            // (2) Recompute front-boundary centers (windows that include the new
            // sample at index 0): centers 0 .. hk*dilation.
            FRONT_BND: for (int_t c = 0; c <= bnd; c++) {
                #pragma HLS LOOP_TRIPCOUNT min=4 max=32
                if (c < time_series_length)
                    convolutions[dil_idx][kernel_idx][c] =
                        single_convolution(time_series, dilation, time_series_length, kernel_idx, c);
            }

            // (3) Recompute back-boundary centers (windows that include the
            // dropped sample): centers time_series_length-1-hk*dilation .. length-1.
            BACK_BND: for (int_t c = time_series_length - 1; c >= time_series_length - 1 - bnd; c--) {
                #pragma HLS LOOP_TRIPCOUNT min=4 max=32
                if (c >= 0)
                    convolutions[dil_idx][kernel_idx][c] =
                        single_convolution(time_series, dilation, time_series_length, kernel_idx, c);
            }

            // PPV: positive proportion of values over the feature map.
            PPV_FEATURES: for (int_t f = 0; f < features_this_dilation; f++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=3
                data_t bias = biases[feature_idx + f];
                int_t positive_count = 0;

                PPV_LOOP: for (int_t i = 0; i < time_series_length; i++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=150 max=192
                    if (convolutions[dil_idx][kernel_idx][i] > bias) {
                        positive_count++;
                    }
                }

                features[feature_idx + f] = (data_t)positive_count / (data_t)time_series_length;
            }

            feature_idx += features_this_dilation;
        }
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

    // Streaming history buffer, persisted across inferences (newest at index 0).
    static data_t local_time_series[MAX_TIME_SERIES_LENGTH] = {0};
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

            // Ingest new sample at index 0; shift history right by one.
            // DESCENDING so each cell reads its un-overwritten left neighbour.
            INGEST_SHIFT: for (int_t i = time_series_length - 1; i >= 1; i--) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=150 max=192
                local_time_series[i] = local_time_series[i - 1];
            }
            local_time_series[0] = new_val;

            // DP-Reuse feature extraction (carry-forward feature map)
            minirocket_feature_extraction_hls(
                local_time_series,
                local_features,
                local_dilations,
                local_num_features_per_dilation,
                local_biases,
                time_series_length,
                num_dilations,
                num_features
            );

            apply_scaler_hls(
                local_features, local_scaled_features,
                local_scaler_mean, local_scaler_scale, num_features
            );

            linear_classifier_predict_hls(
                local_scaled_features, local_predictions,
                local_coefficients, local_intercept,
                num_features, num_classes
            );

            // Write predictions to AXI stream (Bug-B sideband: all fields set)
            for (int_t i = 0; i < num_classes; i++) {
                #pragma HLS PIPELINE II=1
                pkt out_pkt;
                ap_uint< DWIDTH > out_data = *((ap_uint< 32 >*)&local_predictions[i]);
                out_pkt.data = out_data;
                out_pkt.keep = -1;
                out_pkt.last = 1;
                out_pkt.dest = 0;
                out_pkt.user = 0;
                out_pkt.id   = 0;
                out_pkt.strb = -1;
                output_predictions.write(out_pkt);
            }
        }
#if BUILD == 1
    }
#endif
}
