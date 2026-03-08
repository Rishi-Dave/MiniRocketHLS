#include "feature_extraction.hpp"
#include <cstring>

static data_t weights[NUM_KERNELS][KERNEL_SIZE] = {
    #include "../../include/weights.txt"
};

// HLS-optimized convolution with specific kernel and dilation
void apply_kernel_hls(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    int_t kernel_idx,
    int_t dilation,
    int_t time_series_length,
    int_t* output_length
) {
    #pragma HLS INLINE

    const int_t kernel_length = KERNEL_SIZE;
    *output_length = time_series_length;

    if (*output_length <= 0) {
        *output_length = 0;
        return;
    }

    #pragma HLS BIND_STORAGE variable=weights type=rom_1p impl=bram
    #pragma HLS ARRAY_PARTITION variable=weights complete

    CONV_LOOP: for (int_t j = 0; j < MAX_TIME_SERIES_LENGTH; j++) {
        #pragma HLS PIPELINE II=1
        if (j < time_series_length) {
            data_t sliding_window[KERNEL_SIZE] = {0};
            #pragma HLS ARRAY_PARTITION variable=sliding_window complete
            int i = 0;
            CENTERING_KERNEL_LOOP: for (int k = -4; k <= 4; k++) {
                if (j + k * dilation < 0 || j + k * dilation >= time_series_length) {
                    sliding_window[i] = 0.0;
                } else {
                    sliding_window[i] = time_series[j + k * dilation];
                }
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
}

// HLS-optimized MiniRocket feature extraction
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

    int_t feature_idx = 0;

    DILATION_LOOP: for (int_t dil_idx = 0; dil_idx < num_dilations; dil_idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8

        int_t dilation = dilations[dil_idx];
        int_t features_this_dilation = num_features_per_dilation[dil_idx];
        int_t _padding0 = dil_idx % 2;
        int_t padding = ((9-1) * dilation) / 2;

        KERNEL_LOOP: for (int_t kernel_idx = 0; kernel_idx < NUM_KERNELS; kernel_idx++) {
            #pragma HLS UNROLL

            data_t convolutions[MAX_TIME_SERIES_LENGTH];
            #pragma HLS ARRAY_PARTITION variable=convolutions type=cyclic factor=8

            int_t _padding1 = (_padding0 + kernel_idx) % 2;
            int_t feature_idx_for_kernel = feature_idx + kernel_idx * features_this_dilation;

            bool kernel_active = (feature_idx_for_kernel < num_features);

            if (kernel_active) {
                int_t conv_length;
                apply_kernel_hls(time_series, convolutions, kernel_idx, dilation,
                               time_series_length, &conv_length);

                for (int_t f = 0; f < features_this_dilation; f++) {
                    data_t bias = biases[feature_idx_for_kernel + f];
                    int_t positive_count = 0;
                    data_t ppv = 0.0;
                    if (_padding1 == 0) {
                        PPV_LOOP_0: for (int_t i = 0; i < time_series_length; i++) {
                            #pragma HLS PIPELINE II=1
                            if (convolutions[i] > bias) {
                                positive_count++;
                            }
                        }
                        ppv = (data_t)positive_count / (data_t)time_series_length;
                    } else {
                        PPV_LOOP_1: for (int_t i = padding; i < time_series_length - padding; i++) {
                            #pragma HLS PIPELINE II=1
                            if (convolutions[i] > bias) {
                                positive_count++;
                            }
                        }
                        ppv = (data_t)positive_count / (data_t)(time_series_length - 2 * padding);
                    }
                    features[feature_idx_for_kernel + f] = ppv;
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

    // Local arrays
    data_t local_time_series[MAX_TIME_SERIES_LENGTH];
    data_t local_features[MAX_FEATURES];
    data_t local_biases[MAX_FEATURES];
    int_t local_dilations[MAX_DILATIONS];
    int_t local_num_features_per_dilation[MAX_DILATIONS];

    #pragma HLS ARRAY_PARTITION variable=local_time_series type=cyclic factor=84
    #pragma HLS ARRAY_PARTITION variable=local_features type=cyclic factor=84

    // Copy inputs from HBM to local BRAM
    COPY_INPUT: for (int_t i = 0; i < time_series_length; i++) {
        #pragma HLS PIPELINE II=1
        local_time_series[i] = time_series_input[i];
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

    // Run feature extraction
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

    // Write features to AXI-Stream for downstream scaler kernel
    STREAM_OUT: for (int_t i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
        features_out.write(local_features[i]);
    }
}
