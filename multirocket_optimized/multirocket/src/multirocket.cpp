#include "../include/multirocket.hpp"
#include <cstring>


static data_t weights[NUM_KERNELS][KERNEL_SIZE] = {
    #include "../include/weights.txt"
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
    #pragma HLS INLINE off
    
    const int_t kernel_length = KERNEL_SIZE;
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
        
        int i = 0;
        for (int k = -4; k <= 4; k++) {
            if (j + k * dilation < 0 || j + k * dilation >= time_series_length) {
                sliding_window[i] = 0.0;
            } else {
                sliding_window[i] = time_series[j + k * dilation];
            }
            i++;
        }

        data_t value = 0.0;
        KERNEL_LOOP: for (int_t k = 0; k < KERNEL_SIZE; k++) {
            #pragma HLS PIPELINE II=1
            value += sliding_window[k] * weights[kernel_idx][k];
        }
        convolutions[j] = value;
    }
}

// HLS-optimized MultiRocket feature extraction with all 4 pooling operators and 2 representations
void multirocket_feature_extraction_hls(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
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

    // Local arrays for computations
    data_t convolutions[MAX_TIME_SERIES_LENGTH];
    #pragma HLS ARRAY_PARTITION variable=convolutions type=cyclic factor=8

    int_t feature_idx = 0;
    DILATION_LOOP: for (int_t dil_idx = 0; dil_idx < num_dilations; dil_idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8

        int_t dilation = dilations[dil_idx];
        int_t features_this_dilation = num_features_per_dilation[dil_idx];
        int_t _padding0 = dil_idx % 2;
        int_t padding = ((9-1) * dilation) / 2;

        KERNEL_LOOP: for (int_t kernel_idx = 0; kernel_idx < NUM_KERNELS; kernel_idx++) {
            #pragma HLS LOOP_TRIPCOUNT min=84 max=84
            #pragma HLS PIPELINE off

            int_t _padding1 = (_padding0 + kernel_idx) % 2;

            // Apply kernel convolution on current representation
            int_t conv_length;
            apply_kernel_hls(time_series, convolutions, kernel_idx, dilation, time_series_length, &conv_length);

            // Calculate positive proportion of values (PPV)
            for (int_t f = 0; f < features_this_dilation; f++) {

                int_t current_feature_idx = feature_idx + f;
                data_t bias = biases[current_feature_idx];
                PoolingStats stats;

                if (_padding1 == 0) {
                    compute_four_pooling_operators(convolutions, bias, 0, conv_length, &stats);
                } else {
                    compute_four_pooling_operators(convolutions, bias, padding, conv_length - padding, &stats);
                }
                int_t end = current_feature_idx + starting_feature_idx;
                features[end + 0 * num_features] = stats.ppv;
                features[end + 1 * num_features] = stats.mpv;
                features[end + 2 * num_features] = stats.mipv;
                features[end + 3 * num_features] = stats.lspv;
                
            }
            feature_idx+= features_this_dilation;
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
        // Binary classification: use single decision function
        data_t score = intercept[0];
        
        BINARY_FEATURE_LOOP: for (int_t j = 0; j < num_features; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=100 max=10000
            
            score += coefficients[0][j] * scaled_features[j];
        }
        
        predictions[0] = (data_t)0.0 - score;  // Class 0 score
        predictions[1] = score;   // Class 1 score
    } else {
        // Multi-class classification
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

// HLS-optimized top-level function for FPGA
extern "C" void multirocket_inference(
    data_t* time_series_input,      // Input time series
    data_t* prediction_output,      // Output predictions
    data_t* coefficients,           // Model coefficients (flattened)
    data_t* intercept,              // Model intercept
    data_t* scaler_mean,            // Scaler mean values
    data_t* scaler_scale,           // Scaler scale values
    int_t* dilations_0,               // Dilation values
    int_t* num_features_per_dilation_0, // Features per dilation
    data_t* biases_0,                 // Bias values
    int_t num_dilations_0,
    int_t num_features_0,
    int_t* dilations_1,               // Dilation values
    int_t* num_features_per_dilation_1, // Features per dilation
    data_t* biases_1,                 // Bias values
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
    
    // Local arrays for processing
    data_t local_time_series[MAX_TIME_SERIES_LENGTH];
    data_t local_time_series_diff[MAX_TIME_SERIES_LENGTH-1];  
    data_t local_features[MAX_FEATURES];
    data_t local_scaled_features[MAX_FEATURES];
    data_t local_predictions[MAX_CLASSES];
    data_t local_coefficients[MAX_CLASSES][MAX_FEATURES];
    data_t local_scaler_mean[MAX_FEATURES];
    data_t local_scaler_scale[MAX_FEATURES];
    data_t local_intercept[MAX_CLASSES];
    data_t local_biases[MAX_FEATURES];
    int_t local_dilations_0[MAX_DILATIONS];
    int_t local_num_features_per_dilation_0[MAX_DILATIONS];
    data_t local_biases_0[MAX_FEATURES];
    int_t local_dilations_1[MAX_DILATIONS];
    int_t local_num_features_per_dilation_1[MAX_DILATIONS];
    data_t local_biases_1[MAX_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=local_time_series type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=local_time_series_diff type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=local_features type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=local_scaled_features type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=local_predictions type=complete
    #pragma HLS ARRAY_PARTITION variable=local_coefficients type=block factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_intercept type=complete
    
    // Copy input data to local arrays

    COPY_INPUT: for (int_t i = 0; i < time_series_length; i++) {
        #pragma HLS PIPELINE II=1
        local_time_series[i] = time_series_input[i];
    }

    COPY_INPUT_DIFF: for (int i = 0; i < time_series_length - 1; i++) {
        #pragma HLS PIPELINE II=1
        local_time_series_diff[i] = time_series_input[i + 1] - time_series_input[i];
    }
    
    COPY_DILATIONS: for (int_t i = 0; i < num_dilations_0; i++) {
        #pragma HLS PIPELINE II=1
        local_dilations_0[i] = dilations_0[i];
        local_num_features_per_dilation_0[i] = num_features_per_dilation_0[i];
    }
    
    COPY_DILATIONS_1: for (int_t i = 0; i < num_dilations_1; i++) {
        #pragma HLS PIPELINE II=1
        local_dilations_1[i] = dilations_1[i];
        local_num_features_per_dilation_1[i] = num_features_per_dilation_1[i];
    }
    
    for (int i = 0; i < num_features_0; i++) {
        local_biases_0[i] = biases_0[i];
    }

    for (int i = 0; i < num_features_1; i++) {
        local_biases_1[i] = biases_1[i];
    }

    COPY_BIASES: for (int_t i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
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
    

    // Feature extraction
    multirocket_feature_extraction_hls(
        local_time_series,
        local_features,
        local_dilations_0,
        local_num_features_per_dilation_0,
        local_biases_0,
        time_series_length,
        num_dilations_0,
        num_features_0,
        n_feature_per_kernel,
        0
    );
    
    multirocket_feature_extraction_hls(
        local_time_series_diff,
        local_features,
        local_dilations_1,
        local_num_features_per_dilation_1,
        local_biases_1,
        time_series_length - 1,
        num_dilations_1,
        num_features_1,
        n_feature_per_kernel,
        ((num_features_1 + num_features_0) * n_feature_per_kernel) / 2
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