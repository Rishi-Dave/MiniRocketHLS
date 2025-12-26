#include "minirocket_inference_hls.h"
#include <cstring>

// Fixed kernel indices (84 combinations of 3 indices from 0-8)
const int_t kernel_indices[NUM_KERNELS][3] = {
    {0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 1, 5}, {0, 1, 6}, {0, 1, 7}, {0, 1, 8},
    {0, 2, 3}, {0, 2, 4}, {0, 2, 5}, {0, 2, 6}, {0, 2, 7}, {0, 2, 8},
    {0, 3, 4}, {0, 3, 5}, {0, 3, 6}, {0, 3, 7}, {0, 3, 8},
    {0, 4, 5}, {0, 4, 6}, {0, 4, 7}, {0, 4, 8},
    {0, 5, 6}, {0, 5, 7}, {0, 5, 8},
    {0, 6, 7}, {0, 6, 8},
    {0, 7, 8},
    {1, 2, 3}, {1, 2, 4}, {1, 2, 5}, {1, 2, 6}, {1, 2, 7}, {1, 2, 8},
    {1, 3, 4}, {1, 3, 5}, {1, 3, 6}, {1, 3, 7}, {1, 3, 8},
    {1, 4, 5}, {1, 4, 6}, {1, 4, 7}, {1, 4, 8},
    {1, 5, 6}, {1, 5, 7}, {1, 5, 8},
    {1, 6, 7}, {1, 6, 8},
    {1, 7, 8},
    {2, 3, 4}, {2, 3, 5}, {2, 3, 6}, {2, 3, 7}, {2, 3, 8},
    {2, 4, 5}, {2, 4, 6}, {2, 4, 7}, {2, 4, 8},
    {2, 5, 6}, {2, 5, 7}, {2, 5, 8},
    {2, 6, 7}, {2, 6, 8},
    {2, 7, 8},
    {3, 4, 5}, {3, 4, 6}, {3, 4, 7}, {3, 4, 8},
    {3, 5, 6}, {3, 5, 7}, {3, 5, 8},
    {3, 6, 7}, {3, 6, 8},
    {3, 7, 8},
    {4, 5, 6}, {4, 5, 7}, {4, 5, 8},
    {4, 6, 7}, {4, 6, 8},
    {4, 7, 8},
    {5, 6, 7}, {5, 6, 8},
    {5, 7, 8},
    {6, 7, 8}
};

// Cumulative convolution approach matching sktime implementation
// Uses weights alpha=-1 and gamma=+3
void compute_cumulative_convolution_hls(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t C[MAX_TIME_SERIES_LENGTH],
    int_t kernel_idx,
    int_t dilation,
    int_t time_series_length,
    int_t padding
) {
    #pragma HLS INLINE off

    data_t A[MAX_TIME_SERIES_LENGTH];  // A = -X
    data_t G[MAX_TIME_SERIES_LENGTH];  // G = 3X

    #pragma HLS ARRAY_PARTITION variable=A type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=G type=cyclic factor=8

    // Compute A = -X and G = 3X
    COMPUTE_AG: for (int_t i = 0; i < time_series_length; i++) {
        #pragma HLS PIPELINE II=1
        A[i] = (data_t)0.0 - time_series[i];  // -X
        G[i] = time_series[i] + time_series[i] + time_series[i];  // 3X
    }

    // Initialize C_alpha with A
    data_t C_alpha[MAX_TIME_SERIES_LENGTH];
    data_t C_gamma[9][MAX_TIME_SERIES_LENGTH];

    #pragma HLS ARRAY_PARTITION variable=C_alpha type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=C_gamma type=complete dim=1

    INIT_C_ALPHA: for (int_t i = 0; i < time_series_length; i++) {
        #pragma HLS PIPELINE II=1
        C_alpha[i] = A[i];
    }

    // Initialize C_gamma - only middle position (4) has G, others are zero
    INIT_C_GAMMA_OUTER: for (int_t g = 0; g < 9; g++) {
        INIT_C_GAMMA_INNER: for (int_t i = 0; i < time_series_length; i++) {
            #pragma HLS PIPELINE II=1
            C_gamma[g][i] = (g == 4) ? G[i] : (data_t)0.0;
        }
    }

    // Build cumulative arrays
    int_t start = dilation;
    int_t end = time_series_length - padding;

    // First half (gamma_index 0 to 3)
    BUILD_FIRST_HALF: for (int_t gamma_index = 0; gamma_index < 4; gamma_index++) {
        #pragma HLS PIPELINE off

        UPDATE_C_ALPHA_FIRST: for (int_t i = time_series_length - end; i < time_series_length; i++) {
            #pragma HLS PIPELINE II=1
            C_alpha[i] = C_alpha[i] + A[i - (time_series_length - end)];
        }

        UPDATE_C_GAMMA_FIRST: for (int_t i = time_series_length - end; i < time_series_length; i++) {
            #pragma HLS PIPELINE II=1
            C_gamma[gamma_index][i] = G[i - (time_series_length - end)];
        }

        end += dilation;
    }

    start = dilation;

    // Second half (gamma_index 5 to 8)
    BUILD_SECOND_HALF: for (int_t gamma_index = 5; gamma_index < 9; gamma_index++) {
        #pragma HLS PIPELINE off

        UPDATE_C_ALPHA_SECOND: for (int_t i = 0; i < time_series_length - start; i++) {
            #pragma HLS PIPELINE II=1
            C_alpha[i] = C_alpha[i] + A[i + start];
        }

        UPDATE_C_GAMMA_SECOND: for (int_t i = 0; i < time_series_length - start; i++) {
            #pragma HLS PIPELINE II=1
            C_gamma[gamma_index][i] = G[i + start];
        }

        start += dilation;
    }

    // Get kernel indices
    int_t index_0 = kernel_indices[kernel_idx][0];
    int_t index_1 = kernel_indices[kernel_idx][1];
    int_t index_2 = kernel_indices[kernel_idx][2];

    // Compute final C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
    COMPUTE_C: for (int_t i = 0; i < time_series_length; i++) {
        #pragma HLS PIPELINE II=1
        C[i] = C_alpha[i] + C_gamma[index_0][i] + C_gamma[index_1][i] + C_gamma[index_2][i];
    }
}

// HLS-optimized MiniRocket feature extraction using cumulative convolution
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

    // Local arrays for computations
    data_t C[MAX_TIME_SERIES_LENGTH];
    #pragma HLS ARRAY_PARTITION variable=C type=cyclic factor=8

    int_t feature_idx = 0;

    DILATION_LOOP: for (int_t dil_idx = 0; dil_idx < num_dilations; dil_idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8

        int_t dilation = dilations[dil_idx];
        int_t padding = ((9 - 1) * dilation) / 2;
        int_t num_features_this_dilation = num_features_per_dilation[dil_idx];

        // Padding flag based on dilation index
        int_t _padding0 = dil_idx % 2;

        KERNEL_LOOP: for (int_t kernel_idx = 0; kernel_idx < NUM_KERNELS; kernel_idx++) {
            #pragma HLS LOOP_TRIPCOUNT min=84 max=84
            #pragma HLS PIPELINE off

            if (feature_idx >= num_features) break;

            // Compute cumulative convolution
            compute_cumulative_convolution_hls(
                time_series, C, kernel_idx, dilation,
                time_series_length, padding
            );

            // Padding flag for this kernel
            int_t _padding1 = (_padding0 + kernel_idx) % 2;

            // Compute PPV features for this kernel/dilation combination
            FEATURE_LOOP: for (int_t feature_count = 0; feature_count < num_features_this_dilation; feature_count++) {
                #pragma HLS PIPELINE off

                if (feature_idx >= num_features) break;

                data_t bias = biases[feature_idx];
                int_t positive_count = 0;
                int_t total_count = 0;

                // Apply PPV with or without padding
                if (_padding1 == 0) {
                    // No padding - use entire convolution
                    PPV_NO_PAD: for (int_t i = 0; i < time_series_length; i++) {
                        #pragma HLS PIPELINE II=1
                        if (C[i] > bias) {
                            positive_count++;
                        }
                        total_count++;
                    }
                } else {
                    // With padding - exclude padded regions
                    PPV_WITH_PAD: for (int_t i = padding; i < time_series_length - padding; i++) {
                        #pragma HLS PIPELINE II=1
                        if (C[i] > bias) {
                            positive_count++;
                        }
                        total_count++;
                    }
                }

                // Compute PPV feature
                data_t ppv = (total_count > 0) ?
                    ((data_t)positive_count) / ((data_t)total_count) : (data_t)0.0;

                features[feature_idx] = ppv;
                feature_idx++;
            }
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
extern "C" void minirocket_inference_hls_top(
    data_t* time_series_input,      // Input time series
    data_t* prediction_output,      // Output predictions
    data_t* coefficients,           // Model coefficients (flattened)
    data_t* intercept,              // Model intercept
    data_t* scaler_mean,            // Scaler mean values
    data_t* scaler_scale,           // Scaler scale values
    int_t* dilations,               // Dilation values
    int_t* num_features_per_dilation, // Features per dilation
    data_t* biases,                 // Bias values
    int_t time_series_length,
    int_t num_features,
    int_t num_classes,
    int_t num_dilations
) {
    #pragma HLS INTERFACE m_axi port=time_series_input bundle=gmem0 depth=512
    #pragma HLS INTERFACE m_axi port=prediction_output bundle=gmem1 depth=4
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
    
    // Local arrays for processing
    data_t local_time_series[MAX_TIME_SERIES_LENGTH];
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
    
    #pragma HLS ARRAY_PARTITION variable=local_time_series type=cyclic factor=8
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
    
    // Feature extraction
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