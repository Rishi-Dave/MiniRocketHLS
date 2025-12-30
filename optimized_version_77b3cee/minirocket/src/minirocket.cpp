#include "../include/minirocket.hpp"
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
    
    // Local arrays for computations
    data_t convolutions[MAX_TIME_SERIES_LENGTH];
    #pragma HLS ARRAY_PARTITION variable=convolutions type=cyclic factor=8
    
    int_t feature_idx = 0;
    
    DILATION_LOOP: for (int_t dil_idx = 0; dil_idx < num_dilations; dil_idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8
        
        int_t dilation = dilations[dil_idx];
        int_t features_this_dilation = num_features_per_dilation[dil_idx];
        
        KERNEL_LOOP: for (int_t kernel_idx = 0; kernel_idx < NUM_KERNELS; kernel_idx++) {
            #pragma HLS LOOP_TRIPCOUNT min=84 max=84
            #pragma HLS PIPELINE off
            
            if (feature_idx >= num_features) {
                //std::cout << "Warning: feature_idx exceeds num_features!" << std::endl;   
                break;
            }
            // Apply kernel convolution
            int_t conv_length;
            apply_kernel_hls(time_series, convolutions, kernel_idx, dilation, 
                           time_series_length, &conv_length);
            
            // Calculate positive proportion of values (PPV)
            for (int_t f = 0; f < features_this_dilation; f++) {
                data_t bias = biases[feature_idx + f];
                int_t positive_count = 0;
                
                PPV_LOOP: for (int_t i = 0; i < time_series_length; i++) {
                    #pragma HLS PIPELINE II=1
                    if (convolutions[i] > bias) {
                        positive_count++;
                    }
                }
                
                // Compute PPV feature
                data_t ppv = (data_t)positive_count / (data_t)time_series_length;
                features[feature_idx + f] = ppv;
                
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
extern "C" void minirocket_inference(
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