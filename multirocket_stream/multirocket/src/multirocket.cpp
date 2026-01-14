#include "../include/multirocket.hpp"
#include <cstring>


static ap_uint<1> weights[NUM_KERNELS][KERNEL_SIZE] = {
    #include "../include/weights01.txt"
};

static data_t convolutions[MAX_DILATIONS][NUM_KERNELS][MAX_TIME_SERIES_LENGTH];

data_t optimized_fp_multiply(ap_uint<1> x, data_t y) {
    #pragma HLS PIPELINE

    float_num_t yb;
    yb.fp_num = y;

    yb.sign = ~(yb.sign ^ x);
    yb.bexp = yb.bexp + x;
    yb.mant = yb.mant;
    
    return yb.fp_num;
}


data_t single_convolution(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    int_t dilation,
    int_t time_series_length,
    int_t kernel_idx,
    int_t center_idx
) {
    #pragma HLS INLINE off
    
    data_t sliding_window[KERNEL_SIZE] = {0};
    // #pragma HLS ARRAY_PARTITION variable=sliding_window complete
    // #pragma HLS ARRAY_PARTITION variable=weights complete

    int i = 0;
    for (int k = -4; k <= 4; k++) {
        if (center_idx + k * dilation < 0 || center_idx + k * dilation >= time_series_length) {
            sliding_window[i] = 0.0;
        } else {
            sliding_window[i] = time_series[center_idx + k * dilation];
        }
        i++;
    }

    data_t value = 0.0;
    KERNEL_LOOP: for (int_t k = 0; k < KERNEL_SIZE; k++) {
        #pragma HLS PIPELINE II=1
        value += optimized_fp_multiply(weights[kernel_idx][k], sliding_window[k]);
    }
    
    return value;
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
    // data_t convolutions[MAX_TIME_SERIES_LENGTH];
    #pragma HLS ARRAY_PARTITION variable=convolutions complete dim=2
    
    int_t feature_idx = 0;
    int_t hk = KERNEL_SIZE / 2;

    DILATION_LOOP: for (int_t dil_idx = 0; dil_idx < num_dilations; dil_idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8

        int_t dilation = dilations[dil_idx];
        int_t features_this_dilation = num_features_per_dilation[dil_idx];
        int_t _padding0 = dil_idx % 2;
        int_t padding = ((9-1) * dilation) / 2;
        int_t shift = dilation - 1;

        KERNEL_LOOP: for (int_t kernel_idx = 0; kernel_idx < NUM_KERNELS; kernel_idx++) {
            #pragma HLS LOOP_TRIPCOUNT min=84 max=84
            #pragma HLS PIPELINE off

            int_t _padding1 = (_padding0 + kernel_idx) % 2;

            // Shift over indicies that don't need recomputation due to dilation != 1 (beginning)
            
            for (int_t i = 0; i <= hk; i++) {
                for (int_t j = 1; j <= shift; j++) {
                    convolutions[dil_idx][kernel_idx][i * dilation + j] = convolutions[dil_idx][kernel_idx][i * dilation + j - 1];
                }
                convolutions[dil_idx][kernel_idx][i * dilation] = single_convolution(time_series, dilation, time_series_length, kernel_idx, i * dilation);
            }

            // Shift over indicies that don't need recomputation due to dilation != 1 (end)
            for (int_t i = 0; i <= hk; i++) {
                for (int_t j = 1; j <= shift; j++) {
                    convolutions[dil_idx][kernel_idx][time_series_length - i * dilation - j] = convolutions[dil_idx][kernel_idx][time_series_length - i * dilation - j + 1];
                }
                convolutions[dil_idx][kernel_idx][time_series_length - i * dilation] = single_convolution(time_series, dilation, time_series_length, kernel_idx, time_series_length - i * dilation);
            }

            // Shifting over center of convolution not affected by new data point
            for (int_t i = dilation * hk + 1; i < time_series_length - dilation * hk; i++) {
                convolutions[dil_idx][kernel_idx][i] = convolutions[dil_idx][kernel_idx][i-1]; 
            }
            // Calculate positive proportion of values (PPV)
            for (int_t f = 0; f < features_this_dilation; f++) {

                int_t current_feature_idx = feature_idx + f;
                data_t bias = biases[current_feature_idx];
                PoolingStats stats;

                if (_padding1 == 0) {
                    compute_four_pooling_operators(convolutions[dil_idx][kernel_idx], bias, 0, time_series_length, &stats);
                } else {
                    compute_four_pooling_operators(convolutions[dil_idx][kernel_idx], bias, padding, time_series_length - padding, &stats);
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
    hls::stream<data_t> &time_series_input,      // Input time series
    hls::stream<data_t> &prediction_output,      // Output predictions
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
    #pragma HLS INTERFACE axis port=time_series_input
    #pragma HLS INTERFACE axis port=prediction_output
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
    static data_t local_time_series[2][MAX_TIME_SERIES_LENGTH] = {0};
    data_t local_features[MAX_FEATURES];
    data_t local_scaled_features[MAX_FEATURES];
    data_t local_predictions[MAX_CLASSES];
    data_t local_coefficients[MAX_CLASSES][MAX_FEATURES];
    data_t local_scaler_mean[MAX_FEATURES];
    data_t local_scaler_scale[MAX_FEATURES];
    data_t local_intercept[MAX_CLASSES];

    int_t local_dilations[2][MAX_DILATIONS];
    int_t local_num_features_per_dilation[2][MAX_DILATIONS];
    data_t local_biases[2][MAX_FEATURES];

    #pragma HLS ARRAY_PARTITION variable=local_time_series complete dim=1
    // #pragma HLS ARRAY_PARTITION variable=local_scaled_features type=cyclic factor=8
    // #pragma HLS ARRAY_PARTITION variable=local_predictions type=complete
    // #pragma HLS ARRAY_PARTITION variable=local_coefficients type=block factor=4 dim=1
    // #pragma HLS ARRAY_PARTITION variable=local_intercept type=complete
    
    // Copy input data to local arrays
    
    COPY_DILATIONS: for (int_t i = 0; i < num_dilations_0; i++) {
        //#pragma HLS PIPELINE II=1
        local_dilations[0][i] = dilations_0[i];
        local_num_features_per_dilation[0][i] = num_features_per_dilation_0[i];
        local_dilations[1][i] = dilations_1[i];
        local_num_features_per_dilation[1][i] = num_features_per_dilation_1[i];
    }
    
    
    for (int i = 0; i < num_features_0; i++) {
        //#pragma HLS PIPELINE II=1
        local_biases[0][i] = biases_0[i];
        local_biases[1][i] = biases_1[i];
    }

    COPY_BIASES: for (int_t i = 0; i < num_features; i++) {
        //#pragma HLS PIPELINE II=1
        local_scaler_mean[i] = scaler_mean[i];
        local_scaler_scale[i] = scaler_scale[i];
    }
    
    COPY_INTERCEPT: for (int_t i = 0; i < num_classes; i++) {
        //#pragma HLS PIPELINE II=1
        local_intercept[i] = intercept[i];
    }
    
    COPY_COEF: for (int_t i = 0; i < num_classes; i++) {
        for (int_t j = 0; j < num_features; j++) {
            //#pragma HLS PIPELINE II=1
            local_coefficients[i][j] = coefficients[i * num_features + j];
        }
    }
    
    int_t start = ((num_features_1 + num_features_0) * n_feature_per_kernel) / 2;
   
#if BUILD == 1
    while (true) {
#endif 
        if (!time_series_input.empty()) {

            data_t v = time_series_input.read();

            // Read time series data from AXI stream
            for (int_t i = time_series_length; i > 0; i--) {
                #pragma HLS PIPELINE II=1   
                local_time_series[0][i] = local_time_series[0][i-1];
            }
            local_time_series[0][0] = v;
            
            // Compute differenced time series
            for (int_t i = 0; i < time_series_length - 1; i++) {
                #pragma HLS PIPELINE II=1   
                local_time_series[1][i] = local_time_series[0][i+1] - local_time_series[0][i];
            }
            
            // std::cout << "Input OG Series: [";
            // for (int_t i = 0; i < time_series_length; i++) {
            //     std::cout << local_time_series[0][i] << " ";
            // }
            // std::cout << "]" << std::endl;

            // std::cout << "Input Diff Series: [";
            // for (int_t i = 0; i < time_series_length-1; i++) {
            //     std::cout << local_time_series[1][i] << " ";
            // }
            // std::cout << "]" << std::endl;

            // Feature extraction
            for (int i = 0; i < NUM_REPRESENTATIONS; i++) {
                multirocket_feature_extraction_hls(
                    local_time_series[i],
                    local_features,
                    local_dilations[i],
                    local_num_features_per_dilation[i],
                    local_biases[i],
                    (i == 0) ? time_series_length : (int_t) (time_series_length - 1),
                    (i == 0) ? num_dilations_0 : num_dilations_1,
                    (i == 0) ? num_features_0 : num_features_1,
                    n_feature_per_kernel,
                    (i == 0) ? (int_t) 0 : start
                );
            }

            // std::cout << "Extracted Features: [";
            // for (int i = 0; i < num_features; i++) {
            //     std::cout << local_features[i] << " ";
            // }
            // std::cout << std::endl;

            // Apply scaling
            apply_scaler_hls(
                local_features,
                local_scaled_features,
                local_scaler_mean,
                local_scaler_scale,
                num_features
            );
            
            // std::cout << "Scaled Features: [";
            // for (int i = 0; i < num_features; i++) {
            //     std::cout << local_scaled_features[i] << " ";
            // }
            // std::cout << std::endl;

            // Linear classification
            linear_classifier_predict_hls(
                local_scaled_features,
                local_predictions,
                local_coefficients,
                local_intercept,
                num_features,
                num_classes
            );
            
            // std::cout << "Predictions: [";
            // for (int i = 0; i < num_classes; i++) {
            //     std::cout << local_predictions[i] << " ";
            // }
            // std::cout << "]" << std::endl;

            // Copy output
            COPY_OUTPUT: for (int_t i = 0; i < num_classes; i++) {
                #pragma HLS PIPELINE II=1
                prediction_output.write(local_predictions[i]);
            }
        }
#if BUILD == 1
    }
#endif 


}