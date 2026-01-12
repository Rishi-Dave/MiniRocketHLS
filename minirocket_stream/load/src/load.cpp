#include "../include/minirocket.hpp"
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
        #pragma HLS UNROLL factor=512
        if (j < time_series_length) {
            static data_t sliding_window[KERNEL_SIZE] = {0};
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
                #pragma HLS PIPELINE II=1
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
            #pragma HLS UNROLL
            
            int_t _padding1 = (_padding0 + kernel_idx) % 2;
            int_t feature_idx_for_kernel = feature_idx + kernel_idx * features_this_dilation;

            if (feature_idx_for_kernel >= num_features) {
                //std::cout << "Warning: feature_idx exceeds num_features!" << std::endl;   
                break;
            }
            // Apply kernel convolution
            int_t conv_length;
            apply_kernel_hls(time_series, convolutions, kernel_idx, dilation, 
                           time_series_length, &conv_length);
            
            // Calculate positive proportion of values (PPV)
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
        feature_idx+= features_this_dilation * NUM_KERNELS;
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
    
    SCALE_LOOP: for (int_t i = 0; i < MAX_FEATURES; i++) {
        #pragma HLS UNROLL factor=512
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
    
    // if (num_classes == 2) {
    //     // Binary classification: use single decision function
    //     data_t score = intercept[0];
        
    //     BINARY_FEATURE_LOOP: for (int_t j = 0; j < num_features; j++) {
    //         #pragma HLS PIPELINE II=1
    //         #pragma HLS LOOP_TRIPCOUNT min=100 max=10000
            
    //         score += coefficients[0][j] * scaled_features[j];
    //     }
        
    //     predictions[0] = (data_t)0.0 - score;  // Class 0 score
    //     predictions[1] = score;   // Class 1 score
    // } else {
        // Multi-class classification
    CLASS_LOOP: for (int_t i = 0; i < num_classes; i++) {
        #pragma HLS PIPELINE off
        #pragma HLS LOOP_TRIPCOUNT min=2 max=4
        
        data_t score = intercept[i];
        
        FEATURE_LOOP: for (int_t j = 0; j < MAX_FEATURES; j++) {
            #pragma HLS UNROLL factor=512
            #pragma HLS LOOP_TRIPCOUNT min=100 max=10000
            if (j < num_features) score += coefficients[i][j] * scaled_features[j];
        }
        
        predictions[i] = score;
    }

    if (num_classes == 2) {
        // For binary classification, adjust scores to represent both classes
        data_t binary_score = predictions[0];
        predictions[0] = (data_t)0.0 - binary_score;  // Class 0 score
        predictions[1] = binary_score;   // Class 1 score
    }

    //}
}

// HLS-optimized top-level function for FPGA
extern "C" void minirocket_inference(
    data_t* time_series_input,      // Input time series
    hls::stream<data_t>& output,
    int_t num_values
) {
    #pragma HLS INTERFACE m_axi port=time_series_input bundle=gmem0 depth=512
    #pragma HLS INTERFACE axis port=output 
    #pragma HLS INTERFACE s_axilite port=time_series_length bundle=control
    #pragma HLS INTERFACE s_axilite port=num_values bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    for (int i = 0; i < num_values; i++) {
        #pragma HLS PIPELINE
        output.write(time_series_input[i]);
    }
}