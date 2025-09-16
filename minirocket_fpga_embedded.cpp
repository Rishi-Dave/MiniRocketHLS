#include "minirocket_inference_hls.h"
#include "minirocket_model_constants.h"

// FPGA-ready MiniRocket with embedded model constants
// No external memory loading required - all parameters in BRAM/LUTs

extern "C" void minirocket_fpga_inference(
    data_t* input_time_series,      // Input: time series data
    data_t* output_predictions,     // Output: class predictions
    int_t time_series_length        // Input length
) {
    #pragma HLS INTERFACE m_axi port=input_time_series bundle=gmem0 depth=512
    #pragma HLS INTERFACE m_axi port=output_predictions bundle=gmem1 depth=4
    #pragma HLS INTERFACE s_axilite port=time_series_length bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    // Local arrays for computation pipeline
    static data_t local_time_series[MAX_TIME_SERIES_LENGTH];
    static data_t local_features[MAX_FEATURES];
    static data_t local_scaled_features[MAX_FEATURES];
    static data_t local_predictions[MAX_CLASSES];
    
    #pragma HLS ARRAY_PARTITION variable=local_time_series type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=local_features type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=local_scaled_features type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=local_predictions type=complete
    
    // Copy input data
    COPY_INPUT: for (int_t i = 0; i < time_series_length; i++) {
        #pragma HLS PIPELINE II=1
        local_time_series[i] = input_time_series[i];
    }
    
    // Feature extraction using embedded constants
    minirocket_feature_extraction_hls(
        local_time_series,
        local_features,
        (int_t*)MODEL_DILATIONS,                    // Embedded dilations
        (int_t*)MODEL_FEATURES_PER_DILATION,       // Embedded features per dilation
        (data_t*)MODEL_BIASES,                     // Embedded biases
        time_series_length,
        MODEL_NUM_DILATIONS,
        MODEL_NUM_FEATURES
    );
    
    // Scaling using embedded constants
    apply_scaler_hls(
        local_features,
        local_scaled_features,
        (data_t*)MODEL_SCALER_MEAN,                // Embedded scaler mean
        (data_t*)MODEL_SCALER_SCALE,               // Embedded scaler scale
        MODEL_NUM_FEATURES
    );
    
    // Classification using embedded constants
    linear_classifier_predict_hls(
        local_scaled_features,
        local_predictions,
        (data_t(*)[MAX_FEATURES])MODEL_COEFFICIENTS, // Embedded coefficients
        (data_t*)MODEL_INTERCEPT,                   // Embedded intercept
        MODEL_NUM_FEATURES,
        MODEL_NUM_CLASSES
    );
    
    // Copy output predictions
    COPY_OUTPUT: for (int_t i = 0; i < MODEL_NUM_CLASSES; i++) {
        #pragma HLS PIPELINE II=1
        output_predictions[i] = local_predictions[i];
    }
}

// Simple wrapper for single prediction (most common FPGA use case)
extern "C" int_t minirocket_predict_class(
    data_t* input_time_series,
    int_t time_series_length
) {
    #pragma HLS INTERFACE m_axi port=input_time_series bundle=gmem0 depth=512
    #pragma HLS INTERFACE s_axilite port=time_series_length bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    data_t predictions[MAX_CLASSES];
    
    // Run full inference
    minirocket_fpga_inference(input_time_series, predictions, time_series_length);
    
    // Find max prediction
    int_t predicted_class = 0;
    data_t max_score = predictions[0];
    
    FIND_MAX: for (int_t i = 1; i < MODEL_NUM_CLASSES; i++) {
        #pragma HLS UNROLL
        if (predictions[i] > max_score) {
            max_score = predictions[i];
            predicted_class = i;
        }
    }
    
    return predicted_class;
}