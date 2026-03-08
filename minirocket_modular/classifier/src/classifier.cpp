#include "classifier.hpp"

// Ridge regression linear classifier
void linear_classifier_predict_hls(
    data_t scaled_features[MAX_FEATURES],
    data_t predictions[MAX_CLASSES],
    data_t coefficients[MAX_CLASSES][MAX_FEATURES],
    data_t intercept[MAX_CLASSES],
    int_t num_features,
    int_t num_classes
) {
    #pragma HLS INLINE off

    CLASS_LOOP: for (int_t i = 0; i < num_classes; i++) {
        #pragma HLS PIPELINE off
        #pragma HLS LOOP_TRIPCOUNT min=2 max=4

        data_t score = intercept[i];

        FEATURE_LOOP: for (int_t j = 0; j < MAX_FEATURES; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=100 max=1024
            if (j < num_features) score += coefficients[i][j] * scaled_features[j];
        }

        predictions[i] = score;
    }

    if (num_classes == 2) {
        data_t binary_score = predictions[0];
        predictions[0] = (data_t)0.0 - binary_score;
        predictions[1] = binary_score;
    }
}

// Top-level K3 kernel: Classifier with AXI-Stream input and HBM output
extern "C" void classifier(
    data_t* coefficients,
    data_t* intercept,
    data_t* prediction_output,
    hls::stream<data_t>& scaled_features_in,
    int_t num_features,
    int_t num_classes
) {
    #pragma HLS INTERFACE m_axi port=coefficients bundle=gmem6 depth=16384
    #pragma HLS INTERFACE m_axi port=intercept bundle=gmem7 depth=16
    #pragma HLS INTERFACE m_axi port=prediction_output bundle=gmem8 depth=16

    #pragma HLS INTERFACE axis port=scaled_features_in

    #pragma HLS INTERFACE s_axilite port=num_features bundle=control
    #pragma HLS INTERFACE s_axilite port=num_classes bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS DATAFLOW

    // Local arrays
    data_t local_scaled_features[MAX_FEATURES];
    data_t local_predictions[MAX_CLASSES];
    data_t local_coefficients[MAX_CLASSES][MAX_FEATURES];
    data_t local_intercept[MAX_CLASSES];

    #pragma HLS ARRAY_PARTITION variable=local_scaled_features type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=local_predictions type=complete
    #pragma HLS ARRAY_PARTITION variable=local_coefficients type=block factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_intercept type=complete

    // Copy model parameters from HBM
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

    // Read scaled features from AXI-Stream
    STREAM_IN: for (int_t i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
        local_scaled_features[i] = scaled_features_in.read();
    }

    // Run classification
    linear_classifier_predict_hls(
        local_scaled_features,
        local_predictions,
        local_coefficients,
        local_intercept,
        num_features,
        num_classes
    );

    // Write predictions to HBM
    COPY_OUTPUT: for (int_t i = 0; i < num_classes; i++) {
        #pragma HLS PIPELINE II=1
        prediction_output[i] = local_predictions[i];
    }
}
