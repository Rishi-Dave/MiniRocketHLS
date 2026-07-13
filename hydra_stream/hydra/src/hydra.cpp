#include "../include/hydra.hpp"

/**
 * Apply single convolutional kernel with dilation to time series
 *
 * Implements sliding window convolution:
 *   output[t] = sum(window[i] * weights[i]) + bias
 *
 * Uses array partitioning and pipelining for optimal throughput.
 */
void apply_kernel_hls(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t weights[KERNEL_SIZE],
    data_t bias,
    int_t dilation,
    int_t length,
    data_t output[MAX_TIME_SERIES_LENGTH],
    int_t& output_length
) {
    #pragma HLS INLINE off

    // Effective kernel span with dilation
    int_t kernel_span = (KERNEL_SIZE - 1) * dilation + 1;

    if (length < kernel_span) {
        output_length = 0;
        return;
    }

    output_length = length - kernel_span + 1;

    // Sliding window buffer - fully partitioned for parallelism
    data_t sliding_window[KERNEL_SIZE];
    #pragma HLS ARRAY_PARTITION variable=sliding_window complete

    // Main convolution loop
    CONV_LOOP: for (int_t t = 0; t < output_length; t++) {
        #pragma HLS PIPELINE II=1

        // Load sliding window with dilation
        WINDOW_LOAD: for (int_t w = 0; w < KERNEL_SIZE; w++) {
            #pragma HLS UNROLL
            sliding_window[w] = time_series[t + w * dilation];
        }

        // Compute dot product
        data_t sum = bias;
        DOT_PRODUCT: for (int_t w = 0; w < KERNEL_SIZE; w++) {
            #pragma HLS UNROLL
            sum += sliding_window[w] * weights[w];
        }

        output[t] = sum;
    }
}

/**
 * Extract HYDRA features from time series
 *
 * For each of 512 kernels:
 *   1. Apply convolution with dilation
 *   2. Extract max pooling (maximum value)
 *   3. Extract global mean
 *
 * Total features: 512 kernels × 2 pooling operators = 1,024 features
 */
void hydra_feature_extraction_hls(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    data_t kernel_weights[NUM_KERNELS * KERNEL_SIZE],
    data_t biases[NUM_KERNELS],
    int_t dilations[NUM_KERNELS],
    int_t length,
    int_t& num_features
) {
    #pragma HLS INLINE off

    int_t feature_idx = 0;

    // Buffer for convolution output
    data_t conv_output[MAX_TIME_SERIES_LENGTH];

    // Current kernel weights
    data_t current_weights[KERNEL_SIZE];
    #pragma HLS ARRAY_PARTITION variable=current_weights complete

    // Process each kernel
    KERNEL_LOOP: for (int_t k = 0; k < NUM_KERNELS; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=512 max=512

        // Load kernel weights
        WEIGHT_LOAD: for (int_t w = 0; w < KERNEL_SIZE; w++) {
            #pragma HLS UNROLL
            current_weights[w] = kernel_weights[k * KERNEL_SIZE + w];
        }

        // Apply convolution
        int_t conv_length;
        apply_kernel_hls(
            time_series,
            current_weights,
            biases[k],
            dilations[k],
            length,
            conv_output,
            conv_length
        );

        // Extract pooling features
        data_t max_val, mean_val;
        compute_two_pooling_operators(
            conv_output,
            conv_length,
            max_val,
            mean_val
        );

        // Store features
        features[feature_idx++] = max_val;
        features[feature_idx++] = mean_val;
    }

    num_features = feature_idx;
}

/**
 * Apply StandardScaler normalization
 *
 * Normalizes features: (x - mean) / scale
 * In-place operation modifies features array
 */
void apply_scaler_hls(
    data_t features[MAX_FEATURES],
    data_t scaler_mean[MAX_FEATURES],
    data_t scaler_scale[MAX_FEATURES],
    int_t num_features
) {
    #pragma HLS INLINE off

    SCALE_LOOP: for (int_t i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1

        data_t mean = scaler_mean[i];
        data_t scale = scaler_scale[i];

        // Avoid division by zero
        if (scale > 1e-8 || scale < -1e-8) {
            features[i] = (features[i] - mean) / scale;
        } else {
            features[i] = 0.0;
        }
    }
}

/**
 * Ridge linear classifier prediction
 *
 * Computes: prediction = features @ coefficients.T + intercept
 *
 * For each class:
 *   prediction[c] = sum(features[f] * coefficients[f, c]) + intercept[c]
 */
void linear_classifier_predict_hls(
    data_t features[MAX_FEATURES],
    data_t coefficients[MAX_FEATURES * MAX_CLASSES],
    data_t intercept[MAX_CLASSES],
    int_t num_features,
    int_t num_classes,
    data_t prediction[MAX_CLASSES]
) {
    #pragma HLS INLINE off

    // Compute prediction for each class
    CLASS_LOOP: for (int_t c = 0; c < num_classes; c++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=2 max=10

        data_t score = intercept[c];

        // Dot product: features @ coefficients[:, c]
        FEATURE_LOOP: for (int_t f = 0; f < num_features; f++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1024 max=2048

            // Coefficients stored in row-major: [num_features][num_classes]
            score += features[f] * coefficients[f * num_classes + c];
        }

        prediction[c] = score;
    }
}

/**
 * Main HYDRA inference kernel
 *
 * Complete pipeline:
 *   1. Load time series from HBM
 *   2. Extract dictionary-based features
 *   3. Normalize features
 *   4. Classify using Ridge
 *   5. Write predictions to HBM
 *
 * This is the entry point called by the OpenCL host application.
 */
extern "C" void hydra_inference(
    data_t* time_series_input,
    data_t* prediction_output,
    data_t* coefficients,
    data_t* intercept,
    data_t* scaler_mean,
    data_t* scaler_scale,
    data_t* kernel_weights,
    data_t* biases,
    int_t* dilations,
    int_t time_series_length,
    int_t num_features,
    int_t num_classes,
    int_t num_groups
) {
    // HLS interface pragmas for AXI ports
    #pragma HLS INTERFACE m_axi port=time_series_input bundle=gmem0 offset=slave depth=512
    #pragma HLS INTERFACE m_axi port=prediction_output bundle=gmem1 offset=slave depth=10
    #pragma HLS INTERFACE m_axi port=coefficients bundle=gmem2 offset=slave depth=20480
    #pragma HLS INTERFACE m_axi port=intercept bundle=gmem3 offset=slave depth=10
    #pragma HLS INTERFACE m_axi port=scaler_mean bundle=gmem4 offset=slave depth=2048
    #pragma HLS INTERFACE m_axi port=scaler_scale bundle=gmem5 offset=slave depth=2048
    #pragma HLS INTERFACE m_axi port=kernel_weights bundle=gmem6 offset=slave depth=4608
    #pragma HLS INTERFACE m_axi port=biases bundle=gmem7 offset=slave depth=512
    #pragma HLS INTERFACE m_axi port=dilations bundle=gmem8 offset=slave depth=512

    #pragma HLS INTERFACE s_axilite port=time_series_length
    #pragma HLS INTERFACE s_axilite port=num_features
    #pragma HLS INTERFACE s_axilite port=num_classes
    #pragma HLS INTERFACE s_axilite port=num_groups
    #pragma HLS INTERFACE s_axilite port=return

    // Local buffers for computation
    data_t local_time_series[MAX_TIME_SERIES_LENGTH];
    data_t local_features[MAX_FEATURES];
    data_t local_prediction[MAX_CLASSES];

    data_t local_kernel_weights[NUM_KERNELS * KERNEL_SIZE];
    data_t local_biases[NUM_KERNELS];
    int_t local_dilations[NUM_KERNELS];

    data_t local_scaler_mean[MAX_FEATURES];
    data_t local_scaler_scale[MAX_FEATURES];
    data_t local_coefficients[MAX_FEATURES * MAX_CLASSES];
    data_t local_intercept[MAX_CLASSES];

    // Load time series from HBM to local memory
    LOAD_TS: for (int_t i = 0; i < time_series_length; i++) {
        #pragma HLS PIPELINE II=1
        local_time_series[i] = time_series_input[i];
    }

    // Load model parameters from HBM
    LOAD_KERNELS: for (int_t i = 0; i < NUM_KERNELS * KERNEL_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        local_kernel_weights[i] = kernel_weights[i];
    }

    LOAD_BIASES: for (int_t i = 0; i < NUM_KERNELS; i++) {
        #pragma HLS PIPELINE II=1
        local_biases[i] = biases[i];
    }

    LOAD_DILATIONS: for (int_t i = 0; i < NUM_KERNELS; i++) {
        #pragma HLS PIPELINE II=1
        local_dilations[i] = dilations[i];
    }

    LOAD_SCALER_MEAN: for (int_t i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
        local_scaler_mean[i] = scaler_mean[i];
    }

    LOAD_SCALER_SCALE: for (int_t i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
        local_scaler_scale[i] = scaler_scale[i];
    }

    LOAD_COEF: for (int_t i = 0; i < num_features * num_classes; i++) {
        #pragma HLS PIPELINE II=1
        local_coefficients[i] = coefficients[i];
    }

    LOAD_INTERCEPT: for (int_t i = 0; i < num_classes; i++) {
        #pragma HLS PIPELINE II=1
        local_intercept[i] = intercept[i];
    }

    // Feature extraction
    int_t extracted_features;
    hydra_feature_extraction_hls(
        local_time_series,
        local_features,
        local_kernel_weights,
        local_biases,
        local_dilations,
        time_series_length,
        extracted_features
    );

    // Feature normalization
    apply_scaler_hls(
        local_features,
        local_scaler_mean,
        local_scaler_scale,
        num_features
    );

    // Classification
    linear_classifier_predict_hls(
        local_features,
        local_coefficients,
        local_intercept,
        num_features,
        num_classes,
        local_prediction
    );

    // Write predictions to HBM
    WRITE_PRED: for (int_t i = 0; i < num_classes; i++) {
        #pragma HLS PIPELINE II=1
        prediction_output[i] = local_prediction[i];
    }
}
