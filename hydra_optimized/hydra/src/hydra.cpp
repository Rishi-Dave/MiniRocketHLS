#include "../include/hydra.hpp"

/**
 * Optimized HYDRA feature extraction — batch-parallel, UNROLL=16
 *
 * Loop structure:
 *   BATCH_LOOP(32) → CONV_POOL_LOOP(conv_length, II=1) → PARALLEL_KERNELS(16, UNROLL)
 *
 * IMPORTANT: Host must sort kernels by dilation so that all 16 kernels
 * within each batch share the same dilation value. This enables sharing
 * the sliding window reads across the batch.
 *
 * Key optimizations:
 * 1. ap_fixed<32,16> → 1-cycle accumulation (vs 7-cycle float)
 * 2. 5 time series BRAM copies → conflict-free 9-port reads
 * 3. UNROLL=16 → 16 kernels processed per clock cycle
 * 4. Shared sliding window across batch (requires uniform dilation per batch)
 * 5. acc_t<48,32> running_sum → overflow-safe for 5000+ timestep series
 * 6. Fused conv+pooling in single pass
 */
void hydra_feature_extraction_hls(
    data_t ts0[MAX_TIME_SERIES_LENGTH],
    data_t ts1[MAX_TIME_SERIES_LENGTH],
    data_t ts2[MAX_TIME_SERIES_LENGTH],
    data_t ts3[MAX_TIME_SERIES_LENGTH],
    data_t ts4[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    data_t local_weights[NUM_KERNELS][KERNEL_SIZE],
    data_t biases[NUM_KERNELS],
    int_t local_dilations[NUM_KERNELS],
    int_t time_series_length
) {
    #pragma HLS INLINE off

    // Weight array partitioning: enable parallel access for UNROLL_FACTOR kernels
    #pragma HLS ARRAY_PARTITION variable=local_weights complete dim=2
    #pragma HLS ARRAY_PARTITION variable=local_weights cyclic factor=UNROLL_FACTOR dim=1

    // Process all kernels in batches of UNROLL_FACTOR
    // Host sorts kernels by dilation, so all kernels in a batch share the same dilation
    BATCH_LOOP: for (int_t batch = 0; batch < NUM_BATCHES; batch++) {
        #pragma HLS LOOP_TRIPCOUNT min=32 max=32

        int_t kg = batch * UNROLL_FACTOR;  // base kernel index for this batch

        // Get dilation from first kernel in batch (all same after host sorting)
        int_t dilation = local_dilations[kg];
        int_t kernel_span = (KERNEL_SIZE - 1) * dilation + 1;
        int_t conv_length = time_series_length - kernel_span + 1;

        // Pre-load biases into registers
        data_t group_biases[UNROLL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=group_biases complete

        LOAD_BIASES: for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
            #pragma HLS UNROLL
            group_biases[ki] = biases[kg + ki];
        }

        // Pre-load weights into fully-partitioned register array
        data_t reg_weights[UNROLL_FACTOR][KERNEL_SIZE];
        #pragma HLS ARRAY_PARTITION variable=reg_weights complete dim=0

        PRELOAD_WEIGHTS: for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
            #pragma HLS UNROLL
            PRELOAD_W_INNER: for (int_t w = 0; w < KERNEL_SIZE; w++) {
                #pragma HLS UNROLL
                reg_weights[ki][w] = local_weights[kg + ki][w];
            }
        }

        // Initialize accumulators — fully in registers
        acc_t running_sum[UNROLL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=running_sum complete
        data_t running_max[UNROLL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=running_max complete

        INIT_ACC: for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
            #pragma HLS UNROLL
            running_max[ki] = (data_t)(-32000);
            running_sum[ki] = 0;
        }

        // Fused conv + pooling: single pass through time series
        CONV_POOL_LOOP: for (int_t t = 0; t < conv_length; t++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=6 max=5112

            // Read 9 sliding window values from 5 BRAMs (2 ports each = 10)
            data_t sw[KERNEL_SIZE];
            #pragma HLS ARRAY_PARTITION variable=sw complete

            sw[0] = ts0[t];
            sw[1] = ts0[t + dilation];
            sw[2] = ts1[t + 2 * dilation];
            sw[3] = ts1[t + 3 * dilation];
            sw[4] = ts2[t + 4 * dilation];
            sw[5] = ts2[t + 5 * dilation];
            sw[6] = ts3[t + 6 * dilation];
            sw[7] = ts3[t + 7 * dilation];
            sw[8] = ts4[t + 8 * dilation];

            // Process UNROLL_FACTOR kernels in parallel
            PARALLEL_KERNELS: for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
                #pragma HLS UNROLL

                // 9-tap dot product (fully unrolled)
                data_t value = group_biases[ki];
                DOT_PRODUCT: for (int_t w = 0; w < KERNEL_SIZE; w++) {
                    #pragma HLS UNROLL
                    value += sw[w] * reg_weights[ki][w];
                }

                // Streaming max pooling
                if (value > running_max[ki]) {
                    running_max[ki] = value;
                }

                // Streaming sum for mean
                running_sum[ki] += (acc_t)value;
            }
        }

        // Precompute mean values
        data_t mean_vals[UNROLL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=mean_vals complete

        if (conv_length > 0) {
            acc_t inv_conv_length = (acc_t)1 / (acc_t)conv_length;
            COMPUTE_MEAN: for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
                #pragma HLS UNROLL
                mean_vals[ki] = (data_t)(running_sum[ki] * inv_conv_length);
            }
        } else {
            ZERO_MEAN: for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
                #pragma HLS UNROLL
                mean_vals[ki] = (data_t)0;
            }
        }

        // Write 2 features per kernel: max and mean
        WRITE_FEATURES: for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
            #pragma HLS PIPELINE II=1
            int_t abs_k = kg + ki;
            int_t feat_idx = abs_k * 2;

            features[feat_idx]     = (conv_length > 0) ? running_max[ki] : (data_t)0;
            features[feat_idx + 1] = mean_vals[ki];
        }
    }
}

/**
 * Apply StandardScaler normalization
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
        #pragma HLS LOOP_TRIPCOUNT min=1024 max=2048

        data_t scale = scaler_scale[i];
        data_t inv_s = (scale != (data_t)0) ? (data_t)((acc_t)1 / (acc_t)scale) : (data_t)0;
        features[i] = (features[i] - scaler_mean[i]) * inv_s;
    }
}

/**
 * Ridge linear classifier prediction
 * Coefficients in class-major layout: coefficients[c * num_features + f]
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

    CLASS_LOOP: for (int_t c = 0; c < num_classes; c++) {
        #pragma HLS LOOP_TRIPCOUNT min=2 max=10

        acc_t score = (acc_t)intercept[c];

        FEATURE_LOOP: for (int_t f = 0; f < num_features; f++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1024 max=2048

            // Class-major layout: stride-1 access on coefficients for fixed c
            score += (acc_t)features[f] * (acc_t)coefficients[c * num_features + f];
        }

        prediction[c] = (data_t)score;
    }
}

/**
 * Main HYDRA inference kernel — OpenCL entry point
 *
 * IMPORTANT: Host must sort kernels by dilation value before sending data.
 * This ensures all kernels within each UNROLL batch share the same dilation.
 * The host must also reorder scaler/coefficient arrays to match the sorted
 * feature order.
 *
 * AXI ports use float* for host compatibility.
 * Internal computation uses ap_fixed<32,16> with acc_t<48,32> accumulators.
 */
extern "C" void hydra_inference(
    float* time_series_input,
    float* prediction_output,
    float* coefficients,
    float* intercept,
    float* scaler_mean,
    float* scaler_scale,
    float* kernel_weights,
    float* biases,
    int* dilations,
    int time_series_length,
    int num_features,
    int num_classes,
    int num_groups
) {
    #pragma HLS INTERFACE m_axi port=time_series_input bundle=gmem0 offset=slave depth=5120
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

    // ---- Local buffers ----

    // 5 time series BRAM copies for conflict-free 9-port reads
    data_t local_ts0[MAX_TIME_SERIES_LENGTH];
    data_t local_ts1[MAX_TIME_SERIES_LENGTH];
    data_t local_ts2[MAX_TIME_SERIES_LENGTH];
    data_t local_ts3[MAX_TIME_SERIES_LENGTH];
    data_t local_ts4[MAX_TIME_SERIES_LENGTH];

    data_t local_weights[NUM_KERNELS][KERNEL_SIZE];
    data_t local_biases[NUM_KERNELS];
    data_t local_features[MAX_FEATURES];
    data_t local_prediction[MAX_CLASSES];

    data_t local_scaler_mean[MAX_FEATURES];
    data_t local_scaler_scale[MAX_FEATURES];
    data_t local_coefficients[MAX_FEATURES * MAX_CLASSES];
    data_t local_intercept[MAX_CLASSES];

    int_t local_dilations[NUM_KERNELS];

    // ---- Load from HBM (float → ap_fixed conversion) ----

    COPY_TS: for (int i = 0; i < time_series_length; i++) {
        #pragma HLS PIPELINE II=1
        data_t val = (data_t)time_series_input[i];
        local_ts0[i] = val;
        local_ts1[i] = val;
        local_ts2[i] = val;
        local_ts3[i] = val;
        local_ts4[i] = val;
    }

    LOAD_WEIGHTS: for (int i = 0; i < NUM_KERNELS * KERNEL_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        local_weights[i / KERNEL_SIZE][i % KERNEL_SIZE] = (data_t)kernel_weights[i];
    }

    LOAD_BIASES: for (int i = 0; i < NUM_KERNELS; i++) {
        #pragma HLS PIPELINE II=1
        local_biases[i] = (data_t)biases[i];
    }

    LOAD_DILATIONS: for (int i = 0; i < NUM_KERNELS; i++) {
        #pragma HLS PIPELINE II=1
        local_dilations[i] = (int_t)dilations[i];
    }

    LOAD_SCALER_MEAN: for (int i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
        local_scaler_mean[i] = (data_t)scaler_mean[i];
    }

    LOAD_SCALER_SCALE: for (int i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
        local_scaler_scale[i] = (data_t)scaler_scale[i];
    }

    LOAD_COEF: for (int i = 0; i < num_features * num_classes; i++) {
        #pragma HLS PIPELINE II=1
        local_coefficients[i] = (data_t)coefficients[i];
    }

    LOAD_INTERCEPT: for (int i = 0; i < num_classes; i++) {
        #pragma HLS PIPELINE II=1
        local_intercept[i] = (data_t)intercept[i];
    }

    // ---- Feature extraction (UNROLL=16, II=1) ----
    hydra_feature_extraction_hls(
        local_ts0, local_ts1, local_ts2, local_ts3, local_ts4,
        local_features,
        local_weights,
        local_biases,
        local_dilations,
        (int_t)time_series_length
    );

    // ---- Feature normalization ----
    apply_scaler_hls(
        local_features,
        local_scaler_mean,
        local_scaler_scale,
        (int_t)num_features
    );

    // ---- Classification ----
    linear_classifier_predict_hls(
        local_features,
        local_coefficients,
        local_intercept,
        (int_t)num_features,
        (int_t)num_classes,
        local_prediction
    );

    // ---- Write predictions ----
    WRITE_PRED: for (int i = 0; i < num_classes; i++) {
        #pragma HLS PIPELINE II=1
        prediction_output[i] = (float)local_prediction[i];
    }
}
