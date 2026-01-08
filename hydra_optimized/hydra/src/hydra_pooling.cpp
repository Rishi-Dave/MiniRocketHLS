#include "../include/hydra.hpp"

/**
 * Compute max pooling from convolution output
 *
 * Finds the maximum value in the convolution result.
 * This is one of the two pooling operators used in HYDRA.
 */
void compute_max_pooling(
    data_t values[MAX_TIME_SERIES_LENGTH],
    int_t length,
    data_t& max_val
) {
    #pragma HLS INLINE off

    if (length == 0) {
        max_val = 0.0;
        return;
    }

    max_val = values[0];

    MAX_LOOP: for (int_t i = 1; i < length; i++) {
        #pragma HLS PIPELINE II=1
        if (values[i] > max_val) {
            max_val = values[i];
        }
    }
}

/**
 * Compute global mean from convolution output
 *
 * Computes the average value across the entire convolution result.
 * This is the second pooling operator used in HYDRA.
 */
void compute_global_mean(
    data_t values[MAX_TIME_SERIES_LENGTH],
    int_t length,
    data_t& mean_val
) {
    #pragma HLS INLINE off

    if (length == 0) {
        mean_val = 0.0;
        return;
    }

    data_t sum = 0.0;

    MEAN_LOOP: for (int_t i = 0; i < length; i++) {
        #pragma HLS PIPELINE II=1
        sum += values[i];
    }

    mean_val = sum / static_cast<data_t>(length);
}

/**
 * Compute both max and mean pooling in single pass
 *
 * Optimized version that computes both pooling operators
 * in a single loop iteration, reducing latency.
 */
void compute_two_pooling_operators(
    data_t values[MAX_TIME_SERIES_LENGTH],
    int_t length,
    data_t& max_val,
    data_t& mean_val
) {
    #pragma HLS INLINE off

    if (length == 0) {
        max_val = 0.0;
        mean_val = 0.0;
        return;
    }

    max_val = values[0];
    data_t sum = 0.0;

    // Single-pass computation of both statistics
    POOL_LOOP: for (int_t i = 0; i < length; i++) {
        #pragma HLS PIPELINE II=1

        data_t val = values[i];

        // Update max
        if (val > max_val) {
            max_val = val;
        }

        // Update sum for mean
        sum += val;
    }

    mean_val = sum / static_cast<data_t>(length);
}
