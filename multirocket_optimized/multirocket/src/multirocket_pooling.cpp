#include "../include/multirocket.hpp"
#include <cstring>

/**
 * PPV (Proportion of Positive Values)
 * Counts the proportion of convolution values greater than the bias threshold
 * This is the same as MiniRocket's single feature
 */
void compute_ppv(
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    data_t bias,
    int_t length,
    data_t* ppv_out
) {
    #pragma HLS INLINE off

    int_t positive_count = 0;

    PPV_LOOP: for (int_t i = 0; i < length; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=100 max=512 avg=250

        if (convolutions[i] > bias) {
            positive_count++;
        }
    }

    *ppv_out = (data_t)positive_count / (data_t)length;
}

/**
 * MPV (Mean of Positive Values)
 * Computes the average value of all convolution outputs greater than bias
 * Captures the intensity/magnitude of matches
 */
void compute_mpv(
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    data_t bias,
    int_t length,
    data_t* mpv_out
) {
    #pragma HLS INLINE off

    data_t sum = 0.0;
    int_t count = 0;

    MPV_LOOP: for (int_t i = 0; i < length; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=100 max=512 avg=250

        if (convolutions[i] > bias) {
            sum += convolutions[i];
            count++;
        }
    }

    // Avoid division by zero
    if (count > 0) {
        *mpv_out = sum / (data_t)count;
    } else {
        *mpv_out = 0.0;
    }
}

/**
 * MIPV (Mean of Indices of Positive Values)
 * Computes the average index position where convolution > bias
 * Captures information about the location of matches in the time series
 */
void compute_mipv(
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    data_t bias,
    int_t length,
    data_t* mipv_out
) {
    #pragma HLS INLINE off

    int_t index_sum = 0;
    int_t count = 0;

    MIPV_LOOP: for (int_t i = 0; i < length; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=100 max=512 avg=250

        if (convolutions[i] > bias) {
            index_sum += i;
            count++;
        }
    }

    // Normalize by length to get relative position (0 to 1)
    if (count > 0) {
        *mipv_out = (data_t)index_sum / ((data_t)count * (data_t)length);
    } else {
        *mipv_out = 0.0;
    }
}

/**
 * LSPV (Longest Stretch of Positive Values)
 * Finds the maximum consecutive sequence of convolution > bias
 * Captures persistence/duration of pattern matches
 *
 * BOTTLENECK NOTE: This has a data dependency (current_stretch depends on previous iteration)
 * which may limit II=1 pipelining effectiveness
 */
void compute_lspv(
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    data_t bias,
    int_t length,
    data_t* lspv_out
) {
    #pragma HLS INLINE off

    int_t current_stretch = 0;
    int_t max_stretch = 0;

    LSPV_LOOP: for (int_t i = 0; i < length; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=100 max=512 avg=250

        if (convolutions[i] > bias) {
            current_stretch++;
            if (current_stretch > max_stretch) {
                max_stretch = current_stretch;
            }
        } else {
            current_stretch = 0;
        }
    }

    // Normalize by length to get relative stretch (0 to 1)
    *lspv_out = (data_t)max_stretch / (data_t)length;
}

/**
 * Compute all four pooling operators in a single pass
 * This is more efficient than calling each operator separately
 *
 * OPTIMIZATION NOTE: Single-pass computation reduces memory bandwidth
 * All operators computed from same convolution array
 */
void compute_four_pooling_operators(
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    data_t bias,
    int_t length,
    PoolingStats* stats
) {
    #pragma HLS INLINE off
    // Note: Not pipelining outer function, but inner loop is pipelined

    // Accumulators for all 4 operators
    int_t ppv_count = 0;
    data_t mpv_sum = 0.0;
    int_t mpv_count = 0;
    int_t mipv_index_sum = 0;
    int_t mipv_count = 0;
    int_t lspv_current = 0;
    int_t lspv_max = 0;

    // Single pass computes all 4 operators simultaneously
    POOLING_LOOP: for (int_t i = 0; i < length; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=100 max=512 avg=250

        bool is_positive = (convolutions[i] > bias);

        if (is_positive) {
            // PPV
            ppv_count++;

            // MPV
            mpv_sum += convolutions[i];
            mpv_count++;

            // MIPV
            mipv_index_sum += i;
            mipv_count++;

            // LSPV
            lspv_current++;
            if (lspv_current > lspv_max) {
                lspv_max = lspv_current;
            }
        } else {
            lspv_current = 0;
        }
    }

    // Compute final values
    stats->ppv = (data_t)ppv_count / (data_t)length;

    if (mpv_count > 0) {
        stats->mpv = mpv_sum / (data_t)mpv_count;
    } else {
        stats->mpv = 0.0;
    }

    if (mipv_count > 0) {
        stats->mipv = (data_t)mipv_index_sum / ((data_t)mipv_count * (data_t)length);
    } else {
        stats->mipv = 0.0;
    }

    stats->lspv = (data_t)lspv_max / (data_t)length;
}
