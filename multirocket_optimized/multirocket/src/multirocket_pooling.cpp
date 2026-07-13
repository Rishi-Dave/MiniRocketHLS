/*
 * MultiRocket Pooling Operators - Round 2 Verified
 *
 * NOTE: In the optimized v3 kernel (multirocket.cpp), pooling is FUSED inline
 * into the CONV_LOOP via PARALLEL_KERNELS unrolled loop.
 * This file retains the standalone functions for:
 *   - Legacy testbench compatibility
 *   - Reference correctness verification
 *
 * DO NOT MODIFY the Round 2 semantics below (verified 2026-04-07):
 *   - start=0 (no half-dilation padding)
 *   - mipv index = (i - start) = i
 *   - mpv = sum(positives) / count  (not /length)
 *   - mipv = index_sum / count / length
 *   - lspv = max_stretch / length
 */

#include "../include/multirocket.hpp"
#include <cstring>

/**
 * PPV (Proportion of Positive Values)
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
        #pragma HLS LOOP_TRIPCOUNT min=100 max=8192

        if (convolutions[i] > bias) {
            positive_count++;
        }
    }

    *ppv_out = (data_t)positive_count / (data_t)length;
}

/**
 * MPV (Mean of Positive Values)
 */
void compute_mpv(
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    data_t bias,
    int_t length,
    data_t* mpv_out
) {
    #pragma HLS INLINE off

    data_t sum = (data_t)0.0;
    int_t count = 0;

    MPV_LOOP: for (int_t i = 0; i < length; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=100 max=8192

        if (convolutions[i] > bias) {
            sum += convolutions[i];
            count++;
        }
    }

    *mpv_out = (count > 0) ? (data_t)(sum / (data_t)count) : (data_t)0.0;
}

/**
 * MIPV (Mean of Indices of Positive Values)
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
        #pragma HLS LOOP_TRIPCOUNT min=100 max=8192

        if (convolutions[i] > bias) {
            index_sum += i;  // i - start, start=0 per Round 2
            count++;
        }
    }

    *mipv_out = (count > 0) ? (data_t)((data_t)index_sum / (data_t)count / (data_t)length) : (data_t)0.0;
}

/**
 * LSPV (Longest Stretch of Positive Values)
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
        #pragma HLS LOOP_TRIPCOUNT min=100 max=8192

        if (convolutions[i] > bias) {
            current_stretch++;
            if (current_stretch > max_stretch) {
                max_stretch = current_stretch;
            }
        } else {
            current_stretch = 0;
        }
    }

    *lspv_out = (data_t)max_stretch / (data_t)length;
}

/**
 * compute_four_pooling_operators — Round 2 verified single-pass implementation.
 *
 * Semantics (MUST NOT change — verified against HW results 2026-04-07):
 *   start=0, effective_length = length - start = length
 *   ppv  = count(conv > bias) / effective_length
 *   mpv  = sum(conv[i] for i where conv[i] > bias) / count   (0 if count=0)
 *   mipv = sum((i - start) for i where conv[i] > bias) / count / effective_length  (0 if count=0)
 *   lspv = max_consecutive_run / effective_length
 */
void compute_four_pooling_operators(
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    data_t bias,
    int_t start,
    int_t length,
    PoolingStats* stats
) {
    #pragma HLS INLINE off

    int_t ppv_count = 0;
    data_t mpv_sum = (data_t)0.0;
    int_t mipv_index_sum = 0;
    int_t lspv_current = 0;
    int_t lspv_max = 0;

    int_t effective_length = length - start;

    POOLING_LOOP: for (int_t i = start; i < length; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=100 max=8192

        if (convolutions[i] > bias) {
            ppv_count++;
            mpv_sum += convolutions[i];
            mipv_index_sum += (i - start);
            lspv_current++;
            if (lspv_current > lspv_max) {
                lspv_max = lspv_current;
            }
        } else {
            lspv_current = 0;
        }
    }

    data_t inv_length = (data_t)1.0 / (data_t)effective_length;
    stats->ppv  = (data_t)ppv_count * inv_length;
    stats->mpv  = (ppv_count > 0) ? (data_t)(mpv_sum / (data_t)ppv_count) : (data_t)0.0;
    stats->mipv = (ppv_count > 0) ? (data_t)((data_t)mipv_index_sum / (data_t)ppv_count * inv_length) : (data_t)0.0;
    stats->lspv = (data_t)lspv_max * inv_length;
}
