/**
 * HYDRA Optimized Feature Extraction C-Simulation Test
 *
 * Validates that the dilation-grouped, UNROLL=16 loop structure produces
 * identical features to the original per-kernel sequential implementation.
 *
 * Tests:
 * 1. Algorithm equivalence across multiple time series lengths
 * 2. Accumulator overflow stress test (large weights, long series)
 *
 * Build: g++ -O2 -o test_hydra_optimized test_hydra_optimized_csim.cpp -std=c++11
 * Run:   ./test_hydra_optimized
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>

// HYDRA constants
typedef float data_t;
typedef int int_t;

#define MAX_TIME_SERIES_LENGTH 5120
#define NUM_KERNELS 512
#define NUM_GROUPS 8
#define KERNELS_PER_GROUP 64
#define KERNEL_SIZE 9
#define MAX_FEATURES 2048
#define POOLING_OPERATORS 2
#define UNROLL_FACTOR 16
#define NUM_KERNEL_GROUPS (KERNELS_PER_GROUP / UNROLL_FACTOR)  // 4

// ============================================================
// REFERENCE: Original per-kernel sequential implementation
// (same as the proven fused version)
// ============================================================

void hydra_fe_reference(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    data_t kernel_weights[NUM_KERNELS * KERNEL_SIZE],
    data_t biases[NUM_KERNELS],
    int_t dilations[NUM_KERNELS],
    int_t length,
    int_t& num_features
) {
    int_t feature_idx = 0;
    data_t current_weights[KERNEL_SIZE];

    for (int_t k = 0; k < NUM_KERNELS; k++) {
        for (int_t w = 0; w < KERNEL_SIZE; w++) {
            current_weights[w] = kernel_weights[k * KERNEL_SIZE + w];
        }

        int_t local_dilation = dilations[k];
        data_t local_bias = biases[k];
        int_t kernel_span = (KERNEL_SIZE - 1) * local_dilation + 1;
        int_t conv_length = length - kernel_span + 1;

        data_t running_max = -1.0e30f;
        data_t running_sum = 0.0f;

        for (int_t t = 0; t < conv_length; t++) {
            data_t sum = local_bias;
            for (int_t w = 0; w < KERNEL_SIZE; w++) {
                sum += time_series[t + w * local_dilation] * current_weights[w];
            }
            if (sum > running_max) running_max = sum;
            running_sum += sum;
        }

        features[feature_idx]     = (conv_length > 0) ? running_max : 0.0f;
        features[feature_idx + 1] = (conv_length > 0) ? running_sum / (data_t)conv_length : 0.0f;
        feature_idx += 2;
    }
    num_features = feature_idx;
}

// ============================================================
// OPTIMIZED: Dilation-grouped, UNROLL=16 loop structure
// (mirrors the HLS kernel structure, but in pure C++ float)
// ============================================================

void hydra_fe_optimized(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    data_t kernel_weights[NUM_KERNELS * KERNEL_SIZE],
    data_t biases[NUM_KERNELS],
    int_t dilations[NUM_KERNELS],
    int_t length,
    int_t& num_features
) {
    // Reshape weights to 2D for grouped access
    // In HLS this would be a partitioned BRAM; here just a 2D view
    data_t weights_2d[NUM_KERNELS][KERNEL_SIZE];
    for (int_t k = 0; k < NUM_KERNELS; k++) {
        for (int_t w = 0; w < KERNEL_SIZE; w++) {
            weights_2d[k][w] = kernel_weights[k * KERNEL_SIZE + w];
        }
    }

    // Extract per-group dilations (first kernel in each group)
    int_t group_dilations[NUM_GROUPS];
    for (int_t g = 0; g < NUM_GROUPS; g++) {
        group_dilations[g] = dilations[g * KERNELS_PER_GROUP];
    }

    // Process by dilation group (mirrors HLS DILATION_LOOP)
    DILATION_LOOP:
    for (int_t g = 0; g < NUM_GROUPS; g++) {
        int_t dilation = group_dilations[g];
        int_t kernel_span = (KERNEL_SIZE - 1) * dilation + 1;
        int_t conv_length = length - kernel_span + 1;

        if (conv_length <= 0) {
            // Fill with zeros for this group
            for (int_t k = 0; k < KERNELS_PER_GROUP; k++) {
                int_t abs_k = g * KERNELS_PER_GROUP + k;
                features[abs_k * 2]     = 0.0f;
                features[abs_k * 2 + 1] = 0.0f;
            }
            continue;
        }

        data_t inv_conv_len = 1.0f / (data_t)conv_length;

        // Process kernels in batches of UNROLL_FACTOR (mirrors HLS KERNEL_GROUP_LOOP)
        KERNEL_GROUP_LOOP:
        for (int_t b = 0; b < NUM_KERNEL_GROUPS; b++) {
            int_t kg = g * KERNELS_PER_GROUP + b * UNROLL_FACTOR;  // base kernel index

            // Load biases for this batch
            data_t group_biases[UNROLL_FACTOR];
            for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
                group_biases[ki] = biases[kg + ki];
            }

            // Initialize accumulators
            data_t running_max[UNROLL_FACTOR];
            double running_sum[UNROLL_FACTOR];  // double simulates acc_t = ap_fixed<48,32>
            for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
                running_max[ki] = -1.0e30f;
                running_sum[ki] = 0.0;
            }

            // Fused conv + pooling with UNROLL (mirrors HLS CONV_POOL_LOOP)
            CONV_POOL_LOOP:
            for (int_t t = 0; t < conv_length; t++) {
                // Read 9 sliding window values
                // In HLS these come from 5 separate BRAMs; here just array reads
                data_t sw[KERNEL_SIZE];
                for (int_t w = 0; w < KERNEL_SIZE; w++) {
                    sw[w] = time_series[t + w * dilation];
                }

                // Process UNROLL_FACTOR kernels in parallel (mirrors HLS PARALLEL_KERNELS)
                PARALLEL_KERNELS:
                for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
                    // Dot product
                    data_t value = group_biases[ki];
                    for (int_t w = 0; w < KERNEL_SIZE; w++) {
                        value += sw[w] * weights_2d[kg + ki][w];
                    }
                    // Streaming pooling
                    if (value > running_max[ki]) running_max[ki] = value;
                    running_sum[ki] += (double)value;
                }
            }

            // Write features for this batch
            for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
                int_t abs_k = kg + ki;
                features[abs_k * 2]     = running_max[ki];
                features[abs_k * 2 + 1] = (data_t)(running_sum[ki] * (double)inv_conv_len);
            }
        }
    }
    num_features = NUM_KERNELS * POOLING_OPERATORS;
}

// ============================================================
// Test harness
// ============================================================

int test_equivalence(const char* test_name, int ts_length,
                     data_t* time_series, data_t* kernel_weights,
                     data_t* biases, int_t* dilations) {
    static data_t features_ref[MAX_FEATURES];
    static data_t features_opt[MAX_FEATURES];

    int_t nf_ref, nf_opt;
    hydra_fe_reference(time_series, features_ref, kernel_weights, biases,
                       dilations, ts_length, nf_ref);
    hydra_fe_optimized(time_series, features_opt, kernel_weights, biases,
                       dilations, ts_length, nf_opt);

    if (nf_ref != nf_opt) {
        printf("  %s FAIL: num_features mismatch: ref=%d opt=%d\n", test_name, nf_ref, nf_opt);
        return 1;
    }

    int mismatches = 0;
    float max_err = 0.0f;
    int worst_idx = -1;

    for (int f = 0; f < nf_ref; f++) {
        float err = fabsf(features_ref[f] - features_opt[f]);
        float ref_mag = fmaxf(fabsf(features_ref[f]), 1e-6f);
        float rel_err = err / ref_mag;

        if (rel_err > 1e-4f && err > 1e-5f) {
            mismatches++;
            if (err > max_err) {
                max_err = err;
                worst_idx = f;
            }
        }
    }

    if (mismatches == 0) {
        printf("  %s PASS: %d features match\n", test_name, nf_ref);
        return 0;
    } else {
        printf("  %s FAIL: %d/%d mismatches (max_err=%.8f at idx %d, ref=%.8f opt=%.8f)\n",
               test_name, mismatches, nf_ref, max_err, worst_idx,
               features_ref[worst_idx], features_opt[worst_idx]);
        return 1;
    }
}

int main() {
    printf("=== HYDRA Optimized Feature Extraction C-Simulation Test ===\n\n");

    // Allocate arrays
    static data_t time_series[MAX_TIME_SERIES_LENGTH];
    static data_t kernel_weights[NUM_KERNELS * KERNEL_SIZE];
    static data_t biases[NUM_KERNELS];
    static int_t dilations[NUM_KERNELS];

    // Initialize deterministic random weights
    srand(42);
    for (int i = 0; i < NUM_KERNELS * KERNEL_SIZE; i++) {
        kernel_weights[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < NUM_KERNELS; i++) {
        biases[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }

    // HYDRA dilation pattern: 8 groups, each group has fixed dilation
    int dilation_values[] = {1, 2, 4, 8, 1, 2, 4, 8};
    for (int k = 0; k < NUM_KERNELS; k++) {
        int group = k / KERNELS_PER_GROUP;
        dilations[k] = dilation_values[group];
    }

    int total_fail = 0;

    // ---- Test 1: Multiple time series lengths ----
    printf("Test 1: Algorithm equivalence across lengths\n");
    const int TEST_LENGTHS[] = {150, 600, 3750, 5000};
    const int NUM_LENGTHS = 4;

    for (int t = 0; t < NUM_LENGTHS; t++) {
        int len = TEST_LENGTHS[t];
        for (int i = 0; i < len; i++) {
            time_series[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        char name[64];
        snprintf(name, sizeof(name), "length=%d", len);
        total_fail += test_equivalence(name, len, time_series, kernel_weights, biases, dilations);
    }

    // ---- Test 2: Accumulator overflow stress test ----
    printf("\nTest 2: Accumulator stress test (large weights, length=5000)\n");
    // Use weights of magnitude ~3.0 (realistic max for HYDRA)
    for (int i = 0; i < NUM_KERNELS * KERNEL_SIZE; i++) {
        kernel_weights[i] = ((float)rand() / RAND_MAX) * 6.0f - 3.0f;
    }
    for (int i = 0; i < NUM_KERNELS; i++) {
        biases[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < 5000; i++) {
        time_series[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
    }

    // Check max running_sum magnitude
    {
        static data_t features[MAX_FEATURES];
        int_t nf;
        hydra_fe_optimized(time_series, features, kernel_weights, biases, dilations, 5000, nf);

        float max_mean = 0.0f;
        for (int f = 1; f < nf; f += 2) {
            if (fabsf(features[f]) > max_mean) max_mean = fabsf(features[f]);
        }
        // Estimate max running_sum = max_mean * conv_length
        float est_max_sum = max_mean * 5000.0f;
        printf("  Max |mean| = %.4f, estimated max |running_sum| = %.1f\n", max_mean, est_max_sum);
        // ap_fixed<48,32> can hold up to 2^31 ≈ 2.1 billion
        if (est_max_sum < 2.0e9f) {
            printf("  PASS: running_sum within ap_fixed<48,32> range\n");
        } else {
            printf("  WARNING: running_sum may overflow ap_fixed<48,32>!\n");
            total_fail++;
        }
    }

    total_fail += test_equivalence("stress", 5000, time_series, kernel_weights, biases, dilations);

    // ---- Test 3: Edge case — very short series ----
    printf("\nTest 3: Edge case — short series (length=70, dilation=8 gives kernel_span=65)\n");
    for (int i = 0; i < 70; i++) {
        time_series[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    // Reset weights to normal range
    srand(123);
    for (int i = 0; i < NUM_KERNELS * KERNEL_SIZE; i++) {
        kernel_weights[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < NUM_KERNELS; i++) {
        biases[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
    total_fail += test_equivalence("length=70", 70, time_series, kernel_weights, biases, dilations);

    // ---- Summary ----
    printf("\n=== Results: %s ===\n", total_fail == 0 ? "ALL PASSED" : "FAILURES DETECTED");
    return (total_fail > 0) ? 1 : 0;
}
