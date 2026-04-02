/**
 * HYDRA Fused Conv+Pooling C-Simulation Test
 *
 * Compares the original two-pass (conv then pooling) against the fused
 * single-pass implementation. Both must produce identical features.
 *
 * Build: g++ -O2 -o test_hydra_fused test_hydra_fused_csim.cpp -std=c++11
 * Run:   ./test_hydra_fused
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Include HYDRA constants
typedef float data_t;
typedef int int_t;

#define MAX_TIME_SERIES_LENGTH 5120
#define NUM_KERNELS 512
#define NUM_GROUPS 8
#define KERNELS_PER_GROUP 64
#define KERNEL_SIZE 9
#define MAX_FEATURES 2048
#define MAX_DILATIONS 8
#define MAX_CLASSES 10
#define POOLING_OPERATORS 2

// ============================================================
// ORIGINAL implementation (two-pass: conv then pooling)
// ============================================================

void apply_kernel_original(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t weights[KERNEL_SIZE],
    data_t bias,
    int_t dilation,
    int_t length,
    data_t output[MAX_TIME_SERIES_LENGTH],
    int_t& output_length
) {
    int_t kernel_span = (KERNEL_SIZE - 1) * dilation + 1;
    if (length < kernel_span) {
        output_length = 0;
        return;
    }
    output_length = length - kernel_span + 1;

    for (int_t t = 0; t < output_length; t++) {
        data_t sum = bias;
        for (int_t w = 0; w < KERNEL_SIZE; w++) {
            sum += time_series[t + w * dilation] * weights[w];
        }
        output[t] = sum;
    }
}

void compute_two_pooling_original(
    data_t values[MAX_TIME_SERIES_LENGTH],
    int_t length,
    data_t& max_val,
    data_t& mean_val
) {
    if (length == 0) {
        max_val = 0.0f;
        mean_val = 0.0f;
        return;
    }
    max_val = values[0];
    data_t sum = 0.0f;
    for (int_t i = 0; i < length; i++) {
        if (values[i] > max_val) max_val = values[i];
        sum += values[i];
    }
    mean_val = sum / static_cast<data_t>(length);
}

void hydra_fe_original(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    data_t kernel_weights[NUM_KERNELS * KERNEL_SIZE],
    data_t biases[NUM_KERNELS],
    int_t dilations[NUM_KERNELS],
    int_t length,
    int_t& num_features
) {
    int_t feature_idx = 0;
    data_t conv_output[MAX_TIME_SERIES_LENGTH];
    data_t current_weights[KERNEL_SIZE];

    for (int_t k = 0; k < NUM_KERNELS; k++) {
        for (int_t w = 0; w < KERNEL_SIZE; w++) {
            current_weights[w] = kernel_weights[k * KERNEL_SIZE + w];
        }

        int_t conv_length;
        apply_kernel_original(time_series, current_weights, biases[k],
                              dilations[k], length, conv_output, conv_length);

        data_t max_val, mean_val;
        compute_two_pooling_original(conv_output, conv_length, max_val, mean_val);

        features[feature_idx++] = max_val;
        features[feature_idx++] = mean_val;
    }
    num_features = feature_idx;
}

// ============================================================
// FUSED implementation (single-pass: conv + pooling inline)
// ============================================================

void hydra_fe_fused(
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
        features[feature_idx + 1] = (conv_length > 0) ? running_sum / static_cast<data_t>(conv_length) : 0.0f;
        feature_idx += 2;
    }
    num_features = feature_idx;
}

// ============================================================
// Test harness
// ============================================================

int main() {
    printf("=== HYDRA Fused Conv+Pooling C-Simulation Test ===\n\n");

    // Test parameters
    const int NUM_TEST_LENGTHS[] = {150, 600, 3750};
    const int NUM_TESTS = 3;

    // Allocate arrays
    static data_t time_series[MAX_TIME_SERIES_LENGTH];
    static data_t kernel_weights[NUM_KERNELS * KERNEL_SIZE];
    static data_t biases[NUM_KERNELS];
    static int_t dilations[NUM_KERNELS];
    static data_t features_orig[MAX_FEATURES];
    static data_t features_fused[MAX_FEATURES];

    // Initialize random weights (deterministic seed)
    srand(42);
    for (int i = 0; i < NUM_KERNELS * KERNEL_SIZE; i++) {
        kernel_weights[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < NUM_KERNELS; i++) {
        biases[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }

    // HYDRA uses 8 groups with dilations {1, 2, 4, 8, ...}
    int dilation_values[] = {1, 2, 4, 8, 1, 2, 4, 8};
    for (int k = 0; k < NUM_KERNELS; k++) {
        int group = k / KERNELS_PER_GROUP;
        dilations[k] = dilation_values[group % MAX_DILATIONS];
    }

    int total_pass = 0;
    int total_fail = 0;

    for (int t = 0; t < NUM_TESTS; t++) {
        int ts_length = NUM_TEST_LENGTHS[t];
        printf("Test %d: time_series_length=%d\n", t + 1, ts_length);

        // Generate random time series
        for (int i = 0; i < ts_length; i++) {
            time_series[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }

        // Run both implementations
        int_t nf_orig, nf_fused;
        hydra_fe_original(time_series, features_orig, kernel_weights, biases,
                          dilations, ts_length, nf_orig);
        hydra_fe_fused(time_series, features_fused, kernel_weights, biases,
                       dilations, ts_length, nf_fused);

        // Compare
        if (nf_orig != nf_fused) {
            printf("  FAIL: num_features mismatch: orig=%d fused=%d\n", nf_orig, nf_fused);
            total_fail++;
            continue;
        }

        int mismatches = 0;
        float max_err = 0.0f;
        int worst_idx = -1;

        for (int f = 0; f < nf_orig; f++) {
            float err = fabsf(features_orig[f] - features_fused[f]);
            // Use relative tolerance for large values, absolute for small
            float ref = fmaxf(fabsf(features_orig[f]), 1e-6f);
            float rel_err = err / ref;

            if (rel_err > 1e-5f && err > 1e-6f) {
                mismatches++;
                if (err > max_err) {
                    max_err = err;
                    worst_idx = f;
                }
            }
        }

        if (mismatches == 0) {
            printf("  PASS: %d features match exactly\n", nf_orig);
            total_pass++;
        } else {
            printf("  FAIL: %d/%d features mismatch (max_err=%.8f at idx %d, "
                   "orig=%.8f fused=%.8f)\n",
                   mismatches, nf_orig, max_err, worst_idx,
                   features_orig[worst_idx], features_fused[worst_idx]);
            total_fail++;
        }
    }

    printf("\n=== Results: %d/%d PASSED ===\n", total_pass, NUM_TESTS);
    return (total_fail > 0) ? 1 : 0;
}
