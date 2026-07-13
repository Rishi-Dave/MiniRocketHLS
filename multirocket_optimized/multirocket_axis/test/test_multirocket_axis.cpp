/*
 * MultiRocket AXIS-Stream csim smoke test.
 *
 * Same shape as hydra_axis test_hydra_axis.cpp:
 *   1. Confirms the kernel doesn't crash / hang under csim.
 *   2. Confirms output stream produces num_classes non-NaN floats per
 *      incoming beat (once the shift register is primed).
 *   3. Determinism check: two identical input series produce identical
 *      final-beat predictions.
 *
 * Build-gate test, NOT accuracy. Full accuracy parity vs. m_axi
 * multirocket_inference is verified post-OCT-hw-build by the saturation
 * harness with a real model.
 */

#include "../include/multirocket_axis.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>

#define TEST_SERIES_LENGTH 64
#define TEST_NUM_DILATIONS 2
#define TEST_FEATURES_PER_DILATION 1
#define TEST_NUM_FEATURES_PER_REP (TEST_NUM_DILATIONS * NUM_KERNELS * TEST_FEATURES_PER_DILATION)
#define TEST_NUM_FEATURES (TEST_NUM_FEATURES_PER_REP * NUM_POOLING_OPERATORS)  // ×4 pooling, both reps share array
#define TEST_NUM_CLASSES 4
#define TEST_N_FEATURE_PER_KERNEL TEST_FEATURES_PER_DILATION

static void fill_deterministic(data_t* dst, int n, unsigned seed) {
    unsigned state = seed;
    for (int i = 0; i < n; i++) {
        state = state * 1664525u + 1013904223u;
        dst[i] = ((float)(int)(state >> 8) / (float)(1 << 23)) - 1.0f;
    }
}

int main() {
    static data_t coefficients[TEST_NUM_FEATURES * TEST_NUM_CLASSES];
    static data_t intercept[TEST_NUM_CLASSES];
    static data_t scaler_mean[TEST_NUM_FEATURES];
    static data_t scaler_scale[TEST_NUM_FEATURES];
    static int_t  dilations_0[TEST_NUM_DILATIONS];
    static int_t  num_features_per_dilation_0[TEST_NUM_DILATIONS];
    static data_t biases_0[TEST_NUM_FEATURES_PER_REP];
    static int_t  dilations_1[TEST_NUM_DILATIONS];
    static int_t  num_features_per_dilation_1[TEST_NUM_DILATIONS];
    static data_t biases_1[TEST_NUM_FEATURES_PER_REP];

    fill_deterministic(coefficients, TEST_NUM_FEATURES * TEST_NUM_CLASSES, 0xC0FFEEu);
    fill_deterministic(intercept,    TEST_NUM_CLASSES,                     0xBEEFu);
    fill_deterministic(scaler_mean,  TEST_NUM_FEATURES,                    0xF00Du);
    for (int i = 0; i < TEST_NUM_FEATURES; i++) scaler_scale[i] = 1.0f + 0.5f * scaler_mean[i];
    fill_deterministic(biases_0,     TEST_NUM_FEATURES_PER_REP,            0xCAFEu);
    fill_deterministic(biases_1,     TEST_NUM_FEATURES_PER_REP,            0xDEADu);
    dilations_0[0] = 1; dilations_0[1] = 2;
    dilations_1[0] = 1; dilations_1[1] = 2;
    num_features_per_dilation_0[0] = TEST_FEATURES_PER_DILATION;
    num_features_per_dilation_0[1] = TEST_FEATURES_PER_DILATION;
    num_features_per_dilation_1[0] = TEST_FEATURES_PER_DILATION;
    num_features_per_dilation_1[1] = TEST_FEATURES_PER_DILATION;

    static data_t series[TEST_SERIES_LENGTH];
    fill_deterministic(series, TEST_SERIES_LENGTH, 0xDEADBEEFu);

    hls::stream<pkt> in_strm("in");
    hls::stream<pkt> out_strm("out");

    auto push_beat = [&](data_t v) {
        pkt p;
        ap_uint<DWIDTH> w = 0;
        ap_uint<32> bits = *((ap_uint<32>*) &v);
        w.range(31, 0) = bits;
        p.data = w;
        p.keep = -1;
        p.last = 1;
        in_strm.write(p);
    };

    auto pop_logit = [&]() -> data_t {
        pkt p = out_strm.read();
        ap_uint<DWIDTH> w = p.data;
        ap_uint<32> bits = w.range(31, 0);
        data_t v;
        *((ap_uint<32>*) &v) = bits;
        return v;
    };

    data_t last_a[TEST_NUM_CLASSES];
    for (int t = 0; t < TEST_SERIES_LENGTH; t++) {
        push_beat(series[t]);
        multirocket_axis_inference(
            in_strm, out_strm,
            coefficients, intercept,
            scaler_mean, scaler_scale,
            dilations_0, num_features_per_dilation_0, biases_0,
            TEST_NUM_DILATIONS, TEST_NUM_FEATURES_PER_REP,
            dilations_1, num_features_per_dilation_1, biases_1,
            TEST_NUM_DILATIONS, TEST_NUM_FEATURES_PER_REP,
            TEST_SERIES_LENGTH, TEST_NUM_FEATURES, TEST_NUM_CLASSES,
            TEST_N_FEATURE_PER_KERNEL
        );
        for (int c = 0; c < TEST_NUM_CLASSES; c++) last_a[c] = pop_logit();
    }

    int bad = 0;
    for (int c = 0; c < TEST_NUM_CLASSES; c++) {
        if (std::isnan(last_a[c]) || std::isinf(last_a[c])) bad++;
        std::printf("[A] logit[%d] = %f\n", c, (double) last_a[c]);
    }
    if (bad) {
        std::fprintf(stderr, "[FAIL] %d NaN/Inf logits in phase A\n", bad);
        return 1;
    }

    data_t last_b[TEST_NUM_CLASSES];
    for (int t = 0; t < TEST_SERIES_LENGTH; t++) {
        push_beat(series[t]);
        multirocket_axis_inference(
            in_strm, out_strm,
            coefficients, intercept,
            scaler_mean, scaler_scale,
            dilations_0, num_features_per_dilation_0, biases_0,
            TEST_NUM_DILATIONS, TEST_NUM_FEATURES_PER_REP,
            dilations_1, num_features_per_dilation_1, biases_1,
            TEST_NUM_DILATIONS, TEST_NUM_FEATURES_PER_REP,
            TEST_SERIES_LENGTH, TEST_NUM_FEATURES, TEST_NUM_CLASSES,
            TEST_N_FEATURE_PER_KERNEL
        );
        for (int c = 0; c < TEST_NUM_CLASSES; c++) last_b[c] = pop_logit();
    }

    int mismatch = 0;
    for (int c = 0; c < TEST_NUM_CLASSES; c++) {
        if (std::fabs(last_a[c] - last_b[c]) > 1e-3f) mismatch++;
        std::printf("[B] logit[%d] = %f  (delta = %g)\n",
                    c, (double) last_b[c], (double) (last_b[c] - last_a[c]));
    }
    if (mismatch) {
        std::fprintf(stderr,
                     "[FAIL] %d/%d phase-A vs phase-B logits differ\n",
                     mismatch, TEST_NUM_CLASSES);
        return 2;
    }
    std::printf("[PASS] multirocket_axis_inference smoke csim — "
                "L=%d C=%d, %d/%d logits matched\n",
                TEST_SERIES_LENGTH, TEST_NUM_CLASSES,
                TEST_NUM_CLASSES - mismatch, TEST_NUM_CLASSES);
    return 0;
}
