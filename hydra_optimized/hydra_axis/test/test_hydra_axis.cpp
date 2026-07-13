/*
 * HYDRA AXIS-Stream csim smoke test.
 *
 * Drives hydra_axis_inference with synthetic deterministic data and:
 *   1. Confirms the kernel doesn't crash / hang under csim.
 *   2. Confirms the kernel emits EXACTLY ONE pkt of num_classes non-NaN
 *      logits per full window of time_series_length ingested beats
 *      (buffer-then-classify gating: no output on the first
 *      time_series_length-1 calls, one output pkt on the Lth call).
 *   3. Spot-checks that two independent, identical-input windows produce
 *      identical predictions (determinism) — since windows are
 *      non-overlapping and reset after each classify, this also serves
 *      as a functional-equivalence check against the prior shift-register
 *      design: for a buffer starting from all-zero state, a shift
 *      register fed L samples in order holds the same content
 *      (local[i] = series[i]) as an append buffer fed the same L samples,
 *      so the final-beat prediction of the old design and the
 *      window-boundary prediction of the new design are mathematically
 *      identical for the first window.
 *
 * This is a build-gate test, NOT an accuracy test. Full accuracy parity
 * vs. the m_axi hydra_inference variant is verified post-OCT-hw-build by
 * the saturation harness (cpu-network/fpga_network_host.py with the
 * hydra-axis adapter, against a real model).
 */

#include "../include/hydra_axis.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>

// Dimensions used by the smoke test. Small enough that csim runs in
// seconds, large enough that conv path runs through every helper.
#define TEST_SERIES_LENGTH 128
#define TEST_NUM_FEATURES 1024  // = NUM_KERNELS * POOLING_OPERATORS
#define TEST_NUM_CLASSES 4

static void fill_deterministic(data_t* dst, int n, unsigned seed) {
    unsigned state = seed;
    for (int i = 0; i < n; i++) {
        state = state * 1664525u + 1013904223u;
        // Map u32 to (-1, 1).
        dst[i] = ((float)(int)(state >> 8) / (float)(1 << 23)) - 1.0f;
    }
}

static void fill_int_dilations(int_t* dst, int n) {
    // Deterministic small dilations in SORTED blocks of 64 per value —
    // the v6 kernel's UNROLL=16 batches require uniform dilation per
    // batch (host sorts by dilation; 64 is a multiple of 16).
    int dvals[8] = {1, 1, 2, 2, 4, 4, 8, 8};
    for (int i = 0; i < n; i++) dst[i] = dvals[(i / 64) % 8];
}

int main() {
    static data_t coefficients[TEST_NUM_FEATURES * TEST_NUM_CLASSES];
    static data_t intercept[TEST_NUM_CLASSES];
    static data_t scaler_mean[TEST_NUM_FEATURES];
    static data_t scaler_scale[TEST_NUM_FEATURES];
    static data_t kernel_weights[NUM_KERNELS * KERNEL_SIZE];
    static data_t biases[NUM_KERNELS];
    static int_t  dilations[NUM_KERNELS];

    fill_deterministic(coefficients, TEST_NUM_FEATURES * TEST_NUM_CLASSES, 0xC0FFEEu);
    fill_deterministic(intercept,    TEST_NUM_CLASSES,                     0xBEEFu);
    fill_deterministic(scaler_mean,  TEST_NUM_FEATURES,                    0xF00Du);
    for (int i = 0; i < TEST_NUM_FEATURES; i++) scaler_scale[i] = 1.0f + 0.5f * scaler_mean[i];
    fill_deterministic(kernel_weights, NUM_KERNELS * KERNEL_SIZE,           0xCAFEu);
    fill_deterministic(biases,         NUM_KERNELS,                         0xDEADu);
    fill_int_dilations(dilations,      NUM_KERNELS);

    // Build one synthetic time series and push it twice — the second pass
    // is a determinism check.
    static data_t series[TEST_SERIES_LENGTH];
    fill_deterministic(series, TEST_SERIES_LENGTH, 0xDEADBEEFu);

    // Single shared input + output streams. The kernel reads / writes
    // exactly once per call in csim mode.
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

    // v5 packs all num_classes logits into ONE pkt (32 bits per class,
    // class c at bit range [32c+31 : 32c]) — read the pkt ONCE, then
    // unpack each class from its byte range. (A prior version of this TB
    // called out_strm.read() once per class, which only worked against
    // the pre-v5 one-pkt-per-class wire format; fixed here to match the
    // current single-pkt packing.)
    auto unpack_logit = [&](const pkt& p, int c) -> data_t {
        ap_uint<DWIDTH> w = p.data;
        ap_uint<32> bits = w.range(32 * c + 31, 32 * c);
        data_t v;
        *((ap_uint<32>*) &v) = bits;
        return v;
    };

    // Phase A: fill one non-overlapping window by streaming the full
    // series. Each call ingests one beat; only the LAST call (the one that
    // completes the window, fill_count == TEST_SERIES_LENGTH) emits a pkt
    // of TEST_NUM_CLASSES logits. Popping on any earlier call would read
    // from an empty stream, so we pop exactly once, after the loop.
    data_t last_logits_a[TEST_NUM_CLASSES];
    for (int t = 0; t < TEST_SERIES_LENGTH; t++) {
        push_beat(series[t]);
        hydra_axis_inference(
            in_strm, out_strm,
            coefficients, intercept,
            scaler_mean, scaler_scale,
            kernel_weights, biases, dilations,
            TEST_SERIES_LENGTH, TEST_NUM_FEATURES, TEST_NUM_CLASSES, NUM_GROUPS
        );
    }
    if (!out_strm.empty()) {
        pkt out_pkt = out_strm.read();
        for (int c = 0; c < TEST_NUM_CLASSES; c++) last_logits_a[c] = unpack_logit(out_pkt, c);
    } else {
        std::fprintf(stderr,
                      "[FAIL] no output pkt after phase A window fill "
                      "(gating did not fire on the window boundary)\n");
        return 3;
    }
    if (!out_strm.empty()) {
        std::fprintf(stderr,
                      "[FAIL] extra output pkt(s) after phase A — gating "
                      "emitted more than once per window\n");
        return 4;
    }

    // Sanity: no NaN / Inf in the final logits.
    int bad = 0;
    for (int c = 0; c < TEST_NUM_CLASSES; c++) {
        if (std::isnan(last_logits_a[c]) || std::isinf(last_logits_a[c])) bad++;
        std::printf("[A] logit[%d] = %f\n", c, (double) last_logits_a[c]);
    }
    if (bad) {
        std::fprintf(stderr, "[FAIL] %d NaN/Inf logits in phase A\n", bad);
        return 1;
    }

    // Phase B: run the same series again as a second, independent
    // non-overlapping window (fill_count was reset to 0 at the end of
    // phase A). The window-boundary prediction should be identical to
    // phase A's, since it is fed the exact same L samples in the exact
    // same order into a freshly-zeroed-then-filled buffer.
    data_t last_logits_b[TEST_NUM_CLASSES];
    for (int t = 0; t < TEST_SERIES_LENGTH; t++) {
        push_beat(series[t]);
        hydra_axis_inference(
            in_strm, out_strm,
            coefficients, intercept,
            scaler_mean, scaler_scale,
            kernel_weights, biases, dilations,
            TEST_SERIES_LENGTH, TEST_NUM_FEATURES, TEST_NUM_CLASSES, NUM_GROUPS
        );
    }
    if (!out_strm.empty()) {
        pkt out_pkt = out_strm.read();
        for (int c = 0; c < TEST_NUM_CLASSES; c++) last_logits_b[c] = unpack_logit(out_pkt, c);
    } else {
        std::fprintf(stderr,
                      "[FAIL] no output pkt after phase B window fill\n");
        return 5;
    }

    int mismatch = 0;
    for (int c = 0; c < TEST_NUM_CLASSES; c++) {
        if (std::fabs(last_logits_a[c] - last_logits_b[c]) > 1e-3f) mismatch++;
        std::printf("[B] logit[%d] = %f  (delta from A = %g)\n",
                    c, (double) last_logits_b[c],
                    (double) (last_logits_b[c] - last_logits_a[c]));
    }
    if (mismatch) {
        std::fprintf(stderr,
                     "[FAIL] %d/%d phase-A vs phase-B logits differ — "
                     "shift-register state is not deterministic across runs.\n",
                     mismatch, TEST_NUM_CLASSES);
        return 2;
    }

    std::printf("[PASS] hydra_axis_inference smoke csim — "
                "L=%d C=%d, %d/%d logits matched between identical-input runs\n",
                TEST_SERIES_LENGTH, TEST_NUM_CLASSES,
                TEST_NUM_CLASSES - mismatch, TEST_NUM_CLASSES);
    return 0;
}
