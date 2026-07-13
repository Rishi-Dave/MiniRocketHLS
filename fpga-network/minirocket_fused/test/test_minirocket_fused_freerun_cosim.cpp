#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include "../include/minirocket_fused.hpp"
#include "minirocket_hls_testbench_loader.h"

// Bounded FREE-RUN verification harness for minirocket_fused_inference.
//
// v4 (2026-07-06) UPDATE: the DUT now emits ONE packed prediction packet per
// completed window (all num_classes float32 logits in one 512-bit beat, see
// minirocket_fused.cpp file header "v4" FIX 1) instead of the old v2
// per-class loop of `num_classes` separate single-beat packets. This
// harness's output-parsing was updated to match: `output_predictions.size()`
// is now directly the number of completed windows (no more `/ num_classes`
// division), and each window's packet is unpacked via bit ranges
// (`out_data.range(32*c+31, 32*c)`), mirroring test_minirocket_fused.cpp's
// packed-parsing idiom. The pre-load / single-call / golden-comparison
// structure below is otherwise unchanged from the original v2 free-run
// deadlock repro (2026-07-05, see debugging.md "fused free-run deadlock"):
//
//   1. Pre-loads ALL packets for N complete series (N = COSIM_MAX_WINDOWS,
//      overridable via argv[3]) into `input_timeseries` BEFORE calling the
//      kernel -- emulating how real hardware hands the kernel a continuous
//      packet stream via NetLayer/pktDropper.
//   2. Calls the kernel exactly ONCE. Built with `-DBUILD=1
//      -DCOSIM_MAX_WINDOWS=N`, the kernel's internal loop is a FINITE
//      `for` loop bounded to N windows + slack (see minirocket_fused.cpp),
//      so it drains everything pre-loaded above and returns -- IF it
//      doesn't deadlock first. Under RTL cosim, a genuine deadlock shows up
//      as the run stalling/timing out rather than this call returning.
//   3. Drains `output_predictions` and checks how many windows actually
//      completed (liveness check) plus recomputes a GOLDEN reference for
//      each completed window by calling the exact same
//      minirocket_feature_extraction_hls / apply_scaler_hls /
//      linear_classifier_predict_hls functions directly (external linkage,
//      defined in minirocket_fused.cpp) on that window's raw samples --
//      bit-exact by construction, since it's literally the same code the
//      single-shot csim path already exercises, not a re-implementation.
//
// NOTE: BUILD=1 and COSIM_MAX_WINDOWS are wired in by make.tcl.in only for
// this dedicated verification invocation (COSIM_MAX_WINDOWS env var set);
// plain synthesis/ip/hw builds compile with -DBUILD=1 but WITHOUT
// COSIM_MAX_WINDOWS, so the bound is a no-op there (true `while(true)`,
// see minirocket_fused.cpp).
//
// This harness still cannot model real NetLayer/CMAC backpressure (see
// minirocket_fused.cpp file header "HISTORY") -- it validates free-run RTL
// liveness/correctness only, not the real hardware TX interaction.

#ifndef COSIM_MAX_WINDOWS
#define COSIM_MAX_WINDOWS 4
#endif

// Declared here (not in minirocket_fused.hpp) because these are internal
// compute-chain functions, not part of the kernel's external interface --
// see minirocket_fused.cpp. External (non-"C", non-static) linkage lets this
// TB call them directly to build a bit-exact golden reference.
void minirocket_feature_extraction_hls(
    data_t ts0[MAX_TIME_SERIES_LENGTH],
    data_t ts1[MAX_TIME_SERIES_LENGTH],
    data_t ts2[MAX_TIME_SERIES_LENGTH],
    data_t ts3[MAX_TIME_SERIES_LENGTH],
    data_t ts4[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    int_t dilations[MAX_DILATIONS],
    int_t num_features_per_dilation[MAX_DILATIONS],
    data_t biases[MAX_FEATURES],
    int_t time_series_length,
    int_t num_dilations,
    int_t num_features
);

void apply_scaler_hls(
    data_t features[MAX_FEATURES],
    data_t scaled_features[MAX_FEATURES],
    data_t scaler_mean[MAX_FEATURES],
    data_t scaler_scale[MAX_FEATURES],
    int_t num_features
);

void linear_classifier_predict_hls(
    data_t scaled_features[MAX_FEATURES],
    data_t predictions[MAX_CLASSES],
    data_t coefficients[MAX_CLASSES][MAX_FEATURES],
    data_t intercept[MAX_CLASSES],
    int_t num_features,
    int_t num_classes
);

int main(int argc, char* argv[]) {
    std::string model_file = "InsectSound_minirocket_model.json";
    std::string test_file = "InsectSound_test_data_compact.json";
    int requested_windows = COSIM_MAX_WINDOWS;
    if (argc >= 3) {
        model_file = argv[1];
        test_file = argv[2];
    }
    if (argc >= 4) requested_windows = std::atoi(argv[3]);

    std::cout << "HLS MiniRocket FUSED bounded FREE-RUN cosim verification" << std::endl;
    std::cout << "Model file: " << model_file << std::endl;
    std::cout << "Test file: " << test_file << std::endl;
    std::cout << "COSIM_MAX_WINDOWS (compiled into DUT) = " << COSIM_MAX_WINDOWS
              << ", requested_windows (this run) = " << requested_windows << std::endl;

    MiniRocketTestbenchLoader loader;

    data_t (*coefficients)[MAX_FEATURES] = new data_t[MAX_CLASSES][MAX_FEATURES];
    data_t *flattened_coefficients = new data_t[MAX_CLASSES * MAX_FEATURES];
    data_t *intercept = new data_t[MAX_CLASSES];
    data_t *scaler_mean = new data_t[MAX_FEATURES];
    data_t *scaler_scale = new data_t[MAX_FEATURES];
    int_t *dilations = new int_t[MAX_DILATIONS];
    int_t *num_features_per_dilation = new int_t[MAX_DILATIONS];
    data_t *biases = new data_t[MAX_FEATURES];

    int_t num_dilations, num_features, num_classes, time_series_length;

    std::cout << "Loading model..." << std::endl;
    if (!loader.load_model_to_hls_arrays(model_file, coefficients, intercept,
                                        scaler_mean, scaler_scale, dilations,
                                        num_features_per_dilation, biases,
                                        num_dilations, num_features, num_classes,
                                        time_series_length)) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }

    for (int i = 0; i < num_classes * num_features; i++) {
        int row = i / num_features;
        int col = i % num_features;
        flattened_coefficients[i] = coefficients[row][col];
    }

    std::cout << "Loading test data..." << std::endl;
    std::vector<std::vector<float>> test_inputs, expected_outputs;
    if (!loader.load_test_data(test_file, test_inputs, expected_outputs)) {
        std::cerr << "Failed to load test data!" << std::endl;
        return 1;
    }

    std::vector<int> expected_classes;
    for (const auto& output : expected_outputs) {
        if (!output.empty()) expected_classes.push_back((int)output[0]);
    }

    int num_windows = std::min((int)test_inputs.size(), requested_windows);
    if (num_windows < 1) {
        std::cerr << "ERROR: no test windows available (test data has "
                  << test_inputs.size() << " samples)." << std::endl;
        return 1;
    }
    if (num_windows < requested_windows) {
        std::cout << "NOTE: only " << num_windows << " windows available (requested "
                  << requested_windows << "); test data has " << test_inputs.size()
                  << " samples." << std::endl;
    }

    hls::stream<pkt> input_timeseries;
    hls::stream<pkt> output_predictions;

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Pre-loading " << num_windows << " complete windows ("
              << time_series_length << " samples each) BEFORE calling the kernel."
              << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    int total_packets_fed = 0;
    for (int w = 0; w < num_windows; w++) {
        int ts_len = std::min((int)test_inputs[w].size(), (int)time_series_length);
        if (ts_len != (int)time_series_length) {
            std::cerr << "WARNING: window " << w << " only has " << ts_len
                      << " samples (< time_series_length=" << time_series_length
                      << "); it will never complete inside the kernel." << std::endl;
        }
        for (int t = 0; t < ts_len; t++) {
            pkt in_pkt;
            in_pkt.keep = -1;
            in_pkt.last = 1;
            in_pkt.data = *((ap_uint<DWIDTH>*)&test_inputs[w][t]);
            input_timeseries.write(in_pkt);
            total_packets_fed++;
        }
    }
    std::cout << "Pre-loaded " << total_packets_fed << " packets." << std::endl;

    std::cout << "\nCalling minirocket_fused_inference() ONCE (bounded free-run loop "
                 "should drain all pre-loaded packets)..." << std::endl;

    minirocket_fused_inference(
        input_timeseries,
        output_predictions,
        flattened_coefficients,
        intercept,
        scaler_mean,
        scaler_scale,
        dilations,
        num_features_per_dilation,
        biases,
        time_series_length,
        num_features,
        num_classes,
        num_dilations
    );

    std::cout << "minirocket_fused_inference() RETURNED." << std::endl;

    // v4: ONE packed packet per completed window (not num_classes separate
    // packets), so the packet count IS the window count directly.
    int windows_completed = (int)output_predictions.size();

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "output_predictions packet count = " << windows_completed
              << " (expected " << num_windows << " packed packets = "
              << num_windows << " windows, 1 packet/window)" << std::endl;
    std::cout << "windows_completed = " << windows_completed << " / " << num_windows << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    std::vector<std::vector<data_t>> dut_predictions(
        windows_completed, std::vector<data_t>((int)num_classes, (data_t)0.0));
    for (int w = 0; w < windows_completed; w++) {
        pkt out_pkt;
        output_predictions.read(out_pkt);
        ap_uint<DWIDTH> out_data = out_pkt.data;
        for (int c = 0; c < (int)num_classes; c++) {
            ap_uint<32> bits = out_data.range(32 * c + 31, 32 * c);
            dut_predictions[w][c] = *((data_t*)&bits);
        }
    }

    bool all_bit_exact = true;
    int num_correct_vs_expected = 0;

    std::cout << "\nPer-window golden comparison (golden = same core compute "
                 "functions called directly on the pre-loaded raw samples):" << std::endl;

    for (int w = 0; w < windows_completed; w++) {
        static data_t ts0[MAX_TIME_SERIES_LENGTH], ts1[MAX_TIME_SERIES_LENGTH],
                      ts2[MAX_TIME_SERIES_LENGTH], ts3[MAX_TIME_SERIES_LENGTH],
                      ts4[MAX_TIME_SERIES_LENGTH];
        for (int t = 0; t < (int)time_series_length; t++) {
            data_t v = (data_t)test_inputs[w][t];
            ts0[t] = v; ts1[t] = v; ts2[t] = v; ts3[t] = v; ts4[t] = v;
        }

        static data_t features[MAX_FEATURES], scaled[MAX_FEATURES];
        data_t golden_pred[MAX_CLASSES];

        minirocket_feature_extraction_hls(
            ts0, ts1, ts2, ts3, ts4, features,
            dilations, num_features_per_dilation, biases,
            time_series_length, num_dilations, num_features
        );
        apply_scaler_hls(features, scaled, scaler_mean, scaler_scale, num_features);
        linear_classifier_predict_hls(
            scaled, golden_pred, coefficients, intercept, num_features, num_classes
        );

        bool bit_exact = true;
        double max_abs_diff = 0.0;
        for (int c = 0; c < (int)num_classes; c++) {
            double diff = std::fabs((double)dut_predictions[w][c] - (double)golden_pred[c]);
            max_abs_diff = std::max(max_abs_diff, diff);
            if (std::memcmp(&dut_predictions[w][c], &golden_pred[c], sizeof(data_t)) != 0) {
                bit_exact = false;
            }
        }
        all_bit_exact = all_bit_exact && bit_exact;

        int dut_class = 0;
        data_t dut_max = dut_predictions[w][0];
        for (int c = 1; c < (int)num_classes; c++) {
            if (dut_predictions[w][c] > dut_max) { dut_max = dut_predictions[w][c]; dut_class = c; }
        }
        int expected_class = (w < (int)expected_classes.size()) ? expected_classes[w] : -1;
        bool class_match = (dut_class == expected_class);
        if (class_match) num_correct_vs_expected++;

        std::cout << "  Window " << (w + 1) << "/" << num_windows << ": "
                  << (bit_exact ? "BIT-EXACT" : "MISMATCH")
                  << " vs golden (max_abs_diff=" << std::scientific << max_abs_diff
                  << std::fixed << ")"
                  << " dut_class=" << dut_class
                  << " expected_class=" << expected_class
                  << (class_match ? " MATCH" : " CLASS-MISMATCH")
                  << std::endl;
    }

    bool no_stall = (windows_completed == num_windows);
    bool success = no_stall && all_bit_exact;

    std::cout << "\n" << std::string(70, '=') << std::endl;
    if (no_stall) {
        std::cout << "PASS: kernel completed all " << num_windows
                  << " requested windows (no stall)." << std::endl;
    } else {
        std::cout << "FAIL: kernel STALLED -- only completed " << windows_completed
                  << "/" << num_windows << " windows." << std::endl;
    }
    std::cout << (all_bit_exact
                    ? "PASS: all completed windows bit-exact vs golden single-shot compute chain."
                    : "FAIL: at least one completed window's predictions do not match golden.")
              << std::endl;
    std::cout << "Classification match vs expected_classes: " << num_correct_vs_expected
              << "/" << windows_completed << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    delete[] coefficients;
    delete[] flattened_coefficients;
    delete[] intercept;
    delete[] scaler_mean;
    delete[] scaler_scale;
    delete[] dilations;
    delete[] num_features_per_dilation;
    delete[] biases;

    return success ? 0 : 1;
}
