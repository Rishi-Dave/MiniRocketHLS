#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include "../include/minirocket_fused.hpp"
#include "minirocket_hls_testbench_loader.h"

// Buffer-then-classify csim harness for minirocket_fused_inference.
//
// Unlike minirocket_dpr's test_minirocket.cpp (which reads a fresh
// prediction after every single packet, because DP-Reuse classifies every
// packet), this harness feeds all `time_series_length` samples of one
// series as packets, and only reads a prediction back after the LAST
// packet -- that is the one call where fill_count wraps to
// time_series_length inside the kernel and it actually emits ONE packed
// prediction packet (FIX 1, 2026-07-05: num_classes float32 logits packed
// into the low num_classes*4 bytes of one 512-bit beat, mirroring
// hydra_axis.cpp's PACK_PRED -- was previously num_classes separate
// single-beat packets). All earlier calls are pure buffer-fill and produce
// no output packets (output_predictions stays empty), matching the
// buffer-then-classify contract described in minirocket_fused.cpp.
//
// NOTE: BUILD is intentionally left undefined for csim/synthesis/ip (see
// minirocket_fused.cpp's #if BUILD==1 gate and FIX 2 in the same file) so
// the top function returns after each call instead of looping forever --
// this harness calls it once per packet, exactly like minirocket_dpr's
// csim testbench does, and matches the single-shot invocation model the
// kernel is now built with end-to-end.

int main(int argc, char* argv[]) {
    std::string model_file = "InsectSound_minirocket_model.json";
    std::string test_file = "InsectSound_test_data_compact.json";
    bool csim = true;
    if (argc >= 3) {
        model_file = argv[1];
        test_file = argv[2];
    }
    if (argc >= 4) {
        csim = (std::string(argv[3]) == "csim");
    }

    std::cout << "HLS MiniRocket FUSED (buffer-then-classify) Test" << std::endl;
    std::cout << "Model file: " << model_file << std::endl;
    std::cout << "Test file: " << test_file << std::endl;

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

    hls::stream<pkt> input_timeseries;
    hls::stream<pkt> output_predictions;

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

    int num_samples = (int)test_inputs.size();
    if (argc >= 5) num_samples = std::min(num_samples, std::atoi(argv[4]));

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Buffer-then-classify CSIM: feed full series, classify once at the end" << std::endl;
    std::cout << "  Time series length: " << time_series_length << std::endl;
    std::cout << "  Number of test samples: " << num_samples << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    int num_correct = 0;
    int num_scored = 0;
    double max_abs_score_diff = 0.0;

    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        int ts_len = std::min((int)test_inputs[sample_idx].size(), (int)time_series_length);

        for (int t = 0; t < ts_len; t++) {
            pkt in_pkt;
            in_pkt.keep = -1;
            in_pkt.last = 1;
            in_pkt.data = *((ap_uint<DWIDTH>*)&test_inputs[sample_idx][t]);
            input_timeseries.write(in_pkt);

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

            bool is_last_sample_of_series = (t == ts_len - 1) && (ts_len == (int)time_series_length);

            if (is_last_sample_of_series) {
                // Full series buffered on this call -> kernel emitted ONE
                // packed prediction packet (FIX 1: num_classes float32
                // logits in one 512-bit beat, not num_classes separate
                // packets).
                if ((int)output_predictions.size() != 1) {
                    std::cerr << "ERROR: expected 1 packed prediction packet, got "
                              << output_predictions.size()
                              << " (sample " << sample_idx << ")" << std::endl;
                }

                data_t predictions[MAX_CLASSES];
                if (!output_predictions.empty()) {
                    pkt out_pkt;
                    output_predictions.read(out_pkt);
                    ap_uint<DWIDTH> out_data = out_pkt.data;
                    for (int i = 0; i < (int)num_classes; i++) {
                        ap_uint<32> bits = out_data.range(32 * i + 31, 32 * i);
                        predictions[i] = *((data_t*)&bits);
                    }
                }

                int predicted_class = 0;
                data_t max_score = predictions[0];
                for (int i = 1; i < num_classes; i++) {
                    if (predictions[i] > max_score) {
                        max_score = predictions[i];
                        predicted_class = i;
                    }
                }

                int expected_class = (sample_idx < (int)expected_classes.size()) ? expected_classes[sample_idx] : -1;
                bool correct = (predicted_class == expected_class);
                if (correct) num_correct++;
                num_scored++;

                std::cout << "Sample " << sample_idx << ": pred=[";
                for (int i = 0; i < num_classes; i++) std::cout << std::fixed << std::setprecision(4) << predictions[i] << " ";
                std::cout << "] predicted=" << predicted_class
                          << " expected=" << expected_class
                          << (correct ? " MATCH" : " MISMATCH") << std::endl;
            } else {
                // Buffer-fill call: no prediction packets should have been emitted yet.
                if (!output_predictions.empty()) {
                    std::cerr << "ERROR: unexpected prediction packet(s) mid-series at sample "
                              << sample_idx << " t=" << t << std::endl;
                    while (!output_predictions.empty()) { pkt p; output_predictions.read(p); }
                }
            }
        }
    }

    float hls_accuracy = num_scored > 0 ? (float)num_correct / num_scored * 100.0f : 0.0f;

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "RESULTS: " << num_correct << "/" << num_scored
              << " series correctly classified (" << std::fixed << std::setprecision(2)
              << hls_accuracy << "%)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    delete[] coefficients;
    delete[] flattened_coefficients;
    delete[] intercept;
    delete[] scaler_mean;
    delete[] scaler_scale;
    delete[] dilations;
    delete[] num_features_per_dilation;
    delete[] biases;

    bool success = (num_scored > 0) && (num_correct == num_scored);
    std::cout << (success ? "SUCCESS: all scored series matched expected class\n"
                          : "FAILURE: one or more series mismatched expected class\n");
    return success ? 0 : 1;
}
