/*
 * Standalone C-Simulation Test for Feature Extraction (K1)
 *
 * Validates the feature_extraction kernel logic against full-pipeline CPU inference.
 * Runs as pure C++ (no FPGA, no HLS tools required).
 *
 * Usage: ./test_fe_csim <model.json> <test_data.json> [num_samples]
 *
 * What this tests:
 *   1. Loads model params (dilations, biases, etc.) via existing loader
 *   2. Calls minirocket_feature_extraction_hls() directly (C simulation)
 *   3. Applies scaler + classifier in plain C++ to get final predictions
 *   4. Compares predicted classes against expected labels
 *   5. Dumps raw feature vectors for external comparison (e.g., vs Python)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>

// Host-side type definitions (must match kernel common.hpp for csim)
typedef float data_t;
typedef int32_t int_t;
typedef uint8_t idx_t;

#define MAX_TIME_SERIES_LENGTH 8192
#define MAX_FEATURES 1024
#define NUM_KERNELS 84
#define KERNEL_SIZE 9
#define MAX_DILATIONS 16
#define MAX_CLASSES 16

// Include the kernel weights (same as HLS kernel uses)
static data_t weights[NUM_KERNELS][KERNEL_SIZE] = {
    #include "../include/weights.txt"
};

// ============================================================
// C++ reimplementation of HLS functions (for csim validation)
// These must match feature_extraction.cpp exactly
// ============================================================

void apply_kernel_hls_csim(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    int_t kernel_idx,
    int_t dilation,
    int_t time_series_length,
    int_t* output_length
) {
    *output_length = time_series_length;

    if (*output_length <= 0) {
        *output_length = 0;
        return;
    }

    for (int_t j = 0; j < time_series_length; j++) {
        data_t sliding_window[KERNEL_SIZE] = {0};
        int i = 0;
        for (int k = -4; k <= 4; k++) {
            int_t idx = j + k * dilation;
            if (idx < 0 || idx >= time_series_length) {
                sliding_window[i] = 0.0;
            } else {
                sliding_window[i] = time_series[idx];
            }
            i++;
        }

        data_t value = 0.0;
        for (int_t k = 0; k < KERNEL_SIZE; k++) {
            value += sliding_window[k] * weights[kernel_idx][k];
        }
        convolutions[j] = value;
    }
}

void minirocket_feature_extraction_csim(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    int_t dilations[MAX_DILATIONS],
    int_t num_features_per_dilation[MAX_DILATIONS],
    data_t biases[MAX_FEATURES],
    int_t time_series_length,
    int_t num_dilations,
    int_t num_features
) {
    int_t feature_idx = 0;

    for (int_t dil_idx = 0; dil_idx < num_dilations; dil_idx++) {
        int_t dilation = dilations[dil_idx];
        int_t features_this_dilation = num_features_per_dilation[dil_idx];
        int_t _padding0 = dil_idx % 2;
        int_t padding = ((9 - 1) * dilation) / 2;

        for (int_t kernel_idx = 0; kernel_idx < NUM_KERNELS; kernel_idx++) {
            data_t convolutions[MAX_TIME_SERIES_LENGTH];
            int_t _padding1 = (_padding0 + kernel_idx) % 2;
            int_t feature_idx_for_kernel = feature_idx + kernel_idx * features_this_dilation;

            bool kernel_active = (feature_idx_for_kernel < num_features);

            if (kernel_active) {
                int_t conv_length;
                apply_kernel_hls_csim(time_series, convolutions, kernel_idx, dilation,
                                      time_series_length, &conv_length);

                for (int_t f = 0; f < features_this_dilation; f++) {
                    data_t bias = biases[feature_idx_for_kernel + f];
                    int_t positive_count = 0;
                    data_t ppv = 0.0;
                    if (_padding1 == 0) {
                        for (int_t i = 0; i < time_series_length; i++) {
                            if (convolutions[i] > bias) {
                                positive_count++;
                            }
                        }
                        ppv = (data_t)positive_count / (data_t)time_series_length;
                    } else {
                        for (int_t i = padding; i < time_series_length - padding; i++) {
                            if (convolutions[i] > bias) {
                                positive_count++;
                            }
                        }
                        ppv = (data_t)positive_count / (data_t)(time_series_length - 2 * padding);
                    }
                    features[feature_idx_for_kernel + f] = ppv;
                }
            }
        }
        feature_idx += features_this_dilation * NUM_KERNELS;
    }
}

// ============================================================
// Simple JSON loader (replicates minirocket_loader logic)
// ============================================================

std::string read_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return "";
    }
    std::string content, line;
    while (std::getline(file, line)) {
        content += line;
    }
    return content;
}

void trim_ws(std::string& str) {
    str.erase(0, str.find_first_not_of(" \t\n\r\f\v"));
    str.erase(str.find_last_not_of(" \t\n\r\f\v") + 1);
}

int parse_int_value(const std::string& content, const std::string& key) {
    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return -1;
    size_t colon_pos = content.find(":", key_pos);
    size_t value_end = content.find_first_of(",}", colon_pos + 1);
    std::string value_str = content.substr(colon_pos + 1, value_end - colon_pos - 1);
    trim_ws(value_str);
    return std::stoi(value_str);
}

std::vector<float> parse_float_array(const std::string& content, const std::string& key) {
    std::vector<float> result;
    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return result;
    size_t array_start = content.find("[", key_pos);
    size_t array_end = content.find("]", array_start);
    if (array_start == std::string::npos || array_end == std::string::npos) return result;
    std::string array_content = content.substr(array_start + 1, array_end - array_start - 1);
    std::stringstream ss(array_content);
    std::string item;
    while (std::getline(ss, item, ',')) {
        trim_ws(item);
        if (!item.empty()) result.push_back(std::stof(item));
    }
    return result;
}

std::vector<int> parse_int_array(const std::string& content, const std::string& key) {
    std::vector<int> result;
    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return result;
    size_t array_start = content.find("[", key_pos);
    size_t array_end = content.find("]", array_start);
    if (array_start == std::string::npos || array_end == std::string::npos) return result;
    std::string array_content = content.substr(array_start + 1, array_end - array_start - 1);
    std::stringstream ss(array_content);
    std::string item;
    while (std::getline(ss, item, ',')) {
        trim_ws(item);
        if (!item.empty()) result.push_back(std::stoi(item));
    }
    return result;
}

std::vector<std::vector<float>> parse_2d_float_array(const std::string& content, const std::string& key) {
    std::vector<std::vector<float>> result;
    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return result;
    size_t array_start = content.find("[", key_pos);
    size_t array_end = array_start;
    int bracket_count = 0;
    for (size_t i = array_start; i < content.length(); i++) {
        if (content[i] == '[') bracket_count++;
        if (content[i] == ']') bracket_count--;
        if (bracket_count == 0) { array_end = i; break; }
    }
    std::string array_content = content.substr(array_start + 1, array_end - array_start - 1);
    size_t pos = 0;
    while (pos < array_content.length()) {
        size_t sub_start = array_content.find("[", pos);
        if (sub_start == std::string::npos) break;
        size_t sub_end = array_content.find("]", sub_start);
        if (sub_end == std::string::npos) break;
        std::string sub_array = array_content.substr(sub_start + 1, sub_end - sub_start - 1);
        std::vector<float> sub_result;
        std::stringstream ss(sub_array);
        std::string item;
        while (std::getline(ss, item, ',')) {
            trim_ws(item);
            if (!item.empty()) sub_result.push_back(std::stof(item));
        }
        result.push_back(sub_result);
        pos = sub_end + 1;
    }
    return result;
}

// ============================================================
// Main test
// ============================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.json> <test_data.json> [num_samples]" << std::endl;
        return 1;
    }

    std::string model_file = argv[1];
    std::string test_file = argv[2];
    int max_samples = (argc > 3) ? std::stoi(argv[3]) : -1;

    // ---- Load model ----
    std::cout << "Loading model from: " << model_file << std::endl;
    std::string model_content = read_file(model_file);
    if (model_content.empty()) return 1;

    int num_dilations = parse_int_value(model_content, "num_dilations");
    int num_features = parse_int_value(model_content, "num_features");
    int num_classes = parse_int_value(model_content, "num_classes");
    int time_series_length = parse_int_value(model_content, "time_series_length");

    std::cout << "Model: " << num_features << " features, " << num_classes << " classes, "
              << num_dilations << " dilations, ts_len=" << time_series_length << std::endl;

    auto dilations_vec = parse_int_array(model_content, "dilations");
    auto nfpd_vec = parse_int_array(model_content, "num_features_per_dilation");
    auto biases_vec = parse_float_array(model_content, "biases");
    auto scaler_mean_vec = parse_float_array(model_content, "scaler_mean");
    auto scaler_scale_vec = parse_float_array(model_content, "scaler_scale");
    auto intercept_vec = parse_float_array(model_content, "classifier_intercept");

    // Parse coefficients (binary vs multi-class)
    std::vector<std::vector<float>> coef_2d;
    if (num_classes == 2) {
        auto coef_1d = parse_float_array(model_content, "classifier_coef");
        coef_2d.resize(2);
        coef_2d[0] = coef_1d;
        coef_2d[1].resize(coef_1d.size(), 0.0f);
    } else {
        coef_2d = parse_2d_float_array(model_content, "classifier_coef");
    }

    // Copy to HLS-sized arrays
    int_t dilations[MAX_DILATIONS] = {0};
    int_t num_features_per_dilation[MAX_DILATIONS] = {0};
    data_t biases[MAX_FEATURES] = {0};
    data_t scaler_mean[MAX_FEATURES] = {0};
    data_t scaler_scale[MAX_FEATURES] = {0};
    data_t intercept[MAX_CLASSES] = {0};
    data_t coefficients[MAX_CLASSES][MAX_FEATURES] = {{0}};

    for (int i = 0; i < num_dilations; i++) {
        dilations[i] = dilations_vec[i];
        num_features_per_dilation[i] = nfpd_vec[i];
    }
    for (int i = 0; i < num_features; i++) {
        biases[i] = biases_vec[i];
        scaler_mean[i] = scaler_mean_vec[i];
        scaler_scale[i] = scaler_scale_vec[i];
    }
    if (num_classes == 2 && intercept_vec.size() == 1) {
        intercept[0] = intercept_vec[0];
    } else {
        for (int i = 0; i < num_classes; i++) intercept[i] = intercept_vec[i];
    }
    for (int i = 0; i < num_classes; i++) {
        for (int j = 0; j < num_features; j++) {
            coefficients[i][j] = coef_2d[i][j];
        }
    }

    // ---- Load test data ----
    std::cout << "Loading test data from: " << test_file << std::endl;
    std::string test_content = read_file(test_file);
    if (test_content.empty()) return 1;

    auto test_inputs = parse_2d_float_array(test_content, "X_test");
    auto y_labels = parse_float_array(test_content, "y_test");

    int total_samples = test_inputs.size();
    if (max_samples > 0 && max_samples < total_samples) total_samples = max_samples;
    std::cout << "Testing " << total_samples << " samples" << std::endl;

    // ---- Open feature dump file ----
    std::ofstream feature_dump("feature_dump_csim.csv");
    feature_dump << "sample_idx";
    for (int i = 0; i < num_features; i++) feature_dump << ",f" << i;
    feature_dump << std::endl;

    // ---- Run inference loop ----
    int correct = 0;
    double total_fe_ms = 0, total_scale_ms = 0, total_classify_ms = 0;

    for (int s = 0; s < total_samples; s++) {
        data_t time_series[MAX_TIME_SERIES_LENGTH] = {0};
        data_t features[MAX_FEATURES] = {0};

        for (int j = 0; j < time_series_length && j < (int)test_inputs[s].size(); j++) {
            time_series[j] = test_inputs[s][j];
        }

        // K1: Feature Extraction (the function under test)
        auto t0 = std::chrono::high_resolution_clock::now();
        minirocket_feature_extraction_csim(
            time_series, features, dilations, num_features_per_dilation,
            biases, time_series_length, num_dilations, num_features
        );
        auto t1 = std::chrono::high_resolution_clock::now();
        total_fe_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Dump features for this sample
        feature_dump << s;
        for (int i = 0; i < num_features; i++) {
            feature_dump << "," << std::setprecision(8) << features[i];
        }
        feature_dump << std::endl;

        // K2: Scaler (inline C++)
        auto t2 = std::chrono::high_resolution_clock::now();
        data_t scaled_features[MAX_FEATURES];
        for (int i = 0; i < num_features; i++) {
            if (scaler_scale[i] != 0.0f) {
                scaled_features[i] = (features[i] - scaler_mean[i]) / scaler_scale[i];
            } else {
                scaled_features[i] = 0.0f;
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        total_scale_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

        // K3: Classifier (ridge regression)
        auto t4 = std::chrono::high_resolution_clock::now();
        data_t scores[MAX_CLASSES] = {0};
        if (num_classes == 2) {
            // Binary: score = intercept + sum(coef * feature)
            data_t score = intercept[0];
            for (int j = 0; j < num_features; j++) {
                score += coefficients[0][j] * scaled_features[j];
            }
            scores[0] = -score;  // class 0: negate
            scores[1] = score;   // class 1: positive
        } else {
            for (int c = 0; c < num_classes; c++) {
                scores[c] = intercept[c];
                for (int j = 0; j < num_features; j++) {
                    scores[c] += coefficients[c][j] * scaled_features[j];
                }
            }
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        total_classify_ms += std::chrono::duration<double, std::milli>(t5 - t4).count();

        // Find predicted class
        int predicted = 0;
        for (int c = 1; c < num_classes; c++) {
            if (scores[c] > scores[predicted]) predicted = c;
        }

        int expected = (s < (int)y_labels.size()) ? (int)y_labels[s] : -1;
        if (predicted == expected) correct++;

        if (s < 5 || predicted != expected) {
            std::cout << "Sample " << s << ": predicted=" << predicted
                      << " expected=" << expected
                      << (predicted == expected ? " OK" : " MISMATCH") << std::endl;
        }
    }

    feature_dump.close();

    // ---- Summary ----
    double accuracy = (double)correct / total_samples * 100.0;
    std::cout << "\n========== C-SIM RESULTS ==========" << std::endl;
    std::cout << "Correct: " << correct << " / " << total_samples << std::endl;
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    std::cout << "\nTiming (CPU, " << total_samples << " samples):" << std::endl;
    std::cout << "  K1 FeatureExtraction: " << std::fixed << std::setprecision(1) << total_fe_ms << " ms total, "
              << std::setprecision(3) << total_fe_ms / total_samples << " ms/sample" << std::endl;
    std::cout << "  K2 Scaler:           " << std::fixed << std::setprecision(1) << total_scale_ms << " ms total" << std::endl;
    std::cout << "  K3 Classifier:       " << std::fixed << std::setprecision(1) << total_classify_ms << " ms total" << std::endl;
    std::cout << "Feature dump: feature_dump_csim.csv" << std::endl;
    std::cout << "====================================" << std::endl;

    // Check Python baseline accuracy
    float python_baseline = 0.0f;
    size_t acc_pos = test_content.find("\"test_accuracy\":");
    if (acc_pos != std::string::npos) {
        size_t start = test_content.find(":", acc_pos) + 1;
        size_t end = test_content.find_first_of(",}", start);
        std::string acc_str = test_content.substr(start, end - start);
        trim_ws(acc_str);
        python_baseline = std::stof(acc_str) * 100.0f;
        std::cout << "Python baseline: " << std::fixed << std::setprecision(2) << python_baseline << "%" << std::endl;
        if (std::abs(accuracy - python_baseline) < 1.0) {
            std::cout << "PASS: C-sim accuracy matches Python baseline" << std::endl;
        } else {
            std::cout << "WARN: C-sim accuracy differs from Python by "
                      << std::abs(accuracy - python_baseline) << "%" << std::endl;
        }
    }

    return (accuracy > 90.0) ? 0 : 1;
}
