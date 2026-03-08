#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

// HLS types for C simulation
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"

// Shared types and constants
#include "../include/common.hpp"

// Kernel function declarations
#include "../feature_extraction/src/feature_extraction.hpp"
#include "../scaler/src/scaler.hpp"
#include "../classifier/src/classifier.hpp"

// Simple JSON loader (testbench only, not synthesizable)
// Inline minimal loader to avoid header conflicts with minirocket_loader.h

static std::string read_file_contents(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return "";
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}

static std::vector<float> parse_float_array(const std::string& content, const std::string& key) {
    std::vector<float> result;
    size_t pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) return result;
    pos = content.find("[", pos);
    if (pos == std::string::npos) return result;
    size_t end = content.find("]", pos);
    std::string arr = content.substr(pos + 1, end - pos - 1);
    std::stringstream ss(arr);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try { result.push_back(std::stof(token)); } catch (...) {}
    }
    return result;
}

static std::vector<int> parse_int_array(const std::string& content, const std::string& key) {
    std::vector<int> result;
    size_t pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) return result;
    pos = content.find("[", pos);
    if (pos == std::string::npos) return result;
    size_t end = content.find("]", pos);
    std::string arr = content.substr(pos + 1, end - pos - 1);
    std::stringstream ss(arr);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try { result.push_back(std::stoi(token)); } catch (...) {}
    }
    return result;
}

static int parse_int_value(const std::string& content, const std::string& key) {
    size_t pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) return 0;
    pos = content.find(":", pos);
    if (pos == std::string::npos) return 0;
    pos++;
    while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
    size_t end = content.find_first_of(",}\n", pos);
    return std::stoi(content.substr(pos, end - pos));
}

static std::vector<std::vector<float>> parse_2d_float_array(const std::string& content, const std::string& key) {
    std::vector<std::vector<float>> result;
    size_t pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) return result;
    // Find the outer opening bracket
    pos = content.find("[", pos);
    if (pos == std::string::npos) return result;
    pos++; // skip outer [

    int bracket_depth = 1;
    while (pos < content.size() && bracket_depth > 0) {
        // Skip whitespace
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\n' || content[pos] == '\r' || content[pos] == '\t' || content[pos] == ','))
            pos++;
        if (pos >= content.size()) break;

        if (content[pos] == ']') {
            // End of outer array
            break;
        }

        if (content[pos] == '[') {
            // Found inner array start
            pos++;
            size_t end = content.find("]", pos);
            if (end == std::string::npos) break;
            std::string row = content.substr(pos, end - pos);
            std::vector<float> row_vec;
            std::stringstream ss(row);
            std::string token;
            while (std::getline(ss, token, ',')) {
                // Trim whitespace from token
                size_t s = token.find_first_not_of(" \t\n\r");
                if (s == std::string::npos) continue;
                token = token.substr(s);
                try { row_vec.push_back(std::stof(token)); } catch (...) {}
            }
            if (!row_vec.empty()) result.push_back(row_vec);
            pos = end + 1;
        } else {
            pos++;
        }
    }
    return result;
}

int main(int argc, char** argv) {
    std::string model_file = "minirocket_model.json";
    std::string test_file = "minirocket_model_test_data.json";

    if (argc >= 3) {
        model_file = argv[1];
        test_file = argv[2];
    }

    std::cout << "=== MiniRocket Modular Pipeline C Simulation ===" << std::endl;
    std::cout << "Model: " << model_file << std::endl;
    std::cout << "Test:  " << test_file << std::endl;

    // Load model
    std::string model_content = read_file_contents(model_file);
    if (model_content.empty()) {
        std::cerr << "Failed to read model file: " << model_file << std::endl;
        return 1;
    }

    int num_features = parse_int_value(model_content, "num_features");
    int num_classes = parse_int_value(model_content, "num_classes");
    int num_dilations = parse_int_value(model_content, "num_dilations");
    int ts_length = parse_int_value(model_content, "time_series_length");

    std::cout << "Features: " << num_features << ", Classes: " << num_classes
              << ", Dilations: " << num_dilations << ", TS Length: " << ts_length << std::endl;

    // Parse model arrays
    auto biases_vec = parse_float_array(model_content, "biases");
    auto scaler_mean_vec = parse_float_array(model_content, "scaler_mean");
    auto scaler_scale_vec = parse_float_array(model_content, "scaler_scale");
    auto dilations_vec = parse_int_array(model_content, "dilations");
    auto nfpd_vec = parse_int_array(model_content, "num_features_per_dilation");
    auto intercept_vec = parse_float_array(model_content, "classifier_intercept");
    // classifier_coef may be 1D (binary) or 2D (multi-class)
    auto coef_2d = parse_2d_float_array(model_content, "classifier_coef");
    if (coef_2d.empty()) {
        // Flat 1D array — treat as single row for binary classification
        auto coef_flat = parse_float_array(model_content, "classifier_coef");
        if (!coef_flat.empty()) {
            coef_2d.push_back(coef_flat);
        }
    }

    // Populate HLS-compatible arrays
    data_t h_biases[MAX_FEATURES] = {0};
    data_t h_scaler_mean[MAX_FEATURES] = {0};
    data_t h_inv_scale[MAX_FEATURES] = {0};
    int_t  h_dilations[MAX_DILATIONS] = {0};
    int_t  h_nfpd[MAX_DILATIONS] = {0};
    data_t h_intercept[MAX_CLASSES] = {0};
    data_t h_coefficients[MAX_CLASSES * MAX_FEATURES] = {0};

    for (size_t i = 0; i < biases_vec.size() && i < MAX_FEATURES; i++)
        h_biases[i] = biases_vec[i];
    for (size_t i = 0; i < scaler_mean_vec.size() && i < MAX_FEATURES; i++)
        h_scaler_mean[i] = scaler_mean_vec[i];
    for (size_t i = 0; i < scaler_scale_vec.size() && i < MAX_FEATURES; i++)
        h_inv_scale[i] = (scaler_scale_vec[i] != 0.0f) ? (1.0f / scaler_scale_vec[i]) : 0.0f;
    for (size_t i = 0; i < dilations_vec.size() && i < MAX_DILATIONS; i++)
        h_dilations[i] = dilations_vec[i];
    for (size_t i = 0; i < nfpd_vec.size() && i < MAX_DILATIONS; i++)
        h_nfpd[i] = nfpd_vec[i];
    for (size_t i = 0; i < intercept_vec.size() && i < MAX_CLASSES; i++)
        h_intercept[i] = intercept_vec[i];
    for (size_t i = 0; i < coef_2d.size() && i < MAX_CLASSES; i++)
        for (size_t j = 0; j < coef_2d[i].size() && j < MAX_FEATURES; j++)
            h_coefficients[i * num_features + j] = coef_2d[i][j];

    // Load test data
    std::string test_content = read_file_contents(test_file);
    if (test_content.empty()) {
        std::cerr << "Failed to read test file: " << test_file << std::endl;
        return 1;
    }

    auto test_2d = parse_2d_float_array(test_content, "X_test");
    auto y_test = parse_int_array(test_content, "y_test");

    std::cout << "Test samples: " << test_2d.size() << std::endl;

    // Run modular pipeline for each test sample
    int correct = 0;
    data_t h_time_series[MAX_TIME_SERIES_LENGTH] = {0};
    data_t h_predictions[MAX_CLASSES] = {0};

    for (size_t sample = 0; sample < test_2d.size(); sample++) {
        // Prepare input
        for (size_t j = 0; j < test_2d[sample].size() && j < MAX_TIME_SERIES_LENGTH; j++)
            h_time_series[j] = test_2d[sample][j];

        // ---- K1: Feature Extraction ----
        hls::stream<data_t> features_stream("features_stream");

        feature_extraction(
            h_time_series,
            h_dilations,
            h_nfpd,
            h_biases,
            features_stream,
            (int_t)ts_length,
            (int_t)num_features,
            (int_t)num_dilations
        );

        // ---- K2: Scaler ----
        hls::stream<data_t> scaled_features_stream("scaled_features_stream");

        scaler(
            h_scaler_mean,
            h_inv_scale,
            features_stream,
            scaled_features_stream,
            (int_t)num_features
        );

        // ---- K3: Classifier ----
        classifier(
            h_coefficients,
            h_intercept,
            h_predictions,
            scaled_features_stream,
            (int_t)num_features,
            (int_t)num_classes
        );

        // Find predicted class
        int predicted_class = 0;
        float max_score = h_predictions[0];
        for (int c = 0; c < num_classes; c++) {
            if (h_predictions[c] > max_score) {
                max_score = h_predictions[c];
                predicted_class = c;
            }
        }

        int expected_class = (sample < y_test.size()) ? y_test[sample] : 0;
        bool is_correct = (predicted_class == expected_class);
        if (is_correct) correct++;

        std::cout << "Sample[" << sample << "]: predicted=" << predicted_class
                  << " expected=" << expected_class
                  << (is_correct ? " OK" : " FAIL") << std::endl;
    }

    double accuracy = (test_2d.size() > 0) ? ((double)correct / test_2d.size()) * 100.0 : 0.0;
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Correct: " << correct << " / " << test_2d.size() << std::endl;
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy << " %" << std::endl;

    if (accuracy < 90.0) {
        std::cerr << "WARNING: Accuracy below 90% threshold!" << std::endl;
        return 1;
    }

    std::cout << "TEST PASSED" << std::endl;
    return 0;
}
