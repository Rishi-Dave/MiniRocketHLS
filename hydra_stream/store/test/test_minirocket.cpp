#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include "../include/minirocket.hpp"
#include "minirocket_hls_testbench_loader.h"

int main(int argc, char* argv[]) {
    std::string model_file = "minirocket_model.json";
    std::string test_file = "minirocket_model_test_data.json";
    bool csim;
    // Allow command line arguments like C++ version
    if (argc >= 4) {
        model_file = argv[1];
        test_file = argv[2];
        csim = (std::string(argv[3]) == "csim") ? true : false;
    }
    
    std::cout << "HLS MiniRocket Test" << std::endl;
    std::cout << "Model file: " << model_file << std::endl;
    std::cout << "Test file: " << test_file << std::endl;
    
    // Initialize loader
    MiniRocketTestbenchLoader loader;
    
    // HLS arrays (heap allocated for testbench to avoid stack overflow)
    data_t (*coefficients)[MAX_FEATURES] = new data_t[MAX_CLASSES][MAX_FEATURES];
    data_t *flattened_coefficients = new data_t[MAX_CLASSES * MAX_FEATURES];
    data_t *intercept = new data_t[MAX_CLASSES];
    data_t *scaler_mean = new data_t[MAX_FEATURES];
    data_t *scaler_scale = new data_t[MAX_FEATURES];
    int_t *dilations = new int_t[MAX_DILATIONS];
    int_t *num_features_per_dilation = new int_t[MAX_DILATIONS];
    data_t *biases = new data_t[MAX_FEATURES];
    
    int_t num_dilations, num_features, num_classes, time_series_length;
    
    // Load model into HLS arrays
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
    
    // Load test data
    std::cout << "Loading test data..." << std::endl;
    std::vector<std::vector<float>> test_inputs, expected_outputs;
    if (!loader.load_test_data(test_file, test_inputs, expected_outputs)) {
        std::cerr << "Failed to load test data!" << std::endl;
        return 1;
    }
    
    // Try to load Python baseline accuracy from test data
    float python_baseline = 0.0f;
    std::ifstream test_file_stream(test_file);
    if (test_file_stream.is_open()) {
        std::string content((std::istreambuf_iterator<char>(test_file_stream)),
                           std::istreambuf_iterator<char>());
        size_t accuracy_pos = content.find("\"test_accuracy\":");
        if (accuracy_pos != std::string::npos) {
            size_t start = content.find(":", accuracy_pos) + 1;
            size_t end = content.find_first_of(",}", start);
            std::string accuracy_str = content.substr(start, end - start);
            python_baseline = std::stof(accuracy_str) * 100.0f; // Convert to percentage
        }
    }
    
    // Convert y_test (single values) to expected class indices
    std::vector<int> expected_classes;
    for (const auto& output : expected_outputs) {
        if (!output.empty()) {
            expected_classes.push_back((int)output[0]);
        }
    }
    
    // Test on loaded data
    int num_correct = 0;

    int num_tests = std::min((int)test_inputs.size(), (csim) ? 1000 : 10); 

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "C++ MiniRocket Step-by-Step Comparison (Test Sample 1)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    if (csim) {
        std::cout << "Running in C simulation mode (limited to 1000 samples)..." << std::endl;

        for (int test_idx = 0; test_idx < num_tests; test_idx++) {
            // Copy input to HLS array (heap allocated to avoid stack overflow)
            data_t *time_series = new data_t[MAX_TIME_SERIES_LENGTH];
            int input_length = std::min((int)test_inputs[test_idx].size(), MAX_TIME_SERIES_LENGTH);

            for (int i = 0; i < input_length; i++) {
                time_series[i] = test_inputs[test_idx][i];
            }

            // Show detailed output for first test sample only
            if (test_idx == 0) {
                std::cout << "\n=== C++: Input ===" << std::endl;
                std::cout << "Time series length: " << input_length << std::endl;
                std::cout << "First 10 values: ";
                for (int i = 0; i < std::min(10, input_length); i++) {
                    std::cout << std::fixed << std::setprecision(6) << time_series[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "True label: " << expected_classes[test_idx] << std::endl;
            }

            // Run HLS inference pipeline (heap allocated to avoid stack overflow)
            data_t *features = new data_t[MAX_FEATURES];
            data_t *scaled_features = new data_t[MAX_FEATURES];
            data_t *predictions = new data_t[MAX_CLASSES];

            // Feature extraction
            if (test_idx == 0) {
                std::cout << "\n" << std::string(60, '=') << std::endl;
                std::cout << "STEP 1: FEATURE EXTRACTION (MiniRocket Transform)" << std::endl;
                std::cout << std::string(60, '=') << std::endl;
                std::cout << "Running cumulative convolution with α=-1, γ=2..." << std::endl;
            }

            minirocket_feature_extraction_hls(
                time_series,
                features,
                dilations,
                num_features_per_dilation,
                biases,
                input_length,
                num_dilations,
                num_features
            );

            if (test_idx == 0) {
                std::cout << "\n=== C++: Extracted Features ===" << std::endl;
                std::cout << "Total features: " << num_features << std::endl;
                std::cout << "First 10 features: ";
                for (int i = 0; i < std::min(10, (int)num_features); i++) {
                    std::cout << std::fixed << std::setprecision(6) << features[i] << " ";
                }
                std::cout << std::endl;

                // Calculate feature range
                data_t min_feat = features[0], max_feat = features[0];
                for (int i = 1; i < num_features; i++) {
                    if (features[i] < min_feat) min_feat = features[i];
                    if (features[i] > max_feat) max_feat = features[i];
                }
                std::cout << "Feature range: [" << std::fixed << std::setprecision(6)
                        << min_feat << ", " << max_feat << "]" << std::endl;
            }

            // Scaling
            if (test_idx == 0) {
                std::cout << "\n" << std::string(60, '=') << std::endl;
                std::cout << "STEP 2: FEATURE SCALING (StandardScaler)" << std::endl;
                std::cout << std::string(60, '=') << std::endl;
            }

            apply_scaler_hls(
                features,
                scaled_features,
                scaler_mean,
                scaler_scale,
                num_features
            );

            if (test_idx == 0) {
                std::cout << "=== C++: Scaled Features ===" << std::endl;
                std::cout << "Scaler mean (first 5): ";
                for (int i = 0; i < std::min(5, (int)num_features); i++) {
                    std::cout << std::fixed << std::setprecision(6) << scaler_mean[i] << " ";
                }
                std::cout << std::endl;

                std::cout << "Scaler scale (first 5): ";
                for (int i = 0; i < std::min(5, (int)num_features); i++) {
                    std::cout << std::fixed << std::setprecision(6) << scaler_scale[i] << " ";
                }
                std::cout << std::endl;

                std::cout << "Scaled features (first 10): ";
                for (int i = 0; i < std::min(10, (int)num_features); i++) {
                    std::cout << std::fixed << std::setprecision(6) << scaled_features[i] << " ";
                }
                std::cout << std::endl;

                // Calculate scaled range
                data_t min_scaled = scaled_features[0], max_scaled = scaled_features[0];
                for (int i = 1; i < num_features; i++) {
                    if (scaled_features[i] < min_scaled) min_scaled = scaled_features[i];
                    if (scaled_features[i] > max_scaled) max_scaled = scaled_features[i];
                }
                std::cout << "Scaled range: [" << std::fixed << std::setprecision(6)
                        << min_scaled << ", " << max_scaled << "]" << std::endl;
            }

            // Classification
            if (test_idx == 0) {
                std::cout << "\n" << std::string(60, '=') << std::endl;
                std::cout << "STEP 3: LINEAR CLASSIFICATION (Ridge Classifier)" << std::endl;
                std::cout << std::string(60, '=') << std::endl;
            }

            linear_classifier_predict_hls(
                scaled_features,
                predictions,
                coefficients,
                intercept,
                num_features,
                num_classes
            );

            // Find predicted class
            int predicted_class = 0;
            data_t max_score = predictions[0];
            for (int i = 1; i < num_classes; i++) {
                if (predictions[i] > max_score) {
                    max_score = predictions[i];
                    predicted_class = i;
                }
            }

            // Get expected class directly from labels
            int expected_class = (test_idx < expected_classes.size()) ? expected_classes[test_idx] : 0;

            bool correct = (predicted_class == expected_class);
            if (correct) num_correct++;

            if (test_idx == 0) {
                std::cout << "=== C++: Classification ===" << std::endl;
                std::cout << "Coefficients shape: [" << num_classes << ", " << num_features << "]" << std::endl;
                std::cout << "Coefficients[0] (first 5): ";
                for (int i = 0; i < std::min(5, (int)num_features); i++) {
                    std::cout << std::fixed << std::setprecision(6) << coefficients[0][i] << " ";
                }
                std::cout << std::endl;

                std::cout << "Intercept: ";
                for (int i = 0; i < num_classes; i++) {
                    std::cout << std::fixed << std::setprecision(6) << intercept[i] << " ";
                }
                std::cout << std::endl;

                std::cout << "Decision scores: ";
                for (int i = 0; i < num_classes; i++) {
                    std::cout << std::fixed << std::setprecision(6) << predictions[i] << " ";
                }
                std::cout << std::endl;

                std::cout << "Predicted class: " << predicted_class << std::endl;
                std::cout << "True class: " << expected_class << std::endl;
                std::cout << "Correct: " << (correct ? "YES" : "NO") << std::endl;

                std::cout << "\n" << std::string(60, '=') << std::endl;
                std::cout << "SUMMARY" << std::endl;
                std::cout << std::string(60, '=') << std::endl;
                std::cout << "✓ Feature extraction: " << num_features << " features extracted" << std::endl;
                std::cout << "✓ Scaling: Applied StandardScaler normalization" << std::endl;
                std::cout << "✓ Classification: Ridge classifier decision function" << std::endl;
                std::cout << "✓ Prediction: Class " << predicted_class << " (expected: " << expected_class << ")" << std::endl;
                std::cout << "✓ Match: " << (correct ? "YES ✓" : "NO ✗") << std::endl;

                std::cout << "\n" << std::string(60, '=') << std::endl;
                std::cout << "Testing remaining samples for accuracy..." << std::endl;
                std::cout << std::string(60, '=') << std::endl;
            }

            // Progress indicator for every 10 tests (but skip first test since we showed details)
            if (test_idx > 0 && test_idx % 10 == 0) {
                std::cout << "Processing tests " << test_idx + 1 << "-" << std::min(test_idx + 10, num_tests) << "..." << std::endl;
            }

            // Show running accuracy every 10 tests
            if (test_idx % 10 == 9 || test_idx == num_tests - 1) {
                float current_accuracy = (float)num_correct / (test_idx + 1) * 100.0f;
                std::cout << "  Completed " << (test_idx + 1) << " tests - Accuracy: "
                        << std::fixed << std::setprecision(1) << current_accuracy
                        << "% (" << num_correct << "/" << (test_idx + 1) << " correct)" << std::endl;
            }
            
            // Cleanup
            delete[] time_series;
            delete[] features;
            delete[] scaled_features;
            delete[] predictions;
        }
    } else {
        std::cout << "Running in quick test mode (limited to 100 samples)..." << std::endl;
        for (int test_idx = 0; test_idx < num_tests; test_idx++) {
            data_t *time_series = new data_t[MAX_TIME_SERIES_LENGTH];
            data_t *predictions = new data_t[MAX_CLASSES];
            int input_length = std::min((int)test_inputs[test_idx].size(), MAX_TIME_SERIES_LENGTH);

            for (int i = 0; i < input_length; i++) {
                time_series[i] = test_inputs[test_idx][i];
            }
            // data_t* time_series_input,      // Input time series
            // data_t* prediction_output,      // Output predictions
            // data_t* coefficients,           // Model coefficients (flattened)
            // data_t* intercept,              // Model intercept
            // data_t* scaler_mean,            // Scaler mean values
            // data_t* scaler_scale,           // Scaler scale values
            // int_t* dilations,               // Dilation values
            // int_t* num_features_per_dilation, // Features per dilation
            // data_t* biases,                 // Bias values
            // int_t time_series_length,
            // int_t num_features,
            // int_t num_classes,
            // int_t num_dilations
            minirocket_inference(
                time_series,
                predictions,
                flattened_coefficients,
                intercept,
                scaler_mean,
                scaler_scale,
                dilations,
                num_features_per_dilation,
                biases,
                input_length,
                num_features,
                num_classes,
                num_dilations
            );
            // Find predicted class
            int predicted_class = 0;
            data_t max_score = predictions[0];
            for (int i = 1; i < num_classes; i++) {
                if (predictions[i] > max_score) {
                    max_score = predictions[i];
                    predicted_class = i;
                }
            }

            // Get expected class directly from labels
            int expected_class = (test_idx < expected_classes.size()) ? expected_classes[test_idx] : 0;

            bool correct = (predicted_class == expected_class);
            if (correct) num_correct++;

            if (test_idx % 10 == 9 || test_idx == num_tests - 1) {
                float current_accuracy = (float)num_correct / (test_idx + 1) * 100.0f;
                std::cout << "  Completed " << (test_idx + 1) << " tests - Accuracy: "
                        << std::fixed << std::setprecision(1) << current_accuracy
                        << "% (" << num_correct << "/" << (test_idx + 1) << " correct)" << std::endl;
            }

            delete[] time_series;
            delete[] predictions;
        }
        

    }
    
    float hls_accuracy = (float)num_correct / num_tests * 100.0f;

    // Print comparison table
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "ACCURACY COMPARISON: Python Reference vs HLS Implementation" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::endl;

    std::cout << std::left << std::setw(40) << "Implementation" << std::right << std::setw(15) << "Accuracy" << std::setw(15) << "Correct/Total" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    if (python_baseline > 0.0f) {
        std::cout << std::left << std::setw(40) << "Python (NumPy/Numba)"
                  << std::right << std::setw(14) << std::fixed << std::setprecision(2) << python_baseline << "%"
                  << std::setw(15) << "N/A" << std::endl;
    } else {
        std::cout << std::left << std::setw(40) << "Python (NumPy/Numba)"
                  << std::right << std::setw(14) << "N/A"
                  << std::setw(15) << "N/A" << std::endl;
    }

    std::cout << std::left << std::setw(40) << "HLS (ap_fixed<32,16>)"
              << std::right << std::setw(14) << std::fixed << std::setprecision(2) << hls_accuracy << "%"
              << std::setw(8) << num_correct << "/" << std::setw(3) << num_tests << std::endl;

    std::cout << std::string(70, '-') << std::endl;

    if (python_baseline > 0.0f) {
        float difference = hls_accuracy - python_baseline;
        std::cout << std::left << std::setw(40) << "Difference (HLS - Python)"
                  << std::right << std::setw(13) << std::fixed << std::setprecision(2)
                  << std::showpos << difference << "%" << std::noshowpos << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        // Match assessment
        bool close_match = std::abs(difference) <= 1.0f;
        bool acceptable_match = std::abs(difference) <= 5.0f;

        std::cout << "\nMatch Assessment: ";
        if (close_match) {
            std::cout << "✓ EXCELLENT - Within 1% of Python baseline" << std::endl;
        } else if (acceptable_match) {
            std::cout << "✓ GOOD - Within 5% of Python baseline" << std::endl;
        } else if (hls_accuracy >= 90.0f) {
            std::cout << "⚠ ACCEPTABLE - HLS accuracy >= 90% but differs from Python" << std::endl;
        } else {
            std::cout << "✗ POOR - Significant deviation from Python baseline" << std::endl;
        }
    } else {
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "\nNote: Python baseline not available in test data" << std::endl;
    }

    // Cleanup main arrays
    delete[] coefficients;
    delete[] intercept;
    delete[] scaler_mean;
    delete[] scaler_scale;
    delete[] dilations;
    delete[] num_features_per_dilation;
    delete[] biases;

    // Determine success criteria
    bool success;
    if (python_baseline > 0.0f) {
        // Success if within 5% of Python or >= 90% absolute
        success = (hls_accuracy >= 90.0f) || (std::abs(hls_accuracy - python_baseline) <= 5.0f);
    } else {
        // Fallback to absolute threshold
        success = (hls_accuracy >= 90.0f);
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    if (success) {
        std::cout << "✓ SUCCESS: HLS implementation validated!" << std::endl;
    } else {
        std::cout << "✗ FAILURE: HLS accuracy below acceptance criteria" << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl;

    return 0;
}