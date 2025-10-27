#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include "minirocket_inference_hls.h"
#include "minirocket_hls_testbench_loader.h"

int main(int argc, char* argv[]) {
    std::string model_file = "minirocket_model.json";
    std::string test_file = "minirocket_model_test_data.json";
    
    // Allow command line arguments like C++ version
    if (argc >= 3) {
        model_file = argv[1];
        test_file = argv[2];
    }
    
    std::cout << "HLS MiniRocket Test" << std::endl;
    std::cout << "Model file: " << model_file << std::endl;
    std::cout << "Test file: " << test_file << std::endl;
    
    // Initialize loader
    MiniRocketTestbenchLoader loader;
    
    // HLS arrays (heap allocated for testbench to avoid stack overflow)
    data_t (*coefficients)[MAX_FEATURES] = new data_t[MAX_CLASSES][MAX_FEATURES];
    data_t *intercept = new data_t[MAX_CLASSES];
    data_t *scaler_mean = new data_t[MAX_FEATURES];
    data_t *scaler_scale = new data_t[MAX_FEATURES];
    int_t *dilations = new int_t[MAX_DILATIONS];
    int_t *num_features_per_dilation = new int_t[MAX_DILATIONS];
    data_t *biases = new data_t[MAX_FEATURES];
    
    int_t num_dilations, num_features, num_classes, time_series_length;
    
    // Load model into HLS arrays
    if (!loader.load_model_to_hls_arrays(model_file, coefficients, intercept, 
                                        scaler_mean, scaler_scale, dilations,
                                        num_features_per_dilation, biases,
                                        num_dilations, num_features, num_classes,
                                        time_series_length)) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }
    
    // Load test data
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
    int num_tests = std::min((int)test_inputs.size(), 100); // Test first 100 samples
    
    std::cout << "\nTesting HLS implementation with " << num_tests << " samples:" << std::endl;
    std::cout << "Measuring accuracy against ground truth labels (y_test)" << std::endl;
    
    for (int test_idx = 0; test_idx < num_tests; test_idx++) {
        // Progress indicator for every 10 tests
        if (test_idx % 10 == 0) {
            std::cout << "Processing tests " << test_idx + 1 << "-" << std::min(test_idx + 10, num_tests) << "..." << std::endl;
        }
        
        // Copy input to HLS array (heap allocated to avoid stack overflow)
        data_t *time_series = new data_t[MAX_TIME_SERIES_LENGTH];
        int input_length = std::min((int)test_inputs[test_idx].size(), MAX_TIME_SERIES_LENGTH);
        
        for (int i = 0; i < input_length; i++) {
            time_series[i] = test_inputs[test_idx][i];
        }
        
        // Run HLS inference pipeline (heap allocated to avoid stack overflow)
        data_t *features = new data_t[MAX_FEATURES];
        data_t *scaled_features = new data_t[MAX_FEATURES];
        data_t *predictions = new data_t[MAX_CLASSES];
        
        // Feature extraction
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
        
        // Scaling
        apply_scaler_hls(
            features,
            scaled_features,
            scaler_mean,
            scaler_scale,
            num_features
        );
        
        // Classification
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
        
        // Show running accuracy every 10 tests
        if (test_idx % 10 == 9 || test_idx == num_tests - 1) {
            float current_accuracy = (float)num_correct / (test_idx + 1) * 100.0f;
            std::cout << "  Completed " << (test_idx + 1) << " tests - Accuracy: " 
                     << std::fixed << std::setprecision(1) << current_accuracy 
                     << "% (" << num_correct << "/" << (test_idx + 1) << " correct)" << std::endl;
        }
        
        // Show prediction scores for first test only
        if (test_idx == 0) {
            std::cout << "Sample prediction scores - Test 1: ";
            for (int i = 0; i < num_classes; i++) {
                std::cout << "C" << i << "=" << std::fixed << std::setprecision(3) 
                         << predictions[i] << " ";
            }
            std::cout << " → Predicted=" << predicted_class << ", Expected=" << expected_class << std::endl;
        }
        
        // Cleanup
        delete[] time_series;
        delete[] features;
        delete[] scaled_features;
        delete[] predictions;
    }
    
    float accuracy = (float)num_correct / num_tests * 100.0f;
    std::cout << "\n=== HLS IMPLEMENTATION RESULTS ===" << std::endl;
    std::cout << "Ground Truth Accuracy: " << num_correct << "/" << num_tests 
              << " correct (" << std::fixed << std::setprecision(1) 
              << accuracy << "% accuracy)" << std::endl;
    
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
        // If we have baseline, success if within 5% or >= 90%
        success = (accuracy >= 90.0f) || (std::abs(accuracy - python_baseline) <= 5.0f);
        std::cout << "Python baseline: " << std::fixed << std::setprecision(1) << python_baseline << "%" << std::endl;
    } else {
        // Fallback to absolute threshold
        success = (accuracy >= 90.0f);
    }
    
    if (success) {
        std::cout << "SUCCESS: HLS implementation achieves good accuracy!" << std::endl;
        return 0;
    } else {
        std::cout << "WARNING: HLS accuracy significantly below expectations" << std::endl;
        return 1;
    }
}