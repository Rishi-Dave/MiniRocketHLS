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
    
    hls::stream<pkt> input_timeseries;
    hls::stream<pkt> output_predictions;   

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
    
    std::cout << "Input Time Series: [";
    for (int_t i = 0; i < (int)test_inputs[0].size(); i++) {
        std::cout << test_inputs[0][i] << " ";
    }
    std::cout << "]" << std::endl;

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

    int num_tests = std::min((int)test_inputs[0].size(), (csim) ? 1000 : 100); 

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "C++ MiniRocket Step-by-Step Comparison (Test Sample 1)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    int input_length = std::min((int)test_inputs[0].size(), MAX_TIME_SERIES_LENGTH);

    std::cout << "Running in quick test mode (limited to 100 samples)..." << std::endl;
    for (int test_idx = 0; test_idx < num_tests; test_idx++) {

        pkt in_pkt;
        in_pkt.keep = -1;
        in_pkt.last = 1;
        in_pkt.data = *((ap_uint<DWIDTH>*)&test_inputs[0][test_idx]);
        input_timeseries.write(in_pkt);

        minirocket_inference(
            input_timeseries,
            output_predictions,
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

        data_t* predictions = new data_t[num_classes];

        std::cout << "Received predictions: [";
        for (int i = 0; i < num_classes; i++) {
            pkt out_pkt;
            output_predictions.read(out_pkt);
            ap_uint< DWIDTH > out_data = out_pkt.data;
            predictions[i] = *((data_t*)&out_data); 
            std::cout << predictions[i] << " ";
        }
        std::cout << "]" << std::endl;

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

        // if (test_idx % 10 == 9 || test_idx == num_tests - 1) {
        //     float current_accuracy = (float)num_correct / (test_idx + 1) * 100.0f;
        //     std::cout << "  Completed " << (test_idx + 1) << " tests - Accuracy: "
        //             << std::fixed << std::setprecision(1) << current_accuracy
        //             << "% (" << num_correct << "/" << (test_idx + 1) << " correct)" << std::endl;
        // }

        delete[] predictions;
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