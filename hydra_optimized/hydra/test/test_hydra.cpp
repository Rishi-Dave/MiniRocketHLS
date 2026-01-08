#include "../include/hydra.hpp"
#include "hydra_hls_testbench_loader.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

int main(int argc, char* argv[]) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  HYDRA HLS Testbench" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Parse command line arguments
    std::string model_file = "hydra_model.json";
    std::string test_file = "hydra_test.json";
    std::string test_mode = "csim";

    if (argc >= 2) model_file = argv[1];
    if (argc >= 3) test_file = argv[2];
    if (argc >= 4) test_mode = argv[3];

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Model file: " << model_file << std::endl;
    std::cout << "  Test file: " << test_file << std::endl;
    std::cout << "  Test mode: " << test_mode << std::endl;
    std::cout << std::endl;

    // Load model parameters
    HydraModelParams_HLS model;
    if (!load_hydra_model(model_file, &model)) {
        std::cerr << "ERROR: Failed to load model" << std::endl;
        return 1;
    }

    print_model_summary(model);

    // Load test data
    TestTimeSeriesData test_data;
    if (!load_test_data(test_file, &test_data)) {
        std::cerr << "ERROR: Failed to load test data" << std::endl;
        return 1;
    }

    print_test_data_summary(test_data);

    // Allocate buffers for kernel interface
    data_t* time_series_input = new data_t[MAX_TIME_SERIES_LENGTH];
    data_t* prediction_output = new data_t[MAX_CLASSES];

    // Storage for all predictions
    std::vector<std::vector<data_t>> all_predictions;

    std::cout << "Running HYDRA inference..." << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Process each test sample
    for (int_t sample_idx = 0; sample_idx < test_data.num_samples; sample_idx++) {
        std::cout << "\nSample " << sample_idx + 1 << "/" << test_data.num_samples << std::endl;

        // Copy time series to input buffer
        for (int_t t = 0; t < test_data.time_series_length; t++) {
            time_series_input[t] = test_data.X_test[sample_idx][t];
        }

        // Initialize prediction buffer
        for (int_t c = 0; c < model.num_classes; c++) {
            prediction_output[c] = 0.0;
        }

        // Call HYDRA inference kernel
        hydra_inference(
            time_series_input,
            prediction_output,
            model.coefficients,
            model.intercept,
            model.scaler_mean,
            model.scaler_scale,
            model.kernel_weights,
            model.biases,
            model.dilations,
            test_data.time_series_length,
            model.num_features,
            model.num_classes,
            NUM_GROUPS
        );

        // Store predictions
        std::vector<data_t> pred_vec;
        for (int_t c = 0; c < model.num_classes; c++) {
            pred_vec.push_back(prediction_output[c]);
        }
        all_predictions.push_back(pred_vec);

        // Print prediction scores
        std::cout << "  Prediction scores: [";
        for (int_t c = 0; c < model.num_classes; c++) {
            std::cout << std::fixed << std::setprecision(4) << prediction_output[c];
            if (c < model.num_classes - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Find predicted class
        int_t predicted_class = 0;
        data_t max_score = prediction_output[0];
        for (int_t c = 1; c < model.num_classes; c++) {
            if (prediction_output[c] > max_score) {
                max_score = prediction_output[c];
                predicted_class = c;
            }
        }

        int_t true_class = test_data.y_test[sample_idx];
        bool correct = (predicted_class == true_class);

        std::cout << "  Predicted class: " << predicted_class << std::endl;
        std::cout << "  True class: " << true_class << std::endl;
        std::cout << "  " << (correct ? "✓ CORRECT" : "✗ INCORRECT") << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Validation Results" << std::endl;
    std::cout << "========================================\n" << std::endl;

    data_t accuracy = validate_predictions(
        all_predictions,
        test_data.y_test,
        test_data.num_samples,
        model.num_classes
    );

    std::cout << "\n========================================" << std::endl;

    // Determine pass/fail
    bool test_passed = true;

    // For synthetic data, we expect some reasonable predictions
    // (not necessarily 100% accuracy since weights are random)
    data_t min_accuracy = 0.3;  // At least 30% for random test data

    if (accuracy >= min_accuracy) {
        std::cout << "TEST PASSED" << std::endl;
    } else {
        std::cout << "TEST FAILED (accuracy too low)" << std::endl;
        test_passed = false;
    }

    std::cout << "========================================\n" << std::endl;

    // Cleanup
    delete[] time_series_input;
    delete[] prediction_output;

    return test_passed ? 0 : 1;
}
