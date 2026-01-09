#include "hydra_hls_testbench_loader.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

// Simple JSON parser - for production use nlohmann/json library
// This is a minimal implementation for HLS testbench

/**
 * Parse a simple JSON array of floats
 * Format: [1.0, 2.0, 3.0]
 */
std::vector<data_t> parse_float_array(const std::string& json_str) {
    std::vector<data_t> result;
    std::string str = json_str;

    // Remove brackets and whitespace
    size_t start = str.find('[');
    size_t end = str.find(']');
    if (start == std::string::npos || end == std::string::npos) {
        return result;
    }

    str = str.substr(start + 1, end - start - 1);

    // Parse comma-separated values
    std::istringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            result.push_back(std::stof(token));
        } catch (...) {
            // Skip invalid values
        }
    }

    return result;
}

/**
 * Parse a simple JSON array of integers
 */
std::vector<int_t> parse_int_array(const std::string& json_str) {
    std::vector<int_t> result;
    std::string str = json_str;

    size_t start = str.find('[');
    size_t end = str.find(']');
    if (start == std::string::npos || end == std::string::npos) {
        return result;
    }

    str = str.substr(start + 1, end - start - 1);

    std::istringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            result.push_back(std::stoi(token));
        } catch (...) {
            // Skip invalid values
        }
    }

    return result;
}

bool load_hydra_model(
    const std::string& filename,
    HydraModelParams_HLS* model
) {
    std::cout << "Loading HYDRA model from: " << filename << std::endl;

    // Initialize with default values
    model->num_features = 1024;  // 512 kernels × 2 pooling = 1024
    model->num_classes = 2;
    model->time_series_length = 150;

    // Initialize kernel weights with simple pattern
    for (int k = 0; k < NUM_KERNELS; k++) {
        for (int w = 0; w < KERNEL_SIZE; w++) {
            // Simple pattern: alternating positive/negative
            data_t weight = (w % 3 == 0) ? 1.0 : ((w % 3 == 1) ? -0.5 : 0.5);
            // Add slight variation based on kernel index
            weight *= (1.0 + 0.01 * (k % 10));
            model->kernel_weights[k * KERNEL_SIZE + w] = weight;
        }
        model->biases[k] = 0.0;
        model->group_assignments[k] = k / KERNELS_PER_GROUP;  // Group 0-7
        model->dilations[k] = 1 + (k % MAX_DILATIONS);  // Dilations 1-8
    }

    // Initialize scaler (identity scaling for testing)
    for (int f = 0; f < model->num_features; f++) {
        model->scaler_mean[f] = 0.0;
        model->scaler_scale[f] = 1.0;
    }

    // Initialize classifier (simple random weights for testing)
    for (int f = 0; f < model->num_features; f++) {
        for (int c = 0; c < model->num_classes; c++) {
            data_t coef = 0.01 * ((f + c) % 10 - 5);  // Small random-ish values
            model->coefficients[f * model->num_classes + c] = coef;
        }
    }

    for (int c = 0; c < model->num_classes; c++) {
        model->intercept[c] = 0.0;
    }

    std::cout << "Model loaded successfully (using default test values)" << std::endl;
    std::cout << "  Kernels: " << NUM_KERNELS << std::endl;
    std::cout << "  Features: " << model->num_features << std::endl;
    std::cout << "  Classes: " << model->num_classes << std::endl;

    return true;
}

bool load_test_data(
    const std::string& filename,
    TestTimeSeriesData* test_data
) {
    std::cout << "Loading test data from: " << filename << std::endl;

    // Generate simple synthetic test data
    test_data->num_samples = 10;
    test_data->time_series_length = 150;
    test_data->num_classes = 2;

    test_data->X_test.clear();
    test_data->y_test.clear();

    for (int s = 0; s < test_data->num_samples; s++) {
        std::vector<data_t> time_series;

        // Generate sinusoidal pattern with noise
        for (int t = 0; t < test_data->time_series_length; t++) {
            data_t value = std::sin(2.0 * M_PI * t / 30.0);
            value += 0.1 * (rand() % 100 - 50) / 50.0;  // Add noise
            if (s >= 5) value *= -1;  // Second class is inverted
            time_series.push_back(value);
        }

        test_data->X_test.push_back(time_series);
        test_data->y_test.push_back(s < 5 ? 0 : 1);  // Binary classification
    }

    std::cout << "Test data generated successfully" << std::endl;
    std::cout << "  Samples: " << test_data->num_samples << std::endl;
    std::cout << "  Length: " << test_data->time_series_length << std::endl;
    std::cout << "  Classes: " << test_data->num_classes << std::endl;

    return true;
}

data_t validate_predictions(
    const std::vector<std::vector<data_t>>& predictions,
    const std::vector<int_t>& labels,
    int_t num_samples,
    int_t num_classes,
    data_t tolerance
) {
    int_t correct = 0;

    for (int_t i = 0; i < num_samples; i++) {
        // Find predicted class (argmax)
        int_t predicted_class = 0;
        data_t max_score = predictions[i][0];

        for (int_t c = 1; c < num_classes; c++) {
            if (predictions[i][c] > max_score) {
                max_score = predictions[i][c];
                predicted_class = c;
            }
        }

        if (predicted_class == labels[i]) {
            correct++;
        }

        std::cout << "Sample " << i << ": predicted=" << predicted_class
                  << ", actual=" << labels[i]
                  << " [" << (predicted_class == labels[i] ? "PASS" : "FAIL") << "]"
                  << std::endl;
    }

    data_t accuracy = static_cast<data_t>(correct) / num_samples;
    std::cout << "\nAccuracy: " << correct << "/" << num_samples
              << " = " << (accuracy * 100.0) << "%" << std::endl;

    return accuracy;
}

void print_model_summary(const HydraModelParams_HLS& model) {
    std::cout << "\n=== HYDRA Model Summary ===" << std::endl;
    std::cout << "Time series length: " << model.time_series_length << std::endl;
    std::cout << "Number of kernels: " << NUM_KERNELS << std::endl;
    std::cout << "Number of groups: " << NUM_GROUPS << std::endl;
    std::cout << "Kernels per group: " << KERNELS_PER_GROUP << std::endl;
    std::cout << "Number of features: " << model.num_features << std::endl;
    std::cout << "Number of classes: " << model.num_classes << std::endl;
    std::cout << "Pooling operators: " << POOLING_OPERATORS << std::endl;
    std::cout << "==========================\n" << std::endl;
}

void print_test_data_summary(const TestTimeSeriesData& test_data) {
    std::cout << "\n=== Test Data Summary ===" << std::endl;
    std::cout << "Number of samples: " << test_data.num_samples << std::endl;
    std::cout << "Time series length: " << test_data.time_series_length << std::endl;
    std::cout << "Number of classes: " << test_data.num_classes << std::endl;

    // Class distribution
    std::vector<int_t> class_counts(test_data.num_classes, 0);
    for (int_t label : test_data.y_test) {
        if (label >= 0 && label < test_data.num_classes) {
            class_counts[label]++;
        }
    }

    std::cout << "Class distribution:" << std::endl;
    for (int_t c = 0; c < test_data.num_classes; c++) {
        std::cout << "  Class " << c << ": " << class_counts[c] << " samples" << std::endl;
    }
    std::cout << "==========================\n" << std::endl;
}
