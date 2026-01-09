#ifndef HYDRA_HLS_TESTBENCH_LOADER_H
#define HYDRA_HLS_TESTBENCH_LOADER_H

#include "../include/hydra.hpp"
#include <string>
#include <vector>

// Testbench-specific structures for loading JSON data

struct TestTimeSeriesData {
    std::vector<std::vector<data_t>> X_test;  // Test time series
    std::vector<int_t> y_test;                 // Test labels
    int_t time_series_length;
    int_t num_samples;
    int_t num_classes;
};

/**
 * Load HYDRA model parameters from JSON file
 *
 * @param filename Path to model JSON file
 * @param model Output model parameters structure
 * @return true if successful, false otherwise
 */
bool load_hydra_model(
    const std::string& filename,
    HydraModelParams_HLS* model
);

/**
 * Load test dataset from JSON file
 *
 * @param filename Path to test data JSON file
 * @param test_data Output test data structure
 * @return true if successful, false otherwise
 */
bool load_test_data(
    const std::string& filename,
    TestTimeSeriesData* test_data
);

/**
 * Validate predictions against expected labels
 *
 * @param predictions Model predictions [num_samples][num_classes]
 * @param labels Ground truth labels [num_samples]
 * @param num_samples Number of test samples
 * @param num_classes Number of classes
 * @param tolerance Floating point comparison tolerance
 * @return Accuracy as fraction of correct predictions
 */
data_t validate_predictions(
    const std::vector<std::vector<data_t>>& predictions,
    const std::vector<int_t>& labels,
    int_t num_samples,
    int_t num_classes,
    data_t tolerance = 1e-4
);

/**
 * Print model parameters summary
 *
 * @param model Model parameters
 */
void print_model_summary(const HydraModelParams_HLS& model);

/**
 * Print test data summary
 *
 * @param test_data Test dataset
 */
void print_test_data_summary(const TestTimeSeriesData& test_data);

#endif // HYDRA_HLS_TESTBENCH_LOADER_H
