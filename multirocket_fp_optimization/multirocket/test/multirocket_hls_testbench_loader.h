#ifndef MULTIROCKET_HLS_TESTBENCH_LOADER_H
#define MULTIROCKET_HLS_TESTBENCH_LOADER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include "../include/multirocket.hpp"

// Testbench-only loader (NOT synthesizable - for simulation only)
class MiniRocketTestbenchLoader {
private:
    // Simple JSON parser for arrays
    std::vector<float> parse_float_array(const std::string& content, const std::string& key);
    std::vector<int> parse_int_array(const std::string& content, const std::string& key);
    std::vector<std::vector<int>> parse_2d_int_array(const std::string& content, const std::string& key);
    std::vector<std::vector<float>> parse_2d_float_array(const std::string& content, const std::string& key);
    int parse_int_value(const std::string& content, const std::string& key);
    
    std::string read_file(const std::string& filename);
    void trim_whitespace(std::string& str);
    
public:
    // Load model and populate HLS arrays (testbench only)
    bool load_model_to_hls_arrays(
        const std::string& model_filename,
        // Classifier parameters
        data_t coefficients[MAX_CLASSES][MAX_FEATURES],
        data_t intercept[MAX_CLASSES],
        // Scaler parameters
        data_t scaler_mean[MAX_FEATURES],
        data_t scaler_scale[MAX_FEATURES],
        // Model architecture
        int_t dilations_0[MAX_DILATIONS],
        int_t num_features_per_dilation_0[MAX_DILATIONS],
        data_t biases_0[MAX_FEATURES],
        int_t& num_dilations_out_0,
        int_t& num_features_out_0,

        int_t dilations_1[MAX_DILATIONS],
        int_t num_features_per_dilation_1[MAX_DILATIONS],
        data_t biases_1[MAX_FEATURES],
        int_t& num_dilations_out_1,
        int_t& num_features_out_1,

        int_t& num_classes_out,
        int_t& time_series_length_out,
        int_t& n_feature_per_kernel
    );
    
    // Load test data for verification (testbench only)
    bool load_test_data(
        const std::string& test_filename, 
        std::vector<std::vector<float>>& test_inputs,
        std::vector<std::vector<float>>& expected_outputs
    );
};

#endif // MINIROCKET_HLS_TESTBENCH_LOADER_H