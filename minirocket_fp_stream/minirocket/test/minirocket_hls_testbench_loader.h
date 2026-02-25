#ifndef MINIROCKET_HLS_TESTBENCH_LOADER_H
#define MINIROCKET_HLS_TESTBENCH_LOADER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include "../include/minirocket.hpp"

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
        data_t coefficients[MAX_CLASSES][MAX_FEATURES],
        data_t intercept[MAX_CLASSES],
        data_t scaler_mean[MAX_FEATURES],
        data_t scaler_scale[MAX_FEATURES],
        int_t dilations[MAX_DILATIONS],
        int_t num_features_per_dilation[MAX_DILATIONS],
        data_t biases[MAX_FEATURES],
        int_t& num_dilations,
        int_t& num_features,
        int_t& num_classes,
        int_t& time_series_length
    );
    
    // Load test data for verification (testbench only)
    bool load_test_data(
        const std::string& test_filename, 
        std::vector<float>& test_inputs,
        std::vector<std::vector<float>>& expected_outputs
    );
};

#endif // MINIROCKET_HLS_TESTBENCH_LOADER_H