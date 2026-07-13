#ifndef MINIROCKET_FUSED_HLS_TESTBENCH_LOADER_H
#define MINIROCKET_FUSED_HLS_TESTBENCH_LOADER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include "../include/minirocket_fused.hpp"

// Testbench-only loader (NOT synthesizable - for simulation only).
// Same JSON schema / parsing logic as
// fpga-network/minirocket_dpr/test/minirocket_hls_testbench_loader.{h,cpp}
// (model.json: num_dilations/num_features/num_classes/time_series_length/
// dilations/num_features_per_dilation/biases/scaler_mean/scaler_scale/
// classifier_intercept/classifier_coef; test_data.json: X_test/y_pred).
class MiniRocketTestbenchLoader {
private:
    std::vector<float> parse_float_array(const std::string& content, const std::string& key);
    std::vector<int> parse_int_array(const std::string& content, const std::string& key);
    std::vector<std::vector<int>> parse_2d_int_array(const std::string& content, const std::string& key);
    std::vector<std::vector<float>> parse_2d_float_array(const std::string& content, const std::string& key);
    int parse_int_value(const std::string& content, const std::string& key);

    std::string read_file(const std::string& filename);
    void trim_whitespace(std::string& str);

public:
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

    bool load_test_data(
        const std::string& test_filename,
        std::vector<std::vector<float>>& test_inputs,
        std::vector<std::vector<float>>& expected_outputs
    );
};

#endif // MINIROCKET_FUSED_HLS_TESTBENCH_LOADER_H
