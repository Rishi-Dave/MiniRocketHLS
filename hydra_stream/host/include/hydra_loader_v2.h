#ifndef HYDRA_JSON_LOADER_V2_H
#define HYDRA_JSON_LOADER_V2_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

typedef float data_t;
typedef int32_t int_t;

// Constants adjusted for massive UCR datasets
#define MAX_TIME_SERIES_LENGTH 8192  // MosquitoSound=3750, FruitFlies=5000, need buffer
#define MAX_FEATURES 1024            // HYDRA features: 512 kernels * 2 pooling = 1024
#define NUM_KERNELS 512              // HYDRA uses 512 kernels
#define KERNEL_SIZE 9                // HYDRA kernel size
#define NUM_GROUPS 8                 // HYDRA groups
#define MAX_CLASSES 10               // InsectSound has 10 classes

// Testbench-only loader (NOT synthesizable - for simulation only)
class HydraJSONLoader {
private:
    // Simple JSON parser for arrays
    std::vector<float> parse_float_array(const std::string& content, const std::string& key);
    std::vector<int> parse_int_array(const std::string& content, const std::string& key);
    std::vector<std::string> parse_string_array(const std::string& content, const std::string& key);
    std::vector<std::vector<int>> parse_2d_int_array(const std::string& content, const std::string& key);
    std::vector<std::vector<float>> parse_2d_float_array(const std::string& content, const std::string& key);
    int parse_int_value(const std::string& content, const std::string& key);

    std::string read_file(const std::string& filename);
    void trim_whitespace(std::string& str);

    // Helper function to map InsectSound string labels to integers
    int map_insectsound_label(const std::string& label);

public:
    // Load HYDRA model and populate HLS arrays (testbench only)
    bool load_hydra_model(
        const std::string& model_filename,
        data_t kernel_weights[NUM_KERNELS][KERNEL_SIZE],
        data_t biases[NUM_KERNELS],
        int_t dilations[NUM_KERNELS],
        int_t group_assignments[NUM_KERNELS],
        data_t scaler_mean[MAX_FEATURES],
        data_t scaler_scale[MAX_FEATURES],
        data_t coefficients[MAX_CLASSES][MAX_FEATURES],
        data_t intercept[MAX_CLASSES],
        int_t& num_kernels,
        int_t& num_groups,
        int_t& num_features,
        int_t& num_classes,
        int_t& time_series_length
    );

    // Load HYDRA test data for verification (testbench only)
    bool load_hydra_test_data(
        const std::string& test_filename,
        std::vector<std::vector<float>>& test_inputs,
        std::vector<int>& test_labels,
        int_t& num_samples,
        int_t& time_series_length,
        int_t& num_classes
    );
};

#endif // HYDRA_JSON_LOADER_V2_H