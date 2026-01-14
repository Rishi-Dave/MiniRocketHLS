#include "multirocket_hls_testbench_loader.h"

std::string MiniRocketTestbenchLoader::read_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return "";
    }
    
    std::string content, line;
    while (std::getline(file, line)) {
        content += line;
    }
    return content;
}

void MiniRocketTestbenchLoader::trim_whitespace(std::string& str) {
    str.erase(0, str.find_first_not_of(" \t\n\r\f\v"));
    str.erase(str.find_last_not_of(" \t\n\r\f\v") + 1);
}

int MiniRocketTestbenchLoader::parse_int_value(const std::string& content, const std::string& key) {
    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return -1;
    
    size_t colon_pos = content.find(":", key_pos);
    size_t value_start = colon_pos + 1;
    size_t value_end = content.find_first_of(",}", value_start);
    
    std::string value_str = content.substr(value_start, value_end - value_start);
    trim_whitespace(value_str);
    
    return std::stoi(value_str);
}

std::vector<int> MiniRocketTestbenchLoader::parse_int_array(const std::string& content, const std::string& key) {
    std::vector<int> result;
    
    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return result;
    
    size_t array_start = content.find("[", key_pos);
    size_t array_end = content.find("]", array_start);
    
    if (array_start == std::string::npos || array_end == std::string::npos) return result;
    
    std::string array_content = content.substr(array_start + 1, array_end - array_start - 1);
    
    std::stringstream ss(array_content);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        trim_whitespace(item);
        if (!item.empty()) {
            result.push_back(std::stoi(item));
        }
    }
    
    return result;
}

std::vector<float> MiniRocketTestbenchLoader::parse_float_array(const std::string& content, const std::string& key) {
    std::vector<float> result;
    
    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return result;
    
    size_t array_start = content.find("[", key_pos);
    size_t array_end = content.find("]", array_start);
    
    if (array_start == std::string::npos || array_end == std::string::npos) return result;
    
    std::string array_content = content.substr(array_start + 1, array_end - array_start - 1);
    
    std::stringstream ss(array_content);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        trim_whitespace(item);
        if (!item.empty()) {
            result.push_back(std::stof(item));
        }
    }
    
    return result;
}

std::vector<std::vector<int>> MiniRocketTestbenchLoader::parse_2d_int_array(const std::string& content, const std::string& key) {
    std::vector<std::vector<int>> result;
    
    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return result;
    
    size_t array_start = content.find("[", key_pos);
    size_t array_end = array_start;
    int bracket_count = 0;
    
    // Find matching closing bracket
    for (size_t i = array_start; i < content.length(); i++) {
        if (content[i] == '[') bracket_count++;
        if (content[i] == ']') bracket_count--;
        if (bracket_count == 0) {
            array_end = i;
            break;
        }
    }
    
    std::string array_content = content.substr(array_start + 1, array_end - array_start - 1);
    
    // Parse each sub-array
    size_t pos = 0;
    while (pos < array_content.length()) {
        size_t sub_start = array_content.find("[", pos);
        if (sub_start == std::string::npos) break;
        
        size_t sub_end = array_content.find("]", sub_start);
        if (sub_end == std::string::npos) break;
        
        std::string sub_array = array_content.substr(sub_start + 1, sub_end - sub_start - 1);
        
        std::vector<int> sub_result;
        std::stringstream ss(sub_array);
        std::string item;
        
        while (std::getline(ss, item, ',')) {
            trim_whitespace(item);
            if (!item.empty()) {
                sub_result.push_back(std::stoi(item));
            }
        }
        
        result.push_back(sub_result);
        pos = sub_end + 1;
    }
    
    return result;
}

std::vector<std::vector<float>> MiniRocketTestbenchLoader::parse_2d_float_array(const std::string& content, const std::string& key) {
    std::vector<std::vector<float>> result;
    
    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return result;
    
    size_t array_start = content.find("[", key_pos);
    size_t array_end = array_start;
    int bracket_count = 0;
    
    // Find matching closing bracket
    for (size_t i = array_start; i < content.length(); i++) {
        if (content[i] == '[') bracket_count++;
        if (content[i] == ']') bracket_count--;
        if (bracket_count == 0) {
            array_end = i;
            break;
        }
    }
    
    std::string array_content = content.substr(array_start + 1, array_end - array_start - 1);
    
    // Parse each sub-array
    size_t pos = 0;
    while (pos < array_content.length()) {
        size_t sub_start = array_content.find("[", pos);
        if (sub_start == std::string::npos) break;
        
        size_t sub_end = array_content.find("]", sub_start);
        if (sub_end == std::string::npos) break;
        
        std::string sub_array = array_content.substr(sub_start + 1, sub_end - sub_start - 1);
        
        std::vector<float> sub_result;
        std::stringstream ss(sub_array);
        std::string item;
        
        while (std::getline(ss, item, ',')) {
            trim_whitespace(item);
            if (!item.empty()) {
                sub_result.push_back(std::stof(item));
            }
        }
        
        result.push_back(sub_result);
        pos = sub_end + 1;
    }
    
    return result;
}

bool MiniRocketTestbenchLoader::load_model_to_hls_arrays(
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
) {

    std::cout << "Reading model file: " << model_filename << std::endl;
    std::string content = read_file(model_filename);
    if (content.empty()) return false;
    
    num_classes_out = parse_int_value(content, "num_classes");
    time_series_length_out = parse_int_value(content, "time_series_length");
    n_feature_per_kernel = parse_int_value(content, "n_feature_per_kernel");

    // Parse basic parameters
    num_dilations_out_0 = parse_int_value(content, "num_dilations_0");
    num_features_out_0 = parse_int_value(content, "num_features_0");
    num_dilations_out_1 = parse_int_value(content, "num_dilations_1");
    num_features_out_1 = parse_int_value(content, "num_features_1");

    std::cout << "Parsed model parameters:" << std::endl;
    std::cout << "  num_dilations: " << num_dilations_out_0 << ", " << num_dilations_out_1 << std::endl;
    std::cout << "  num_features: " << num_features_out_0 << ", " << num_features_out_1 << std::endl;
    std::cout << "  num_classes: " << num_classes_out << std::endl;
    std::cout << "  time_series_length: " << time_series_length_out << std::endl;
    
    if (num_dilations_out_0 > MAX_DILATIONS || num_features_out_0 > MAX_FEATURES || num_classes_out > MAX_CLASSES) {
        std::cerr << "Error: Model parameters exceed HLS limits" << std::endl;
        return false;
    }
    
    // Parse arrays
    auto dilations_vec_0 = parse_int_array(content, "dilations_0");
    auto num_features_per_dilation_vec_0 = parse_int_array(content, "num_features_per_dilation_0");
    auto biases_vec_0 = parse_float_array(content, "biases_0");
    auto dilations_vec_1 = parse_int_array(content, "dilations_1");
    auto num_features_per_dilation_vec_1 = parse_int_array(content, "num_features_per_dilation_1");
    auto biases_vec_1 = parse_float_array(content, "biases_1");

    auto scaler_mean_vec = parse_float_array(content, "scaler_mean");
    auto scaler_scale_vec = parse_float_array(content, "scaler_scale");
    auto intercept_vec = parse_float_array(content, "classifier_intercept");
    
    std::cout << "Parsed model arrays." << std::endl;
    std::cout << "  dilations_0 size: " << dilations_vec_0.size() << std::endl;
    std::cout << "  num_features_per_dilation_0 size: " << num_features_per_dilation_vec_0.size() << std::endl;
    std::cout << "  biases_0 size: " << biases_vec_0.size() << std::endl;
    std::cout << "  dilations_1 size: " << dilations_vec_1.size() << std::endl;
    std::cout << "  num_features_per_dilation_1 size: " << num_features_per_dilation_vec_1.size() << std::endl;
    std::cout << "  biases_1 size: " << biases_vec_1.size() << std::endl;
    
    std::cout << "  scaler_mean size: " << scaler_mean_vec.size() << std::endl;
    std::cout << "  scaler_scale size: " << scaler_scale_vec.size() << std::endl;
    std::cout << "  intercept size: " << intercept_vec.size() << std::endl;

    // Handle binary vs multi-class classification for coefficients
    std::vector<std::vector<float>> coef_2d;
    if (num_classes_out == 2) {
        // Binary classification: classifier_coef is 1D array
        std::cout << "Parsing binary classifier coefficients..." << std::endl;
        auto coef_1d = parse_float_array(content, "classifier_coef");
        coef_2d.resize(2);
        coef_2d[0] = coef_1d;  // Use the 1D coefficients for class 0 decision function
        coef_2d[1].resize(coef_1d.size(), 0.0f);  // Not used but keep for consistency
    } else {
        // Multi-class: classifier_coef is 2D array
        std::cout << "Parsing multi-class classifier coefficients..." << std::endl;
        coef_2d = parse_2d_float_array(content, "classifier_coef");
    }
    
    std::cout << "Parsed classifier coefficients." << std::endl;
    std::cout << "  Coefficients size: " << coef_2d.size() << " x " << (coef_2d.empty() ? 0 : coef_2d[0].size()) << std::endl;

    // Copy to HLS arrays
    for (int i = 0; i < num_dilations_out_0; i++) {
        dilations_0[i] = dilations_vec_0[i];
        num_features_per_dilation_0[i] = num_features_per_dilation_vec_0[i];
    }

    for (int i = 0; i < num_dilations_out_1; i++) {
        dilations_1[i] = dilations_vec_1[i];
        num_features_per_dilation_1[i] = num_features_per_dilation_vec_1[i];
    }
    
    for (int i = 0; i < num_features_out_0; i++) {
        biases_0[i] = biases_vec_0[i];
    }

    for (int i = 0; i < num_features_out_1; i++) {
        biases_1[i] = biases_vec_1[i];
    }

    for (int i = 0; i < (num_features_out_0 + num_features_out_1) * n_feature_per_kernel; i++) {
        scaler_mean[i] = scaler_mean_vec[i];
        scaler_scale[i] = scaler_scale_vec[i];
    }
    
    // Handle intercept for binary vs multi-class
    if (num_classes_out == 2 && intercept_vec.size() == 1) {
        // Binary classification: single intercept
        intercept[0] = intercept_vec[0];
        intercept[1] = 0.0f;  // Not used
    } else {
        // Multi-class: one intercept per class
        for (int i = 0; i < num_classes_out; i++) {
            intercept[i] = intercept_vec[i];
        }
    }
    
    for (int i = 0; i < num_classes_out; i++) {
        for (int j = 0; j < (num_features_out_0 + num_features_out_1) * n_feature_per_kernel; j++) {
            coefficients[i][j] = coef_2d[i][j];
        }
    }
    
    std::cout << "Model loaded successfully into HLS arrays:" << std::endl;
    std::cout << "  Features: " << num_features_out_0 << ", " << num_features_out_1 << std::endl;
    std::cout << "  Classes: " << num_classes_out << std::endl;
    std::cout << "  Dilations: " << num_dilations_out_0 << ", " << num_dilations_out_1 << std::endl;
    
    return true;
}

bool MiniRocketTestbenchLoader::load_test_data(
    const std::string& test_filename, 
    std::vector<std::vector<float>>& test_inputs,
    std::vector<std::vector<float>>& expected_outputs
) {
    std::string content = read_file(test_filename);
    if (content.empty()) return false;
    
    test_inputs = parse_2d_float_array(content, "X_test");
    
    // Parse y_test as 1D array and convert to 2D format
    auto y_labels = parse_float_array(content, "y_pred");
    expected_outputs.clear();
    for (float label : y_labels) {
        expected_outputs.push_back({label});
    }
    
    std::cout << "Test data loaded: " << test_inputs.size() << " samples" << std::endl;
    return true;
}