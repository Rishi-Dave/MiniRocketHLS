#include "hydra_loader_v2.h"
#include <map>
#include <algorithm>

std::string HydraJSONLoader::read_file(const std::string& filename) {
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

void HydraJSONLoader::trim_whitespace(std::string& str) {
    str.erase(0, str.find_first_not_of(" \t\n\r\f\v"));
    str.erase(str.find_last_not_of(" \t\n\r\f\v") + 1);
}

int HydraJSONLoader::parse_int_value(const std::string& content, const std::string& key) {
    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return -1;
    
    size_t colon_pos = content.find(":", key_pos);
    size_t value_start = colon_pos + 1;
    size_t value_end = content.find_first_of(",}", value_start);
    
    std::string value_str = content.substr(value_start, value_end - value_start);
    trim_whitespace(value_str);
    
    return std::stoi(value_str);
}

std::vector<int> HydraJSONLoader::parse_int_array(const std::string& content, const std::string& key) {
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
            // Skip if item contains quotes (it's a string, not an int)
            if (item.find('"') != std::string::npos) {
                return result;  // Return empty vector to signal this is not an int array
            }
            try {
                result.push_back(std::stoi(item));
            } catch (const std::invalid_argument&) {
                // Not a valid integer, return empty result
                return std::vector<int>();
            } catch (const std::out_of_range&) {
                // Integer out of range, return empty result
                return std::vector<int>();
            }
        }
    }

    return result;
}

std::vector<float> HydraJSONLoader::parse_float_array(const std::string& content, const std::string& key) {
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

std::vector<std::vector<int>> HydraJSONLoader::parse_2d_int_array(const std::string& content, const std::string& key) {
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

std::vector<std::vector<float>> HydraJSONLoader::parse_2d_float_array(const std::string& content, const std::string& key) {
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

std::vector<std::string> HydraJSONLoader::parse_string_array(const std::string& content, const std::string& key) {
    std::vector<std::string> result;

    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return result;

    size_t array_start = content.find("[", key_pos);
    size_t array_end = content.find("]", array_start);

    if (array_start == std::string::npos || array_end == std::string::npos) return result;

    std::string array_content = content.substr(array_start + 1, array_end - array_start - 1);

    // Parse string array - each element is quoted
    size_t pos = 0;
    while (pos < array_content.length()) {
        size_t quote_start = array_content.find("\"", pos);
        if (quote_start == std::string::npos) break;

        size_t quote_end = array_content.find("\"", quote_start + 1);
        if (quote_end == std::string::npos) break;

        std::string str_value = array_content.substr(quote_start + 1, quote_end - quote_start - 1);
        result.push_back(str_value);

        pos = quote_end + 1;
    }

    return result;
}

int HydraJSONLoader::map_insectsound_label(const std::string& label) {
    static const std::map<std::string, int> label_map = {
        {"aedes_female", 0},
        {"aedes_male", 1},
        {"fruit_flies", 2},
        {"house_flies", 3},
        {"quinx_female", 4},
        {"quinx_male", 5},
        {"stigma_female", 6},
        {"stigma_male", 7},
        {"tarsalis_female", 8},
        {"tarsalis_male", 9}
    };

    auto it = label_map.find(label);
    if (it != label_map.end()) {
        return it->second;
    }

    std::cerr << "Warning: Unknown label '" << label << "', returning -1" << std::endl;
    return -1;
}

bool HydraJSONLoader::load_hydra_model(
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
) {
    std::cout << "Reading HYDRA model file: " << model_filename << std::endl;
    std::string content = read_file(model_filename);
    if (content.empty()) return false;

    // Parse basic parameters
    num_kernels = parse_int_value(content, "num_kernels");
    num_groups = parse_int_value(content, "num_groups");
    num_features = parse_int_value(content, "num_features");
    num_classes = parse_int_value(content, "num_classes");
    time_series_length = parse_int_value(content, "time_series_length");

    std::cout << "Parsed HYDRA model parameters:" << std::endl;
    std::cout << "  num_kernels: " << num_kernels << std::endl;
    std::cout << "  num_groups: " << num_groups << std::endl;
    std::cout << "  num_features: " << num_features << std::endl;
    std::cout << "  num_classes: " << num_classes << std::endl;
    std::cout << "  time_series_length: " << time_series_length << std::endl;

    if (num_kernels > NUM_KERNELS || num_features > MAX_FEATURES || num_classes > MAX_CLASSES) {
        std::cerr << "Error: Model parameters exceed HLS limits" << std::endl;
        std::cerr << "  num_kernels: " << num_kernels << " (max: " << NUM_KERNELS << ")" << std::endl;
        std::cerr << "  num_features: " << num_features << " (max: " << MAX_FEATURES << ")" << std::endl;
        std::cerr << "  num_classes: " << num_classes << " (max: " << MAX_CLASSES << ")" << std::endl;
        return false;
    }

    // Parse kernel_weights (flat array of 512*9 = 4608 floats)
    auto kernel_weights_flat = parse_float_array(content, "kernel_weights");
    std::cout << "  kernel_weights size: " << kernel_weights_flat.size() << " (expected: " << (num_kernels * KERNEL_SIZE) << ")" << std::endl;

    if (kernel_weights_flat.size() != static_cast<size_t>(num_kernels * KERNEL_SIZE)) {
        std::cerr << "Error: kernel_weights size mismatch" << std::endl;
        return false;
    }

    // Reshape kernel_weights from flat array to 2D array [NUM_KERNELS][KERNEL_SIZE]
    for (int i = 0; i < num_kernels; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            kernel_weights[i][j] = kernel_weights_flat[i * KERNEL_SIZE + j];
        }
    }

    // Parse biases and dilations
    auto biases_vec = parse_float_array(content, "biases");
    auto dilations_vec = parse_int_array(content, "dilations");

    std::cout << "  biases size: " << biases_vec.size() << std::endl;
    std::cout << "  dilations size: " << dilations_vec.size() << std::endl;

    if (biases_vec.size() != static_cast<size_t>(num_kernels) || dilations_vec.size() != static_cast<size_t>(num_kernels)) {
        std::cerr << "Error: biases or dilations size mismatch" << std::endl;
        return false;
    }

    for (int i = 0; i < num_kernels; i++) {
        biases[i] = biases_vec[i];
        dilations[i] = dilations_vec[i];
    }

    // Calculate group_assignments: kernel_idx / (num_kernels / num_groups)
    int kernels_per_group = num_kernels / num_groups;
    for (int i = 0; i < num_kernels; i++) {
        group_assignments[i] = i / kernels_per_group;
    }

    // Parse scaler parameters
    auto scaler_mean_vec = parse_float_array(content, "scaler_mean");
    auto scaler_scale_vec = parse_float_array(content, "scaler_scale");

    std::cout << "  scaler_mean size: " << scaler_mean_vec.size() << std::endl;
    std::cout << "  scaler_scale size: " << scaler_scale_vec.size() << std::endl;

    if (scaler_mean_vec.size() != static_cast<size_t>(num_features) || scaler_scale_vec.size() != static_cast<size_t>(num_features)) {
        std::cerr << "Error: scaler parameters size mismatch" << std::endl;
        return false;
    }

    for (int i = 0; i < num_features; i++) {
        scaler_mean[i] = scaler_mean_vec[i];
        scaler_scale[i] = scaler_scale_vec[i];
    }

    // Parse classifier coefficients (stored as 1D flat array in JSON)
    auto coef_flat = parse_float_array(content, "coefficients");
    std::cout << "  coefficients size (flat): " << coef_flat.size() << " (expected: " << (num_classes * num_features) << ")" << std::endl;

    if (coef_flat.size() != static_cast<size_t>(num_classes * num_features)) {
        std::cerr << "Error: coefficients size mismatch (expected " << (num_classes * num_features) << ", got " << coef_flat.size() << ")" << std::endl;
        return false;
    }

    // Reshape flat array to 2D [num_classes][num_features]
    // Model JSON stores coefficients as coef_.T.flatten() (feature-major):
    //   [f0_c0, f0_c1, ..., f0_cN, f1_c0, f1_c1, ..., f1_cN, ...]
    // So coef_flat[f * num_classes + c] = coefficient for class c, feature f
    for (int c = 0; c < num_classes; c++) {
        for (int f = 0; f < num_features; f++) {
            coefficients[c][f] = coef_flat[f * num_classes + c];
        }
    }

    // Parse intercept
    auto intercept_vec = parse_float_array(content, "intercept");
    std::cout << "  intercept size: " << intercept_vec.size() << std::endl;

    if (intercept_vec.size() != static_cast<size_t>(num_classes)) {
        std::cerr << "Error: intercept size mismatch" << std::endl;
        return false;
    }

    for (int i = 0; i < num_classes; i++) {
        intercept[i] = intercept_vec[i];
    }

    std::cout << "HYDRA model loaded successfully into HLS arrays:" << std::endl;
    std::cout << "  Kernels: " << num_kernels << std::endl;
    std::cout << "  Groups: " << num_groups << std::endl;
    std::cout << "  Features: " << num_features << std::endl;
    std::cout << "  Classes: " << num_classes << std::endl;

    return true;
}

bool HydraJSONLoader::load_hydra_test_data(
    const std::string& test_filename,
    std::vector<std::vector<float>>& test_inputs,
    std::vector<int>& test_labels,
    int_t& num_samples,
    int_t& time_series_length,
    int_t& num_classes
) {
    std::cout << "Reading HYDRA test data file: " << test_filename << std::endl;
    std::string content = read_file(test_filename);
    if (content.empty()) return false;

    // Parse basic parameters
    num_samples = parse_int_value(content, "num_samples");
    time_series_length = parse_int_value(content, "time_series_length");
    num_classes = parse_int_value(content, "num_classes");

    std::cout << "Parsed test data parameters:" << std::endl;
    std::cout << "  num_samples: " << num_samples << std::endl;
    std::cout << "  time_series_length: " << time_series_length << std::endl;
    std::cout << "  num_classes: " << num_classes << std::endl;

    // Parse time_series (2D array: num_samples x time_series_length)
    test_inputs = parse_2d_float_array(content, "time_series");
    std::cout << "  time_series size: " << test_inputs.size() << " x " << (test_inputs.empty() ? 0 : test_inputs[0].size()) << std::endl;

    if (test_inputs.size() != static_cast<size_t>(num_samples)) {
        std::cerr << "Error: time_series size mismatch (expected " << num_samples << " samples, got " << test_inputs.size() << ")" << std::endl;
        return false;
    }

    // Try to parse labels as integers first (for MosquitoSound, FruitFlies, etc.)
    test_labels = parse_int_array(content, "labels");
    std::cout << "  labels size: " << test_labels.size() << std::endl;

    // If integer parsing didn't work, try string parsing (for InsectSound)
    if (test_labels.empty()) {
        auto labels_str = parse_string_array(content, "labels");
        std::cout << "  Trying string labels, size: " << labels_str.size() << std::endl;

        if (labels_str.empty()) {
            std::cerr << "Error: No labels found in test data" << std::endl;
            return false;
        }

        // Map string labels to integers using InsectSound mapping
        test_labels.reserve(labels_str.size());
        for (const auto& label_str : labels_str) {
            int label_int = map_insectsound_label(label_str);
            if (label_int < 0) {
                std::cerr << "Error: Failed to map label '" << label_str << "'" << std::endl;
                return false;
            }
            test_labels.push_back(label_int);
        }
    }

    if (test_labels.size() != static_cast<size_t>(num_samples)) {
        std::cerr << "Error: labels size mismatch (expected " << num_samples << " labels, got " << test_labels.size() << ")" << std::endl;
        return false;
    }

    std::cout << "HYDRA test data loaded successfully:" << std::endl;
    std::cout << "  Samples: " << num_samples << std::endl;
    std::cout << "  Time series length: " << time_series_length << std::endl;
    std::cout << "  Classes: " << num_classes << std::endl;

    return true;
}