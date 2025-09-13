#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include "minirocket_inference_hls.h"

class HLSModelLoader {
private:
    std::vector<float> load_json_float_array(const std::string& filename, const std::string& key) {
        // Simple JSON parser for arrays - assumes clean format
        std::ifstream file(filename);
        std::string line, content;
        
        while (std::getline(file, line)) {
            content += line;
        }
        
        // Find the key
        size_t key_pos = content.find("\"" + key + "\"");
        if (key_pos == std::string::npos) {
            std::cerr << "Key " << key << " not found!" << std::endl;
            return {};
        }
        
        // Find array start
        size_t array_start = content.find("[", key_pos);
        size_t array_end = content.find("]", array_start);
        
        if (array_start == std::string::npos || array_end == std::string::npos) {
            std::cerr << "Array for " << key << " not found!" << std::endl;
            return {};
        }
        
        std::string array_content = content.substr(array_start + 1, array_end - array_start - 1);
        
        std::vector<float> result;
        size_t pos = 0;
        
        while (pos < array_content.length()) {
            size_t comma_pos = array_content.find(",", pos);
            if (comma_pos == std::string::npos) comma_pos = array_content.length();
            
            std::string value_str = array_content.substr(pos, comma_pos - pos);
            
            // Clean whitespace and brackets
            value_str.erase(0, value_str.find_first_not_of(" \t\n\r\f\v"));
            value_str.erase(value_str.find_last_not_of(" \t\n\r\f\v") + 1);
            
            if (!value_str.empty()) {
                try {
                    result.push_back(std::stof(value_str));
                } catch (...) {
                    // Skip invalid values
                }
            }
            
            pos = comma_pos + 1;
        }
        
        return result;
    }
    
    std::vector<int> load_json_int_array(const std::string& filename, const std::string& key) {
        std::vector<float> float_vals = load_json_float_array(filename, key);
        std::vector<int> result;
        for (float val : float_vals) {
            result.push_back(static_cast<int>(val));
        }
        return result;
    }

public:
    bool load_model_for_hls(const std::string& filename, MiniRocketModelParams_HLS& params) {
        std::cout << "Loading model parameters..." << std::endl;
        
        // Load dilations
        auto dilations = load_json_int_array(filename, "dilations");
        params.num_dilations = std::min(static_cast<int>(dilations.size()), MAX_DILATIONS);
        for (int i = 0; i < params.num_dilations; i++) {
            params.dilations[i] = dilations[i];
        }
        
        // Load num_features_per_dilation
        auto features_per_dil = load_json_int_array(filename, "num_features_per_dilation");
        for (int i = 0; i < params.num_dilations; i++) {
            params.num_features_per_dilation[i] = (i < features_per_dil.size()) ? features_per_dil[i] : NUM_KERNELS;
        }
        
        // Load basic parameters
        auto temp_vec = load_json_int_array(filename, "num_features");
        params.num_features = temp_vec.empty() ? 420 : temp_vec[0];
        
        temp_vec = load_json_int_array(filename, "num_classes");
        params.num_classes = temp_vec.empty() ? 4 : temp_vec[0];
        
        // Load biases
        auto biases = load_json_float_array(filename, "biases");
        int num_biases = std::min(static_cast<int>(biases.size()), MAX_FEATURES);
        for (int i = 0; i < num_biases; i++) {
            params.biases[i] = biases[i];
        }
        
        // Load scaler parameters
        auto scaler_mean = load_json_float_array(filename, "scaler_mean");
        auto scaler_scale = load_json_float_array(filename, "scaler_scale");
        
        for (int i = 0; i < params.num_features && i < MAX_FEATURES; i++) {
            params.scaler_mean[i] = (i < scaler_mean.size()) ? scaler_mean[i] : 0.0;
            params.scaler_scale[i] = (i < scaler_scale.size()) ? scaler_scale[i] : 1.0;
        }
        
        // Load intercept
        auto intercept = load_json_float_array(filename, "classifier_intercept");
        for (int i = 0; i < params.num_classes && i < MAX_CLASSES; i++) {
            params.intercept[i] = (i < intercept.size()) ? intercept[i] : 0.0;
        }
        
        // Load coefficients (this is a simplified approach)
        auto coefficients = load_json_float_array(filename, "classifier_coef");
        
        // Coefficients are stored as [class0_features, class1_features, ...]
        for (int class_idx = 0; class_idx < params.num_classes && class_idx < MAX_CLASSES; class_idx++) {
            for (int feat_idx = 0; feat_idx < params.num_features && feat_idx < MAX_FEATURES; feat_idx++) {
                int coef_idx = class_idx * params.num_features + feat_idx;
                params.coefficients[class_idx][feat_idx] = 
                    (coef_idx < coefficients.size()) ? coefficients[coef_idx] : 0.0;
            }
        }
        
        std::cout << "Model loaded: " << params.num_features << " features, " 
                  << params.num_classes << " classes, " 
                  << params.num_dilations << " dilations" << std::endl;
        
        return true;
    }
};

void test_hls_implementation() {
    std::cout << "Testing HLS implementation..." << std::endl;
    
    // Load model
    HLSModelLoader loader;
    MiniRocketModelParams_HLS params;
    
    if (!loader.load_model_for_hls("minirocket_model.json", params)) {
        std::cerr << "Failed to load model!" << std::endl;
        return;
    }
    
    // Create sample time series (sine wave)
    data_t time_series[MAX_TIME_SERIES_LENGTH];
    int_t time_series_length = 128;
    
    for (int i = 0; i < time_series_length; i++) {
        time_series[i] = sin(2.0 * M_PI * i / 32.0) + 0.1 * ((i % 10) - 5) / 5.0;
    }
    
    std::cout << "Input time series (first 10 values): ";
    for (int i = 0; i < 10; i++) {
        std::cout << std::fixed << std::setprecision(3) << time_series[i] << " ";
    }
    std::cout << std::endl;
    
    // Test feature extraction
    data_t features[MAX_FEATURES];
    
    std::cout << "Running feature extraction..." << std::endl;
    minirocket_feature_extraction_hls(
        time_series,
        features,
        params.dilations,
        params.num_features_per_dilation,
        params.biases,
        time_series_length,
        params.num_dilations,
        params.num_features
    );
    
    std::cout << "Feature extraction complete. First 10 features: ";
    for (int i = 0; i < 10 && i < params.num_features; i++) {
        std::cout << std::fixed << std::setprecision(4) << features[i] << " ";
    }
    std::cout << std::endl;
    
    // Test scaling
    data_t scaled_features[MAX_FEATURES];
    
    std::cout << "Applying feature scaling..." << std::endl;
    apply_scaler_hls(
        features,
        scaled_features,
        params.scaler_mean,
        params.scaler_scale,
        params.num_features
    );
    
    std::cout << "Scaling complete. First 10 scaled features: ";
    for (int i = 0; i < 10 && i < params.num_features; i++) {
        std::cout << std::fixed << std::setprecision(4) << scaled_features[i] << " ";
    }
    std::cout << std::endl;
    
    // Test classification
    data_t predictions[MAX_CLASSES];
    
    std::cout << "Running classification..." << std::endl;
    linear_classifier_predict_hls(
        scaled_features,
        predictions,
        params.coefficients,
        params.intercept,
        params.num_features,
        params.num_classes
    );
    
    // Find predicted class
    int predicted_class = 0;
    data_t max_score = predictions[0];
    
    std::cout << "Classification scores: ";
    for (int i = 0; i < params.num_classes; i++) {
        std::cout << "Class" << i << "=" << std::fixed << std::setprecision(4) << predictions[i] << " ";
        if (predictions[i] > max_score) {
            max_score = predictions[i];
            predicted_class = i;
        }
    }
    std::cout << std::endl;
    
    std::cout << "Predicted class: " << predicted_class 
              << " (score: " << std::fixed << std::setprecision(4) << max_score << ")" << std::endl;
    std::cout << "HLS implementation test complete!" << std::endl;
}

int main() {
    test_hls_implementation();
    return 0;
}