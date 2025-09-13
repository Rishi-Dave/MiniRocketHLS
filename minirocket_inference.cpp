#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "json/json.h"

class MiniRocketInference {
private:
    int num_kernels;
    int num_dilations;
    int num_features;
    int num_classes;
    int time_series_length;
    
    std::vector<std::vector<int>> kernel_indices;
    std::vector<int> dilations;
    std::vector<int> num_features_per_dilation;
    std::vector<float> biases;
    std::vector<float> scaler_mean;
    std::vector<float> scaler_scale;
    std::vector<std::vector<float>> classifier_coef;
    std::vector<float> classifier_intercept;

public:
    bool load_model(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open model file " << filename << std::endl;
            return false;
        }

        Json::Value root;
        Json::Reader reader;
        
        if (!reader.parse(file, root)) {
            std::cerr << "Error: Failed to parse JSON" << std::endl;
            return false;
        }

        // Load basic parameters
        num_kernels = root["num_kernels"].asInt();
        num_dilations = root["num_dilations"].asInt();
        num_features = root["num_features"].asInt();
        num_classes = root["num_classes"].asInt();

        // Load kernel indices
        kernel_indices.resize(num_kernels);
        for (int i = 0; i < num_kernels; i++) {
            kernel_indices[i].resize(3);
            for (int j = 0; j < 3; j++) {
                kernel_indices[i][j] = root["kernel_indices"][i][j].asInt();
            }
        }

        // Load dilations
        dilations.resize(num_dilations);
        for (int i = 0; i < num_dilations; i++) {
            dilations[i] = root["dilations"][i].asInt();
        }

        // Load features per dilation
        num_features_per_dilation.resize(num_dilations);
        for (int i = 0; i < num_dilations; i++) {
            num_features_per_dilation[i] = root["num_features_per_dilation"][i].asInt();
        }

        // Load biases
        biases.resize(num_features);
        for (int i = 0; i < num_features; i++) {
            biases[i] = root["biases"][i].asFloat();
        }

        // Load scaler parameters
        scaler_mean.resize(num_features);
        scaler_scale.resize(num_features);
        for (int i = 0; i < num_features; i++) {
            scaler_mean[i] = root["scaler_mean"][i].asFloat();
            scaler_scale[i] = root["scaler_scale"][i].asFloat();
        }

        // Load classifier parameters
        classifier_coef.resize(num_classes);
        for (int i = 0; i < num_classes; i++) {
            classifier_coef[i].resize(num_features);
            for (int j = 0; j < num_features; j++) {
                classifier_coef[i][j] = root["classifier_coef"][i][j].asFloat();
            }
        }

        classifier_intercept.resize(num_classes);
        for (int i = 0; i < num_classes; i++) {
            classifier_intercept[i] = root["classifier_intercept"][i].asFloat();
        }

        std::cout << "Model loaded successfully:" << std::endl;
        std::cout << "  Kernels: " << num_kernels << std::endl;
        std::cout << "  Dilations: " << num_dilations << std::endl;
        std::cout << "  Features: " << num_features << std::endl;
        std::cout << "  Classes: " << num_classes << std::endl;

        return true;
    }

    std::vector<float> apply_kernel_single_series(const std::vector<float>& time_series, 
                                                 const std::vector<int>& kernel_idx, 
                                                 int dilation) {
        int length = time_series.size();
        int kernel_length = 9;
        int output_length = length - (kernel_length - 1) * dilation;
        
        std::vector<float> result;
        if (output_length <= 0) {
            return result;
        }
        
        result.resize(output_length);
        
        for (int j = 0; j < output_length; j++) {
            float value = 0.0f;
            for (int k = 0; k < 3; k++) {
                int pos = j + kernel_idx[k] * dilation;
                if (pos < length) {
                    // Use weights: -1, 0, 1 pattern
                    float weight = (k == 0) ? -1.0f : ((k == 2) ? 1.0f : 0.0f);
                    value += time_series[pos] * weight;
                }
            }
            result[j] = value;
        }
        
        return result;
    }

    std::vector<float> extract_features(const std::vector<float>& time_series) {
        std::vector<float> features(num_features, 0.0f);
        time_series_length = time_series.size();
        
        int feature_idx = 0;
        
        for (int dil_idx = 0; dil_idx < num_dilations; dil_idx++) {
            int dilation = dilations[dil_idx];
            
            for (int kernel_idx = 0; kernel_idx < num_kernels; kernel_idx++) {
                // Apply kernel
                std::vector<float> convolutions = apply_kernel_single_series(
                    time_series, kernel_indices[kernel_idx], dilation);
                
                // Calculate positive proportion of values (PPV)
                float bias = biases[feature_idx];
                int positive_count = 0;
                
                for (float conv : convolutions) {
                    if (conv > bias) {
                        positive_count++;
                    }
                }
                
                float ppv = (convolutions.size() > 0) ? 
                    static_cast<float>(positive_count) / convolutions.size() : 0.0f;
                
                features[feature_idx] = ppv;
                feature_idx++;
            }
        }
        
        return features;
    }

    std::vector<float> apply_scaler(const std::vector<float>& features) {
        std::vector<float> scaled_features(num_features);
        
        for (int i = 0; i < num_features; i++) {
            scaled_features[i] = (features[i] - scaler_mean[i]) / scaler_scale[i];
        }
        
        return scaled_features;
    }

    std::vector<float> predict_proba(const std::vector<float>& scaled_features) {
        std::vector<float> predictions(num_classes);
        
        for (int i = 0; i < num_classes; i++) {
            float score = classifier_intercept[i];
            for (int j = 0; j < num_features; j++) {
                score += classifier_coef[i][j] * scaled_features[j];
            }
            predictions[i] = score;
        }
        
        return predictions;
    }

    int predict_class(const std::vector<float>& time_series) {
        // Feature extraction
        std::vector<float> features = extract_features(time_series);
        
        // Apply scaling
        std::vector<float> scaled_features = apply_scaler(features);
        
        // Predict
        std::vector<float> scores = predict_proba(scaled_features);
        
        // Find class with highest score
        int predicted_class = 0;
        float max_score = scores[0];
        for (int i = 1; i < num_classes; i++) {
            if (scores[i] > max_score) {
                max_score = scores[i];
                predicted_class = i;
            }
        }
        
        return predicted_class;
    }

    std::vector<float> predict_scores(const std::vector<float>& time_series) {
        std::vector<float> features = extract_features(time_series);
        std::vector<float> scaled_features = apply_scaler(features);
        return predict_proba(scaled_features);
    }
};

// Test function to verify against Python results
bool test_against_python(const std::string& model_file, const std::string& test_data_file) {
    MiniRocketInference model;
    
    if (!model.load_model(model_file)) {
        return false;
    }

    // Load test data
    std::ifstream file(test_data_file);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open test data file " << test_data_file << std::endl;
        return false;
    }

    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(file, root)) {
        std::cerr << "Error: Failed to parse test data JSON" << std::endl;
        return false;
    }

    // Load test samples
    int num_test_samples = root["X_test"].size();
    int series_length = root["X_test"][0].size();
    
    std::cout << "Testing on " << num_test_samples << " samples of length " << series_length << std::endl;
    
    int correct_predictions = 0;
    int total_predictions = 0;
    
    for (int i = 0; i < std::min(num_test_samples, 100); i++) {  // Test first 100 samples
        // Load time series
        std::vector<float> time_series(series_length);
        for (int j = 0; j < series_length; j++) {
            time_series[j] = root["X_test"][i][j].asFloat();
        }
        
        // Get prediction
        int cpp_prediction = model.predict_class(time_series);
        int python_prediction = root["y_pred"][i].asInt();
        int true_label = root["y_test"][i].asInt();
        
        if (cpp_prediction == python_prediction) {
            correct_predictions++;
        }
        
        // Also check against true label
        if (cpp_prediction == true_label) {
            // This matches the true label
        }
        
        total_predictions++;
        
        if (i < 10) {  // Print first 10 for debugging
            std::cout << "Sample " << i << ": C++=" << cpp_prediction 
                     << ", Python=" << python_prediction 
                     << ", True=" << true_label << std::endl;
        }
    }
    
    float agreement = static_cast<float>(correct_predictions) / total_predictions;
    std::cout << "C++ vs Python agreement: " << std::fixed << std::setprecision(4) 
              << agreement << " (" << correct_predictions << "/" << total_predictions << ")" << std::endl;
    
    return agreement > 0.95;  // Expect very high agreement
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <model_file> <test_data_file>" << std::endl;
        return 1;
    }
    
    std::string model_file = argv[1];
    std::string test_data_file = argv[2];
    
    std::cout << "Testing MiniRocket C++ implementation..." << std::endl;
    
    bool success = test_against_python(model_file, test_data_file);
    
    if (success) {
        std::cout << "SUCCESS: C++ implementation matches Python results!" << std::endl;
        return 0;
    } else {
        std::cout << "FAILED: C++ implementation does not match Python results." << std::endl;
        return 1;
    }
}