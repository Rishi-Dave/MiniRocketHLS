#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <iomanip>
#include "json.hpp"

using json = nlohmann::json;

// Configuration
#define NUM_KERNELS 84
#define KERNEL_SIZE 9
#define MAX_DILATIONS 8
#define MAX_TIME_SERIES_LENGTH 512
#define POOLING_OPS 4
#define NUM_REPRESENTATIONS 2

// Kernel indices (all C(9,3) combinations)
const int KERNEL_INDICES[NUM_KERNELS][3] = {
    {0,1,2}, {0,1,3}, {0,1,4}, {0,1,5}, {0,1,6}, {0,1,7}, {0,1,8},
    {0,2,3}, {0,2,4}, {0,2,5}, {0,2,6}, {0,2,7}, {0,2,8},
    {0,3,4}, {0,3,5}, {0,3,6}, {0,3,7}, {0,3,8},
    {0,4,5}, {0,4,6}, {0,4,7}, {0,4,8},
    {0,5,6}, {0,5,7}, {0,5,8},
    {0,6,7}, {0,6,8},
    {0,7,8},
    {1,2,3}, {1,2,4}, {1,2,5}, {1,2,6}, {1,2,7}, {1,2,8},
    {1,3,4}, {1,3,5}, {1,3,6}, {1,3,7}, {1,3,8},
    {1,4,5}, {1,4,6}, {1,4,7}, {1,4,8},
    {1,5,6}, {1,5,7}, {1,5,8},
    {1,6,7}, {1,6,8},
    {1,7,8},
    {2,3,4}, {2,3,5}, {2,3,6}, {2,3,7}, {2,3,8},
    {2,4,5}, {2,4,6}, {2,4,7}, {2,4,8},
    {2,5,6}, {2,5,7}, {2,5,8},
    {2,6,7}, {2,6,8},
    {2,7,8},
    {3,4,5}, {3,4,6}, {3,4,7}, {3,4,8},
    {3,5,6}, {3,5,7}, {3,5,8},
    {3,6,7}, {3,6,8},
    {3,7,8},
    {4,5,6}, {4,5,7}, {4,5,8},
    {4,6,7}, {4,6,8},
    {4,7,8},
    {5,6,7}, {5,6,8},
    {5,7,8},
    {6,7,8}
};

struct PoolingStats {
    float ppv;   // Proportion of Positive Values
    float mpv;   // Mean of Positive Values
    float mipv;  // Mean of Indices of Positive Values
    float lspv;  // Longest Stretch of Positive Values
};

struct ModelParams {
    int num_kernels;
    int num_features;
    int num_classes;
    int num_dilations_orig;
    int num_dilations_diff;

    std::vector<int> dilations_orig;
    std::vector<int> dilations_diff;
    std::vector<float> biases_orig;
    std::vector<float> biases_diff;

    std::vector<float> scaler_mean;
    std::vector<float> scaler_scale;

    std::vector<std::vector<float>> coefficients;
    std::vector<float> intercept;

    float weights[NUM_KERNELS][KERNEL_SIZE];
};

// Generate simplified kernel weights (-1, 0, +2 pattern)
void generate_kernel_weights(float weights[NUM_KERNELS][KERNEL_SIZE]) {
    for (int k = 0; k < NUM_KERNELS; k++) {
        // Initialize all to -1
        for (int i = 0; i < KERNEL_SIZE; i++) {
            weights[k][i] = -1.0f;
        }

        // Set selected positions to +2
        weights[k][KERNEL_INDICES[k][0]] = 2.0f;
        weights[k][KERNEL_INDICES[k][1]] = 2.0f;
        weights[k][KERNEL_INDICES[k][2]] = 2.0f;
    }
}

// Apply kernel convolution
void apply_kernel(const float* X, int length, const float* kernel_weights,
                 int dilation, float* C) {
    for (int j = 0; j < length; j++) {
        float value = 0.0f;

        for (int k = -4; k <= 4; k++) {
            int idx = j + k * dilation;
            if (idx >= 0 && idx < length) {
                value += X[idx] * kernel_weights[k + 4];
            }
        }

        C[j] = value;
    }
}

// Compute PPV (Proportion of Positive Values)
float compute_ppv(const float* C, float bias, int length) {
    int count = 0;
    for (int i = 0; i < length; i++) {
        if (C[i] > bias) {
            count++;
        }
    }
    return (float)count / (float)length;
}

// Compute MPV (Mean of Positive Values)
float compute_mpv(const float* C, float bias, int length) {
    float sum = 0.0f;
    int count = 0;

    for (int i = 0; i < length; i++) {
        if (C[i] > bias) {
            sum += C[i];
            count++;
        }
    }

    return (count > 0) ? (sum / (float)count) : 0.0f;
}

// Compute MIPV (Mean of Indices of Positive Values, normalized)
float compute_mipv(const float* C, float bias, int length) {
    int index_sum = 0;
    int count = 0;

    for (int i = 0; i < length; i++) {
        if (C[i] > bias) {
            index_sum += i;
            count++;
        }
    }

    return (count > 0) ? ((float)index_sum / ((float)count * (float)length)) : 0.0f;
}

// Compute LSPV (Longest Stretch of Positive Values, normalized)
float compute_lspv(const float* C, float bias, int length) {
    int max_stretch = 0;
    int current_stretch = 0;

    for (int i = 0; i < length; i++) {
        if (C[i] > bias) {
            current_stretch++;
            if (current_stretch > max_stretch) {
                max_stretch = current_stretch;
            }
        } else {
            current_stretch = 0;
        }
    }

    return (float)max_stretch / (float)length;
}

// Compute all four pooling operators in single pass
void compute_four_pooling(const float* C, float bias, int length, PoolingStats* stats) {
    int ppv_count = 0;
    float mpv_sum = 0.0f;
    int mpv_count = 0;
    int mipv_index_sum = 0;
    int mipv_count = 0;
    int lspv_current = 0;
    int lspv_max = 0;

    for (int i = 0; i < length; i++) {
        bool is_positive = (C[i] > bias);

        if (is_positive) {
            ppv_count++;
            mpv_sum += C[i];
            mpv_count++;
            mipv_index_sum += i;
            mipv_count++;
            lspv_current++;
            if (lspv_current > lspv_max) {
                lspv_max = lspv_current;
            }
        } else {
            lspv_current = 0;
        }
    }

    stats->ppv = (float)ppv_count / (float)length;
    stats->mpv = (mpv_count > 0) ? (mpv_sum / (float)mpv_count) : 0.0f;
    stats->mipv = (mipv_count > 0) ? ((float)mipv_index_sum / ((float)mipv_count * (float)length)) : 0.0f;
    stats->lspv = (float)lspv_max / (float)length;
}

// Extract features from single representation
void extract_features_single_repr(const float* X, int length,
                                  const std::vector<int>& dilations,
                                  const std::vector<float>& biases,
                                  const float weights[NUM_KERNELS][KERNEL_SIZE],
                                  std::vector<float>& features) {
    float C[MAX_TIME_SERIES_LENGTH];
    int num_dilations = dilations.size();
    int feature_idx = 0;

    for (int d_idx = 0; d_idx < num_dilations; d_idx++) {
        int dilation = dilations[d_idx];

        for (int k_idx = 0; k_idx < NUM_KERNELS; k_idx++) {
            // Apply convolution
            apply_kernel(X, length, weights[k_idx], dilation, C);

            // Get bias
            float bias = biases[d_idx * NUM_KERNELS + k_idx];

            // Compute 4 pooling features
            PoolingStats stats;
            compute_four_pooling(C, bias, length, &stats);

            features[feature_idx++] = stats.ppv;
            features[feature_idx++] = stats.mpv;
            features[feature_idx++] = stats.mipv;
            features[feature_idx++] = stats.lspv;
        }
    }
}

// Compute first-order difference
void compute_diff(const float* original, float* diff, int length) {
    for (int i = 0; i < length - 1; i++) {
        diff[i] = original[i + 1] - original[i];
    }
}

// Extract MultiRocket features (dual representation)
void extract_multirocket_features(const float* X, int length,
                                 const ModelParams& params,
                                 std::vector<float>& features) {
    int num_feat_orig = NUM_KERNELS * params.num_dilations_orig * POOLING_OPS;
    int num_feat_diff = NUM_KERNELS * params.num_dilations_diff * POOLING_OPS;

    features.resize(num_feat_orig + num_feat_diff);

    // Extract features from original representation
    std::vector<float> features_orig(num_feat_orig);
    extract_features_single_repr(X, length, params.dilations_orig,
                                params.biases_orig, params.weights, features_orig);

    // Compute first-order difference
    float X_diff[MAX_TIME_SERIES_LENGTH];
    compute_diff(X, X_diff, length);

    // Extract features from diff representation
    std::vector<float> features_diff(num_feat_diff);
    extract_features_single_repr(X_diff, length - 1, params.dilations_diff,
                                params.biases_diff, params.weights, features_diff);

    // Concatenate features
    std::copy(features_orig.begin(), features_orig.end(), features.begin());
    std::copy(features_diff.begin(), features_diff.end(), features.begin() + num_feat_orig);
}

// Apply StandardScaler
void apply_scaler(std::vector<float>& features, const ModelParams& params) {
    for (size_t i = 0; i < features.size(); i++) {
        features[i] = (features[i] - params.scaler_mean[i]) / params.scaler_scale[i];
    }
}

// Classify using Ridge classifier
int classify(const std::vector<float>& features, const ModelParams& params) {
    std::vector<float> scores(params.num_classes, 0.0f);

    for (int c = 0; c < params.num_classes; c++) {
        scores[c] = params.intercept[c];

        for (size_t f = 0; f < features.size(); f++) {
            scores[c] += params.coefficients[c][f] * features[f];
        }
    }

    // Return class with highest score
    return std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
}

// Load model from JSON
bool load_model(const std::string& filename, ModelParams& params) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open model file: " << filename << std::endl;
        return false;
    }

    json j;
    try {
        file >> j;

        std::cout << "  Loading basic parameters..." << std::endl;
        params.num_kernels = j["num_kernels"];
        params.num_features = j["num_features"];
        params.num_classes = j["num_classes"];
        params.num_dilations_orig = j["num_dilations_orig"];
        params.num_dilations_diff = j["num_dilations_diff"];

        std::cout << "  Loading dilations..." << std::endl;
        params.dilations_orig = j["dilations_orig"].get<std::vector<int>>();
        params.dilations_diff = j["dilations_diff"].get<std::vector<int>>();

        std::cout << "  Loading biases..." << std::endl;
        params.biases_orig = j["biases_orig"].get<std::vector<float>>();
        params.biases_diff = j["biases_diff"].get<std::vector<float>>();

        std::cout << "  Loading scaler parameters..." << std::endl;
        params.scaler_mean = j["scaler_mean"].get<std::vector<float>>();
        params.scaler_scale = j["scaler_scale"].get<std::vector<float>>();

        std::cout << "  Loading classifier..." << std::endl;
        params.coefficients = j["coefficients"].get<std::vector<std::vector<float>>>();
        params.intercept = j["intercept"].get<std::vector<float>>();

        // Generate kernel weights
        std::cout << "  Generating kernel weights..." << std::endl;
        generate_kernel_weights(params.weights);

        std::cout << "✓ Model loaded successfully" << std::endl;
        std::cout << "  Num features: " << params.num_features << std::endl;
        std::cout << "  Num classes: " << params.num_classes << std::endl;
        std::cout << "  Dilations (orig): " << params.num_dilations_orig << std::endl;
        std::cout << "  Dilations (diff): " << params.num_dilations_diff << std::endl;

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

// Load test data from JSON
bool load_test_data(const std::string& filename,
                   std::vector<std::vector<float>>& time_series,
                   std::vector<std::string>& labels,
                   std::vector<std::vector<float>>& python_features,
                   std::vector<int>& python_predictions) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open test data file: " << filename << std::endl;
        return false;
    }

    json j;
    file >> j;

    int num_samples = j["num_samples"];

    time_series = j["time_series"].get<std::vector<std::vector<float>>>();
    labels = j["labels"].get<std::vector<std::string>>();
    python_features = j["features_python"].get<std::vector<std::vector<float>>>();
    python_predictions = j["predictions_python"].get<std::vector<int>>();

    std::cout << "✓ Test data loaded successfully" << std::endl;
    std::cout << "  Num samples: " << num_samples << std::endl;
    std::cout << "  Time series length: " << time_series[0].size() << std::endl;

    return true;
}

// Compute feature-wise error
float compute_feature_error(const std::vector<float>& cpp_features,
                           const std::vector<float>& python_features) {
    float max_error = 0.0f;
    float sum_error = 0.0f;

    for (size_t i = 0; i < cpp_features.size(); i++) {
        float error = std::abs(cpp_features[i] - python_features[i]);
        sum_error += error;
        if (error > max_error) {
            max_error = error;
        }
    }

    return sum_error / cpp_features.size();
}

int main(int argc, char** argv) {
    std::cout << "=" << std::string(79, '=') << std::endl;
    std::cout << "MultiRocket C++ Testbench" << std::endl;
    std::cout << "=" << std::string(79, '=') << std::endl;

    std::string model_file = "multirocket84_gunpoint_model.json";
    std::string test_file = "multirocket84_gunpoint_test.json";

    if (argc >= 2) model_file = argv[1];
    if (argc >= 3) test_file = argv[2];

    // Load model
    std::cout << "\n1. Loading model..." << std::endl;
    ModelParams params;
    if (!load_model(model_file, params)) {
        return 1;
    }

    // Load test data
    std::cout << "\n2. Loading test data..." << std::endl;
    std::vector<std::vector<float>> time_series;
    std::vector<std::string> labels;
    std::vector<std::vector<float>> python_features;
    std::vector<int> python_predictions;

    if (!load_test_data(test_file, time_series, labels, python_features, python_predictions)) {
        return 1;
    }

    // Extract features and validate
    std::cout << "\n3. Extracting features and validating..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    int num_correct = 0;
    float total_avg_error = 0.0f;
    float total_max_error = 0.0f;

    for (size_t i = 0; i < time_series.size(); i++) {
        // Extract features
        std::vector<float> cpp_features;
        extract_multirocket_features(time_series[i].data(), time_series[i].size(),
                                     params, cpp_features);

        // Apply scaler
        apply_scaler(cpp_features, params);

        // Classify
        int cpp_prediction = classify(cpp_features, params);

        // Compare with Python
        float avg_error = compute_feature_error(cpp_features, python_features[i]);

        float max_error = 0.0f;
        for (size_t f = 0; f < cpp_features.size(); f++) {
            float error = std::abs(cpp_features[f] - python_features[i][f]);
            if (error > max_error) max_error = error;
        }

        total_avg_error += avg_error;
        total_max_error = std::max(total_max_error, max_error);

        bool match = (cpp_prediction == python_predictions[i]);
        if (match) num_correct++;

        std::cout << "Sample " << i << ": "
                  << "Label=" << labels[i] << ", "
                  << "C++=" << cpp_prediction << ", "
                  << "Python=" << python_predictions[i] << ", "
                  << (match ? "✓" : "✗") << " | "
                  << "Avg Error=" << std::fixed << std::setprecision(6) << avg_error << ", "
                  << "Max Error=" << max_error << std::endl;
    }

    std::cout << std::string(80, '-') << std::endl;

    // Summary
    std::cout << "\n4. Validation Summary" << std::endl;
    std::cout << "=" << std::string(79, '=') << std::endl;

    float accuracy = (float)num_correct / time_series.size() * 100.0f;
    float avg_error = total_avg_error / time_series.size();

    std::cout << "\nClassification:" << std::endl;
    std::cout << "  Correct: " << num_correct << "/" << time_series.size() << std::endl;
    std::cout << "  Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;

    std::cout << "\nFeature Error:" << std::endl;
    std::cout << "  Average error: " << std::scientific << avg_error << std::endl;
    std::cout << "  Maximum error: " << total_max_error << std::endl;

    std::cout << "\nValidation Status:" << std::endl;
    if (accuracy == 100.0f && avg_error < 0.01f) {
        std::cout << "  ✓ PASS - C++ implementation matches Python reference" << std::endl;
    } else if (accuracy == 100.0f) {
        std::cout << "  ⚠ PARTIAL - Classification correct but features differ" << std::endl;
    } else {
        std::cout << "  ✗ FAIL - C++ implementation does not match Python" << std::endl;
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;

    return (accuracy == 100.0f && avg_error < 0.01f) ? 0 : 1;
}
