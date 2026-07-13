#include "hydra_hls_testbench_loader.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <map>
#include <stdexcept>

// =========================================================================
// Minimal but real JSON parser for HYDRA testbench
// Handles the flat arrays and nested 2D array in the model/test JSON files.
// =========================================================================

// Read entire file into a string
static bool read_file(const std::string& filename, std::string& content) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "ERROR: Cannot open file: " << filename << std::endl;
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    content = ss.str();
    return true;
}

// Find the value string for a given key in JSON content.
// Returns the raw JSON value (could be a number, string, array, nested array).
// Looks for "key": <value>
static size_t find_key(const std::string& content, const std::string& key, size_t start = 0) {
    std::string search = "\"" + key + "\"";
    size_t pos = content.find(search, start);
    if (pos == std::string::npos) return std::string::npos;
    // Find the colon after the key
    size_t colon = content.find(':', pos + search.size());
    if (colon == std::string::npos) return std::string::npos;
    // Skip whitespace
    size_t val_start = colon + 1;
    while (val_start < content.size() && (content[val_start] == ' ' || content[val_start] == '\n' ||
           content[val_start] == '\r' || content[val_start] == '\t')) {
        val_start++;
    }
    return val_start;
}

// Parse a scalar integer at position pos in content
static int parse_int_at(const std::string& content, size_t pos) {
    size_t end = pos;
    while (end < content.size() && (isdigit(content[end]) || content[end] == '-' || content[end] == '+'))
        end++;
    return std::stoi(content.substr(pos, end - pos));
}

// Parse a flat 1D array of floats starting at '[' position
static std::vector<float> parse_flat_float_array(const std::string& content, size_t bracket_pos) {
    std::vector<float> result;
    if (bracket_pos == std::string::npos || content[bracket_pos] != '[') return result;

    size_t pos = bracket_pos + 1;
    size_t end = content.size();

    while (pos < end) {
        // Skip whitespace
        while (pos < end && (content[pos] == ' ' || content[pos] == '\n' ||
               content[pos] == '\r' || content[pos] == '\t' || content[pos] == ','))
            pos++;
        if (pos >= end || content[pos] == ']') break;
        // Parse float
        size_t num_end = pos;
        while (num_end < end && (isdigit(content[num_end]) || content[num_end] == '-' ||
               content[num_end] == '+' || content[num_end] == '.' || content[num_end] == 'e' ||
               content[num_end] == 'E'))
            num_end++;
        if (num_end > pos) {
            try {
                result.push_back(std::stof(content.substr(pos, num_end - pos)));
            } catch (...) {}
        }
        pos = num_end;
    }
    return result;
}

// Parse a flat 1D array of ints starting at '[' position
static std::vector<int> parse_flat_int_array(const std::string& content, size_t bracket_pos) {
    std::vector<int> result;
    if (bracket_pos == std::string::npos || content[bracket_pos] != '[') return result;

    size_t pos = bracket_pos + 1;
    size_t end = content.size();

    while (pos < end) {
        while (pos < end && (content[pos] == ' ' || content[pos] == '\n' ||
               content[pos] == '\r' || content[pos] == '\t' || content[pos] == ','))
            pos++;
        if (pos >= end || content[pos] == ']') break;
        size_t num_end = pos;
        while (num_end < end && (isdigit(content[num_end]) || content[num_end] == '-' ||
               content[num_end] == '+'))
            num_end++;
        if (num_end > pos) {
            try {
                result.push_back(std::stoi(content.substr(pos, num_end - pos)));
            } catch (...) {}
        }
        pos = num_end;
    }
    return result;
}

// Parse a 2D array of floats: [[...], [...], ...]
// Returns flattened row-major vector, sets num_rows and num_cols
static std::vector<float> parse_2d_float_array(const std::string& content, size_t outer_bracket,
                                                 int& num_rows, int& num_cols) {
    std::vector<float> result;
    num_rows = 0;
    num_cols = -1;

    if (outer_bracket == std::string::npos || content[outer_bracket] != '[') return result;

    size_t pos = outer_bracket + 1;
    size_t end = content.size();

    while (pos < end) {
        // Skip whitespace and commas
        while (pos < end && (content[pos] == ' ' || content[pos] == '\n' ||
               content[pos] == '\r' || content[pos] == '\t' || content[pos] == ','))
            pos++;

        if (pos >= end) break;
        if (content[pos] == ']') break;  // End of outer array
        if (content[pos] != '[') { pos++; continue; }  // Skip unexpected chars

        // Parse inner array
        std::vector<float> row = parse_flat_float_array(content, pos);
        if (num_cols < 0) num_cols = (int)row.size();

        result.insert(result.end(), row.begin(), row.end());
        num_rows++;

        // Skip to end of inner array ']'
        int depth = 1;
        pos++;
        while (pos < end && depth > 0) {
            if (content[pos] == '[') depth++;
            else if (content[pos] == ']') depth--;
            pos++;
        }
    }
    return result;
}

// Parse array of quoted strings: ["foo", "bar", ...]
static std::vector<std::string> parse_string_array(const std::string& content, size_t bracket_pos) {
    std::vector<std::string> result;
    if (bracket_pos == std::string::npos || content[bracket_pos] != '[') return result;

    size_t pos = bracket_pos + 1;
    size_t end = content.size();

    while (pos < end) {
        while (pos < end && content[pos] != '"' && content[pos] != ']') pos++;
        if (pos >= end || content[pos] == ']') break;
        // Found opening quote
        pos++;
        size_t str_start = pos;
        while (pos < end && content[pos] != '"') pos++;
        result.push_back(content.substr(str_start, pos - str_start));
        pos++;  // Skip closing quote
    }
    return result;
}

// Map InsectSound string label to integer class index (alphabetical sort = sklearn LabelEncoder order)
static int map_insectsound_label(const std::string& label) {
    static const std::map<std::string, int> label_map = {
        {"aedes_female", 0},
        {"aedes_male",   1},
        {"fruit_flies",  2},
        {"house_flies",  3},
        {"quinx_female", 4},
        {"quinx_male",   5},
        {"stigma_female",6},
        {"stigma_male",  7},
        {"tarsalis_female", 8},
        {"tarsalis_male",   9}
    };
    auto it = label_map.find(label);
    if (it != label_map.end()) return it->second;
    // Try numeric label
    try { return std::stoi(label); } catch (...) {}
    std::cerr << "WARNING: Unknown label '" << label << "', defaulting to 0" << std::endl;
    return 0;
}

// =========================================================================
// Public API
// =========================================================================

bool load_hydra_model(const std::string& filename, HydraModelParams_HLS* model) {
    std::cout << "Loading HYDRA model from: " << filename << std::endl;
    std::string content;
    if (!read_file(filename, content)) return false;

    // Parse scalar metadata
    {
        size_t pos = find_key(content, "num_features");
        if (pos != std::string::npos)
            model->num_features = parse_int_at(content, pos);
        else
            model->num_features = 1024;
    }
    {
        size_t pos = find_key(content, "num_classes");
        if (pos != std::string::npos)
            model->num_classes = parse_int_at(content, pos);
        else
            model->num_classes = 2;
    }
    {
        size_t pos = find_key(content, "time_series_length");
        if (pos != std::string::npos)
            model->time_series_length = parse_int_at(content, pos);
        else
            model->time_series_length = 150;
    }

    // Parse kernel_weights (flat array, length NUM_KERNELS * KERNEL_SIZE)
    {
        size_t pos = find_key(content, "kernel_weights");
        if (pos != std::string::npos && content[pos] == '[') {
            auto vals = parse_flat_float_array(content, pos);
            int n = std::min((int)vals.size(), NUM_KERNELS * KERNEL_SIZE);
            for (int i = 0; i < n; i++) model->kernel_weights[i] = vals[i];
            std::cout << "  kernel_weights: " << n << " values loaded" << std::endl;
        } else {
            std::cerr << "ERROR: kernel_weights not found in model file" << std::endl;
            return false;
        }
    }

    // Parse biases
    {
        size_t pos = find_key(content, "biases");
        if (pos != std::string::npos && content[pos] == '[') {
            auto vals = parse_flat_float_array(content, pos);
            int n = std::min((int)vals.size(), NUM_KERNELS);
            for (int i = 0; i < n; i++) model->biases[i] = vals[i];
            std::cout << "  biases: " << n << " values loaded" << std::endl;
        }
    }

    // Parse dilations (integers)
    {
        size_t pos = find_key(content, "dilations");
        if (pos != std::string::npos && content[pos] == '[') {
            auto vals = parse_flat_int_array(content, pos);
            int n = std::min((int)vals.size(), NUM_KERNELS);
            for (int i = 0; i < n; i++) model->dilations[i] = vals[i];
            // Fill group_assignments based on index (group = kernel / KERNELS_PER_GROUP)
            for (int i = 0; i < NUM_KERNELS; i++)
                model->group_assignments[i] = i / KERNELS_PER_GROUP;
            std::cout << "  dilations: " << n << " values loaded" << std::endl;
        }
    }

    // Parse scaler_mean
    {
        size_t pos = find_key(content, "scaler_mean");
        if (pos != std::string::npos && content[pos] == '[') {
            auto vals = parse_flat_float_array(content, pos);
            int n = std::min((int)vals.size(), (int)MAX_FEATURES);
            for (int i = 0; i < n; i++) model->scaler_mean[i] = vals[i];
            std::cout << "  scaler_mean: " << n << " values loaded" << std::endl;
        }
    }

    // Parse scaler_scale
    {
        size_t pos = find_key(content, "scaler_scale");
        if (pos != std::string::npos && content[pos] == '[') {
            auto vals = parse_flat_float_array(content, pos);
            int n = std::min((int)vals.size(), (int)MAX_FEATURES);
            for (int i = 0; i < n; i++) model->scaler_scale[i] = vals[i];
            std::cout << "  scaler_scale: " << n << " values loaded" << std::endl;
        }
    }

    // Parse coefficients (flat, length num_features * num_classes)
    {
        size_t pos = find_key(content, "coefficients");
        if (pos != std::string::npos && content[pos] == '[') {
            auto vals = parse_flat_float_array(content, pos);
            int n = std::min((int)vals.size(), (int)(MAX_FEATURES * MAX_CLASSES));
            for (int i = 0; i < n; i++) model->coefficients[i] = vals[i];
            std::cout << "  coefficients: " << n << " values loaded" << std::endl;
        }
    }

    // Parse intercept
    {
        size_t pos = find_key(content, "intercept");
        if (pos != std::string::npos && content[pos] == '[') {
            auto vals = parse_flat_float_array(content, pos);
            int n = std::min((int)vals.size(), (int)MAX_CLASSES);
            for (int i = 0; i < n; i++) model->intercept[i] = vals[i];
            std::cout << "  intercept: " << n << " values loaded" << std::endl;
        }
    }

    std::cout << "Model loaded successfully from JSON" << std::endl;
    std::cout << "  num_kernels: " << NUM_KERNELS << std::endl;
    std::cout << "  num_features: " << model->num_features << std::endl;
    std::cout << "  num_classes: " << model->num_classes << std::endl;
    std::cout << "  time_series_length: " << model->time_series_length << std::endl;
    return true;
}

bool load_test_data(const std::string& filename, TestTimeSeriesData* test_data) {
    std::cout << "Loading test data from: " << filename << std::endl;
    std::string content;
    if (!read_file(filename, content)) return false;

    // Parse metadata
    {
        size_t pos = find_key(content, "num_samples");
        test_data->num_samples = (pos != std::string::npos) ? parse_int_at(content, pos) : 0;
    }
    {
        size_t pos = find_key(content, "time_series_length");
        test_data->time_series_length = (pos != std::string::npos) ? parse_int_at(content, pos) : 0;
    }
    {
        size_t pos = find_key(content, "num_classes");
        test_data->num_classes = (pos != std::string::npos) ? parse_int_at(content, pos) : 2;
    }

    // Parse time_series: 2D array [[...], [...], ...]
    {
        size_t pos = find_key(content, "time_series");
        if (pos == std::string::npos || content[pos] != '[') {
            std::cerr << "ERROR: time_series not found in test file" << std::endl;
            return false;
        }

        int num_rows = 0, num_cols = 0;
        std::cout << "  Parsing time_series 2D array (this may take a moment)..." << std::endl;
        std::vector<float> flat = parse_2d_float_array(content, pos, num_rows, num_cols);

        if (num_rows == 0) {
            std::cerr << "ERROR: No time series samples parsed" << std::endl;
            return false;
        }

        test_data->num_samples = num_rows;
        test_data->time_series_length = num_cols;
        test_data->X_test.clear();
        test_data->X_test.resize(num_rows);
        for (int i = 0; i < num_rows; i++) {
            test_data->X_test[i].resize(num_cols);
            for (int j = 0; j < num_cols; j++) {
                test_data->X_test[i][j] = flat[i * num_cols + j];
            }
        }
        std::cout << "  time_series: " << num_rows << " samples x " << num_cols << " timesteps" << std::endl;
    }

    // Parse labels — try integer array first (MosquitoSound/FruitFlies have numeric labels)
    {
        size_t pos = find_key(content, "labels");
        if (pos == std::string::npos) {
            std::cerr << "ERROR: labels not found in test file" << std::endl;
            return false;
        }

        test_data->y_test.clear();

        if (content[pos] == '[') {
            // Try to determine if labels are strings or numbers by looking at first non-whitespace char after '['
            size_t scan = pos + 1;
            while (scan < content.size() && (content[scan] == ' ' || content[scan] == '\n' ||
                   content[scan] == '\r' || content[scan] == '\t')) scan++;

            if (scan < content.size() && content[scan] == '"') {
                // String labels (InsectSound)
                auto str_labels = parse_string_array(content, pos);
                std::cout << "  labels: " << str_labels.size() << " string labels" << std::endl;
                for (const auto& lbl : str_labels) {
                    test_data->y_test.push_back(map_insectsound_label(lbl));
                }
            } else {
                // Integer labels
                auto int_labels = parse_flat_int_array(content, pos);
                std::cout << "  labels: " << int_labels.size() << " integer labels" << std::endl;
                for (int lbl : int_labels) {
                    test_data->y_test.push_back(lbl);
                }
            }
        }

        if (test_data->y_test.size() != (size_t)test_data->num_samples) {
            std::cerr << "WARNING: labels size " << test_data->y_test.size()
                      << " != num_samples " << test_data->num_samples << std::endl;
        }
    }

    std::cout << "Test data loaded from JSON" << std::endl;
    std::cout << "  Samples: " << test_data->num_samples << std::endl;
    std::cout << "  Length: " << test_data->time_series_length << std::endl;
    std::cout << "  Classes: " << test_data->num_classes << std::endl;
    return true;
}

data_t validate_predictions(
    const std::vector<std::vector<data_t>>& predictions,
    const std::vector<int_t>& labels,
    int_t num_samples,
    int_t num_classes,
    data_t tolerance
) {
    int_t correct = 0;

    for (int_t i = 0; i < num_samples; i++) {
        // Find predicted class (argmax)
        int_t predicted_class = 0;
        data_t max_score = predictions[i][0];

        for (int_t c = 1; c < (int_t)predictions[i].size(); c++) {
            if (predictions[i][c] > max_score) {
                max_score = predictions[i][c];
                predicted_class = c;
            }
        }

        if (predicted_class == labels[i]) {
            correct++;
        }

        // Only print first 20 samples to avoid flooding output
        if (i < 20) {
            std::cout << "Sample " << i << ": predicted=" << predicted_class
                      << ", actual=" << labels[i]
                      << " [" << (predicted_class == labels[i] ? "PASS" : "FAIL") << "]"
                      << std::endl;
        }
    }
    if (num_samples > 20) {
        std::cout << "... (suppressed output for samples 20-" << num_samples - 1 << ")" << std::endl;
    }

    data_t accuracy = static_cast<data_t>(correct) / num_samples;
    std::cout << "\nAccuracy: " << correct << "/" << num_samples
              << " = " << (accuracy * 100.0) << "%" << std::endl;

    return accuracy;
}

void print_model_summary(const HydraModelParams_HLS& model) {
    std::cout << "\n=== HYDRA Model Summary ===" << std::endl;
    std::cout << "Time series length: " << model.time_series_length << std::endl;
    std::cout << "Number of kernels: " << NUM_KERNELS << std::endl;
    std::cout << "Number of groups: " << NUM_GROUPS << std::endl;
    std::cout << "Kernels per group: " << KERNELS_PER_GROUP << std::endl;
    std::cout << "Number of features: " << model.num_features << std::endl;
    std::cout << "Number of classes: " << model.num_classes << std::endl;
    std::cout << "Pooling operators: " << POOLING_OPERATORS << std::endl;
    std::cout << "==========================\n" << std::endl;
}

void print_test_data_summary(const TestTimeSeriesData& test_data) {
    std::cout << "\n=== Test Data Summary ===" << std::endl;
    std::cout << "Number of samples: " << test_data.num_samples << std::endl;
    std::cout << "Time series length: " << test_data.time_series_length << std::endl;
    std::cout << "Number of classes: " << test_data.num_classes << std::endl;

    // Class distribution
    std::vector<int_t> class_counts(test_data.num_classes, 0);
    for (int_t label : test_data.y_test) {
        if (label >= 0 && label < test_data.num_classes) {
            class_counts[label]++;
        }
    }

    std::cout << "Class distribution:" << std::endl;
    for (int_t c = 0; c < test_data.num_classes; c++) {
        std::cout << "  Class " << c << ": " << class_counts[c] << " samples" << std::endl;
    }
    std::cout << "==========================\n" << std::endl;
}
