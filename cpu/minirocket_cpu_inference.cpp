// MiniRocket CPU Baseline Inference (C++)
// Implements the same MiniRocket feature extraction + Ridge classifier
// as the Python aeon library, for fair CPU baseline comparison.
//
// Compile: g++ -O3 -march=native -o minirocket_cpu minirocket_cpu_inference.cpp
// Usage:   ./minirocket_cpu <model.json> <test_data.json> [output.csv]

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iomanip>

// ============================================================
// Minimal JSON parser (no external dependencies)
// Handles the specific model/test_data JSON format
// ============================================================

struct JsonValue {
    enum Type { NONE, NUMBER, STRING, ARRAY, OBJECT };
    Type type = NONE;
    double number = 0;
    std::string str;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    double as_number() const { return number; }
    const std::string& as_string() const { return str; }

    const JsonValue& operator[](const std::string& key) const {
        for (auto& p : obj)
            if (p.first == key) return p.second;
        static JsonValue empty;
        return empty;
    }

    std::vector<double> as_double_array() const {
        std::vector<double> out;
        out.reserve(arr.size());
        for (auto& v : arr) out.push_back(v.number);
        return out;
    }

    std::vector<int> as_int_array() const {
        std::vector<int> out;
        out.reserve(arr.size());
        for (auto& v : arr) out.push_back((int)v.number);
        return out;
    }

    std::vector<std::vector<double>> as_2d_double_array() const {
        std::vector<std::vector<double>> out;
        out.reserve(arr.size());
        for (auto& row : arr) out.push_back(row.as_double_array());
        return out;
    }

    std::vector<std::vector<int>> as_2d_int_array() const {
        std::vector<std::vector<int>> out;
        out.reserve(arr.size());
        for (auto& row : arr) out.push_back(row.as_int_array());
        return out;
    }
};

class JsonParser {
    const std::string& s;
    size_t pos;

    void skip_ws() {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t'))
            pos++;
    }

    JsonValue parse_value() {
        skip_ws();
        if (pos >= s.size()) return {};
        char c = s[pos];
        if (c == '"') return parse_string();
        if (c == '[') return parse_array();
        if (c == '{') return parse_object();
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        if (s.substr(pos, 4) == "true") { pos += 4; JsonValue v; v.type = JsonValue::NUMBER; v.number = 1; return v; }
        if (s.substr(pos, 5) == "false") { pos += 5; JsonValue v; v.type = JsonValue::NUMBER; v.number = 0; return v; }
        if (s.substr(pos, 4) == "null") { pos += 4; return {}; }
        return {};
    }

    JsonValue parse_string() {
        pos++; // skip "
        JsonValue v;
        v.type = JsonValue::STRING;
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\') { pos++; v.str += s[pos++]; }
            else v.str += s[pos++];
        }
        pos++; // skip closing "
        return v;
    }

    JsonValue parse_number() {
        JsonValue v;
        v.type = JsonValue::NUMBER;
        size_t start = pos;
        if (s[pos] == '-') pos++;
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') pos++;
        if (pos < s.size() && s[pos] == '.') {
            pos++;
            while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') pos++;
        }
        if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
            pos++;
            if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) pos++;
            while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') pos++;
        }
        v.number = std::stod(s.substr(start, pos - start));
        return v;
    }

    JsonValue parse_array() {
        pos++; // skip [
        JsonValue v;
        v.type = JsonValue::ARRAY;
        skip_ws();
        if (pos < s.size() && s[pos] == ']') { pos++; return v; }
        while (true) {
            v.arr.push_back(parse_value());
            skip_ws();
            if (pos >= s.size() || s[pos] == ']') { pos++; break; }
            pos++; // skip ,
        }
        return v;
    }

    JsonValue parse_object() {
        pos++; // skip {
        JsonValue v;
        v.type = JsonValue::OBJECT;
        skip_ws();
        if (pos < s.size() && s[pos] == '}') { pos++; return v; }
        while (true) {
            skip_ws();
            auto key = parse_string();
            skip_ws();
            pos++; // skip :
            auto val = parse_value();
            v.obj.push_back({key.str, val});
            skip_ws();
            if (pos >= s.size() || s[pos] == '}') { pos++; break; }
            pos++; // skip ,
        }
        return v;
    }

public:
    JsonParser(const std::string& str) : s(str), pos(0) {}
    JsonValue parse() { return parse_value(); }
};

JsonValue load_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "ERROR: Cannot open " << path << std::endl;
        exit(1);
    }
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    JsonParser parser(content);
    return parser.parse();
}

// ============================================================
// MiniRocket Model
// ============================================================

struct MiniRocketModel {
    int num_kernels;       // 84
    int num_dilations;     // 9
    int num_features;      // 840
    int num_classes;
    int time_series_length;

    std::vector<std::vector<int>> kernel_indices; // [84][3]
    std::vector<int> dilations;                   // [num_dilations]
    std::vector<int> num_features_per_dilation;   // [num_dilations]
    std::vector<double> biases;                   // [num_features]

    // Scaler
    std::vector<double> scaler_mean;  // [num_features]
    std::vector<double> scaler_scale; // [num_features]

    // Ridge classifier
    std::vector<std::vector<double>> classifier_coef; // [num_classes][num_features]
    std::vector<double> classifier_intercept;          // [num_classes]
    std::vector<int> classes;

    void load(const std::string& path) {
        std::cout << "Loading model from: " << path << std::endl;
        auto j = load_json(path);

        num_kernels = (int)j["num_kernels"].as_number();
        num_dilations = (int)j["num_dilations"].as_number();
        num_features = (int)j["num_features"].as_number();
        num_classes = (int)j["num_classes"].as_number();
        time_series_length = (int)j["time_series_length"].as_number();

        kernel_indices = j["kernel_indices"].as_2d_int_array();
        dilations = j["dilations"].as_int_array();
        num_features_per_dilation = j["num_features_per_dilation"].as_int_array();
        biases = j["biases"].as_double_array();
        scaler_mean = j["scaler_mean"].as_double_array();
        scaler_scale = j["scaler_scale"].as_double_array();
        // Handle binary vs multi-class: binary has coef as 1D [num_features]
        auto& coef_val = j["classifier_coef"];
        if (coef_val.arr.size() > 0 && coef_val.arr[0].type == JsonValue::ARRAY) {
            classifier_coef = coef_val.as_2d_double_array();
        } else {
            // Binary: single coefficient vector, wrap in 2D
            classifier_coef.push_back(coef_val.as_double_array());
        }
        classifier_intercept = j["classifier_intercept"].as_double_array();
        classes = j["classes"].as_int_array();

        std::cout << "  num_kernels: " << num_kernels << std::endl;
        std::cout << "  num_dilations: " << num_dilations << std::endl;
        std::cout << "  num_features: " << num_features << std::endl;
        std::cout << "  num_classes: " << num_classes << std::endl;
        std::cout << "  time_series_length: " << time_series_length << std::endl;
    }
};

// ============================================================
// MiniRocket Feature Extraction
//
// For each dilation d:
//   For each kernel k (84 kernels, each with 3 indices from {0..8}):
//     weights[9] = {-1,-1,-1,-1,-1,-1,-1,-1,-1} initially
//     then set weights[indices[0]] += 3, weights[indices[1]] += 3, weights[indices[2]] += 3
//     => weights at selected indices become +2, rest remain -1
//     Convolve time_series with these 9 weights at stride=dilation
//     For each bias in this kernel's bias set:
//       feature = PPV (proportion of conv output > bias)
// ============================================================

// The 9 fixed weights: -1 everywhere, +2 at the 3 selected indices
// This matches the MiniRocket paper: W = {-1, +2} with 3 positive positions
static const double WEIGHT_NEG = -1.0;
static const double WEIGHT_POS = 2.0;

void extract_features(const MiniRocketModel& model,
                      const std::vector<double>& time_series,
                      std::vector<double>& features) {
    const int L = model.time_series_length;
    features.resize(model.num_features);

    int feature_idx = 0;

    // Aeon's MiniRocket centers kernels at position 4 and alternates
    // between padded (full output) and unpadded (valid-only) per kernel.
    // _padding0 = dilation_index % 2
    // _padding1 = (_padding0 + kernel_index) % 2
    // When _padding1 == 0: padded output (length = L, zero-pad boundaries)
    // When _padding1 == 1: unpadded output (length = L - 8*dilation)

    for (int d = 0; d < model.num_dilations; d++) {
        int dilation = model.dilations[d];
        int n_feat_this_dil = model.num_features_per_dilation[d];
        int padding0 = d % 2;
        int half_pad = 4 * dilation; // padding on each side

        for (int k = 0; k < 84; k++) {
            // Build weights: -1 everywhere, +2 at selected indices
            double weights[9];
            for (int i = 0; i < 9; i++) weights[i] = WEIGHT_NEG;
            weights[model.kernel_indices[k][0]] = WEIGHT_POS;
            weights[model.kernel_indices[k][1]] = WEIGHT_POS;
            weights[model.kernel_indices[k][2]] = WEIGHT_POS;

            int padding1 = (padding0 + k) % 2;

            // Determine convolution range
            int t_start, t_end, conv_length;
            if (padding1 == 0) {
                // Padded: full output, kernel centered at position 4
                t_start = 0;
                t_end = L;
                conv_length = L;
            } else {
                // Unpadded: valid only, skip boundary positions
                t_start = half_pad;
                t_end = L - half_pad;
                conv_length = t_end - t_start;
            }

            if (conv_length <= 0) {
                for (int f = 0; f < n_feat_this_dil; f++) {
                    features[feature_idx++] = 0.0;
                }
                continue;
            }

            for (int f = 0; f < n_feat_this_dil; f++) {
                double bias = model.biases[feature_idx];
                int count_positive = 0;

                for (int t = t_start; t < t_end; t++) {
                    double conv_val = 0.0;
                    for (int w = 0; w < 9; w++) {
                        // Kernel centered at position 4: offset = (w - 4) * dilation
                        int idx = t + (w - 4) * dilation;
                        if (idx >= 0 && idx < L) {
                            conv_val += weights[w] * time_series[idx];
                        }
                        // else: zero-padding (contributes 0)
                    }
                    if (conv_val > bias) {
                        count_positive++;
                    }
                }

                features[feature_idx] = (double)count_positive / (double)conv_length;
                feature_idx++;
            }
        }
    }
}

void apply_scaler(const MiniRocketModel& model, std::vector<double>& features) {
    for (int i = 0; i < model.num_features; i++) {
        features[i] = (features[i] - model.scaler_mean[i]) / model.scaler_scale[i];
    }
}

int classify(const MiniRocketModel& model, const std::vector<double>& features) {
    if (model.classifier_coef.size() == 1) {
        // Binary classification: sign(coef @ features + intercept)
        double score = model.classifier_intercept[0];
        for (int f = 0; f < model.num_features; f++) {
            score += model.classifier_coef[0][f] * features[f];
        }
        return model.classes[score > 0 ? 1 : 0];
    }

    // Multi-class: argmax(coef @ features + intercept)
    int best_class = 0;
    double best_score = -1e30;

    for (int c = 0; c < model.num_classes; c++) {
        double score = model.classifier_intercept[c];
        for (int f = 0; f < model.num_features; f++) {
            score += model.classifier_coef[c][f] * features[f];
        }
        if (score > best_score) {
            best_score = score;
            best_class = c;
        }
    }

    return model.classes[best_class];
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.json> <test_data.json> [output.csv]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string test_path = argv[2];
    std::string csv_path = (argc > 3) ? argv[3] : "";

    // Load model
    MiniRocketModel model;
    model.load(model_path);

    // Load test data
    std::cout << "Loading test data from: " << test_path << std::endl;
    auto test_json = load_json(test_path);

    // Handle varying field names across test data JSONs
    std::string dataset_name;
    if (test_json["dataset_name"].type != JsonValue::NONE)
        dataset_name = test_json["dataset_name"].as_string();
    else
        dataset_name = "unknown";

    int num_samples = (int)test_json["num_samples"].as_number();

    int series_length = 0;
    if (test_json["series_length"].type != JsonValue::NONE)
        series_length = (int)test_json["series_length"].as_number();
    else if (test_json["time_series_length"].type != JsonValue::NONE)
        series_length = (int)test_json["time_series_length"].as_number();
    else
        series_length = (int)test_json["X_test"].arr[0].arr.size();

    auto X_test_2d = test_json["X_test"].as_2d_double_array();
    auto y_test = test_json["y_test"].as_int_array();

    std::cout << "Dataset: " << dataset_name << std::endl;
    std::cout << "  Samples: " << num_samples << std::endl;
    std::cout << "  Series length: " << series_length << std::endl;

    assert(series_length == model.time_series_length);
    assert((int)X_test_2d.size() == num_samples);
    assert((int)y_test.size() == num_samples);

    // Per-sample inference with timing
    std::cout << "\nRunning per-sample inference..." << std::endl;
    std::vector<double> latencies_ms(num_samples);
    std::vector<int> predictions(num_samples);
    std::vector<double> features;
    int correct = 0;

    // Warmup (3 samples)
    for (int i = 0; i < std::min(3, num_samples); i++) {
        extract_features(model, X_test_2d[i], features);
        apply_scaler(model, features);
        classify(model, features);
    }

    for (int i = 0; i < num_samples; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        extract_features(model, X_test_2d[i], features);
        apply_scaler(model, features);
        int pred = classify(model, features);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        latencies_ms[i] = ms;
        predictions[i] = pred;

        if (pred == y_test[i]) correct++;

        if ((i + 1) % 5000 == 0 || i == num_samples - 1) {
            std::cout << "  Sample " << (i + 1) << "/" << num_samples << "..." << std::endl;
        }
    }

    // Compute statistics
    double accuracy = (double)correct / num_samples;

    std::vector<double> sorted_lat(latencies_ms);
    std::sort(sorted_lat.begin(), sorted_lat.end());

    double sum = 0, sum2 = 0;
    for (double v : latencies_ms) { sum += v; sum2 += v * v; }
    double mean = sum / num_samples;
    double std_dev = std::sqrt(sum2 / num_samples - mean * mean);

    auto percentile = [&](double p) -> double {
        double idx = p / 100.0 * (num_samples - 1);
        int lo = (int)idx;
        int hi = std::min(lo + 1, num_samples - 1);
        double frac = idx - lo;
        return sorted_lat[lo] * (1 - frac) + sorted_lat[hi] * frac;
    };

    double p50 = percentile(50);
    double p95 = percentile(95);
    double p99 = percentile(99);

    std::cout << "\n========== RESULTS ==========" << std::endl;
    std::cout << "Dataset:     " << dataset_name << std::endl;
    std::cout << "Accuracy:    " << std::fixed << std::setprecision(4) << (accuracy * 100)
              << "% (" << correct << "/" << num_samples << ")" << std::endl;
    std::cout << "Throughput:  " << std::fixed << std::setprecision(1) << (1000.0 / mean)
              << " inferences/sec" << std::endl;
    std::cout << "\nLatency distribution (ms):" << std::endl;
    std::cout << "  Mean:  " << std::fixed << std::setprecision(3) << mean << std::endl;
    std::cout << "  P50:   " << p50 << std::endl;
    std::cout << "  P95:   " << p95 << std::endl;
    std::cout << "  P99:   " << p99 << std::endl;
    std::cout << "  Min:   " << sorted_lat.front() << std::endl;
    std::cout << "  Max:   " << sorted_lat.back() << std::endl;
    std::cout << "  Std:   " << std_dev << std::endl;
    std::cout << "=============================" << std::endl;

    // Write CSV
    if (csv_path.empty()) {
        csv_path = "../results/MiniRocket_CPU_cpp_" + dataset_name + "_per_sample.csv";
    }
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "sample_id,total_ms,predicted,actual,correct" << std::endl;
        for (int i = 0; i < num_samples; i++) {
            csv << i << "," << std::fixed << std::setprecision(3) << latencies_ms[i]
                << "," << predictions[i] << "," << y_test[i]
                << "," << (predictions[i] == y_test[i] ? 1 : 0) << std::endl;
        }
        csv.close();
        std::cout << "\nPer-sample CSV written to: " << csv_path << std::endl;
    } else {
        std::cerr << "WARNING: Could not open " << csv_path << " for writing" << std::endl;
    }

    return 0;
}
