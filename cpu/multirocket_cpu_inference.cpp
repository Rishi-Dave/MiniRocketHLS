// MultiRocket84 CPU Baseline Inference (C++)
// Implements the same MultiRocket84 feature extraction + Ridge classifier
// as the custom_multirocket84.py, for fair CPU baseline comparison.
//
// Algorithm: 84 kernels (C(9,3)), 4 pooling operators (PPV, MPV, MIPV, LSPV),
// 2 representations (original + first-order difference).
// Features: 84 × num_dilations × 4 × 2 (typically 4704 for 7 dilations)
//
// Compile: g++ -O3 -march=native -o multirocket_cpu multirocket_cpu_inference.cpp
// Usage:   ./multirocket_cpu <model.json> <test_data.json> [output.csv]

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
// Minimal JSON parser (reused from minirocket_cpu_inference.cpp)
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

    std::vector<std::string> as_string_array() const {
        std::vector<std::string> out;
        out.reserve(arr.size());
        for (auto& v : arr) out.push_back(v.str);
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
// MultiRocket84 Model
// ============================================================

struct MultiRocket84Model {
    int num_kernels;           // 84
    int num_features;          // 4704 typically
    int num_classes;
    int time_series_length;

    int num_dilations_orig;
    int num_dilations_diff;

    std::vector<int> dilations_orig;
    std::vector<int> dilations_diff;

    std::vector<double> biases_orig;   // [num_kernels * num_dilations_orig]
    std::vector<double> biases_diff;   // [num_kernels * num_dilations_diff]

    // Kernel structure: 84 kernels, each with 3 indices from {0..8}
    std::vector<std::vector<int>> kernel_indices;  // [84][3]
    // Kernel weights: [84][9] — -1 everywhere, +2 at selected indices
    std::vector<std::vector<double>> weights;      // [84][9]

    // Scaler
    std::vector<double> scaler_mean;   // [num_features]
    std::vector<double> scaler_scale;  // [num_features]

    // Ridge classifier
    std::vector<std::vector<double>> classifier_coef; // [num_classes][num_features]
    std::vector<double> classifier_intercept;          // [num_classes]
    std::vector<std::string> classes;

    void load(const std::string& path) {
        std::cout << "Loading MultiRocket84 model from: " << path << std::endl;
        auto j = load_json(path);

        num_kernels = (int)j["num_kernels"].as_number();
        num_features = (int)j["num_features"].as_number();
        num_classes = (int)j["num_classes"].as_number();
        time_series_length = (int)j["time_series_length"].as_number();

        num_dilations_orig = (int)j["num_dilations_orig"].as_number();
        num_dilations_diff = (int)j["num_dilations_diff"].as_number();

        dilations_orig = j["dilations_orig"].as_int_array();
        dilations_diff = j["dilations_diff"].as_int_array();

        biases_orig = j["biases_orig"].as_double_array();
        biases_diff = j["biases_diff"].as_double_array();

        kernel_indices = j["kernel_indices"].as_2d_int_array();
        weights = j["weights"].as_2d_double_array();

        scaler_mean = j["scaler_mean"].as_double_array();
        scaler_scale = j["scaler_scale"].as_double_array();

        classifier_coef = j["coefficients"].as_2d_double_array();
        classifier_intercept = j["intercept"].as_double_array();
        classes = j["classes"].as_string_array();

        // Verify dimensions
        assert(num_kernels == 84);
        assert((int)dilations_orig.size() == num_dilations_orig);
        assert((int)dilations_diff.size() == num_dilations_diff);
        assert((int)biases_orig.size() == num_kernels * num_dilations_orig);
        assert((int)biases_diff.size() == num_kernels * num_dilations_diff);
        int expected_features = num_kernels * num_dilations_orig * 4
                              + num_kernels * num_dilations_diff * 4;
        assert(num_features == expected_features);
        assert((int)scaler_mean.size() == num_features);
        assert((int)scaler_scale.size() == num_features);
        assert((int)classifier_coef.size() == num_classes);
        assert((int)classifier_coef[0].size() == num_features);
        assert((int)classifier_intercept.size() == num_classes);

        std::cout << "  num_kernels: " << num_kernels << std::endl;
        std::cout << "  num_features: " << num_features << std::endl;
        std::cout << "  num_classes: " << num_classes << std::endl;
        std::cout << "  time_series_length: " << time_series_length << std::endl;
        std::cout << "  dilations_orig (" << num_dilations_orig << "): ";
        for (int d : dilations_orig) std::cout << d << " ";
        std::cout << std::endl;
        std::cout << "  dilations_diff (" << num_dilations_diff << "): ";
        for (int d : dilations_diff) std::cout << d << " ";
        std::cout << std::endl;
    }
};

// ============================================================
// MultiRocket84 Feature Extraction
//
// For each representation (original series, first-order difference):
//   For each dilation d:
//     For each kernel k (84 kernels):
//       1. Convolve with 9-tap kernel at stride=dilation
//       2. Apply 4 pooling operators with bias threshold:
//          PPV  = proportion of conv values > bias
//          MPV  = mean of conv values where value > bias
//          MIPV = mean of indices where conv value > bias (normalized)
//          LSPV = longest consecutive stretch where conv value > bias (normalized)
// ============================================================

void apply_kernel(const std::vector<double>& series, int series_len,
                  const std::vector<double>& kernel_weights,
                  int dilation, std::vector<double>& conv_output) {
    conv_output.resize(series_len);
    for (int t = 0; t < series_len; t++) {
        double val = 0.0;
        for (int w = 0; w < 9; w++) {
            int idx = t + (w - 4) * dilation;
            if (idx >= 0 && idx < series_len) {
                val += kernel_weights[w] * series[idx];
            }
        }
        conv_output[t] = val;
    }
}

void compute_four_pooling(const std::vector<double>& conv_output, int conv_len,
                          double bias,
                          double& ppv, double& mpv, double& mipv, double& lspv) {
    int count_positive = 0;
    double sum_positive = 0.0;
    double sum_indices = 0.0;
    int max_stretch = 0;
    int current_stretch = 0;

    for (int t = 0; t < conv_len; t++) {
        if (conv_output[t] > bias) {
            count_positive++;
            sum_positive += conv_output[t];
            sum_indices += t;
            current_stretch++;
            if (current_stretch > max_stretch)
                max_stretch = current_stretch;
        } else {
            current_stretch = 0;
        }
    }

    ppv = (double)count_positive / (double)conv_len;

    if (count_positive > 0) {
        mpv = sum_positive / (double)count_positive;
        mipv = (sum_indices / (double)count_positive) / (double)conv_len;
    } else {
        mpv = 0.0;
        mipv = 0.0;
    }

    lspv = (double)max_stretch / (double)conv_len;
}

void extract_features_single_repr(const MultiRocket84Model& model,
                                  const std::vector<double>& series, int series_len,
                                  const std::vector<int>& dilations, int num_dilations,
                                  const std::vector<double>& biases,
                                  std::vector<double>& features, int& feature_idx) {
    std::vector<double> conv_output;

    for (int d = 0; d < num_dilations; d++) {
        int dilation = dilations[d];
        for (int k = 0; k < model.num_kernels; k++) {
            apply_kernel(series, series_len, model.weights[k], dilation, conv_output);

            double bias = biases[d * model.num_kernels + k];
            double ppv, mpv, mipv, lspv;
            compute_four_pooling(conv_output, series_len, bias, ppv, mpv, mipv, lspv);

            features[feature_idx++] = ppv;
            features[feature_idx++] = mpv;
            features[feature_idx++] = mipv;
            features[feature_idx++] = lspv;
        }
    }
}

void extract_features(const MultiRocket84Model& model,
                      const std::vector<double>& time_series,
                      std::vector<double>& features) {
    features.resize(model.num_features);
    int feature_idx = 0;

    // Original representation
    extract_features_single_repr(model, time_series, model.time_series_length,
                                 model.dilations_orig, model.num_dilations_orig,
                                 model.biases_orig, features, feature_idx);

    // First-order difference representation
    int diff_len = model.time_series_length - 1;
    std::vector<double> diff_series(diff_len);
    for (int i = 0; i < diff_len; i++) {
        diff_series[i] = time_series[i + 1] - time_series[i];
    }

    extract_features_single_repr(model, diff_series, diff_len,
                                 model.dilations_diff, model.num_dilations_diff,
                                 model.biases_diff, features, feature_idx);

    assert(feature_idx == model.num_features);
}

void apply_scaler(const MultiRocket84Model& model, std::vector<double>& features) {
    for (int i = 0; i < model.num_features; i++) {
        features[i] = (features[i] - model.scaler_mean[i]) / model.scaler_scale[i];
    }
}

std::string classify(const MultiRocket84Model& model, const std::vector<double>& features) {
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
    MultiRocket84Model model;
    model.load(model_path);

    // Load test data
    std::cout << "Loading test data from: " << test_path << std::endl;
    auto test_json = load_json(test_path);

    int num_samples = (int)test_json["num_samples"].as_number();
    int series_length = (int)test_json["time_series_length"].as_number();

    // Handle varying field names
    auto X_test_2d = test_json["time_series"].as_2d_double_array();
    if (X_test_2d.empty()) {
        X_test_2d = test_json["X_test"].as_2d_double_array();
    }

    // Labels can be strings or ints
    std::vector<std::string> y_test;
    auto& labels_val = test_json["labels"];
    if (labels_val.type == JsonValue::NONE) {
        auto& yt = test_json["y_test"];
        for (auto& v : yt.arr) {
            if (v.type == JsonValue::STRING) y_test.push_back(v.str);
            else y_test.push_back(std::to_string((int)v.number));
        }
    } else {
        for (auto& v : labels_val.arr) {
            if (v.type == JsonValue::STRING) y_test.push_back(v.str);
            else y_test.push_back(std::to_string((int)v.number));
        }
    }

    // Dataset name
    std::string dataset_name = "unknown";
    if (test_json["dataset"].type != JsonValue::NONE)
        dataset_name = test_json["dataset"].as_string();
    else if (test_json["dataset_name"].type != JsonValue::NONE)
        dataset_name = test_json["dataset_name"].as_string();

    std::cout << "Dataset: " << dataset_name << std::endl;
    std::cout << "  Samples: " << num_samples << std::endl;
    std::cout << "  Series length: " << series_length << std::endl;

    assert(series_length == model.time_series_length);
    assert((int)X_test_2d.size() == num_samples);
    assert((int)y_test.size() == num_samples);

    // Per-sample inference with timing
    std::cout << "\nRunning per-sample inference..." << std::endl;
    std::vector<double> latencies_ms(num_samples);
    std::vector<std::string> predictions(num_samples);
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
        std::string pred = classify(model, features);
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
    std::cout << "Algorithm:   MultiRocket84" << std::endl;
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
        csv_path = "../results/MultiRocket84_CPU_cpp_" + dataset_name + "_per_sample.csv";
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
