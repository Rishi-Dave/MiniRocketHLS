// MiniRocket CPU Baseline Inference (C++) — Multithreaded (v2)
// Adds OpenMP-based multithreading, to make CPU throughput comparisons
// against the paper's baseline meaningful: the paper's own CPU baseline
// is multi-threaded Python (aeon, parallelized via Python's
// `multiprocessing` across convolutions/pooling) — the original
// single-threaded minirocket_cpu_inference.cpp is NOT directly
// comparable to that on throughput, only on correctness/accuracy.
//
// Design:
//   - Per-sample LATENCY benchmark stays single-threaded and sequential,
//     exactly like the original file. Running this multithreaded would
//     add scheduling/contention noise to the P50/P95/P99 numbers,
//     making single-sample latency comparisons (vs GPU/FPGA) less
//     meaningful. Latency is inherently a single-thread-doing-one-thing
//     measurement.
//   - A NEW batched THROUGHPUT benchmark parallelizes across samples
//     using OpenMP (`#pragma omp parallel for`), each thread processing
//     a different sample. This is the number to compare against the
//     paper's multi-threaded Python execution times/throughput.
//
// Compile (note the added -fopenmp):
//   g++ -O3 -march=native -std=c++17 -fopenmp -o minirocket_cpu_mt minirocket_cpu_inference_v2.cpp
//
// Usage:
//   ./minirocket_cpu_mt <model.json> <test_data.json> [output.csv] [num_threads]
//
//   num_threads (optional): defaults to all logical cores available
//   (omp_get_max_threads()). Pass a smaller number to see how throughput
//   scales with thread count, e.g. to compare against the paper's
//   reported core count.

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
#include <omp.h>

// ============================================================
// Minimal JSON parser (identical to the original CPU file)
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
// MiniRocket Model (identical to the original CPU file)
// ============================================================

struct MiniRocketModel {
    int num_kernels;
    int num_dilations;
    int num_features;
    int num_classes;
    int time_series_length;

    std::vector<std::vector<int>> kernel_indices;
    std::vector<int> dilations;
    std::vector<int> num_features_per_dilation;
    std::vector<double> biases;

    std::vector<double> scaler_mean;
    std::vector<double> scaler_scale;

    std::vector<std::vector<double>> classifier_coef;
    std::vector<double> classifier_intercept;
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
        auto& coef_val = j["classifier_coef"];
        if (coef_val.arr.size() > 0 && coef_val.arr[0].type == JsonValue::ARRAY) {
            classifier_coef = coef_val.as_2d_double_array();
        } else {
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
// MiniRocket Feature Extraction (identical algorithm/logic to
// the original CPU file — this is a pure function of its inputs and
// writes only to the `features` vector passed in by the caller, so
// it is already safe to call concurrently from multiple threads as
// long as each thread passes its OWN `features` vector, never a
// shared one. That's exactly how the OpenMP throughput benchmark
// below uses it.)
// ============================================================

static const double WEIGHT_NEG = -1.0;
static const double WEIGHT_POS = 2.0;

void extract_features(const MiniRocketModel& model,
                      const std::vector<double>& time_series,
                      std::vector<double>& features) {
    const int L = model.time_series_length;
    features.resize(model.num_features);

    int feature_idx = 0;

    for (int d = 0; d < model.num_dilations; d++) {
        int dilation = model.dilations[d];
        int n_feat_this_dil = model.num_features_per_dilation[d];
        int padding0 = d % 2;
        int half_pad = 4 * dilation;

        for (int k = 0; k < 84; k++) {
            double weights[9];
            for (int i = 0; i < 9; i++) weights[i] = WEIGHT_NEG;
            weights[model.kernel_indices[k][0]] = WEIGHT_POS;
            weights[model.kernel_indices[k][1]] = WEIGHT_POS;
            weights[model.kernel_indices[k][2]] = WEIGHT_POS;

            int padding1 = (padding0 + k) % 2;

            int t_start, t_end, conv_length;
            if (padding1 == 0) {
                t_start = 0;
                t_end = L;
                conv_length = L;
            } else {
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
                        int idx = t + (w - 4) * dilation;
                        if (idx >= 0 && idx < L) {
                            conv_val += weights[w] * time_series[idx];
                        }
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
        double score = model.classifier_intercept[0];
        for (int f = 0; f < model.num_features; f++) {
            score += model.classifier_coef[0][f] * features[f];
        }
        return model.classes[score > 0 ? 1 : 0];
    }

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
        std::cerr << "Usage: " << argv[0] << " <model.json> <test_data.json> [output.csv] [num_threads]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string test_path = argv[2];
    std::string csv_path = (argc > 3) ? argv[3] : "";
    int num_threads = (argc > 4) ? std::stoi(argv[4]) : omp_get_max_threads();

    omp_set_num_threads(num_threads);
    std::cout << "OpenMP threads: " << num_threads
              << " (max available: " << omp_get_max_threads() << ")" << std::endl;

    MiniRocketModel model;
    model.load(model_path);

    std::cout << "Loading test data from: " << test_path << std::endl;
    auto test_json = load_json(test_path);

    std::string dataset_name = (test_json["dataset_name"].type != JsonValue::NONE)
        ? test_json["dataset_name"].as_string() : "unknown";

    int num_samples = 0;
    if (test_json["num_samples"].type != JsonValue::NONE)
        num_samples = (int)test_json["num_samples"].as_number();
    else
        num_samples = (int)test_json["X_test"].arr.size();

    int series_length = (test_json["series_length"].type != JsonValue::NONE)
        ? (int)test_json["series_length"].as_number()
        : (test_json["time_series_length"].type != JsonValue::NONE)
            ? (int)test_json["time_series_length"].as_number()
            : (int)test_json["X_test"].arr[0].arr.size();

    auto X_test_2d = test_json["X_test"].as_2d_double_array();
    auto y_test = test_json["y_test"].as_int_array();
    bool has_labels = ((int)y_test.size() == num_samples);
    if (!has_labels) {
        y_test.assign(num_samples, -1);
    }

    std::cout << "Dataset: " << dataset_name << std::endl;
    std::cout << "  Samples: " << num_samples << std::endl;
    std::cout << "  Series length: " << series_length << std::endl;

    assert(series_length == model.time_series_length);
    assert((int)X_test_2d.size() == num_samples);

    // ------------------------------------------------------------
    // PART 1: Single-threaded per-sample LATENCY benchmark
    // (identical to the original file — kept sequential deliberately,
    // see the file-level comment at the top for why)
    // ------------------------------------------------------------
    std::cout << "\n[1/2] Running SINGLE-THREADED per-sample latency benchmark..." << std::endl;
    std::vector<double> latencies_ms(num_samples);
    std::vector<int> predictions(num_samples);
    std::vector<double> features;
    int correct = 0;

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
        if (has_labels && pred == y_test[i]) correct++;

        if ((i + 1) % 5000 == 0 || i == num_samples - 1) {
            std::cout << "  Sample " << (i + 1) << "/" << num_samples << "..." << std::endl;
        }
    }

    double accuracy = has_labels ? ((double)correct / num_samples) : 0.0;

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

    std::cout << "\n========== SINGLE-THREADED LATENCY RESULTS ==========" << std::endl;
    std::cout << "Dataset:     " << dataset_name << std::endl;
    if (has_labels) {
        std::cout << "Accuracy:    " << std::fixed << std::setprecision(4) << (accuracy * 100)
                  << "% (" << correct << "/" << num_samples << ")" << std::endl;
    } else {
        std::cout << "Accuracy:    N/A (no ground truth y_test labels in JSON)" << std::endl;
    }
    std::cout << "Throughput:  " << std::fixed << std::setprecision(1) << (1000.0 / mean)
              << " inferences/sec (single-threaded)" << std::endl;
    std::cout << "\nLatency distribution (ms):" << std::endl;
    std::cout << "  Mean:  " << std::fixed << std::setprecision(3) << mean << std::endl;
    std::cout << "  P50:   " << percentile(50) << std::endl;
    std::cout << "  P95:   " << percentile(95) << std::endl;
    std::cout << "  P99:   " << percentile(99) << std::endl;
    std::cout << "  Min:   " << sorted_lat.front() << std::endl;
    std::cout << "  Max:   " << sorted_lat.back() << std::endl;
    std::cout << "  Std:   " << std_dev << std::endl;
    std::cout << "======================================================" << std::endl;

    // ------------------------------------------------------------
    // PART 2: Multithreaded batched THROUGHPUT benchmark
    // This is the number comparable to the paper's multi-threaded
    // Python (aeon) execution times — it parallelizes across samples,
    // the same way aeon's `multiprocessing`-based parallelism does,
    // just using OpenMP threads instead of Python processes.
    // ------------------------------------------------------------
    std::cout << "\n[2/2] Running MULTITHREADED batched throughput benchmark ("
              << num_threads << " threads)..." << std::endl;

    std::vector<int> mt_predictions(num_samples);
    int mt_correct = 0;

    auto mt_t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        // Each thread gets its OWN features buffer — this is what makes
        // calling extract_features/apply_scaler/classify safe here even
        // though they're being called concurrently from many threads.
        std::vector<double> thread_features;
        #pragma omp for schedule(static) reduction(+:mt_correct)
        for (int i = 0; i < num_samples; i++) {
            extract_features(model, X_test_2d[i], thread_features);
            apply_scaler(model, thread_features);
            int pred = classify(model, thread_features);
            mt_predictions[i] = pred;
            if (has_labels && pred == y_test[i]) mt_correct++;
        }
    }

    auto mt_t1 = std::chrono::high_resolution_clock::now();
    double mt_total_ms = std::chrono::duration<double, std::milli>(mt_t1 - mt_t0).count();
    double mt_throughput = num_samples / (mt_total_ms / 1000.0);
    double mt_accuracy = has_labels ? ((double)mt_correct / num_samples) : 0.0;

    std::cout << "\n========== MULTITHREADED THROUGHPUT RESULTS ==========" << std::endl;
    std::cout << "Threads:     " << num_threads << std::endl;
    std::cout << "Total time:  " << std::fixed << std::setprecision(1) << mt_total_ms << " ms" << std::endl;
    std::cout << "Throughput:  " << std::fixed << std::setprecision(1) << mt_throughput
              << " inferences/sec (multithreaded)" << std::endl;
    if (has_labels) {
        std::cout << "Accuracy:    " << std::fixed << std::setprecision(4) << (mt_accuracy * 100)
                  << "% (" << mt_correct << "/" << num_samples
                  << ") — should match single-threaded accuracy above" << std::endl;
    }
    std::cout << "Speedup vs. single-threaded: "
              << std::fixed << std::setprecision(2) << (mt_throughput / (1000.0 / mean)) << "x" << std::endl;
    std::cout << "=======================================================" << std::endl;

    // ------------------------------------------------------------
    // CSV output — from the single-threaded latency run, same format
    // as the original CPU file and the CUDA file, for consistency.
    // ------------------------------------------------------------
    if (csv_path.empty()) {
        csv_path = "../results/MiniRocket_CPU_cpp_mt_" + dataset_name + "_per_sample.csv";
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
