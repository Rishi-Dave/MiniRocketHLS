// Standalone optimized-C++ baseline for HYDRA inference.
//
// Compiles the SAME kernel source used by HLS (../hydra/src/hydra.cpp,
// ../hydra/src/hydra_pooling.cpp) as a native C++ program with -O3
// -march=native. HLS pragmas are no-ops outside Vitis HLS (no
// __SYNTHESIS__ macro), so the loops run as plain CPU code.
//
// Loads the same JSON model + test files used by the FPGA host so the
// accuracy and inference workload are identical to the HW run, then
// times steady-state inference throughput.
//
// Usage:
//   hydra_cpp_baseline <model.json> <test.json> [warmup=20] [out.json]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../hydra/include/hydra.hpp"

extern "C" void hydra_inference(
    data_t* time_series_input,
    data_t* prediction_output,
    data_t* coefficients,
    data_t* intercept,
    data_t* scaler_mean,
    data_t* scaler_scale,
    data_t* kernel_weights,
    data_t* biases,
    int_t*  dilations,
    int_t   time_series_length,
    int_t   num_features,
    int_t   num_classes,
    int_t   num_groups);

// ---------------------------------------------------------------------------
// Minimal JSON parser (mirrors HydraJSONLoader behavior, but self-contained).
// ---------------------------------------------------------------------------
namespace json {

static std::string read_file(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        return {};
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static void trim(std::string& s) {
    s.erase(0, s.find_first_not_of(" \t\n\r\f\v"));
    auto e = s.find_last_not_of(" \t\n\r\f\v");
    if (e != std::string::npos) s.erase(e + 1);
}

static int parse_int(const std::string& c, const std::string& key) {
    size_t p = c.find("\"" + key + "\"");
    if (p == std::string::npos) return -1;
    size_t colon = c.find(':', p);
    size_t end = c.find_first_of(",}", colon + 1);
    std::string v = c.substr(colon + 1, end - colon - 1);
    trim(v);
    return std::stoi(v);
}

static std::vector<float> parse_float_array(const std::string& c, const std::string& key) {
    std::vector<float> out;
    size_t p = c.find("\"" + key + "\"");
    if (p == std::string::npos) return out;
    size_t a = c.find('[', p);
    size_t b = c.find(']', a);
    if (a == std::string::npos || b == std::string::npos) return out;
    std::string body = c.substr(a + 1, b - a - 1);
    std::stringstream ss(body);
    std::string item;
    while (std::getline(ss, item, ',')) {
        trim(item);
        if (item.empty()) continue;
        if (item.find('"') != std::string::npos) { out.clear(); return out; }
        try { out.push_back(std::stof(item)); }
        catch (...) { out.clear(); return out; }
    }
    return out;
}

static std::vector<int> parse_int_array(const std::string& c, const std::string& key) {
    std::vector<int> out;
    size_t p = c.find("\"" + key + "\"");
    if (p == std::string::npos) return out;
    size_t a = c.find('[', p);
    size_t b = c.find(']', a);
    if (a == std::string::npos || b == std::string::npos) return out;
    std::string body = c.substr(a + 1, b - a - 1);
    std::stringstream ss(body);
    std::string item;
    while (std::getline(ss, item, ',')) {
        trim(item);
        if (item.empty()) continue;
        if (item.find('"') != std::string::npos) { out.clear(); return out; }
        try { out.push_back(std::stoi(item)); }
        catch (...) { out.clear(); return out; }
    }
    return out;
}

static std::vector<std::string> parse_string_array(const std::string& c, const std::string& key) {
    std::vector<std::string> out;
    size_t p = c.find("\"" + key + "\"");
    if (p == std::string::npos) return out;
    size_t a = c.find('[', p);
    size_t b = c.find(']', a);
    if (a == std::string::npos || b == std::string::npos) return out;
    std::string body = c.substr(a + 1, b - a - 1);
    size_t i = 0;
    while (i < body.size()) {
        size_t qs = body.find('"', i);
        if (qs == std::string::npos) break;
        size_t qe = body.find('"', qs + 1);
        if (qe == std::string::npos) break;
        out.push_back(body.substr(qs + 1, qe - qs - 1));
        i = qe + 1;
    }
    return out;
}

// 2D float array: [num_samples][time_series_length]. Same balanced-bracket
// strategy as the FPGA host loader (parse_2d_float_array) so behavior is
// identical.
static std::vector<std::vector<float>> parse_2d_float_array(
    const std::string& c, const std::string& key)
{
    std::vector<std::vector<float>> out;
    size_t p = c.find("\"" + key + "\"");
    if (p == std::string::npos) return out;
    size_t a = c.find('[', p);
    int depth = 0;
    size_t b = a;
    for (size_t i = a; i < c.size(); ++i) {
        if (c[i] == '[') ++depth;
        else if (c[i] == ']') { --depth; if (depth == 0) { b = i; break; } }
    }
    std::string body = c.substr(a + 1, b - a - 1);
    size_t pos = 0;
    while (pos < body.size()) {
        size_t s = body.find('[', pos);
        if (s == std::string::npos) break;
        size_t e = body.find(']', s);
        if (e == std::string::npos) break;
        std::string sub = body.substr(s + 1, e - s - 1);
        std::vector<float> row;
        std::stringstream ss(sub);
        std::string item;
        while (std::getline(ss, item, ',')) {
            trim(item);
            if (!item.empty()) row.push_back(std::stof(item));
        }
        out.push_back(std::move(row));
        pos = e + 1;
    }
    return out;
}

}  // namespace json

// InsectSound stores labels as strings; map them to ints. Same mapping as
// hydra_loader_v2.cpp::map_insectsound_label.
static int map_insectsound_label(const std::string& s) {
    static const std::map<std::string, int> m = {
        {"aedes_female", 0}, {"aedes_male", 1}, {"fruit_flies", 2},
        {"house_flies", 3}, {"quinx_female", 4}, {"quinx_male", 5},
        {"stigma_female", 6}, {"stigma_male", 7},
        {"tarsalis_female", 8}, {"tarsalis_male", 9}};
    auto it = m.find(s);
    return it == m.end() ? -1 : it->second;
}

// ---------------------------------------------------------------------------
// Model + test data containers (heap to keep the stack small).
// ---------------------------------------------------------------------------
struct HydraModel {
    std::vector<float> kernel_weights;   // NUM_KERNELS * KERNEL_SIZE
    std::vector<float> biases;           // NUM_KERNELS
    std::vector<int>   dilations;        // NUM_KERNELS
    std::vector<float> scaler_mean;      // num_features
    std::vector<float> scaler_scale;     // num_features
    // Coefficients in the JSON are stored as coef_.T.flatten(), i.e.
    // index = f * num_classes + c. The kernel expects the same layout
    // (see linear_classifier_predict_hls), so we pass the flat array
    // through unchanged.
    std::vector<float> coefficients;     // num_features * num_classes
    std::vector<float> intercept;        // num_classes
    int num_kernels = 0;
    int num_groups = 0;
    int num_features = 0;
    int num_classes = 0;
    int time_series_length = 0;
};

static bool load_model(const std::string& path, HydraModel& m) {
    std::string c = json::read_file(path);
    if (c.empty()) return false;
    m.num_kernels         = json::parse_int(c, "num_kernels");
    m.num_groups          = json::parse_int(c, "num_groups");
    m.num_features        = json::parse_int(c, "num_features");
    m.num_classes         = json::parse_int(c, "num_classes");
    m.time_series_length  = json::parse_int(c, "time_series_length");

    if (m.num_kernels  > NUM_KERNELS  ||
        m.num_features > MAX_FEATURES ||
        m.num_classes  > MAX_CLASSES) {
        std::cerr << "Model exceeds compile-time limits.\n";
        return false;
    }

    m.kernel_weights = json::parse_float_array(c, "kernel_weights");
    m.biases         = json::parse_float_array(c, "biases");
    m.dilations      = json::parse_int_array(c, "dilations");
    m.scaler_mean    = json::parse_float_array(c, "scaler_mean");
    m.scaler_scale   = json::parse_float_array(c, "scaler_scale");
    m.coefficients   = json::parse_float_array(c, "coefficients");
    m.intercept      = json::parse_float_array(c, "intercept");

    auto check = [&](const char* n, size_t got, size_t exp) {
        if (got != exp) {
            std::cerr << n << " size mismatch: got " << got << " expected " << exp << "\n";
            return false;
        }
        return true;
    };
    if (!check("kernel_weights", m.kernel_weights.size(), (size_t)m.num_kernels * KERNEL_SIZE)) return false;
    if (!check("biases",         m.biases.size(),         (size_t)m.num_kernels)) return false;
    if (!check("dilations",      m.dilations.size(),      (size_t)m.num_kernels)) return false;
    if (!check("scaler_mean",    m.scaler_mean.size(),    (size_t)m.num_features)) return false;
    if (!check("scaler_scale",   m.scaler_scale.size(),   (size_t)m.num_features)) return false;
    if (!check("coefficients",   m.coefficients.size(),   (size_t)m.num_features * m.num_classes)) return false;
    if (!check("intercept",      m.intercept.size(),      (size_t)m.num_classes)) return false;
    return true;
}

struct HydraTest {
    std::vector<std::vector<float>> inputs;  // [num_samples][T]
    std::vector<int> labels;
    int num_samples = 0;
    int time_series_length = 0;
    int num_classes = 0;
};

static bool load_test(const std::string& path, HydraTest& t) {
    std::string c = json::read_file(path);
    if (c.empty()) return false;
    t.num_samples         = json::parse_int(c, "num_samples");
    t.time_series_length  = json::parse_int(c, "time_series_length");
    t.num_classes         = json::parse_int(c, "num_classes");
    t.inputs              = json::parse_2d_float_array(c, "time_series");
    if ((int)t.inputs.size() != t.num_samples) {
        std::cerr << "time_series sample count mismatch.\n";
        return false;
    }

    t.labels = json::parse_int_array(c, "labels");
    if (t.labels.empty()) {  // InsectSound stores string labels
        auto strs = json::parse_string_array(c, "labels");
        for (const auto& s : strs) {
            int v = map_insectsound_label(s);
            if (v < 0) {
                std::cerr << "Unknown label: " << s << "\n";
                return false;
            }
            t.labels.push_back(v);
        }
    }
    if ((int)t.labels.size() != t.num_samples) {
        std::cerr << "label count mismatch.\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Benchmark driver
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.json> <test.json> [warmup=20] [results.json]\n";
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string test_path  = argv[2];
    const int warmup = (argc > 3) ? std::atoi(argv[3]) : 20;
    const std::string out_path = (argc > 4) ? argv[4] : "";

    HydraModel model;
    HydraTest  test;
    if (!load_model(model_path, model)) return 1;
    if (!load_test(test_path,  test))   return 1;

    if (test.time_series_length != model.time_series_length) {
        std::cerr << "Warning: test T=" << test.time_series_length
                  << " differs from model T=" << model.time_series_length
                  << " (using test value).\n";
    }
    const int T = test.time_series_length;
    if (T > MAX_TIME_SERIES_LENGTH) {
        std::cerr << "Time series length " << T
                  << " exceeds MAX_TIME_SERIES_LENGTH=" << MAX_TIME_SERIES_LENGTH
                  << ". Recompile with a larger MAX.\n";
        return 1;
    }

    std::cout << "Model:  " << model_path << "\n"
              << "Test:   " << test_path  << "\n"
              << "Samples: " << test.num_samples
              << "  T=" << T
              << "  features=" << model.num_features
              << "  classes=" << model.num_classes
              << "  kernels=" << model.num_kernels
              << "  groups=" << model.num_groups << "\n";

    // Per-call output buffer (kernel writes num_classes floats).
    std::vector<float> pred(model.num_classes);

    // ---------- Warmup ----------
    int n_warm = std::min(warmup, test.num_samples);
    for (int i = 0; i < n_warm; ++i) {
        hydra_inference(
            test.inputs[i].data(),
            pred.data(),
            model.coefficients.data(),
            model.intercept.data(),
            model.scaler_mean.data(),
            model.scaler_scale.data(),
            model.kernel_weights.data(),
            model.biases.data(),
            model.dilations.data(),
            T,
            model.num_features,
            model.num_classes,
            model.num_groups);
    }

    // ---------- Timed full pass ----------
    int correct = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < test.num_samples; ++i) {
        hydra_inference(
            test.inputs[i].data(),
            pred.data(),
            model.coefficients.data(),
            model.intercept.data(),
            model.scaler_mean.data(),
            model.scaler_scale.data(),
            model.kernel_weights.data(),
            model.biases.data(),
            model.dilations.data(),
            T,
            model.num_features,
            model.num_classes,
            model.num_groups);

        int argmax = 0;
        for (int c = 1; c < model.num_classes; ++c) {
            if (pred[c] > pred[argmax]) argmax = c;
        }
        if (argmax == test.labels[i]) ++correct;
    }
    auto t1 = std::chrono::steady_clock::now();

    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    double throughput = test.num_samples / elapsed_s;
    double latency_ms = 1000.0 * elapsed_s / test.num_samples;
    double accuracy   = 100.0 * correct / test.num_samples;

    std::cout << "----------------------------------------\n"
              << "Samples:    " << test.num_samples << "\n"
              << "Elapsed:    " << elapsed_s << " s\n"
              << "Throughput: " << throughput << " inf/s\n"
              << "Latency:    " << latency_ms << " ms/sample\n"
              << "Accuracy:   " << correct << "/" << test.num_samples
              << " = " << accuracy << "%\n";

    if (!out_path.empty()) {
        std::ofstream f(out_path);
        f << "{\n"
          << "  \"model\": \"" << model_path << "\",\n"
          << "  \"test\":  \"" << test_path  << "\",\n"
          << "  \"samples\": " << test.num_samples << ",\n"
          << "  \"time_series_length\": " << T << ",\n"
          << "  \"num_features\": " << model.num_features << ",\n"
          << "  \"num_classes\":  " << model.num_classes << ",\n"
          << "  \"warmup\": " << n_warm << ",\n"
          << "  \"elapsed_s\": " << elapsed_s << ",\n"
          << "  \"throughput_inf_per_s\": " << throughput << ",\n"
          << "  \"latency_ms_per_sample\": " << latency_ms << ",\n"
          << "  \"accuracy_pct\": " << accuracy << "\n"
          << "}\n";
    }
    return 0;
}
