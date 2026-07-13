/*
 * csim_v3.cpp — C++ csim harness for MultiRocket Optimized v3
 *
 * Compiles and links against multirocket.cpp + multirocket_pooling.cpp
 * Uses nlohmann/json (header-only) or a minimal JSON parser.
 *
 * Build:
 *   g++ -O2 -std=c++17 \
 *       -I multirocket/include \
 *       -I /home/Xilinx/Vitis_HLS/2023.2/include \
 *       csim_v3.cpp multirocket/src/multirocket.cpp multirocket/src/multirocket_pooling.cpp \
 *       -o csim_v3
 *
 * Usage:
 *   ./csim_v3
 */

// C++ csim harness: compile with standard g++, no Vitis HLS required.
// multirocket.hpp detects absence of __SYNTHESIS__ and uses float/int32
// instead of ap_fixed<32,16>/ap_int<32>, avoiding bit-pattern mismatches.

// Include the actual header — it will use float for data_t in g++ mode
#include "multirocket/include/multirocket.hpp"

// Forward declare the top-level function (uses float* since data_t=float in csim)
extern "C" void multirocket_inference(
    data_t*, data_t*, data_t*, data_t*, data_t*, data_t*,
    int_t*, int_t*, data_t*, int_t, int_t,
    int_t*, int_t*, data_t*, int_t, int_t,
    int_t, int_t, int_t, int_t
);
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

// ---- Minimal JSON parser (no external deps) ----
// We use a simple approach: call Python to pre-parse and write flat binary files.
// Actually: use Python to convert JSON to a flat CSV that C++ can read easily.
// To avoid this complexity, we use a basic recursive descent JSON parser below.

#include <cassert>
#include <map>
#include <variant>
#include <stdexcept>

// Forward declaration
struct JsonVal;
using JsonArr = std::vector<JsonVal>;
using JsonObj = std::map<std::string, JsonVal>;

struct JsonVal {
    enum Type { Null, Bool, Int, Float, Str, Array, Object } type;
    bool b;
    long long i;
    double f;
    std::string s;
    JsonArr arr;
    JsonObj obj;
    JsonVal() : type(Null), b(false), i(0), f(0) {}
};

static void skip_ws(const char*& p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
}

static JsonVal parse_val(const char*& p);

static std::string parse_string(const char*& p) {
    assert(*p == '"'); p++;
    std::string s;
    while (*p != '"') {
        if (*p == '\\') { p++; }
        s += *p++;
    }
    p++; // closing "
    return s;
}

static JsonVal parse_array(const char*& p) {
    assert(*p == '['); p++;
    JsonVal v; v.type = JsonVal::Array;
    skip_ws(p);
    if (*p == ']') { p++; return v; }
    while (true) {
        skip_ws(p);
        v.arr.push_back(parse_val(p));
        skip_ws(p);
        if (*p == ']') { p++; break; }
        if (*p == ',') p++;
    }
    return v;
}

static JsonVal parse_object(const char*& p) {
    assert(*p == '{'); p++;
    JsonVal v; v.type = JsonVal::Object;
    skip_ws(p);
    if (*p == '}') { p++; return v; }
    while (true) {
        skip_ws(p);
        std::string key = parse_string(p);
        skip_ws(p); assert(*p == ':'); p++;
        skip_ws(p);
        v.obj[key] = parse_val(p);
        skip_ws(p);
        if (*p == '}') { p++; break; }
        if (*p == ',') p++;
    }
    return v;
}

static JsonVal parse_val(const char*& p) {
    skip_ws(p);
    if (*p == '"') { JsonVal v; v.type=JsonVal::Str; v.s=parse_string(p); return v; }
    if (*p == '[') return parse_array(p);
    if (*p == '{') return parse_object(p);
    if (*p == 't') { p+=4; JsonVal v; v.type=JsonVal::Bool; v.b=true; return v; }
    if (*p == 'f') { p+=5; JsonVal v; v.type=JsonVal::Bool; v.b=false; return v; }
    if (*p == 'n') { p+=4; return JsonVal(); }
    // number
    char* end;
    double d = strtod(p, &end);
    JsonVal v;
    // Check if it's integer-like
    bool is_int = true;
    for (const char* q = p; q < end; q++) {
        if (*q == '.' || *q == 'e' || *q == 'E') { is_int = false; break; }
    }
    if (is_int) { v.type=JsonVal::Int; v.i=(long long)d; }
    else { v.type=JsonVal::Float; v.f=d; }
    p = end;
    return v;
}

static JsonVal parse_json_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    std::string s((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    const char* p = s.c_str();
    return parse_val(p);
}

static double get_num(const JsonVal& v) {
    if (v.type == JsonVal::Int) return (double)v.i;
    if (v.type == JsonVal::Float) return v.f;
    return 0.0;
}

static std::vector<double> get_arr_double(const JsonVal& v) {
    std::vector<double> out;
    out.reserve(v.arr.size());
    for (auto& x : v.arr) out.push_back(get_num(x));
    return out;
}

static std::vector<int> get_arr_int(const JsonVal& v) {
    std::vector<int> out;
    out.reserve(v.arr.size());
    for (auto& x : v.arr) out.push_back((int)get_num(x));
    return out;
}

// ---- Static arrays for multirocket_inference ----
// Use global allocation to avoid stack overflow (50K floats × multiple arrays)

static data_t g_ts_input[MAX_TIME_SERIES_LENGTH];
static data_t g_pred_output[MAX_CLASSES];
static data_t g_coefficients[MAX_CLASSES * MAX_FEATURES];
static data_t g_intercept[MAX_CLASSES];
static data_t g_scaler_mean[MAX_FEATURES];
static data_t g_scaler_scale[MAX_FEATURES];
static int_t  g_dilations_0[MAX_DILATIONS];
static int_t  g_nfpd_0[MAX_DILATIONS];
static data_t g_biases_0[MAX_FEATURES];
static int_t  g_dilations_1[MAX_DILATIONS];
static int_t  g_nfpd_1[MAX_DILATIONS];
static data_t g_biases_1[MAX_FEATURES];

struct Dataset {
    std::string name;
    std::string model_path;
    std::string test_path;
    double expected_acc;
};

static double run_dataset(const Dataset& ds) {
    printf("\n=== %s ===\n", ds.name.c_str());
    fflush(stdout);

    // Load model
    JsonVal model = parse_json_file(ds.model_path);
    auto& mobj = model.obj;

    int num_features  = (int)get_num(mobj["num_features"]);
    int num_classes   = (int)get_num(mobj["num_classes"]);
    int ts_length     = (int)get_num(mobj["time_series_length"]);
    int num_dil_0     = (int)get_num(mobj["num_dilations_orig"]);
    int num_dil_1     = (int)get_num(mobj["num_dilations_diff"]);
    int n_feature_per_kernel = 4;  // MultiRocket fixed

    auto dilations_0_v = get_arr_int(mobj["dilations_orig"]);
    auto dilations_1_v = get_arr_int(mobj["dilations_diff"]);
    auto biases_0_v    = get_arr_double(mobj["biases_orig"]);
    auto biases_1_v    = get_arr_double(mobj["biases_diff"]);
    auto scaler_mean_v = get_arr_double(mobj["scaler_mean"]);
    auto scaler_scale_v= get_arr_double(mobj["scaler_scale"]);
    auto intercept_v   = get_arr_double(mobj["intercept"]);

    int num_features_0 = (int)biases_0_v.size();
    int num_features_1 = (int)biases_1_v.size();

    // Compute num_features_per_dilation (uniform: biases / dilations / NUM_KERNELS)
    int slots_orig = num_features_0 / (num_dil_0 * NUM_KERNELS);
    int slots_diff = num_features_1 / (num_dil_1 * NUM_KERNELS);

    // Load coefficients (num_classes × num_features, stored row-major in JSON)
    auto& coef_arr = mobj["coefficients"].arr;
    for (int c = 0; c < num_classes && c < MAX_CLASSES; c++) {
        auto row = get_arr_double(coef_arr[c]);
        for (int f = 0; f < num_features && f < MAX_FEATURES; f++) {
            g_coefficients[c * num_features + f] = (data_t)row[f];
        }
    }

    for (int c = 0; c < num_classes && c < MAX_CLASSES; c++)
        g_intercept[c] = (data_t)intercept_v[c];

    for (int f = 0; f < num_features && f < MAX_FEATURES; f++) {
        g_scaler_mean[f]  = (data_t)scaler_mean_v[f];
        g_scaler_scale[f] = (data_t)scaler_scale_v[f];
    }

    for (int i = 0; i < num_dil_0 && i < MAX_DILATIONS; i++) {
        g_dilations_0[i] = (int_t)dilations_0_v[i];
        g_nfpd_0[i]      = (int_t)slots_orig;
    }
    for (int i = 0; i < num_dil_1 && i < MAX_DILATIONS; i++) {
        g_dilations_1[i] = (int_t)dilations_1_v[i];
        g_nfpd_1[i]      = (int_t)slots_diff;
    }
    for (int i = 0; i < num_features_0 && i < MAX_FEATURES; i++)
        g_biases_0[i] = (data_t)biases_0_v[i];
    for (int i = 0; i < num_features_1 && i < MAX_FEATURES; i++)
        g_biases_1[i] = (data_t)biases_1_v[i];

    // Get class list for string label mapping
    auto& classes_arr = mobj["classes"].arr;
    std::map<std::string, int> class_to_idx;
    for (int c = 0; c < (int)classes_arr.size(); c++)
        class_to_idx[classes_arr[c].s] = c;

    printf("  num_features=%d, num_classes=%d, ts_length=%d\n",
           num_features, num_classes, ts_length);
    printf("  num_features_0=%d (slots=%d), num_features_1=%d (slots=%d)\n",
           num_features_0, slots_orig, num_features_1, slots_diff);
    fflush(stdout);

    // Load test data
    JsonVal test_data = parse_json_file(ds.test_path);
    auto& tobj = test_data.obj;
    auto& ts_arr  = tobj["time_series"].arr;
    auto& lbl_arr = tobj["labels"].arr;
    int num_samples = (int)ts_arr.size();

    printf("  Samples: %d, expected ~%.0f%%\n", num_samples, ds.expected_acc);
    fflush(stdout);

    int correct = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (int si = 0; si < num_samples; si++) {
        if (si % 100 == 0) {
            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            double eta = (elapsed / (si + 1)) * (num_samples - si);
            printf("  [%d/%d] elapsed=%.1fs eta=%.0fs\r", si, num_samples, elapsed, eta);
            fflush(stdout);
        }

        // Load time series
        auto& row = ts_arr[si].arr;
        int actual_len = (int)row.size();
        if (actual_len > MAX_TIME_SERIES_LENGTH) actual_len = MAX_TIME_SERIES_LENGTH;
        for (int t = 0; t < actual_len; t++)
            g_ts_input[t] = (data_t)get_num(row[t]);

        // Run inference
        multirocket_inference(
            g_ts_input,
            g_pred_output,
            g_coefficients,
            g_intercept,
            g_scaler_mean,
            g_scaler_scale,
            g_dilations_0,
            g_nfpd_0,
            g_biases_0,
            (int_t)num_dil_0,
            (int_t)num_features_0,
            g_dilations_1,
            g_nfpd_1,
            g_biases_1,
            (int_t)num_dil_1,
            (int_t)num_features_1,
            (int_t)actual_len,
            (int_t)num_features,
            (int_t)num_classes,
            (int_t)n_feature_per_kernel
        );

        // Argmax prediction
        int pred_class = 0;
        data_t max_val = g_pred_output[0];
        for (int c = 1; c < num_classes; c++) {
            if (g_pred_output[c] > max_val) {
                max_val = g_pred_output[c];
                pred_class = c;
            }
        }

        // Get true label
        int true_class;
        auto& lv = lbl_arr[si];
        if (lv.type == JsonVal::Str) {
            true_class = class_to_idx[lv.s];
        } else {
            true_class = (int)get_num(lv);
        }

        if (pred_class == true_class) correct++;
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double acc = (double)correct / num_samples * 100.0;
    printf("  [%d/%d] done in %.1fs                              \n", num_samples, num_samples, elapsed);
    printf("  Accuracy: %d/%d = %.2f%% (expected ~%.0f%%)\n",
           correct, num_samples, acc, ds.expected_acc);
    return acc;
}

int main() {
    std::string base = "/home/rdave009/minirocket-hls/MiniRocketHLS/multirocket_optimized/models";

    Dataset datasets[] = {
        {"InsectSound",
         base + "/multirocket84_insectsound_model.json",
         base + "/multirocket84_insectsound_test_shuf.json",
         74.60},
        {"MosquitoSound",
         base + "/multirocket84_mosquitosound_model.json",
         base + "/multirocket84_mosquitosound_test_shuf.json",
         86.60},
        {"FruitFlies",
         base + "/multirocket84_fruitflies_model.json",
         base + "/multirocket84_fruitflies_test_shuf.json",
         97.80},
    };

    double accs[3];
    for (int i = 0; i < 3; i++) {
        accs[i] = run_dataset(datasets[i]);
    }

    printf("\n=== SUMMARY ===\n");
    bool all_pass = true;
    const char* names[] = {"InsectSound", "MosquitoSound", "FruitFlies"};
    double expected[] = {74.60, 86.60, 97.80};
    for (int i = 0; i < 3; i++) {
        double diff = fabs(accs[i] - expected[i]);
        bool pass = diff < 2.0;  // within 2% of Round 2 HW result
        if (!pass) all_pass = false;
        printf("  %s: %.2f%% (Round2 HW: %.2f%%) diff=%.2f%% [%s]\n",
               names[i], accs[i], expected[i], diff, pass ? "PASS" : "FAIL");
    }
    printf("\n");
    if (all_pass) {
        printf("VERDICT: v3 MATCHES ROUND 2 HW ACCURACY -- READY FOR XCLBIN BUILD\n");
    } else {
        printf("VERDICT: ACCURACY MISMATCH -- DO NOT BUILD XCLBIN\n");
    }
    return all_pass ? 0 : 1;
}
