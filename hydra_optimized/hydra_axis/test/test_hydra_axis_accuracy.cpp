/*
 * HYDRA AXIS accuracy csim testbench (v6 v2_fixed port).
 *
 * Loads a REAL model + test-data JSON, applies the SAME dilation-sort +
 * per-feature permutation contract as saturation_harness/load_hydra_hbm.cpp
 * (parser code copied from there so the data path is identical), streams
 * each test series through hydra_axis_inference one beat at a time, and
 * gates on classification accuracy vs the published float-kernel numbers
 * (InsectSound 69.41%, MosquitoSound 70.05%, FruitFlies 87.61%).
 *
 * Config via environment (so make.tcl needs no per-dataset edits):
 *   HYDRA_MODEL  path to hydra_<ds>_model.json         (required)
 *   HYDRA_TEST   path to hydra_<ds>_test_1000.json     (required)
 *   HYDRA_MAXN   max samples to run (default 1000)
 *   HYDRA_GATE   min accuracy %, FAIL below (default 60.0)
 */

#include "../include/hydra_axis.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------- Minimal JSON extractor (verbatim from load_hydra_hbm.cpp) ------

static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

static size_t find_key(const std::string& s, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t p = 0;
    while (true) {
        p = s.find(needle, p);
        if (p == std::string::npos)
            throw std::runtime_error("missing key: " + key);
        size_t after = p + needle.size();
        while (after < s.size() && std::isspace((unsigned char)s[after])) after++;
        if (after < s.size() && s[after] == ':') return after + 1;
        p = after;
    }
}

static double get_scalar(const std::string& s, const std::string& key) {
    size_t p = find_key(s, key);
    while (p < s.size() && std::isspace((unsigned char)s[p])) p++;
    char* end = nullptr;
    double v = std::strtod(s.c_str() + p, &end);
    if (end == s.c_str() + p)
        throw std::runtime_error("bad scalar for key: " + key);
    return v;
}

static std::vector<double> get_array(const std::string& s, const std::string& key) {
    size_t p = find_key(s, key);
    while (p < s.size() && std::isspace((unsigned char)s[p])) p++;
    if (p >= s.size() || s[p] != '[')
        throw std::runtime_error("expected '[' for key: " + key);
    int depth = 0;
    std::vector<double> out;
    while (p < s.size()) {
        char c = s[p];
        if (c == '[') { depth++; p++; continue; }
        if (c == ']') { depth--; p++; if (depth == 0) return out; continue; }
        if (c == ',' || std::isspace((unsigned char)c)) { p++; continue; }
        char* end = nullptr;
        double v = std::strtod(s.c_str() + p, &end);
        if (end == s.c_str() + p)
            throw std::runtime_error("bad number in array: " + key);
        out.push_back(v);
        p = (size_t)(end - s.c_str());
    }
    throw std::runtime_error("unterminated array: " + key);
}

// Labels may be ints (MosquitoSound/FruitFlies) or strings (InsectSound).
static std::vector<int> get_labels(const std::string& s) {
    static const std::map<std::string, int> label_map = {
        {"aedes_female", 0}, {"aedes_male", 1}, {"fruit_flies", 2},
        {"house_flies", 3}, {"quinx_female", 4}, {"quinx_male", 5},
        {"stigma_female", 6}, {"stigma_male", 7}, {"tarsalis_female", 8},
        {"tarsalis_male", 9}
    };
    size_t p = find_key(s, "labels");
    while (p < s.size() && std::isspace((unsigned char)s[p])) p++;
    if (p >= s.size() || s[p] != '[')
        throw std::runtime_error("expected '[' for labels");
    p++;
    std::vector<int> out;
    while (p < s.size()) {
        char c = s[p];
        if (c == ']') return out;
        if (c == ',' || std::isspace((unsigned char)c)) { p++; continue; }
        if (c == '"') {
            size_t q = s.find('"', p + 1);
            if (q == std::string::npos) throw std::runtime_error("bad label string");
            std::string lab = s.substr(p + 1, q - p - 1);
            auto it = label_map.find(lab);
            if (it == label_map.end())
                throw std::runtime_error("unknown label: " + lab);
            out.push_back(it->second);
            p = q + 1;
        } else {
            char* end = nullptr;
            double v = std::strtod(s.c_str() + p, &end);
            if (end == s.c_str() + p) throw std::runtime_error("bad label number");
            out.push_back((int)v);
            p = (size_t)(end - s.c_str());
        }
    }
    throw std::runtime_error("unterminated labels array");
}

int main() {
    const char* model_path = std::getenv("HYDRA_MODEL");
    const char* test_path  = std::getenv("HYDRA_TEST");
    if (!model_path || !test_path) {
        std::fprintf(stderr, "set HYDRA_MODEL and HYDRA_TEST env vars\n");
        return 2;
    }
    int   max_n  = std::getenv("HYDRA_MAXN") ? std::atoi(std::getenv("HYDRA_MAXN")) : 1000;
    float gate   = std::getenv("HYDRA_GATE") ? (float)std::atof(std::getenv("HYDRA_GATE")) : 60.0f;
    // Test JSONs are CLASS-SORTED; stride>1 samples across all classes so
    // small-N accuracy is comparable to the published full-set numbers.
    int   stride = std::getenv("HYDRA_STRIDE") ? std::atoi(std::getenv("HYDRA_STRIDE")) : 1;
    if (stride < 1) stride = 1;

    // --- Model (same fields + sanity as load_hydra_hbm.cpp) ---
    std::string js = slurp(model_path);
    int num_kernels        = (int)get_scalar(js, "num_kernels");
    int num_groups         = (int)get_scalar(js, "num_groups");
    int num_features       = (int)get_scalar(js, "num_features");
    int num_classes        = (int)get_scalar(js, "num_classes");
    int time_series_length = (int)get_scalar(js, "time_series_length");

    auto coefs_d   = get_array(js, "coefficients");
    auto intercept = get_array(js, "intercept");
    auto sm        = get_array(js, "scaler_mean");
    auto ss        = get_array(js, "scaler_scale");
    auto kw        = get_array(js, "kernel_weights");
    auto bi        = get_array(js, "biases");
    auto dil_d     = get_array(js, "dilations");

    if ((int)coefs_d.size() != num_classes * num_features ||
        (int)kw.size() != num_kernels * 9 ||
        (int)dil_d.size() != num_kernels) {
        std::fprintf(stderr, "model array size mismatch\n");
        return 2;
    }

    std::vector<float>   coefs_f(coefs_d.begin(), coefs_d.end());
    std::vector<float>   inter_f(intercept.begin(), intercept.end());
    std::vector<float>   sm_f(sm.begin(), sm.end());
    std::vector<float>   ss_f(ss.begin(), ss.end());
    std::vector<float>   kw_f(kw.begin(), kw.end());
    std::vector<float>   bi_f(bi.begin(), bi.end());
    std::vector<int32_t> dil_i(dil_d.begin(), dil_d.end());

    // --- Dilation sort + feature permutation (same as load_hydra_hbm) ---
    {
        std::vector<int> order(num_kernels);
        for (int i = 0; i < num_kernels; i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return dil_i[a] < dil_i[b]; });

        std::vector<float>   kw_s(kw_f.size()), bi_s(bi_f.size());
        std::vector<int32_t> dil_s(dil_i.size());
        std::vector<float>   sm_s(sm_f.size()), ss_s(ss_f.size()), coefs_s(coefs_f.size());
        for (int i = 0; i < num_kernels; i++) {
            int src = order[i];
            for (int w = 0; w < 9; w++) kw_s[i * 9 + w] = kw_f[src * 9 + w];
            bi_s[i] = bi_f[src];
            dil_s[i] = dil_i[src];
            for (int p2 = 0; p2 < 2; p2++) {
                int fd = i * 2 + p2, fs = src * 2 + p2;
                sm_s[fd] = sm_f[fs];
                ss_s[fd] = ss_f[fs];
                for (int c = 0; c < num_classes; c++)
                    coefs_s[fd * num_classes + c] = coefs_f[fs * num_classes + c];
            }
        }
        kw_f.swap(kw_s); bi_f.swap(bi_s); dil_i.swap(dil_s);
        sm_f.swap(sm_s); ss_f.swap(ss_s); coefs_f.swap(coefs_s);

        for (int b = 0; b < num_kernels / 16; b++)
            for (int k = 1; k < 16; k++)
                if (dil_i[b * 16 + k] != dil_i[b * 16]) {
                    std::fprintf(stderr, "[FAIL] batch %d mixed dilations after sort\n", b);
                    return 3;
                }
        std::printf("[sort] %d batches of 16 verified dilation-uniform\n", num_kernels / 16);
    }

    // --- Test data ---
    std::printf("[info] loading test data %s (this can take a minute)\n", test_path);
    std::string ts_js = slurp(test_path);
    int num_samples = (int)get_scalar(ts_js, "num_samples");
    auto series_flat = get_array(ts_js, "time_series");  // row-major flat
    auto labels = get_labels(ts_js);
    if ((int)series_flat.size() != num_samples * time_series_length ||
        (int)labels.size() != num_samples) {
        std::fprintf(stderr, "test data size mismatch (%zu vs %d*%d, labels %zu)\n",
                     series_flat.size(), num_samples, time_series_length, labels.size());
        return 2;
    }
    ts_js.clear(); ts_js.shrink_to_fit();

    // Build the strided sample index list (class-mixed when stride>1).
    std::vector<int> picks;
    for (int n = 0; n < num_samples && (int)picks.size() < max_n; n += stride)
        picks.push_back(n);
    num_samples = (int)picks.size();
    std::printf("[info] evaluating %d samples (stride %d)\n", num_samples, stride);

    // --- Stream through the kernel ---
    hls::stream<pkt> in_strm("in"), out_strm("out");
    int correct = 0;

    for (int ni = 0; ni < num_samples; ni++) {
        int n = picks[ni];
        for (int t = 0; t < time_series_length; t++) {
            pkt p;
            float v = (float)series_flat[(size_t)n * time_series_length + t];
            ap_uint<DWIDTH> w = 0;
            ap_uint<32> bits = *((ap_uint<32>*)&v);
            w.range(31, 0) = bits;
            p.data = w; p.keep = -1; p.last = 1;
            in_strm.write(p);
            hydra_axis_inference(
                in_strm, out_strm,
                coefs_f.data(), inter_f.data(), sm_f.data(), ss_f.data(),
                kw_f.data(), bi_f.data(), (int_t*)dil_i.data(),
                time_series_length, num_features, num_classes, num_groups);
        }
        if (out_strm.empty()) {
            std::fprintf(stderr, "[FAIL] sample %d: no output after full window\n", n);
            return 4;
        }
        pkt out_pkt = out_strm.read();
        int best = 0; float best_v = -1e30f;
        for (int c = 0; c < num_classes; c++) {
            ap_uint<32> bits = out_pkt.data.range(32 * c + 31, 32 * c);
            float logit; *((ap_uint<32>*)&logit) = bits;
            if (std::isnan(logit) || std::isinf(logit)) {
                std::fprintf(stderr, "[FAIL] sample %d: NaN/Inf logit c=%d\n", n, c);
                return 5;
            }
            if (logit > best_v) { best_v = logit; best = c; }
        }
        if (best == labels[n]) correct++;
        if ((ni + 1) % 50 == 0)
            std::printf("[prog] %d/%d acc so far %.2f%%\n", ni + 1, num_samples,
                        100.0 * correct / (ni + 1));
    }

    float acc = 100.0f * correct / num_samples;
    std::printf("[RESULT] accuracy %.2f%% (%d/%d), gate %.2f%%\n",
                acc, correct, num_samples, gate);
    if (acc < gate) {
        std::fprintf(stderr, "[FAIL] accuracy below gate\n");
        return 1;
    }
    std::printf("[PASS] hydra_axis v6 accuracy csim\n");
    return 0;
}
