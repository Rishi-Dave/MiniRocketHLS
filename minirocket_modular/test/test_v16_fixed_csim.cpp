/*
 * C-Simulation test for the restructured v16_fixed feature extraction.
 * Validates that the loop-nest restructuring produces identical results to the original.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <chrono>
#include <string>
#include <sstream>

typedef float data_t;
typedef int32_t int_t;
typedef uint8_t idx_t;

#define MAX_TIME_SERIES_LENGTH 8192
#define MAX_FEATURES 1024
#define NUM_KERNELS 84
#define KERNEL_SIZE 9
#define MAX_DILATIONS 16
#define MAX_CLASSES 16

static data_t weights[NUM_KERNELS][KERNEL_SIZE] = {
    #include "../include/weights.txt"
};

#define UNROLL_FACTOR 16
#define NUM_KERNEL_GROUPS (NUM_KERNELS / UNROLL_FACTOR)
#define REMAINDER_KERNELS (NUM_KERNELS % UNROLL_FACTOR)

// Restructured version (matches feature_extraction_v16_fixed.cpp logic)
void minirocket_fe_v16_fixed(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    int_t dilations[MAX_DILATIONS],
    int_t num_features_per_dilation[MAX_DILATIONS],
    data_t biases[MAX_FEATURES],
    int_t time_series_length,
    int_t num_dilations,
    int_t num_features
) {
    int_t feature_idx = 0;

    for (int_t dil_idx = 0; dil_idx < num_dilations; dil_idx++) {
        int_t dilation = dilations[dil_idx];
        int_t features_this_dilation = num_features_per_dilation[dil_idx];
        int_t _padding0 = dil_idx % 2;
        int_t padding = ((9 - 1) * dilation) / 2;

        for (int_t kg = 0; kg < NUM_KERNELS; kg += UNROLL_FACTOR) {
            int_t group_size = (kg + UNROLL_FACTOR <= NUM_KERNELS) ? UNROLL_FACTOR : (NUM_KERNELS - kg);

            data_t convolutions[UNROLL_FACTOR][MAX_TIME_SERIES_LENGTH];
            memset(convolutions, 0, sizeof(convolutions));

            // Phase 1: Convolution with shared sliding window
            for (int_t j = 0; j < time_series_length; j++) {
                data_t sw[KERNEL_SIZE];
                for (int k = 0; k < KERNEL_SIZE; k++) {
                    int_t idx = j + (k - 4) * dilation;
                    if (idx < 0 || idx >= time_series_length) {
                        sw[k] = 0.0;
                    } else {
                        sw[k] = time_series[idx];
                    }
                }

                for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
                    int_t kernel_idx = kg + ki;
                    data_t value = 0.0;
                    for (int_t k = 0; k < KERNEL_SIZE; k++) {
                        value += sw[k] * weights[kernel_idx][k];
                    }
                    convolutions[ki][j] = value;
                }
            }

            // Phase 2: PPV
            for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
                int_t kernel_idx = kg + ki;
                if (kernel_idx >= NUM_KERNELS) break;

                int_t _padding1 = (_padding0 + kernel_idx) % 2;
                int_t feature_idx_for_kernel = feature_idx + kernel_idx * features_this_dilation;

                if (feature_idx_for_kernel >= num_features) continue;

                for (int_t f = 0; f < features_this_dilation; f++) {
                    data_t bias = biases[feature_idx_for_kernel + f];
                    int_t positive_count = 0;

                    if (_padding1 == 0) {
                        for (int_t i = 0; i < time_series_length; i++) {
                            if (convolutions[ki][i] > bias) positive_count++;
                        }
                        features[feature_idx_for_kernel + f] =
                            (data_t)positive_count / (data_t)time_series_length;
                    } else {
                        for (int_t i = padding; i < time_series_length - padding; i++) {
                            if (convolutions[ki][i] > bias) positive_count++;
                        }
                        features[feature_idx_for_kernel + f] =
                            (data_t)positive_count / (data_t)(time_series_length - 2 * padding);
                    }
                }
            }
        }
        feature_idx += features_this_dilation * NUM_KERNELS;
    }
}

// Original version (reference)
void apply_kernel_original(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    int_t kernel_idx, int_t dilation,
    int_t time_series_length, int_t* output_length
) {
    *output_length = time_series_length;
    for (int_t j = 0; j < time_series_length; j++) {
        data_t sw[KERNEL_SIZE] = {0};
        int i = 0;
        for (int k = -4; k <= 4; k++) {
            int_t idx = j + k * dilation;
            sw[i++] = (idx < 0 || idx >= time_series_length) ? 0.0f : time_series[idx];
        }
        data_t v = 0.0;
        for (int_t k = 0; k < KERNEL_SIZE; k++) v += sw[k] * weights[kernel_idx][k];
        convolutions[j] = v;
    }
}

void minirocket_fe_original(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    int_t dilations[MAX_DILATIONS],
    int_t num_features_per_dilation[MAX_DILATIONS],
    data_t biases[MAX_FEATURES],
    int_t time_series_length, int_t num_dilations, int_t num_features
) {
    int_t feature_idx = 0;
    for (int_t dil_idx = 0; dil_idx < num_dilations; dil_idx++) {
        int_t dilation = dilations[dil_idx];
        int_t features_this_dilation = num_features_per_dilation[dil_idx];
        int_t _padding0 = dil_idx % 2;
        int_t padding = ((9-1) * dilation) / 2;
        for (int_t kernel_idx = 0; kernel_idx < NUM_KERNELS; kernel_idx++) {
            data_t convolutions[MAX_TIME_SERIES_LENGTH];
            int_t _padding1 = (_padding0 + kernel_idx) % 2;
            int_t fidx = feature_idx + kernel_idx * features_this_dilation;
            if (fidx >= num_features) continue;
            int_t cl;
            apply_kernel_original(time_series, convolutions, kernel_idx, dilation, time_series_length, &cl);
            for (int_t f = 0; f < features_this_dilation; f++) {
                data_t bias = biases[fidx + f];
                int_t pc = 0;
                if (_padding1 == 0) {
                    for (int_t i = 0; i < time_series_length; i++) if (convolutions[i] > bias) pc++;
                    features[fidx + f] = (data_t)pc / (data_t)time_series_length;
                } else {
                    for (int_t i = padding; i < time_series_length - padding; i++) if (convolutions[i] > bias) pc++;
                    features[fidx + f] = (data_t)pc / (data_t)(time_series_length - 2 * padding);
                }
            }
        }
        feature_idx += features_this_dilation * NUM_KERNELS;
    }
}

// JSON helpers (minimal)
std::string read_file(const std::string& f) {
    std::ifstream file(f); std::string c, l;
    while (std::getline(file, l)) c += l; return c;
}
void trim(std::string& s) { s.erase(0, s.find_first_not_of(" \t\n\r")); s.erase(s.find_last_not_of(" \t\n\r") + 1); }
int piv(const std::string& c, const std::string& k) {
    size_t p = c.find("\"" + k + "\""); if (p == std::string::npos) return -1;
    size_t cp = c.find(":", p); size_t ve = c.find_first_of(",}", cp + 1);
    std::string v = c.substr(cp + 1, ve - cp - 1); trim(v); return std::stoi(v);
}
std::vector<float> pfa(const std::string& c, const std::string& k) {
    std::vector<float> r; size_t p = c.find("\"" + k + "\""); if (p == std::string::npos) return r;
    size_t s = c.find("[", p), e = c.find("]", s); std::string a = c.substr(s+1, e-s-1);
    std::stringstream ss(a); std::string i; while (std::getline(ss, i, ',')) { trim(i); if (!i.empty()) r.push_back(std::stof(i)); }
    return r;
}
std::vector<int> pia(const std::string& c, const std::string& k) {
    std::vector<int> r; size_t p = c.find("\"" + k + "\""); if (p == std::string::npos) return r;
    size_t s = c.find("[", p), e = c.find("]", s); std::string a = c.substr(s+1, e-s-1);
    std::stringstream ss(a); std::string i; while (std::getline(ss, i, ',')) { trim(i); if (!i.empty()) r.push_back(std::stoi(i)); }
    return r;
}
std::vector<std::vector<float>> p2f(const std::string& c, const std::string& k) {
    std::vector<std::vector<float>> r; size_t p = c.find("\"" + k + "\""); if (p == std::string::npos) return r;
    size_t s = c.find("[", p), e = s; int bc = 0;
    for (size_t i = s; i < c.length(); i++) { if (c[i]=='[') bc++; if (c[i]==']') bc--; if (!bc) { e=i; break; } }
    std::string a = c.substr(s+1, e-s-1); size_t pos = 0;
    while (pos < a.length()) { size_t ss2 = a.find("[", pos); if (ss2 == std::string::npos) break;
        size_t se = a.find("]", ss2); std::string sa = a.substr(ss2+1, se-ss2-1);
        std::vector<float> sr; std::stringstream sss(sa); std::string it;
        while (std::getline(sss, it, ',')) { trim(it); if (!it.empty()) sr.push_back(std::stof(it)); }
        r.push_back(sr); pos = se + 1; } return r;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "Usage: " << argv[0] << " <model.json> <test_data.json>" << std::endl; return 1; }

    std::string mc = read_file(argv[1]);
    int nd = piv(mc, "num_dilations"), nf = piv(mc, "num_features"), tsl = piv(mc, "time_series_length");
    auto dv = pia(mc, "dilations"), nfpd = pia(mc, "num_features_per_dilation");
    auto bv = pfa(mc, "biases");

    int_t dilations[MAX_DILATIONS]={0}, num_fpd[MAX_DILATIONS]={0};
    data_t biases[MAX_FEATURES]={0};
    for (int i = 0; i < nd; i++) { dilations[i] = dv[i]; num_fpd[i] = nfpd[i]; }
    for (int i = 0; i < nf; i++) biases[i] = bv[i];

    std::string tc = read_file(argv[2]);
    auto test_inputs = p2f(tc, "X_test");

    int num_test = std::min((int)test_inputs.size(), 10); // test first 10 samples
    int mismatches = 0;

    std::cout << "Testing v16_fixed vs original on " << num_test << " samples..." << std::endl;

    for (int s = 0; s < num_test; s++) {
        data_t ts[MAX_TIME_SERIES_LENGTH] = {0};
        for (int j = 0; j < tsl && j < (int)test_inputs[s].size(); j++) ts[j] = test_inputs[s][j];

        data_t feat_orig[MAX_FEATURES] = {0};
        data_t feat_fixed[MAX_FEATURES] = {0};

        minirocket_fe_original(ts, feat_orig, dilations, num_fpd, biases, tsl, nd, nf);
        minirocket_fe_v16_fixed(ts, feat_fixed, dilations, num_fpd, biases, tsl, nd, nf);

        // Compare features
        float max_diff = 0;
        int diff_count = 0;
        for (int i = 0; i < nf; i++) {
            float diff = std::abs(feat_orig[i] - feat_fixed[i]);
            if (diff > 1e-6) {
                diff_count++;
                if (diff > max_diff) max_diff = diff;
            }
        }

        if (diff_count > 0) {
            std::cout << "Sample " << s << ": " << diff_count << " features differ, max_diff=" << max_diff << " FAIL" << std::endl;
            mismatches++;
        } else {
            std::cout << "Sample " << s << ": all " << nf << " features match exactly. PASS" << std::endl;
        }
    }

    std::cout << "\n========== V16 FIXED VALIDATION ==========" << std::endl;
    if (mismatches == 0) {
        std::cout << "PASS: Restructured v16_fixed produces identical results to original" << std::endl;
    } else {
        std::cout << "FAIL: " << mismatches << "/" << num_test << " samples had feature mismatches" << std::endl;
    }
    return mismatches > 0 ? 1 : 0;
}
