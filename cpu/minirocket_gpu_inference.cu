// MiniRocket GPU Inference (CUDA C++)
// ====================================
// GPU port of minirocket_cpu_inference.cpp. Implements the SAME feature
// extraction algorithm (padding scheme, per-feature bias, kernel_indices
// loaded from the model file) so results should match the CPU version
// almost exactly (tiny float/double rounding aside).
//
// WHAT MOVED TO THE GPU: only extract_features() — the expensive part
// (84 kernels x ~9 dilations x full time-series length, per sample).
// The scaler and Ridge classifier are cheap (840 elementwise ops / one
// dot product) so they stay on the CPU for simplicity, exactly like in
// the CPU file. See the comment above run_scaler_and_classify_on_host()
// for how to move those to the GPU too, if you want a fully-device
// pipeline later.
//
// Compile:
//   nvcc -O3 -arch=sm_80 -std=c++17 -o minirocket_gpu minirocket_gpu_inference.cu
//   (change -arch=sm_80 to match your GPU: sm_75 = T4, sm_80 = A100, sm_86 = RTX 30xx, sm_89 = RTX 40xx)
//
// Usage:
//   ./minirocket_gpu <model.json> <test_data.json> [output.csv] [batch_size]
//
//   batch_size (optional, default 1):
//     1   -> matches the CPU benchmark exactly: one sample per kernel
//            launch, so per-sample latency is directly comparable to
//            the CPU/FPGA numbers.
//     >1  -> also reports batched THROUGHPUT (many samples processed
//            in one kernel launch), which is where the GPU actually
//            wins — see the "batch=1 vs batched" discussion you
//            already benchmarked with the Python script.

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <cuda_runtime.h>

// ============================================================
// CUDA error-checking helper
// Every CUDA call is wrapped in this so mistakes fail loudly with a
// file/line number instead of silently corrupting results.
// ============================================================
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " -> " << cudaGetErrorString(err) << std::endl;      \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ============================================================
// Minimal JSON parser (identical to the CPU file — copied verbatim
// so this file has zero dependency on minirocket_cpu_inference.cpp
// and can be compiled completely standalone with nvcc).
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
// MiniRocket Model — identical fields to the CPU version.
// ============================================================

// Upper bounds for the __constant__ memory arrays below. MiniRocket's
// kernel count (84) is fixed by the algorithm; dilation count and total
// feature count vary per dataset, so we just need bounds big enough for
// any realistic model. GPU constant memory is 64KB total, so we have
// plenty of room (84*3 ints + 32 ints + 4096 doubles is well under that).
#define MAX_DILATIONS 32

struct MiniRocketModel {
    int num_kernels;       // 84
    int num_dilations;
    int num_features;
    int num_classes;
    int time_series_length;

    std::vector<std::vector<int>> kernel_indices; // [84][3]
    std::vector<int> dilations;                   // [num_dilations]
    std::vector<int> num_features_per_dilation;    // [num_dilations]
    std::vector<double> biases;                    // [num_features]

    std::vector<double> scaler_mean;
    std::vector<double> scaler_scale;

    std::vector<std::vector<double>> classifier_coef;
    std::vector<double> classifier_intercept;
    std::vector<int> classes;

    // Precomputed per-dilation feature offsets: where in the flat
    // `features[]` array does dilation d's block of features start?
    // (features are laid out dilation-by-dilation, then kernel-by-kernel,
    // then bias-by-bias within a kernel — same nested order as the CPU
    // loop `for d: for k: for f:` so feature indices line up exactly
    // with model.biases / scaler_mean / scaler_scale / classifier_coef.)
    std::vector<int> dilation_feature_offset;

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

        // Sanity checks: this GPU port assumes 84 kernels (fixed by the
        // MiniRocket algorithm) and bounds we sized the constant memory for.
        assert(num_kernels == 84 && "This CUDA port assumes the standard 84 MiniRocket kernels");
        assert(num_dilations <= MAX_DILATIONS && "Increase MAX_DILATIONS for this model");
        // Guards the counts[256] buffer in extract_features_kernel — see
        // the comment there. A silent overflow here is what caused the
        // earlier chance-level-accuracy bug on the 9996-feature model.
        for (int d = 0; d < num_dilations; d++) {
            assert(num_features_per_dilation[d] <= 256 &&
                   "A dilation needs more than 256 features/kernel — bump counts[256] "
                   "in extract_features_kernel AND this bound together, or predictions "
                   "will silently corrupt again.");
        }
        // NOTE: no cap on num_features — biases live in regular (global)
        // device memory, not __constant__ memory, specifically so this
        // scales to any model size (see the comment above d_biases_ptr).

        // Build the per-dilation feature offsets (see comment on the
        // field above). offset[0] = 0, offset[d] = offset[d-1] + 84 * num_features_per_dilation[d-1]
        dilation_feature_offset.resize(num_dilations);
        int running = 0;
        for (int d = 0; d < num_dilations; d++) {
            dilation_feature_offset[d] = running;
            running += 84 * num_features_per_dilation[d];
        }
        assert(running == num_features && "Feature count mismatch — check num_features_per_dilation");

        std::cout << "  num_kernels: " << num_kernels << std::endl;
        std::cout << "  num_dilations: " << num_dilations << std::endl;
        std::cout << "  num_features: " << num_features << std::endl;
        std::cout << "  num_classes: " << num_classes << std::endl;
        std::cout << "  time_series_length: " << time_series_length << std::endl;
    }
};

// ============================================================
// Device (GPU) constant memory
// ============================================================
// __constant__ memory is cached and broadcast-read by all threads at
// once when they access the same address — perfect fit here, since
// every sample's threads read the SAME kernel_indices/dilations/biases,
// just applied to different input data. This is much faster than
// reading the same values from regular global memory over and over.
__constant__ int   d_kernel_indices[84 * 3];       // [k][0..2] flattened
__constant__ int   d_dilations[MAX_DILATIONS];
__constant__ int   d_num_features_per_dilation[MAX_DILATIONS];
__constant__ int   d_dilation_feature_offset[MAX_DILATIONS];

// Biases are NOT in __constant__ memory, unlike the small arrays above.
// GPU constant memory is hard-capped at 64KB total across ALL __constant__
// variables combined — a 9996-feature model's biases alone would need
// ~78KB of doubles, blowing past that ceiling. So biases live in regular
// (global) device memory instead, sized dynamically to whatever the
// model actually needs. This is a plain pointer (not a fixed-size array)
// allocated once in upload_model_to_gpu() and reused across all kernel
// launches for the lifetime of the program.
static double* g_d_biases = nullptr;

// ============================================================
// GPU feature-extraction kernel
// ============================================================
// Thread mapping (this is the key design decision):
//   blockIdx.x  = which sample in the batch
//   blockIdx.y  = which dilation (there are only ~9-10 of these)
//   threadIdx.x = which of the 84 fixed kernels
//
// So each thread is responsible for ONE (sample, dilation, kernel)
// combination, and internally loops over that kernel's small number of
// bias thresholds (num_features_per_dilation[d], typically single
// digits) to produce that many output features.
//
// Performance note: the CPU code recomputes the convolution value at
// each time step once PER bias threshold (nested loop: for each bias,
// re-scan the whole series). Here we flip the loop order — scan the
// series ONCE per thread, and for each time step, check it against all
// of this kernel's bias thresholds at the same time. Same math, same
// result, just avoids repeating the expensive convolution sum.
__global__ void extract_features_kernel(
    const double* __restrict__ X,       // [batch_size, L] input time series, row-major
    double* __restrict__ features_out,  // [batch_size, num_features] output, row-major
    const double* __restrict__ biases,  // [num_features] — global memory, see g_d_biases comment above
    int L,
    int num_dilations,
    int batch_size)
{
    int sample = blockIdx.x;
    int d      = blockIdx.y;
    int k      = threadIdx.x;

    if (sample >= batch_size || d >= num_dilations || k >= 84) return;

    int dilation       = d_dilations[d];
    int n_feat_this_dil = d_num_features_per_dilation[d];
    int padding0       = d % 2;
    int padding1       = (padding0 + k) % 2;
    int half_pad       = 4 * dilation;

    // Build this kernel's 9 weights: -1 everywhere except +2 at the
    // 3 positions given in kernel_indices (loaded from the model file,
    // NOT regenerated — different models can order/pick these differently).
    double weights[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) weights[i] = -1.0;
    weights[d_kernel_indices[k * 3 + 0]] = 2.0;
    weights[d_kernel_indices[k * 3 + 1]] = 2.0;
    weights[d_kernel_indices[k * 3 + 2]] = 2.0;

    // Padded vs. unpadded convolution range — identical logic to the
    // CPU version's alternating padding scheme.
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

    // Where in features_out do this thread's results go?
    int feature_base = d_dilation_feature_offset[d] + k * n_feat_this_dil;
    const double* ts = X + (size_t)sample * L;
    double* out = features_out + (size_t)sample * (d_dilation_feature_offset[num_dilations - 1]
                                                      + 84 * d_num_features_per_dilation[num_dilations - 1]);

    if (conv_length <= 0) {
        for (int f = 0; f < n_feat_this_dil; f++) out[feature_base + f] = 0.0;
        return;
    }

    // Per-thread local accumulator for this kernel's bias thresholds.
    // MUST be large enough for the largest num_features_per_dilation[d]
    // this model actually has — a too-small bound here causes a SILENT
    // stack buffer overflow (no crash, just corrupted predictions), which
    // is exactly what happened with 16: it was fine for an 840-feature
    // model (~1-2 features/kernel/dilation) but silently broke on a
    // 9996-feature model (up to ~100+ features/kernel for the lowest
    // dilation), collapsing accuracy to chance level. 256 is generous
    // headroom; MiniRocketModel::load() below asserts this bound isn't
    // exceeded, so a future larger model fails loudly here instead of
    // silently corrupting results again.
    int counts[256] = {0};

    for (int t = t_start; t < t_end; t++) {
        double conv_val = 0.0;
        #pragma unroll
        for (int w = 0; w < 9; w++) {
            int idx = t + (w - 4) * dilation;
            if (idx >= 0 && idx < L) {
                conv_val += weights[w] * ts[idx];
            }
        }
        for (int f = 0; f < n_feat_this_dil; f++) {
            double bias = biases[feature_base + f];
            if (conv_val > bias) counts[f]++;
        }
    }

    for (int f = 0; f < n_feat_this_dil; f++) {
        out[feature_base + f] = (double)counts[f] / (double)conv_length;
    }
}

// ============================================================
// Scaler + Ridge classifier — kept on the CPU (host), same as the
// CPU baseline. These are cheap (840 elementwise ops, one dot product
// per class) so moving them to the GPU wouldn't meaningfully change
// throughput — the convolution above is where all the time goes.
//
// To make this a fully-device pipeline later (avoiding the
// device -> host copy of features), you'd add two more small kernels:
//   1. one thread per feature: features[i] = (features[i]-mean[i])/scale[i]
//   2. one thread per class: dot product of scaled features with that
//      class's coefficient row, then an argmax reduction.
// Not done here to keep this file focused and easy to follow.
// ============================================================

void apply_scaler(const MiniRocketModel& model, std::vector<double>& features) {
    for (int i = 0; i < model.num_features; i++) {
        features[i] = (features[i] - model.scaler_mean[i]) / model.scaler_scale[i];
    }
}

int classify(const MiniRocketModel& model, const std::vector<double>& features) {
    if (model.classifier_coef.size() == 1) {
        double score = model.classifier_intercept[0];
        for (int f = 0; f < model.num_features; f++)
            score += model.classifier_coef[0][f] * features[f];
        return model.classes[score > 0 ? 1 : 0];
    }
    int best_class = 0;
    double best_score = -1e30;
    for (int c = 0; c < model.num_classes; c++) {
        double score = model.classifier_intercept[c];
        for (int f = 0; f < model.num_features; f++)
            score += model.classifier_coef[c][f] * features[f];
        if (score > best_score) { best_score = score; best_class = c; }
    }
    return model.classes[best_class];
}

// ============================================================
// Upload the model's constant data to GPU constant memory. Done once,
// not per-sample, since it doesn't change between samples.
// ============================================================
void upload_model_to_gpu(const MiniRocketModel& model) {
    std::vector<int> flat_kernel_indices(84 * 3);
    for (int k = 0; k < 84; k++)
        for (int i = 0; i < 3; i++)
            flat_kernel_indices[k * 3 + i] = model.kernel_indices[k][i];

    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_indices, flat_kernel_indices.data(),
                                   84 * 3 * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_dilations, model.dilations.data(),
                                   model.num_dilations * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_features_per_dilation, model.num_features_per_dilation.data(),
                                   model.num_dilations * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_dilation_feature_offset, model.dilation_feature_offset.data(),
                                   model.num_dilations * sizeof(int)));

    // Biases: regular global memory, sized to this model's actual feature
    // count (no 64KB constant-memory ceiling to worry about here).
    CUDA_CHECK(cudaMalloc(&g_d_biases, model.num_features * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(g_d_biases, model.biases.data(),
                          model.num_features * sizeof(double), cudaMemcpyHostToDevice));
}

// Runs feature extraction on GPU for a batch of `batch_size` samples
// starting at `X_test_2d[start_idx]`. Returns the extracted features
// for each sample (still needs scaler + classifier applied on host).
std::vector<std::vector<double>> run_gpu_feature_extraction(
    const MiniRocketModel& model,
    const std::vector<std::vector<double>>& X_test_2d,
    int start_idx,
    int batch_size)
{
    int L = model.time_series_length;
    int num_features = model.num_features;

    // Flatten the batch's input series into one contiguous host buffer.
    std::vector<double> h_X(batch_size * L);
    for (int i = 0; i < batch_size; i++)
        std::copy(X_test_2d[start_idx + i].begin(), X_test_2d[start_idx + i].end(),
                  h_X.begin() + (size_t)i * L);

    double *d_X, *d_features;
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)batch_size * L * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_features, (size_t)batch_size * num_features * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), (size_t)batch_size * L * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Grid: (batch_size, num_dilations), Block: 84 threads (one per kernel).
    // Note: 84 isn't a multiple of the 32-thread warp size, so ~1/4 of the
    // last warp is idle. That's a known small inefficiency you could fix
    // by padding to 96 threads/block and returning early for threadIdx >= 84
    // (already done via the `if (k >= 84) return` guard) — left as-is here
    // for clarity; revisit if you're squeezing out the last bit of speed.
    dim3 grid(batch_size, model.num_dilations);
    dim3 block(84);
    extract_features_kernel<<<grid, block>>>(d_X, d_features, g_d_biases, L, model.num_dilations, batch_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_features(batch_size * num_features);
    CUDA_CHECK(cudaMemcpy(h_features.data(), d_features,
                          (size_t)batch_size * num_features * sizeof(double),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_features));

    std::vector<std::vector<double>> result(batch_size);
    for (int i = 0; i < batch_size; i++)
        result[i].assign(h_features.begin() + (size_t)i * num_features,
                          h_features.begin() + (size_t)(i + 1) * num_features);
    return result;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.json> <test_data.json> [output.csv] [batch_size]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string test_path = argv[2];
    std::string csv_path = (argc > 3) ? argv[3] : "";
    int throughput_batch_size = (argc > 4) ? std::stoi(argv[4]) : 1;

    // Report which GPU we're running on (helpful when comparing runs
    // across machines, same idea as the Python script printing this).
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "ERROR: No CUDA-capable GPU found on this machine." << std::endl;
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (" << (prop.totalGlobalMem / 1e9) << " GB)" << std::endl;

    MiniRocketModel model;
    model.load(model_path);
    upload_model_to_gpu(model);

    std::cout << "Loading test data from: " << test_path << std::endl;
    auto test_json = load_json(test_path);

    std::string dataset_name = (test_json["dataset_name"].type != JsonValue::NONE)
        ? test_json["dataset_name"].as_string() : "unknown";

    // num_samples may be absent from "compact" test JSON files — fall back
    // to counting the actual rows in X_test rather than trusting a
    // possibly-missing metadata field (same fix as the CPU file).
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
    // Ground-truth labels are optional in compact test files. If missing
    // or the wrong length, fall back to placeholder -1s and report
    // accuracy as N/A instead of asserting/crashing.
    bool has_labels = ((int)y_test.size() == num_samples);
    if (!has_labels) {
        y_test.assign(num_samples, -1);
    }

    std::cout << "Dataset: " << dataset_name << std::endl;
    std::cout << "  Samples: " << num_samples << std::endl;
    std::cout << "  Series length: " << series_length << std::endl;

    assert(series_length == model.time_series_length);
    assert((int)X_test_2d.size() == num_samples);

    // ---- Warmup (a few samples) — first CUDA kernel launch always
    // pays a one-time initialization cost, so we exclude that from timing,
    // same idea as the CPU file's warmup loop and the Python script's
    // num_warmup runs. ----
    for (int i = 0; i < std::min(3, num_samples); i++) {
        run_gpu_feature_extraction(model, X_test_2d, i, 1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Batch=1 latency benchmark (directly comparable to the CPU
    // and FPGA per-sample numbers) ----
    std::cout << "\nRunning per-sample (batch=1) inference..." << std::endl;
    std::vector<double> latencies_ms(num_samples);
    std::vector<int> predictions(num_samples);
    int correct = 0;

    for (int i = 0; i < num_samples; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        auto features_batch = run_gpu_feature_extraction(model, X_test_2d, i, 1);
        std::vector<double> features = features_batch[0];
        apply_scaler(model, features);
        int pred = classify(model, features);

        auto t1 = std::chrono::high_resolution_clock::now();

        latencies_ms[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        predictions[i] = pred;
        if (has_labels && pred == y_test[i]) correct++;

        if ((i + 1) % 5000 == 0 || i == num_samples - 1)
            std::cout << "  Sample " << (i + 1) << "/" << num_samples << "..." << std::endl;
    }

    // ---- Stats (same formulas as the CPU file, for apples-to-apples comparison) ----
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

    std::cout << "\n========== RESULTS (GPU, batch=1) ==========" << std::endl;
    std::cout << "Dataset:     " << dataset_name << std::endl;
    if (has_labels) {
        double accuracy = (double)correct / num_samples;
        std::cout << "Accuracy:    " << std::fixed << std::setprecision(4) << (accuracy * 100)
                  << "% (" << correct << "/" << num_samples << ")" << std::endl;
    } else {
        std::cout << "Accuracy:    N/A (no ground truth y_test labels in JSON)" << std::endl;
    }
    std::cout << "Throughput:  " << std::fixed << std::setprecision(1) << (1000.0 / mean)
              << " inferences/sec" << std::endl;
    std::cout << "\nLatency distribution (ms):" << std::endl;
    std::cout << "  Mean:  " << std::fixed << std::setprecision(3) << mean << std::endl;
    std::cout << "  P50:   " << percentile(50) << std::endl;
    std::cout << "  P95:   " << percentile(95) << std::endl;
    std::cout << "  P99:   " << percentile(99) << std::endl;
    std::cout << "  Min:   " << sorted_lat.front() << std::endl;
    std::cout << "  Max:   " << sorted_lat.back() << std::endl;
    std::cout << "  Std:   " << std_dev << std::endl;
    std::cout << "==============================================" << std::endl;

    // ---- Optional: batched throughput number, if batch_size > 1 was
    // requested. This is the number that will actually beat the FPGA/CPU
    // (see your Colab results — GPU wins big on throughput, loses on
    // single-sample latency due to kernel-launch overhead). ----
    if (throughput_batch_size > 1) {
        int n_bench = std::min(throughput_batch_size, num_samples);
        auto t0 = std::chrono::high_resolution_clock::now();
        run_gpu_feature_extraction(model, X_test_2d, 0, n_bench);
        auto t1 = std::chrono::high_resolution_clock::now();
        double batch_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "\n[Batched throughput, batch_size=" << n_bench << "]" << std::endl;
        std::cout << "  Total time: " << batch_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << (n_bench / (batch_ms / 1000.0)) << " inferences/sec" << std::endl;
    }

    // ---- CSV output, same format as the CPU file so both can feed the
    // same downstream comparison/plotting scripts. ----
    if (csv_path.empty())
        csv_path = "../results/MiniRocket_GPU_cuda_" + dataset_name + "_per_sample.csv";
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "sample_id,total_ms,predicted,actual,correct" << std::endl;
        for (int i = 0; i < num_samples; i++) {
            int actual = has_labels ? y_test[i] : -1;
            int is_correct = has_labels ? (predictions[i] == y_test[i] ? 1 : 0) : -1;
            csv << i << "," << std::fixed << std::setprecision(3) << latencies_ms[i]
                << "," << predictions[i] << "," << actual << "," << is_correct << std::endl;
        }
        csv.close();
        std::cout << "\nPer-sample CSV written to: " << csv_path << std::endl;
    } else {
        std::cerr << "WARNING: Could not open " << csv_path << " for writing" << std::endl;
    }

    return 0;
}
