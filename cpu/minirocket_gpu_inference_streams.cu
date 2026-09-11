// MiniRocket GPU Inference (CUDA C++) — Multi-stream version
// ============================================================
// Same model, same kernel, same algorithm as minirocket_gpu_inference.cu.
// The only change: the per-sample latency benchmark now pipelines
// multiple in-flight samples across N concurrent CUDA streams instead
// of launching one sample, blocking (cudaDeviceSynchronize), then
// launching the next.
//
// WHY THIS MATTERS: a single kernel launch already uses thousands of
// GPU threads (see the "CPU vs MT-CPU vs GPU" discussion) — there's no
// "single-threaded GPU" to begin with. What DOES cost real time is
// per-launch overhead: the H2D copy, kernel dispatch, and D2H copy for
// one tiny (single-sample) launch are mostly fixed latency, not
// compute. The original file pays that overhead serially, once per
// sample. This version overlaps it: while stream A's kernel computes,
// stream B's H2D transfer can already be in flight, etc. This is the
// GPU analog of your CPU single-threaded -> multithreaded upgrade —
// same hardware, more concurrency extracted from it.
//
// Compile:
//   nvcc -O3 -arch=sm_75 -std=c++17 -o minirocket_gpu_streams minirocket_gpu_inference_streams.cu
//
// Usage:
//   ./minirocket_gpu_streams <model.json> <test_data.json> [output.csv] [throughput_batch_size] [num_streams]
//
//   num_streams (optional, default 4): how many concurrent CUDA streams
//   pipeline the per-sample latency benchmark. Try 1 to see the
//   single-stream (no overlap) baseline for comparison, then 4/8 to see
//   the effect of overlap.

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
// Minimal JSON parser (identical to the original CPU/GPU files)
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
        pos++;
        JsonValue v;
        v.type = JsonValue::STRING;
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\') { pos++; v.str += s[pos++]; }
            else v.str += s[pos++];
        }
        pos++;
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
        pos++;
        JsonValue v;
        v.type = JsonValue::ARRAY;
        skip_ws();
        if (pos < s.size() && s[pos] == ']') { pos++; return v; }
        while (true) {
            v.arr.push_back(parse_value());
            skip_ws();
            if (pos >= s.size() || s[pos] == ']') { pos++; break; }
            pos++;
        }
        return v;
    }

    JsonValue parse_object() {
        pos++;
        JsonValue v;
        v.type = JsonValue::OBJECT;
        skip_ws();
        if (pos < s.size() && s[pos] == '}') { pos++; return v; }
        while (true) {
            skip_ws();
            auto key = parse_string();
            skip_ws();
            pos++;
            auto val = parse_value();
            v.obj.push_back({key.str, val});
            skip_ws();
            if (pos >= s.size() || s[pos] == '}') { pos++; break; }
            pos++;
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
// MiniRocket Model (identical to minirocket_gpu_inference.cu)
// ============================================================

#define MAX_DILATIONS 32

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

        assert(num_kernels == 84 && "This CUDA port assumes the standard 84 MiniRocket kernels");
        assert(num_dilations <= MAX_DILATIONS && "Increase MAX_DILATIONS for this model");
        for (int d = 0; d < num_dilations; d++) {
            assert(num_features_per_dilation[d] <= 256 &&
                   "A dilation needs more than 256 features/kernel — bump counts[256] "
                   "in extract_features_kernel AND this bound together.");
        }

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
// Device constant memory (identical to minirocket_gpu_inference.cu)
// ============================================================
__constant__ int   d_kernel_indices[84 * 3];
__constant__ int   d_dilations[MAX_DILATIONS];
__constant__ int   d_num_features_per_dilation[MAX_DILATIONS];
__constant__ int   d_dilation_feature_offset[MAX_DILATIONS];
static double* g_d_biases = nullptr;

// ============================================================
// Feature extraction kernel — IDENTICAL to minirocket_gpu_inference.cu.
// Streams change WHEN/HOW this kernel is launched, not what it computes.
// ============================================================
__global__ void extract_features_kernel(
    const double* __restrict__ X,
    double* __restrict__ features_out,
    const double* __restrict__ biases,
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

    double weights[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) weights[i] = -1.0;
    weights[d_kernel_indices[k * 3 + 0]] = 2.0;
    weights[d_kernel_indices[k * 3 + 1]] = 2.0;
    weights[d_kernel_indices[k * 3 + 2]] = 2.0;

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

    int feature_base = d_dilation_feature_offset[d] + k * n_feat_this_dil;
    const double* ts = X + (size_t)sample * L;
    double* out = features_out + (size_t)sample * (d_dilation_feature_offset[num_dilations - 1]
                                                      + 84 * d_num_features_per_dilation[num_dilations - 1]);

    if (conv_length <= 0) {
        for (int f = 0; f < n_feat_this_dil; f++) out[feature_base + f] = 0.0;
        return;
    }

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

void upload_model_to_gpu(const MiniRocketModel& model) {
    std::vector<int> flat_kernel_indices(84 * 3);
    for (int k = 0; k < 84; k++)
        for (int i = 0; i < 3; i++)
            flat_kernel_indices[k * 3 + i] = model.kernel_indices[k][i];

    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_indices, flat_kernel_indices.data(), 84 * 3 * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_dilations, model.dilations.data(), model.num_dilations * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_features_per_dilation, model.num_features_per_dilation.data(),
                                   model.num_dilations * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_dilation_feature_offset, model.dilation_feature_offset.data(),
                                   model.num_dilations * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&g_d_biases, model.num_features * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(g_d_biases, model.biases.data(),
                          model.num_features * sizeof(double), cudaMemcpyHostToDevice));
}

// Non-streamed batched throughput helper — identical in spirit to the
// original file's run_gpu_feature_extraction, used only for the large
// "[Batched throughput]" demonstration section (already-large launches
// don't benefit from stream overlap the way tiny single-sample launches do).
std::vector<std::vector<double>> run_gpu_feature_extraction_batched(
    const MiniRocketModel& model,
    const std::vector<std::vector<double>>& X_test_2d,
    int start_idx,
    int batch_size)
{
    int L = model.time_series_length;
    int num_features = model.num_features;

    std::vector<double> h_X(batch_size * L);
    for (int i = 0; i < batch_size; i++)
        std::copy(X_test_2d[start_idx + i].begin(), X_test_2d[start_idx + i].end(),
                  h_X.begin() + (size_t)i * L);

    double *d_X, *d_features;
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)batch_size * L * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_features, (size_t)batch_size * num_features * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), (size_t)batch_size * L * sizeof(double), cudaMemcpyHostToDevice));

    dim3 grid(batch_size, model.num_dilations);
    dim3 block(84);
    extract_features_kernel<<<grid, block>>>(d_X, d_features, g_d_biases, L, model.num_dilations, batch_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_features(batch_size * num_features);
    CUDA_CHECK(cudaMemcpy(h_features.data(), d_features, (size_t)batch_size * num_features * sizeof(double),
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
        std::cerr << "Usage: " << argv[0]
                  << " <model.json> <test_data.json> [output.csv] [throughput_batch_size] [num_streams]"
                  << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string test_path = argv[2];
    std::string csv_path = (argc > 3) ? argv[3] : "";
    int throughput_batch_size = (argc > 4) ? std::stoi(argv[4]) : 1;
    int num_streams = (argc > 5) ? std::stoi(argv[5]) : 4;

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "ERROR: No CUDA-capable GPU found on this machine." << std::endl;
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (" << (prop.totalGlobalMem / 1e9) << " GB)" << std::endl;
    std::cout << "Streams: " << num_streams << std::endl;

    MiniRocketModel model;
    model.load(model_path);
    upload_model_to_gpu(model);

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
    if (!has_labels) y_test.assign(num_samples, -1);

    std::cout << "Dataset: " << dataset_name << std::endl;
    std::cout << "  Samples: " << num_samples << std::endl;
    std::cout << "  Series length: " << series_length << std::endl;

    assert(series_length == model.time_series_length);
    assert((int)X_test_2d.size() == num_samples);

    int L = model.time_series_length;
    int num_features = model.num_features;

    // ------------------------------------------------------------
    // Multi-stream pipeline setup. Each stream gets its own PINNED
    // host buffers (pinned = page-locked, required for true async
    // cudaMemcpyAsync — a regular std::vector's memory can't be
    // safely used for async transfers) and its own device buffers,
    // so streams never contend with each other for memory.
    // ------------------------------------------------------------
    std::vector<cudaStream_t> streams(num_streams);
    std::vector<double*> h_X_pinned(num_streams), h_feat_pinned(num_streams);
    std::vector<double*> d_X_buf(num_streams), d_feat_buf(num_streams);
    std::vector<cudaEvent_t> start_evt(num_streams), stop_evt(num_streams);
    std::vector<int> pending_sample(num_streams, -1);  // which sample idx is in-flight on this stream

    for (int s = 0; s < num_streams; s++) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMallocHost(&h_X_pinned[s], L * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_feat_pinned[s], num_features * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_X_buf[s], L * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_feat_buf[s], num_features * sizeof(double)));
        CUDA_CHECK(cudaEventCreate(&start_evt[s]));
        CUDA_CHECK(cudaEventCreate(&stop_evt[s]));
    }

    std::vector<double> latencies_ms(num_samples);
    std::vector<int> predictions(num_samples);
    int correct = 0;

    // Helper: finalize whatever sample is currently pending on stream s
    // (blocks only on THIS stream — other streams keep running).
    auto finalize_stream = [&](int s) {
        int i = pending_sample[s];
        if (i < 0) return;  // nothing pending yet
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_evt[s], stop_evt[s]));

        std::vector<double> features(h_feat_pinned[s], h_feat_pinned[s] + num_features);
        apply_scaler(model, features);
        int pred = classify(model, features);

        latencies_ms[i] = (double)ms;
        predictions[i] = pred;
        if (has_labels && pred == y_test[i]) correct++;
        pending_sample[s] = -1;
    };

    dim3 block(84);
    dim3 grid_single(1, model.num_dilations);  // one sample per launch

    std::cout << "\nRunning MULTI-STREAM per-sample latency benchmark ("
              << num_streams << " streams)..." << std::endl;

    // Warmup: push one dummy sample through each stream and drain it,
    // so the first real measurements aren't paying one-time CUDA
    // context/JIT costs.
    for (int s = 0; s < num_streams && s < num_samples; s++) {
        std::copy(X_test_2d[s].begin(), X_test_2d[s].end(), h_X_pinned[s]);
        CUDA_CHECK(cudaMemcpyAsync(d_X_buf[s], h_X_pinned[s], L * sizeof(double),
                                    cudaMemcpyHostToDevice, streams[s]));
        extract_features_kernel<<<grid_single, block, 0, streams[s]>>>(
            d_X_buf[s], d_feat_buf[s], g_d_biases, L, model.num_dilations, 1);
        CUDA_CHECK(cudaMemcpyAsync(h_feat_pinned[s], d_feat_buf[s], num_features * sizeof(double),
                                    cudaMemcpyDeviceToHost, streams[s]));
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    }

    // TRUE overlapped-throughput timer: starts right before the pipelined
    // loop begins issuing real (non-warmup) work, ends after the final
    // drain — this captures actual wall-clock time INCLUDING whatever
    // concurrency the streams achieved, unlike 1000/mean_latency below
    // (which assumes serial execution and understates real throughput
    // when streams are genuinely overlapping work).
    auto pipeline_t0 = std::chrono::high_resolution_clock::now();

    // Main pipelined loop: for each sample, pick a stream round-robin.
    // If that stream still has a PREVIOUS sample in flight, finalize it
    // first (this is where per-stream buffer reuse becomes safe — the
    // sync guarantees the old async copy has fully landed before we
    // overwrite the pinned buffer with new data).
    for (int i = 0; i < num_samples; i++) {
        int s = i % num_streams;

        finalize_stream(s);  // reads out the PREVIOUS sample assigned to this stream, if any

        std::copy(X_test_2d[i].begin(), X_test_2d[i].end(), h_X_pinned[s]);

        CUDA_CHECK(cudaEventRecord(start_evt[s], streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_X_buf[s], h_X_pinned[s], L * sizeof(double),
                                    cudaMemcpyHostToDevice, streams[s]));
        extract_features_kernel<<<grid_single, block, 0, streams[s]>>>(
            d_X_buf[s], d_feat_buf[s], g_d_biases, L, model.num_dilations, 1);
        CUDA_CHECK(cudaMemcpyAsync(h_feat_pinned[s], d_feat_buf[s], num_features * sizeof(double),
                                    cudaMemcpyDeviceToHost, streams[s]));
        CUDA_CHECK(cudaEventRecord(stop_evt[s], streams[s]));

        pending_sample[s] = i;  // this sample's result will be read out next time we revisit stream s

        if ((i + 1) % 5000 == 0 || i == num_samples - 1) {
            std::cout << "  Sample " << (i + 1) << "/" << num_samples << "..." << std::endl;
        }
    }

    // Drain: finalize whatever's still pending on every stream after
    // the loop ends (the tail — up to num_streams samples).
    for (int s = 0; s < num_streams; s++) finalize_stream(s);

    auto pipeline_t1 = std::chrono::high_resolution_clock::now();
    double pipeline_total_ms = std::chrono::duration<double, std::milli>(pipeline_t1 - pipeline_t0).count();
    double pipeline_throughput = num_samples / (pipeline_total_ms / 1000.0);

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

    std::cout << "\n========== RESULTS (GPU, " << num_streams << " streams, per-sample) ==========" << std::endl;
    std::cout << "Dataset:     " << dataset_name << std::endl;
    if (has_labels) {
        std::cout << "Accuracy:    " << std::fixed << std::setprecision(4) << (accuracy * 100)
                  << "% (" << correct << "/" << num_samples << ")" << std::endl;
    } else {
        std::cout << "Accuracy:    N/A (no ground truth y_test labels in JSON)" << std::endl;
    }
    std::cout << "Throughput:  " << std::fixed << std::setprecision(1) << (1000.0 / mean)
              << " inferences/sec (per-event latency basis — assumes serial, likely UNDERSTATES real throughput)" << std::endl;
    std::cout << "\nTRUE PIPELINE WALL-CLOCK (this is the real overlap-benefit number):" << std::endl;
    std::cout << "  Total time:  " << std::fixed << std::setprecision(1) << pipeline_total_ms << " ms" << std::endl;
    std::cout << "  Throughput:  " << std::fixed << std::setprecision(1) << pipeline_throughput
              << " inferences/sec" << std::endl;
    std::cout << "\nLatency distribution (ms):" << std::endl;
    std::cout << "  Mean:  " << std::fixed << std::setprecision(3) << mean << std::endl;
    std::cout << "  P50:   " << percentile(50) << std::endl;
    std::cout << "  P95:   " << percentile(95) << std::endl;
    std::cout << "  P99:   " << percentile(99) << std::endl;
    std::cout << "  Min:   " << sorted_lat.front() << std::endl;
    std::cout << "  Max:   " << sorted_lat.back() << std::endl;
    std::cout << "  Std:   " << std_dev << std::endl;
    std::cout << "===================================================================" << std::endl;

    // Same large-batch throughput demonstration as the single-stream file
    // (already maximally parallel within one launch, so not stream-pipelined).
    if (throughput_batch_size > 1) {
        int n_bench = std::min(throughput_batch_size, num_samples);
        auto t0 = std::chrono::high_resolution_clock::now();
        run_gpu_feature_extraction_batched(model, X_test_2d, 0, n_bench);
        auto t1 = std::chrono::high_resolution_clock::now();
        double batch_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "\n[Batched throughput, batch_size=" << n_bench << "]" << std::endl;
        std::cout << "  Total time: " << batch_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << (n_bench / (batch_ms / 1000.0)) << " inferences/sec" << std::endl;
    }

    if (csv_path.empty())
        csv_path = "../results/MiniRocket_GPU_streams_" + dataset_name + "_per_sample.csv";
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

    for (int s = 0; s < num_streams; s++) {
        cudaFreeHost(h_X_pinned[s]);
        cudaFreeHost(h_feat_pinned[s]);
        cudaFree(d_X_buf[s]);
        cudaFree(d_feat_buf[s]);
        cudaEventDestroy(start_evt[s]);
        cudaEventDestroy(stop_evt[s]);
        cudaStreamDestroy(streams[s]);
    }

    return 0;
}
