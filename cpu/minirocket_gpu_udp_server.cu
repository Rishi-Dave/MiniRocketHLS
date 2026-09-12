// MiniRocket GPU UDP Streaming Server — NAIVE full-recompute (NO DP-Reuse)
//==============================================================================
// Phase (a) correctness baseline for streaming MiniRocket on GPU.
//
// On EVERY new sample: shift the sliding window left by 1, append the sample,
// then run a FULL MiniRocket feature extract over the entire window of length
// L = time_series_length. This is intentionally the naive approach — DP-Reuse
// (incremental convolution / PPV updates) is future work.
//
// Algorithm / structure reused from the validated cpu/minirocket_gpu_inference.cu
// (JSON parser, MiniRocketModel, extract_features_kernel, host scaler+Ridge,
// g_d_biases in global memory — NOT __constant__).
//
// Protocol (IPv4 UDP SOCK_DGRAM):
//   Client → server: exactly 4 bytes, little-endian float32 sample.
//   Server → client (to source addr): exactly 4 bytes, little-endian int32
//                                    predicted class label (= model.classes[argmax]).
//   Sliding window of length L starts as zeros; memmove left by 1, append new
//   sample; FULL naive recompute; scaler+classify; reply.
//
// Compile (sm_75 = RTX 2080 Ti on cerebro):
//   nvcc -O3 -arch=sm_75 -std=c++17 -o minirocket_gpu_udp_server minirocket_gpu_udp_server.cu
//
// Usage:
//   ./minirocket_gpu_udp_server <model.json> [bind_host] [bind_port]
//   defaults: 127.0.0.1 9000
//==============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <cstring>
#include <csignal>
#include <atomic>
#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

// ============================================================
// CUDA error-checking helper
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

static std::atomic<bool> g_running{true};

static void on_signal(int) {
    g_running = false;
}

// ============================================================
// Minimal JSON parser (identical to existing GPU files)
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
// MiniRocket Model
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
                   "in extract_features_kernel AND this bound together, or predictions "
                   "will silently corrupt again.");
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
// Device constant memory (+ biases in global memory)
// ============================================================
__constant__ int   d_kernel_indices[84 * 3];
__constant__ int   d_dilations[MAX_DILATIONS];
__constant__ int   d_num_features_per_dilation[MAX_DILATIONS];
__constant__ int   d_dilation_feature_offset[MAX_DILATIONS];

// Biases in GLOBAL device memory (g_d_biases), NOT __constant__ —
// constant mem overflow bug on large feature counts (see existing GPU files).
static double* g_d_biases = nullptr;

// ============================================================
// GPU feature-extraction kernel (same as batch_size=1 offline GPU)
// Weights: -1 everywhere except +2 at three kernel_indices positions.
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

    int dilation        = d_dilations[d];
    int n_feat_this_dil = d_num_features_per_dilation[d];
    int padding0        = d % 2;
    int padding1        = (padding0 + k) % 2;
    int half_pad        = 4 * dilation;

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
    // num_features = offset[last] + 84 * n_feat[last]
    double* out = features_out + (size_t)sample * (d_dilation_feature_offset[num_dilations - 1]
                                                      + 84 * d_num_features_per_dilation[num_dilations - 1]);

    if (conv_length <= 0) {
        for (int f = 0; f < n_feat_this_dil; f++) out[feature_base + f] = 0.0;
        return;
    }

    // counts[256] — asserted in MiniRocketModel::load via num_features_per_dilation
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
// Host apply_scaler + classify (Ridge) — matching existing GPU files
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

    CUDA_CHECK(cudaMalloc(&g_d_biases, model.num_features * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(g_d_biases, model.biases.data(),
                          model.num_features * sizeof(double), cudaMemcpyHostToDevice));
}

// ============================================================
// Main — UDP server, NAIVE full-recompute per packet
// ============================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.json> [bind_host] [bind_port]" << std::endl;
        std::cerr << "  defaults: 127.0.0.1 9000" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string bind_host = (argc > 2) ? argv[2] : "127.0.0.1";
    int bind_port = (argc > 3) ? std::stoi(argv[3]) : 9000;

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "ERROR: No CUDA-capable GPU found." << std::endl;
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    MiniRocketModel model;
    model.load(model_path);
    upload_model_to_gpu(model);

    const int L = model.time_series_length;
    const int num_features = model.num_features;

    // Allocate once; reuse across packets (correctness-phase simplicity:
    // cudaMemcpy H2D/D2H each inference).
    double *d_X = nullptr, *d_features = nullptr;
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)L * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_features, (size_t)num_features * sizeof(double)));

    std::vector<double> h_window((size_t)L, 0.0);  // starts as zeros
    std::vector<double> h_features((size_t)num_features, 0.0);

    // UDP IPv4 SOCK_DGRAM server
    int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("socket");
        return 1;
    }
    int yes = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(bind_port));
    if (inet_pton(AF_INET, bind_host.c_str(), &addr.sin_addr) != 1) {
        std::cerr << "ERROR: invalid bind_host " << bind_host << std::endl;
        return 1;
    }
    if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind");
        return 1;
    }

    // Startup banner
    std::cout << "========================================" << std::endl;
    std::cout << "MiniRocket GPU UDP Server" << std::endl;
    std::cout << "  model:         " << model_path << std::endl;
    std::cout << "  L:             " << L << std::endl;
    std::cout << "  num_features:  " << num_features << std::endl;
    std::cout << "  bind:          " << bind_host << ":" << bind_port << std::endl;
    std::cout << "  GPU:           " << prop.name << std::endl;
    std::cout << "  mode:          NAIVE full-recompute (no DP-Reuse)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Waiting for float32 samples..." << std::endl;

    const int progress_every = 1000;
    uint64_t n_samples = 0;
    int last_pred = 0;
    double last_inf_ms = 0.0;

    // Optional short recv timeout so Ctrl+C is noticed promptly
    timeval tv{};
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    while (g_running) {
        uint8_t buf[4];
        sockaddr_in client{};
        socklen_t client_len = sizeof(client);
        ssize_t n = recvfrom(sock, buf, 4, 0,
                             reinterpret_cast<sockaddr*>(&client), &client_len);
        if (n < 0) {
            if (!g_running) break;
            continue;  // timeout / EINTR
        }
        if (n != 4) {
            std::cerr << "WARNING: expected 4-byte float32, got " << n << " bytes" << std::endl;
            continue;
        }

        float sample_f32;
        std::memcpy(&sample_f32, buf, 4);  // little-endian on x86/ARM LE hosts
        double sample = static_cast<double>(sample_f32);

        // Sliding window: memmove left by 1, append new sample at end
        if (L > 1) {
            std::memmove(h_window.data(), h_window.data() + 1,
                         (size_t)(L - 1) * sizeof(double));
        }
        h_window[(size_t)L - 1] = sample;

        auto t0 = std::chrono::high_resolution_clock::now();

        // NAIVE: FULL MiniRocket feature extract on the whole window
        CUDA_CHECK(cudaMemcpy(d_X, h_window.data(), (size_t)L * sizeof(double),
                              cudaMemcpyHostToDevice));

        dim3 grid(1, model.num_dilations);  // batch_size = 1
        dim3 block(84);
        extract_features_kernel<<<grid, block>>>(
            d_X, d_features, g_d_biases, L, model.num_dilations, /*batch_size=*/1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_features.data(), d_features,
                              (size_t)num_features * sizeof(double),
                              cudaMemcpyDeviceToHost));

        apply_scaler(model, h_features);
        int pred = classify(model, h_features);

        auto t1 = std::chrono::high_resolution_clock::now();
        last_inf_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        last_pred = pred;
        n_samples++;

        // Reply: little-endian int32 class label
        int32_t pred_i32 = static_cast<int32_t>(pred);
        uint8_t out_buf[4];
        std::memcpy(out_buf, &pred_i32, 4);
        ssize_t sn = sendto(sock, out_buf, 4, 0,
                            reinterpret_cast<sockaddr*>(&client), client_len);
        if (sn != 4) {
            perror("sendto");
        }

        if (progress_every > 0 && (n_samples % (uint64_t)progress_every) == 0) {
            std::cout << "  samples=" << n_samples
                      << " last_pred=" << last_pred
                      << " inf_ms=" << std::fixed << std::setprecision(3) << last_inf_ms
                      << std::endl;
        }
    }

    std::cout << "Shutting down after " << n_samples << " samples." << std::endl;
    ::close(sock);
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_features));
    if (g_d_biases) {
        CUDA_CHECK(cudaFree(g_d_biases));
        g_d_biases = nullptr;
    }
    return 0;
}
