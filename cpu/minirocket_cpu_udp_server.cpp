// MiniRocket CPU UDP Streaming Server — NAIVE full-recompute (NO DP-Reuse)
//==============================================================================
// Fair GPU-vs-CPU streaming curve baseline: same UDP protocol + RESET as
// minirocket_gpu_udp_server.cu, but feature extract runs on the host (single
// thread). OpenMP / MT can be added later — clarity first.
//
// Algorithm matches the non-CUDA path / GPU kernel logic:
//   for each dilation d, for each of 84 kernels:
//     build weights (-1, +2 at three kernel_indices),
//     convolve over window with dilation, count PPV vs biases,
//   then scaler + Ridge classify.
//
// Protocol (IPv4 UDP SOCK_DGRAM), little-endian explicit memcpy:
//   len==4: float32 sample → infer → reply int32 class
//   len==8: control  magic=0x4D525354 ('MRST') + cmd
//           cmd=1 RESET → zero window, reply int32 0 ACK
//           unknown cmd → reply int32 -1
//
// Compile:
//   g++ -O3 -march=native -std=c++17 -o minirocket_cpu_udp_server \
//       minirocket_cpu_udp_server.cpp
//
// Usage:
//   ./minirocket_cpu_udp_server <model.json> [bind_host] [bind_port]
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
#include <cstdint>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

static constexpr uint32_t kCtrlMagic = 0x4D525354u;  // 'MRST'
static constexpr uint32_t kCmdReset  = 1u;

static std::atomic<bool> g_running{true};

static void on_signal(int) {
    g_running = false;
}

// ============================================================
// Minimal JSON parser (same style as GPU / client files)
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
// MiniRocket Model + host feature extract (NAIVE, single-thread)
// ============================================================

struct MiniRocketModel {
    int num_kernels = 0;
    int num_dilations = 0;
    int num_features = 0;
    int num_classes = 0;
    int time_series_length = 0;

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

        assert(num_kernels == 84 && "Assumes standard 84 MiniRocket kernels");

        dilation_feature_offset.resize(num_dilations);
        int running = 0;
        for (int d = 0; d < num_dilations; d++) {
            dilation_feature_offset[d] = running;
            running += 84 * num_features_per_dilation[d];
        }
        assert(running == num_features && "Feature count mismatch");

        std::cout << "  num_kernels: " << num_kernels << std::endl;
        std::cout << "  num_dilations: " << num_dilations << std::endl;
        std::cout << "  num_features: " << num_features << std::endl;
        std::cout << "  num_classes: " << num_classes << std::endl;
        std::cout << "  time_series_length: " << time_series_length << std::endl;
    }
};

// Host-side MiniRocket feature extract — port of GPU kernel / CPU inference.
// NAIVE: full recompute over entire window. Single-thread (MT/OpenMP later).
void extract_features_cpu(const MiniRocketModel& model,
                          const std::vector<double>& ts,
                          std::vector<double>& features_out) {
    const int L = model.time_series_length;
    std::fill(features_out.begin(), features_out.end(), 0.0);

    for (int d = 0; d < model.num_dilations; d++) {
        const int dilation = model.dilations[d];
        const int n_feat_this_dil = model.num_features_per_dilation[d];
        const int padding0 = d % 2;
        const int half_pad = 4 * dilation;
        const int feat_off = model.dilation_feature_offset[d];

        for (int k = 0; k < 84; k++) {
            double weights[9];
            for (int i = 0; i < 9; i++) weights[i] = -1.0;
            weights[model.kernel_indices[k][0]] = 2.0;
            weights[model.kernel_indices[k][1]] = 2.0;
            weights[model.kernel_indices[k][2]] = 2.0;

            const int padding1 = (padding0 + k) % 2;
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

            const int feature_base = feat_off + k * n_feat_this_dil;
            if (conv_length <= 0) {
                for (int f = 0; f < n_feat_this_dil; f++)
                    features_out[(size_t)feature_base + f] = 0.0;
                continue;
            }

            std::vector<int> counts((size_t)n_feat_this_dil, 0);
            for (int t = t_start; t < t_end; t++) {
                double conv_val = 0.0;
                for (int w = 0; w < 9; w++) {
                    int idx = t + (w - 4) * dilation;
                    if (idx >= 0 && idx < L)
                        conv_val += weights[w] * ts[(size_t)idx];
                }
                for (int f = 0; f < n_feat_this_dil; f++) {
                    if (conv_val > model.biases[(size_t)feature_base + f])
                        counts[(size_t)f]++;
                }
            }
            for (int f = 0; f < n_feat_this_dil; f++) {
                features_out[(size_t)feature_base + f] =
                    (double)counts[(size_t)f] / (double)conv_length;
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

static double percentile_sorted(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (double)(v.size() - 1);
    size_t i = (size_t)idx;
    double frac = idx - (double)i;
    if (i + 1 < v.size())
        return v[i] * (1.0 - frac) + v[i + 1] * frac;
    return v[i];
}

static void send_i32_reply(int sock, const sockaddr_in& client, socklen_t client_len,
                           int32_t value) {
    uint8_t out_buf[4];
    std::memcpy(out_buf, &value, 4);
    ssize_t sn = sendto(sock, out_buf, 4, 0,
                        reinterpret_cast<const sockaddr*>(&client), client_len);
    if (sn != 4) perror("sendto");
}

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

    MiniRocketModel model;
    model.load(model_path);

    const int L = model.time_series_length;
    const int num_features = model.num_features;

    std::vector<double> h_window((size_t)L, 0.0);
    std::vector<double> h_features((size_t)num_features, 0.0);

    int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { perror("socket"); return 1; }
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

    std::cout << "========================================" << std::endl;
    std::cout << "MiniRocket CPU UDP Server" << std::endl;
    std::cout << "  model:         " << model_path << std::endl;
    std::cout << "  L:             " << L << std::endl;
    std::cout << "  num_features:  " << num_features << std::endl;
    std::cout << "  bind:          " << bind_host << ":" << bind_port << std::endl;
    std::cout << "  mode:          NAIVE full-recompute (no DP-Reuse)" << std::endl;
    std::cout << "  threads:       1 (OpenMP optional later)" << std::endl;
    std::cout << "  protocol:      4B float32 sample | 8B MRST control (RESET=1)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Waiting for packets..." << std::endl;

    const int progress_every = 1000;
    uint64_t n_samples = 0;
    uint64_t n_resets = 0;
    int last_pred = 0;
    double last_inf_ms = 0.0;
    std::vector<double> infer_ms_hist;
    infer_ms_hist.reserve(1 << 16);

    timeval tv{};
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    uint8_t buf[16];
    while (g_running) {
        sockaddr_in client{};
        socklen_t client_len = sizeof(client);
        ssize_t n = recvfrom(sock, buf, sizeof(buf), 0,
                             reinterpret_cast<sockaddr*>(&client), &client_len);
        if (n < 0) {
            if (!g_running) break;
            continue;
        }

        if (n == 8) {
            uint32_t magic = 0, cmd = 0;
            std::memcpy(&magic, buf + 0, 4);
            std::memcpy(&cmd,   buf + 4, 4);
            if (magic != kCtrlMagic) {
                std::cerr << "WARNING: 8-byte packet with bad magic — dropping\n";
                continue;
            }
            if (cmd == kCmdReset) {
                std::fill(h_window.begin(), h_window.end(), 0.0);
                n_resets++;
                send_i32_reply(sock, client, client_len, 0);
            } else {
                send_i32_reply(sock, client, client_len, -1);
            }
            continue;
        }

        if (n != 4) {
            std::cerr << "WARNING: expected 4 or 8 bytes, got " << n << std::endl;
            continue;
        }

        float sample_f32;
        std::memcpy(&sample_f32, buf, 4);
        double sample = static_cast<double>(sample_f32);

        if (L > 1) {
            std::memmove(h_window.data(), h_window.data() + 1,
                         (size_t)(L - 1) * sizeof(double));
        }
        h_window[(size_t)L - 1] = sample;

        auto t0 = std::chrono::high_resolution_clock::now();

        // NAIVE: FULL MiniRocket feature extract on the whole window (no DP-Reuse)
        extract_features_cpu(model, h_window, h_features);
        apply_scaler(model, h_features);
        int pred = classify(model, h_features);

        auto t1 = std::chrono::high_resolution_clock::now();
        last_inf_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        last_pred = pred;
        n_samples++;
        infer_ms_hist.push_back(last_inf_ms);

        send_i32_reply(sock, client, client_len, static_cast<int32_t>(pred));

        if (progress_every > 0 && (n_samples % (uint64_t)progress_every) == 0) {
            std::cout << "  samples=" << n_samples
                      << " resets=" << n_resets
                      << " last_pred=" << last_pred
                      << " inf_ms=" << std::fixed << std::setprecision(3) << last_inf_ms
                      << std::endl;
        }
    }

    std::cout << "\n========== SERVER STATS (SIGINT) ==========" << std::endl;
    std::cout << "  samples:  " << n_samples << std::endl;
    std::cout << "  resets:   " << n_resets << std::endl;
    if (!infer_ms_hist.empty()) {
        double sum = 0.0;
        for (double x : infer_ms_hist) sum += x;
        double mean = sum / (double)infer_ms_hist.size();
        double p50 = percentile_sorted(infer_ms_hist, 0.50);
        double p95 = percentile_sorted(infer_ms_hist, 0.95);
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  infer_ms mean: " << mean << std::endl;
        std::cout << "  infer_ms p50:  " << p50 << std::endl;
        std::cout << "  infer_ms p95:  " << p95 << std::endl;
    }
    std::cout << "===========================================" << std::endl;

    std::cout << "Shutting down after " << n_samples << " samples." << std::endl;
    ::close(sock);
    return 0;
}
