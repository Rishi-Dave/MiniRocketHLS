// MiniRocket UDP full-dataset streaming bench — offline-equivalent report
//==============================================================================
// Streams every test series to a MiniRocket UDP server (GPU or CPU) with
// RESET between series. Scores accuracy using ONLY the final-after-L prediction
// (offline-equivalent). Prints Accuracy / Throughput / RTT percentiles in the
// same style as offline minirocket_gpu.
//
// Protocol:
//   RESET: 8-byte LE (magic=0x4D525354, cmd=1) → wait int32 ACK 0
//   Sample: 4-byte LE float32 → recv 4-byte LE int32 class; RTT = send→recv
//
// Note: early in-window preds are NOT scored; RESET between series is required
// for independence (otherwise leftover window contaminates the next series).
//
// Compile:
//   g++ -O3 -march=native -std=c++17 -o minirocket_udp_bench minirocket_udp_bench.cpp
//
// Usage:
//   ./minirocket_udp_bench <test_data.json> [host] [port] [max_series] [csv_out]
//   defaults: 127.0.0.1 9000 max_series=0 (all) csv_out=empty
//==============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

static constexpr uint32_t kCtrlMagic = 0x4D525354u;
static constexpr uint32_t kCmdReset  = 1u;

// ============================================================
// Minimal JSON parser (same style as existing client)
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

struct RttStats {
    double mean = 0, p50 = 0, p95 = 0, p99 = 0, min_v = 0, max_v = 0, std_v = 0;
    size_t n = 0;
};

static RttStats compute_rtt_stats(std::vector<double> v) {
    RttStats s;
    s.n = v.size();
    if (v.empty()) return s;
    double sum = 0.0, sumsq = 0.0;
    s.min_v = v[0];
    s.max_v = v[0];
    for (double x : v) {
        sum += x;
        sumsq += x * x;
        if (x < s.min_v) s.min_v = x;
        if (x > s.max_v) s.max_v = x;
    }
    s.mean = sum / (double)v.size();
    s.std_v = std::sqrt(std::max(0.0, sumsq / (double)v.size() - s.mean * s.mean));
    std::sort(v.begin(), v.end());
    auto pct = [&](double p) {
        double idx = p * (double)(v.size() - 1);
        size_t i = (size_t)idx;
        double frac = idx - (double)i;
        if (i + 1 < v.size()) return v[i] * (1.0 - frac) + v[i + 1] * frac;
        return v[i];
    };
    s.p50 = pct(0.50);
    s.p95 = pct(0.95);
    s.p99 = pct(0.99);
    return s;
}

static void print_rtt_block(const char* title, const RttStats& s) {
    std::cout << title << " (n=" << s.n << "):\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Mean: " << s.mean << "  P50: " << s.p50
              << "  P95: " << s.p95 << "  P99: " << s.p99 << "\n";
    std::cout << "  Min:  " << s.min_v << "  Max: " << s.max_v
              << "  Std: " << s.std_v << "\n";
}

static bool send_reset(int sock, const sockaddr_in& server) {
    uint8_t pkt[8];
    uint32_t magic = kCtrlMagic;
    uint32_t cmd = kCmdReset;
    std::memcpy(pkt + 0, &magic, 4);
    std::memcpy(pkt + 4, &cmd, 4);
    ssize_t sn = sendto(sock, pkt, 8, 0,
                        reinterpret_cast<const sockaddr*>(&server), sizeof(server));
    if (sn != 8) { perror("sendto RESET"); return false; }

    uint8_t in_buf[4];
    sockaddr_in from{};
    socklen_t from_len = sizeof(from);
    ssize_t rn = recvfrom(sock, in_buf, 4, 0,
                          reinterpret_cast<sockaddr*>(&from), &from_len);
    if (rn != 4) {
        perror("recvfrom RESET ACK");
        return false;
    }
    int32_t ack = 0;
    std::memcpy(&ack, in_buf, 4);
    if (ack != 0) {
        std::cerr << "WARNING: RESET ACK=" << ack << " (expected 0)\n";
    }
    return true;
}

static bool send_sample_recv_pred(int sock, const sockaddr_in& server,
                                  float sample_f32, int32_t& pred_out,
                                  double& rtt_ms_out) {
    uint8_t out_buf[4];
    std::memcpy(out_buf, &sample_f32, 4);

    auto t0 = std::chrono::steady_clock::now();
    ssize_t sn = sendto(sock, out_buf, 4, 0,
                        reinterpret_cast<const sockaddr*>(&server), sizeof(server));
    if (sn != 4) { perror("sendto sample"); return false; }

    uint8_t in_buf[4];
    sockaddr_in from{};
    socklen_t from_len = sizeof(from);
    ssize_t rn = recvfrom(sock, in_buf, 4, 0,
                          reinterpret_cast<sockaddr*>(&from), &from_len);
    auto t1 = std::chrono::steady_clock::now();
    if (rn != 4) {
        perror("recvfrom pred");
        return false;
    }
    int32_t pred = 0;
    std::memcpy(&pred, in_buf, 4);
    pred_out = pred;
    rtt_ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <test_data.json> [host] [port] [max_series] [csv_out]\n"
                  << "  defaults: 127.0.0.1 9000 max_series=0 (all) csv_out=empty\n";
        return 1;
    }

    std::string test_path = argv[1];
    std::string host = (argc > 2) ? argv[2] : "127.0.0.1";
    int port = (argc > 3) ? std::stoi(argv[3]) : 9000;
    int max_series = (argc > 4) ? std::stoi(argv[4]) : 0;
    std::string csv_out = (argc > 5) ? argv[5] : "";

    std::cout << "========== MiniRocket UDP Bench ==========\n";
    std::cout << "Mode: GPU/CPU UDP streaming, naive, final-after-L scoring\n";
    std::cout << "Note: early in-window preds are NOT scored; RESET between\n"
              << "      series is required for independence.\n";

    auto test_json = load_json(test_path);
    std::string dataset_name = (test_json["dataset_name"].type != JsonValue::NONE)
        ? test_json["dataset_name"].as_string() : "unknown";
    auto X_test = test_json["X_test"].as_2d_double_array();
    auto y_test = test_json["y_test"].as_int_array();
    bool has_labels = !y_test.empty();

    int N = (int)X_test.size();
    if (max_series > 0 && max_series < N) N = max_series;
    if (N <= 0) {
        std::cerr << "ERROR: no series to score\n";
        return 1;
    }

    std::cout << "Dataset: " << dataset_name << "\n";
    std::cout << "Series to score: " << N << " / " << X_test.size() << "\n";
    std::cout << "Server: " << host << ":" << port << "\n";
    std::cout << "==========================================\n";

    int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { perror("socket"); return 1; }
    timeval tv{};
    tv.tv_sec = 30;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    sockaddr_in server{};
    server.sin_family = AF_INET;
    server.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &server.sin_addr) != 1) {
        std::cerr << "ERROR: invalid host " << host << "\n";
        return 1;
    }

    std::ofstream csv;
    if (!csv_out.empty()) {
        csv.open(csv_out);
        if (!csv.is_open()) {
            std::cerr << "ERROR: cannot open csv_out " << csv_out << "\n";
            return 1;
        }
        csv << "series_id,final_pred,label,correct,final_rtt_ms,mean_step_rtt_ms\n";
    }

    std::vector<double> all_step_rtts;
    std::vector<double> final_rtts;
    all_step_rtts.reserve((size_t)N * (X_test.empty() ? 0 : X_test[0].size()));
    final_rtts.reserve((size_t)N);

    int correct = 0;
    uint64_t total_samples = 0;

    auto wall0 = std::chrono::steady_clock::now();

    for (int i = 0; i < N; i++) {
        if (!send_reset(sock, server)) {
            std::cerr << "ERROR: RESET failed at series " << i << "\n";
            ::close(sock);
            return 1;
        }

        const auto& series = X_test[(size_t)i];
        const int L = (int)series.size();
        int32_t final_pred = 0;
        double final_rtt = 0.0;
        double sum_step_rtt = 0.0;

        for (int t = 0; t < L; t++) {
            float sample_f32 = static_cast<float>(series[(size_t)t]);
            int32_t pred = 0;
            double rtt_ms = 0.0;
            if (!send_sample_recv_pred(sock, server, sample_f32, pred, rtt_ms)) {
                std::cerr << "ERROR: sample failed series=" << i << " t=" << t << "\n";
                ::close(sock);
                return 1;
            }
            all_step_rtts.push_back(rtt_ms);
            sum_step_rtt += rtt_ms;
            total_samples++;
            if (t == L - 1) {
                final_pred = pred;
                final_rtt = rtt_ms;
            }
        }
        final_rtts.push_back(final_rtt);

        int label = (has_labels && i < (int)y_test.size()) ? y_test[(size_t)i] : -1;
        int ok = (has_labels && final_pred == label) ? 1 : 0;
        if (ok) correct++;
        double mean_step = (L > 0) ? (sum_step_rtt / (double)L) : 0.0;

        if (csv.is_open()) {
            csv << i << "," << final_pred << "," << label << "," << ok << ","
                << std::fixed << std::setprecision(6) << final_rtt << ","
                << mean_step << "\n";
        }

        if (((i + 1) % 500) == 0 || (i + 1) == N) {
            std::cout << "  progress: " << (i + 1) << "/" << N
                      << " correct_so_far=" << correct << std::endl;
        }
    }

    auto wall1 = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(wall1 - wall0).count();
    ::close(sock);
    if (csv.is_open()) csv.close();

    double accuracy_pct = has_labels ? (100.0 * (double)correct / (double)N) : 0.0;
    double series_per_s = (wall_s > 0) ? ((double)N / wall_s) : 0.0;
    double samples_per_s = (wall_s > 0) ? ((double)total_samples / wall_s) : 0.0;

    RttStats final_stats = compute_rtt_stats(final_rtts);
    RttStats step_stats = compute_rtt_stats(all_step_rtts);

    std::cout << "\n========== RESULTS (GPU UDP streaming, naive, final-after-L) ==========\n";
    std::cout << "Dataset: " << dataset_name << "\n";
    std::cout << "Series scored: " << N << "\n";
    if (has_labels) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Accuracy: " << accuracy_pct << "% (" << correct << "/" << N << ")\n";
    } else {
        std::cout << "Accuracy: n/a (no y_test)\n";
    }
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Throughput: " << series_per_s << " series/sec  ("
              << samples_per_s << " samples/sec)\n";
    std::cout << "Wall clock: " << wall_s << " s  total_samples=" << total_samples << "\n";
    std::cout << "\nFinal-pred RTT / per-sample RTT distribution (ms):\n";
    std::cout << "  (1) Final-pred RTT = RTT of the L-th sample only, across series\n";
    std::cout << "  (2) Per-sample RTT = all step RTTs across all series\n";
    print_rtt_block("  Final-pred RTT", final_stats);
    print_rtt_block("  Per-sample RTT", step_stats);
    if (!csv_out.empty())
        std::cout << "CSV written: " << csv_out << "\n";
    std::cout << "=======================================================================\n";

    return 0;
}
