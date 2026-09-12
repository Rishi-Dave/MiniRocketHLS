// MiniRocket UDP Client — correctness testing against the GPU streaming server
//==============================================================================
// Sends one float32 sample at a time from a test series; receives int32 class
// predictions. After L samples the window is full and the prediction is
// comparable to offline whole-series MiniRocket inference.
//
// Compile:
//   g++ -O3 -std=c++17 -o minirocket_udp_client minirocket_udp_client.cpp
//
// Usage:
//   ./minirocket_udp_client <test_data.json> [server_host] [server_port] [series_index]
//   defaults: 127.0.0.1 9000 series_index=0
//==============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cassert>
#include <iomanip>
#include <cstdint>
#include <chrono>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

// ============================================================
// Minimal JSON parser (same style as GPU / CPU inference files)
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <test_data.json> [server_host] [server_port] [series_index]"
                  << std::endl;
        std::cerr << "  defaults: 127.0.0.1 9000 series_index=0" << std::endl;
        return 1;
    }

    std::string test_path = argv[1];
    std::string host = (argc > 2) ? argv[2] : "127.0.0.1";
    int port = (argc > 3) ? std::stoi(argv[3]) : 9000;
    int series_index = (argc > 4) ? std::stoi(argv[4]) : 0;

    std::cout << "Loading test data from: " << test_path << std::endl;
    auto test_json = load_json(test_path);

    std::string dataset_name = (test_json["dataset_name"].type != JsonValue::NONE)
        ? test_json["dataset_name"].as_string() : "unknown";

    auto X_test_2d = test_json["X_test"].as_2d_double_array();
    auto y_test = test_json["y_test"].as_int_array();
    bool has_labels = !y_test.empty();

    int series_length = (test_json["series_length"].type != JsonValue::NONE)
        ? (int)test_json["series_length"].as_number()
        : (test_json["time_series_length"].type != JsonValue::NONE)
            ? (int)test_json["time_series_length"].as_number()
            : (int)X_test_2d[0].size();

    if (series_index < 0 || series_index >= (int)X_test_2d.size()) {
        std::cerr << "ERROR: series_index " << series_index
                  << " out of range [0, " << X_test_2d.size() << ")" << std::endl;
        return 1;
    }

    const std::vector<double>& series = X_test_2d[series_index];
    const int L = (int)series.size();
    if (L != series_length) {
        std::cerr << "WARNING: series.size()=" << L
                  << " vs metadata series_length=" << series_length
                  << " — using series.size()" << std::endl;
    }

    std::cout << "Dataset: " << dataset_name << std::endl;
    std::cout << "  series_index: " << series_index << " / " << X_test_2d.size() << std::endl;
    std::cout << "  L (samples to stream): " << L << std::endl;
    std::cout << "  server: " << host << ":" << port << std::endl;
    std::cout << "Note: early windows are zero-padded; only the prediction after"
              << " the L-th sample is comparable to offline whole-series inference."
              << std::endl;

    int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("socket");
        return 1;
    }

    // Reasonable receive timeout (5s)
    timeval tv{};
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    if (setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        perror("setsockopt SO_RCVTIMEO");
        return 1;
    }

    sockaddr_in server{};
    server.sin_family = AF_INET;
    server.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &server.sin_addr) != 1) {
        std::cerr << "ERROR: invalid server_host " << host << std::endl;
        return 1;
    }

    std::vector<int> predictions;
    predictions.reserve((size_t)L);

    // Stream the whole series; record prediction after each sample.
    // The prediction after the L-th sample is the offline-equivalent result.
    for (int i = 0; i < L; i++) {
        float sample_f32 = static_cast<float>(series[(size_t)i]);
        uint8_t out_buf[4];
        std::memcpy(out_buf, &sample_f32, 4);

        ssize_t sn = sendto(sock, out_buf, 4, 0,
                            reinterpret_cast<sockaddr*>(&server), sizeof(server));
        if (sn != 4) {
            perror("sendto");
            ::close(sock);
            return 1;
        }

        uint8_t in_buf[4];
        sockaddr_in from{};
        socklen_t from_len = sizeof(from);
        ssize_t rn = recvfrom(sock, in_buf, 4, 0,
                              reinterpret_cast<sockaddr*>(&from), &from_len);
        if (rn != 4) {
            if (rn < 0)
                perror("recvfrom (timeout or error — is the server running?)");
            else
                std::cerr << "ERROR: expected 4-byte int32 reply, got " << rn << std::endl;
            ::close(sock);
            return 1;
        }

        int32_t pred_i32 = 0;
        std::memcpy(&pred_i32, in_buf, 4);
        int pred = static_cast<int>(pred_i32);
        predictions.push_back(pred);

        // Per-step progress for first few and last
        bool show = (i < 5) || (i >= L - 3) || ((i + 1) % 100 == 0);
        if (show) {
            std::cout << "  step " << (i + 1) << "/" << L
                      << " sample=" << std::fixed << std::setprecision(6) << sample_f32
                      << " pred=" << pred << std::endl;
        }
    }

    ::close(sock);

    std::cout << "\n========== STREAM COMPLETE ==========" << std::endl;
    std::cout << "All predictions (" << predictions.size() << "):" << std::endl;
    for (size_t i = 0; i < predictions.size(); i++) {
        if (i > 0) std::cout << " ";
        std::cout << predictions[i];
        if ((i + 1) % 40 == 0) std::cout << "\n";
    }
    std::cout << std::endl;

    int final_pred = predictions.back();
    std::cout << "Final prediction (after L=" << L << " samples, offline-equivalent): "
              << final_pred << std::endl;

    if (has_labels && series_index < (int)y_test.size()) {
        int label = y_test[series_index];
        bool ok = (final_pred == label);
        std::cout << "Ground-truth label: " << label << std::endl;
        std::cout << "Final prediction vs label: " << (ok ? "MATCH" : "MISMATCH")
                  << " (accuracy of final only: " << (ok ? "1/1" : "0/1") << ")"
                  << std::endl;
        std::cout << "Note: early windows are zero-padded so only the last prediction"
                  << " after L samples is comparable to offline whole-series inference."
                  << std::endl;
    } else {
        std::cout << "No y_test labels available for accuracy check." << std::endl;
    }
    std::cout << "=====================================" << std::endl;

    return 0;
}
