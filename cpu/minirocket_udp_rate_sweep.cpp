// MiniRocket UDP rate-sweep harness — knee/plateau curve (NETWORK_RESULTS style)
//==============================================================================
// Offered-pps vs processed-pps / loss% / RTT percentiles. Hypothesis (banner):
// GPU plateau expected BETWEEN CPU (~paper Table V 11.0 ms RTT / ~5k pps
 // loss-free in NETWORK_RESULTS CPU baselines) and FPGA (paper 4.12 ms RTT /
// multi-million pps F2F). F2F FPGA pps is a different (much higher) regime.
//
// Modes:
//   rr (request-response, Table V style): send sample, wait reply; measure RTT.
//     Sustained rate ≈ 1/mean_RTT. Good for low-rate / RTT comparison.
//   openloop (default): pace sends at target offered_pps; recv non-blocking into
//     a queue. After dwell: offered/received/loss%/processed_pps + RTT stats.
//
// Sequence matching WITHOUT protocol change (openloop RTT):
//   Protocol only carries float32 payload — no sequence number. We approximate
 //   RTT by keeping send timestamps in a ring buffer and pairing replies in
 //   order (FIFO). ASSUMPTION: in-order localhost UDP delivery. Documented here
 //   and in the banner; do not trust openloop RTT on lossy / reordering paths.
//
// Before each rate point: RESET window once, then stream `samples` datapoints
 // by cycling through flattened X_test.
//
// Compile:
 //   g++ -O3 -march=native -std=c++17 -o minirocket_udp_rate_sweep \
 //       minirocket_udp_rate_sweep.cpp
//
// Usage:
//   ./minirocket_udp_rate_sweep <test_data.json> [host] [port] [csv_out]
 //   Optional argv after csv:
 //     --rates 100,500,1000,2000,5000,10000,20000
 //     --samples 5000
 //     --mode rr|openloop
 //     --warmup 200
//==============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <thread>
#include <iomanip>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

static constexpr uint32_t kCtrlMagic = 0x4D525354u;
static constexpr uint32_t kCmdReset  = 1u;

// ============================================================
// Minimal JSON parser
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
    double mean = 0, p50 = 0, p95 = 0, p99 = 0;
};

static RttStats compute_rtt_stats(std::vector<double> v) {
    RttStats s;
    if (v.empty()) return s;
    double sum = 0.0;
    for (double x : v) sum += x;
    s.mean = sum / (double)v.size();
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

static std::vector<double> parse_rates(const std::string& s) {
    std::vector<double> rates;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) rates.push_back(std::stod(tok));
    }
    return rates;
}

static bool send_reset_blocking(int sock, const sockaddr_in& server) {
    // Temporarily blocking recv for RESET ACK
    uint8_t pkt[8];
    uint32_t magic = kCtrlMagic, cmd = kCmdReset;
    std::memcpy(pkt + 0, &magic, 4);
    std::memcpy(pkt + 4, &cmd, 4);
    if (sendto(sock, pkt, 8, 0,
               reinterpret_cast<const sockaddr*>(&server), sizeof(server)) != 8) {
        perror("sendto RESET");
        return false;
    }
    uint8_t in_buf[4];
    sockaddr_in from{};
    socklen_t from_len = sizeof(from);
    // Wait up to 5s
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(sock, &rfds);
    timeval tv{};
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    int sel = select(sock + 1, &rfds, nullptr, nullptr, &tv);
    if (sel <= 0) {
        std::cerr << "ERROR: RESET ACK timeout\n";
        return false;
    }
    ssize_t rn = recvfrom(sock, in_buf, 4, 0,
                          reinterpret_cast<sockaddr*>(&from), &from_len);
    if (rn != 4) {
        perror("recvfrom RESET");
        return false;
    }
    return true;
}

static void set_nonblocking(int sock, bool nb) {
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags < 0) return;
    if (nb) fcntl(sock, F_SETFL, flags | O_NONBLOCK);
    else    fcntl(sock, F_SETFL, flags & ~O_NONBLOCK);
}

static void drain_socket(int sock) {
    uint8_t buf[64];
    while (true) {
        ssize_t n = recvfrom(sock, buf, sizeof(buf), MSG_DONTWAIT, nullptr, nullptr);
        if (n < 0) break;
    }
}

struct SweepRow {
    double rate_target_pps = 0;
    std::string mode;
    uint64_t offered = 0;
    uint64_t received = 0;
    double loss_pct = 0;
    double processed_pps = 0;
    RttStats rtt;
};

static SweepRow run_rr(int sock, const sockaddr_in& server,
                       const std::vector<float>& stream,
                       double /*rate_target*/, int n_samples, int warmup) {
    SweepRow row;
    row.mode = "rr";
    set_nonblocking(sock, false);
    timeval tv{};
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    std::vector<double> rtts;
    rtts.reserve((size_t)n_samples);

    auto wall0 = std::chrono::steady_clock::now();
    uint64_t offered = 0, received = 0;

    for (int i = 0; i < n_samples; i++) {
        float sample = stream[(size_t)i % stream.size()];
        uint8_t out_buf[4];
        std::memcpy(out_buf, &sample, 4);

        auto t0 = std::chrono::steady_clock::now();
        if (sendto(sock, out_buf, 4, 0,
                   reinterpret_cast<const sockaddr*>(&server), sizeof(server)) != 4) {
            perror("sendto");
            break;
        }
        offered++;

        uint8_t in_buf[4];
        sockaddr_in from{};
        socklen_t from_len = sizeof(from);
        ssize_t rn = recvfrom(sock, in_buf, 4, 0,
                              reinterpret_cast<sockaddr*>(&from), &from_len);
        auto t1 = std::chrono::steady_clock::now();
        if (rn == 4) {
            received++;
            if (i >= warmup) {
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                rtts.push_back(ms);
            }
        }
    }
    auto wall1 = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(wall1 - wall0).count();

    row.offered = offered;
    row.received = received;
    row.loss_pct = (offered > 0) ? (100.0 * (1.0 - (double)received / (double)offered)) : 0.0;
    row.processed_pps = (wall_s > 0) ? ((double)received / wall_s) : 0.0;
    row.rtt = compute_rtt_stats(rtts);
    // For rr, sustained ≈ processed; also ≈ 1000/mean_rtt
    row.rate_target_pps = row.processed_pps;
    return row;
}

static SweepRow run_openloop(int sock, const sockaddr_in& server,
                             const std::vector<float>& stream,
                             double rate_target, int n_samples, int warmup) {
    SweepRow row;
    row.mode = "openloop";
    row.rate_target_pps = rate_target;

    // FIFO send-time ring for in-order RTT pairing (localhost assumption)
    const size_t RING = 1 << 16;
    std::vector<std::chrono::steady_clock::time_point> send_ts(RING);
    size_t send_idx = 0;
    size_t recv_idx = 0;  // next unmatched send

    set_nonblocking(sock, true);
    drain_socket(sock);

    std::vector<double> rtts;
    rtts.reserve((size_t)n_samples);

    const double period_s = (rate_target > 0) ? (1.0 / rate_target) : 0.0;
    auto wall0 = std::chrono::steady_clock::now();
    auto next_send = wall0;

    uint64_t offered = 0, received = 0;

    auto try_recv = [&]() {
        while (true) {
            uint8_t in_buf[4];
            ssize_t rn = recvfrom(sock, in_buf, 4, MSG_DONTWAIT, nullptr, nullptr);
            if (rn < 0) break;
            if (rn != 4) continue;
            auto tnow = std::chrono::steady_clock::now();
            if (recv_idx < send_idx) {
                size_t slot = recv_idx % RING;
                // Only trust if ring hasn't wrapped over this slot
                if (send_idx - recv_idx < RING) {
                    double ms = std::chrono::duration<double, std::milli>(
                        tnow - send_ts[slot]).count();
                    // skip warmup sends
                    if ((int)recv_idx >= warmup && ms >= 0.0 && ms < 60000.0)
                        rtts.push_back(ms);
                }
                recv_idx++;
                received++;
            } else {
                // orphan reply
                received++;
            }
        }
    };

    for (int i = 0; i < n_samples; i++) {
        // Pace
        auto now = std::chrono::steady_clock::now();
        if (period_s > 0) {
            if (now < next_send) {
                // brief poll while waiting
                try_recv();
                now = std::chrono::steady_clock::now();
                if (now < next_send)
                    std::this_thread::sleep_until(next_send);
            }
            next_send += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(period_s));
            // If we fell behind, don't accumulate infinite debt — catch up
            now = std::chrono::steady_clock::now();
            if (next_send + std::chrono::milliseconds(100) < now)
                next_send = now;
        }

        float sample = stream[(size_t)i % stream.size()];
        uint8_t out_buf[4];
        std::memcpy(out_buf, &sample, 4);

        auto tsend = std::chrono::steady_clock::now();
        ssize_t sn = sendto(sock, out_buf, 4, 0,
                            reinterpret_cast<const sockaddr*>(&server), sizeof(server));
        if (sn == 4) {
            send_ts[send_idx % RING] = tsend;
            send_idx++;
            offered++;
        }
        try_recv();
    }

    // Drain remaining replies briefly
    auto drain_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (std::chrono::steady_clock::now() < drain_deadline) {
        try_recv();
        if (received >= offered) break;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    auto wall1 = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(wall1 - wall0).count();

    row.offered = offered;
    row.received = received;
    row.loss_pct = (offered > 0) ? (100.0 * (1.0 - (double)received / (double)offered)) : 0.0;
    row.processed_pps = (wall_s > 0) ? ((double)received / wall_s) : 0.0;
    row.rtt = compute_rtt_stats(rtts);
    return row;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <test_data.json> [host] [port] [csv_out]\n"
                  << "  --rates 100,500,...  --samples 5000  --mode rr|openloop  --warmup 200\n";
        return 1;
    }

    std::string test_path = argv[1];
    std::string host = "127.0.0.1";
    int port = 9000;
    std::string csv_out;
    std::vector<double> rates = {100, 500, 1000, 2000, 5000, 10000, 20000};
    int n_samples = 5000;
    std::string mode = "openloop";
    int warmup = 200;

    // Positional: host port csv, then flags
    int positional = 0;
    for (int a = 2; a < argc; a++) {
        std::string arg = argv[a];
        if (arg == "--rates" && a + 1 < argc) {
            rates = parse_rates(argv[++a]);
        } else if (arg == "--samples" && a + 1 < argc) {
            n_samples = std::stoi(argv[++a]);
        } else if (arg == "--mode" && a + 1 < argc) {
            mode = argv[++a];
        } else if (arg == "--warmup" && a + 1 < argc) {
            warmup = std::stoi(argv[++a]);
        } else if (arg.rfind("--", 0) == 0) {
            std::cerr << "Unknown flag: " << arg << "\n";
            return 1;
        } else {
            if (positional == 0) host = arg;
            else if (positional == 1) port = std::stoi(arg);
            else if (positional == 2) csv_out = arg;
            positional++;
        }
    }

    std::cout << "========== MiniRocket UDP Rate Sweep ==========\n";
    std::cout << "HYPOTHESIS: GPU plateau expected BETWEEN\n"
              << "  CPU  (~paper Table V 11.0 ms RTT / ~5k pps loss-free CPU baselines)\n"
              << "  FPGA (paper Table V 4.12 ms RTT / multi-million pps F2F)\n"
              << "F2F FPGA pps is a different (much higher) regime than host UDP.\n";
    std::cout << "Openloop RTT assumes in-order localhost UDP (FIFO send-ts pairing;\n"
              << "protocol has no sequence number).\n";
    std::cout << "Mode: " << mode << "  samples/point: " << n_samples
              << "  warmup: " << warmup << "\n";
    std::cout << "Server: " << host << ":" << port << "\n";
    std::cout << "================================================\n";

    auto test_json = load_json(test_path);
    auto X_test = test_json["X_test"].as_2d_double_array();
    if (X_test.empty()) {
        std::cerr << "ERROR: empty X_test\n";
        return 1;
    }
    // Flatten / cycle series into a sample stream
    std::vector<float> stream;
    for (auto& row : X_test)
        for (double v : row)
            stream.push_back(static_cast<float>(v));
    if (stream.empty()) {
        std::cerr << "ERROR: no samples in X_test\n";
        return 1;
    }
    std::cout << "Stream pool: " << stream.size() << " samples from "
              << X_test.size() << " series\n";

    int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { perror("socket"); return 1; }
    // Enlarge buffers a bit for openloop bursts
    int bufsz = 4 * 1024 * 1024;
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &bufsz, sizeof(bufsz));
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &bufsz, sizeof(bufsz));

    sockaddr_in server{};
    server.sin_family = AF_INET;
    server.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &server.sin_addr) != 1) {
        std::cerr << "ERROR: invalid host\n";
        return 1;
    }

    std::ofstream csv;
    if (!csv_out.empty()) {
        csv.open(csv_out);
        if (!csv.is_open()) {
            std::cerr << "ERROR: cannot open " << csv_out << "\n";
            return 1;
        }
        csv << "rate_target_pps,mode,offered,received,loss_pct,processed_pps,"
               "rtt_mean_ms,rtt_p50_ms,rtt_p95_ms,rtt_p99_ms\n";
    }

    std::cout << std::left
              << std::setw(12) << "target_pps"
              << std::setw(10) << "mode"
              << std::setw(10) << "offered"
              << std::setw(10) << "recv"
              << std::setw(10) << "loss%"
              << std::setw(12) << "proc_pps"
              << std::setw(10) << "rtt_mean"
              << std::setw(10) << "rtt_p50"
              << std::setw(10) << "rtt_p95"
              << std::setw(10) << "rtt_p99"
              << "\n";

    // For rr mode, rates list is ignored as "targets" — we run once (or once
    // per entry with same semantics). Still iterate: first entry drives rr once
    // if rates empty fallthrough; for clarity run rr once using rates[0] label.
    if (mode == "rr") {
        set_nonblocking(sock, false);
        if (!send_reset_blocking(sock, server)) {
            std::cerr << "RESET failed\n";
            return 1;
        }
        drain_socket(sock);
        SweepRow row = run_rr(sock, server, stream, /*unused*/0, n_samples, warmup);
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(12) << row.processed_pps
                  << std::setw(10) << row.mode
                  << std::setw(10) << row.offered
                  << std::setw(10) << row.received
                  << std::setw(10) << row.loss_pct
                  << std::setw(12) << row.processed_pps
                  << std::setw(10) << row.rtt.mean
                  << std::setw(10) << row.rtt.p50
                  << std::setw(10) << row.rtt.p95
                  << std::setw(10) << row.rtt.p99
                  << "\n";
        if (csv.is_open()) {
            csv << std::fixed << std::setprecision(6)
                << row.processed_pps << "," << row.mode << ","
                << row.offered << "," << row.received << ","
                << row.loss_pct << "," << row.processed_pps << ","
                << row.rtt.mean << "," << row.rtt.p50 << ","
                << row.rtt.p95 << "," << row.rtt.p99 << "\n";
        }
        std::cout << "\n(rr note: sustained ≈ 1/mean_RTT; compare to Table V "
                     "CPU 11.0 ms / FPGA 4.12 ms)\n";
    } else {
        for (double rate : rates) {
            set_nonblocking(sock, false);
            if (!send_reset_blocking(sock, server)) {
                std::cerr << "RESET failed at rate " << rate << "\n";
                return 1;
            }
            drain_socket(sock);
            SweepRow row = run_openloop(sock, server, stream, rate, n_samples, warmup);
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(12) << row.rate_target_pps
                      << std::setw(10) << row.mode
                      << std::setw(10) << row.offered
                      << std::setw(10) << row.received
                      << std::setw(10) << row.loss_pct
                      << std::setw(12) << row.processed_pps
                      << std::setw(10) << row.rtt.mean
                      << std::setw(10) << row.rtt.p50
                      << std::setw(10) << row.rtt.p95
                      << std::setw(10) << row.rtt.p99
                      << "\n";
            if (csv.is_open()) {
                csv << std::fixed << std::setprecision(6)
                    << row.rate_target_pps << "," << row.mode << ","
                    << row.offered << "," << row.received << ","
                    << row.loss_pct << "," << row.processed_pps << ","
                    << row.rtt.mean << "," << row.rtt.p50 << ","
                    << row.rtt.p95 << "," << row.rtt.p99 << "\n";
            }
        }
    }

    if (csv.is_open()) {
        std::cout << "CSV written: " << csv_out << "\n";
        csv.close();
    }
    ::close(sock);
    return 0;
}
