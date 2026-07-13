// tp_client.cpp — C++ port of throughput.py.
//
// Single-process, two-thread UDP rate-sweep client:
//   sender thread paces sendto() to a target pps; receiver thread drains
//   any replies (counts but doesn't block sender). Writes sweep.csv +
//   summary.json on exit.
//
// Build:
//   g++ -std=c++17 -O2 -pthread -o tp_client tp_client.cpp

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/stat.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

struct Args {
    std::string fpga_ip   = "192.168.40.10";
    uint16_t    fpga_port = 62177;
    uint16_t    listen_port = 62178;
    double      rate_min  = 1024;
    double      rate_max  = 262144;
    int         rate_steps = 8;
    double      duration_s = 2.0;
    double      abort_loss_pct = 80.0;
    int         pkt_bytes = 64;
    std::string out;
};

struct StepResult {
    double  target_pps;
    long long sent;
    long long recv;
    double  achieved_pps;
    double  loss_pct;
    double  bps;
};

static StepResult run_step(int sock, sockaddr_in fpga, double target_pps,
                           double duration_s, int pkt_bytes) {
    using clk = std::chrono::steady_clock;
    long long interval_ns = (target_pps > 0)
        ? static_cast<long long>(1e9 / target_pps) : 0;

    std::vector<char> payload(pkt_bytes, 0);
    char rxbuf[2048];

    std::atomic<long long> sent_n{0};
    std::atomic<long long> recv_n{0};
    std::atomic<bool> stop{false};

    std::thread rx([&]() {
        struct timeval tv{0, 50000};  // 50 ms recv timeout
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        while (!stop.load(std::memory_order_relaxed)) {
            sockaddr_in src{}; socklen_t sl = sizeof(src);
            ssize_t n = ::recvfrom(sock, rxbuf, sizeof(rxbuf), 0,
                                   reinterpret_cast<sockaddr*>(&src), &sl);
            if (n > 0) {
                recv_n.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    auto t_start = clk::now();
    auto t_end   = t_start + std::chrono::nanoseconds(
                                static_cast<long long>(duration_s * 1e9));
    auto next    = t_start;
    while (clk::now() < t_end) {
        auto now = clk::now();
        if (now < next) {
            auto slack = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            next - now).count();
            if (slack > 50000) {
                std::this_thread::sleep_for(
                    std::chrono::nanoseconds(slack - 50000));
            }
            // spin remainder
            while (clk::now() < next) { /* busy-wait */ }
        }
        ssize_t s = ::sendto(sock, payload.data(), pkt_bytes, 0,
                             reinterpret_cast<sockaddr*>(&fpga), sizeof(fpga));
        if (s > 0) {
            sent_n.fetch_add(1, std::memory_order_relaxed);
        }
        next += std::chrono::nanoseconds(interval_ns);
    }
    // Let recv thread drain straggler echoes for ~50 ms
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop.store(true, std::memory_order_relaxed);
    rx.join();

    StepResult r;
    r.target_pps   = target_pps;
    r.sent         = sent_n.load();
    r.recv         = recv_n.load();
    r.achieved_pps = r.sent / duration_s;
    r.loss_pct     = 100.0 * (r.sent - r.recv) / std::max(r.sent, 1LL);
    r.bps          = r.achieved_pps * pkt_bytes * 8.0;
    return r;
}

int main(int argc, char** argv) {
    Args a;
    static option opts[] = {
        {"fpga-ip",        required_argument, nullptr, 'i'},
        {"fpga-port",      required_argument, nullptr, 'p'},
        {"listen-port",    required_argument, nullptr, 'l'},
        {"rate-min",       required_argument, nullptr, 'a'},
        {"rate-max",       required_argument, nullptr, 'b'},
        {"rate-steps",     required_argument, nullptr, 's'},
        {"duration-s",     required_argument, nullptr, 'd'},
        {"abort-loss-pct", required_argument, nullptr, 'L'},
        {"packet-bytes",   required_argument, nullptr, 'B'},
        {"out",            required_argument, nullptr, 'o'},
        {nullptr, 0, nullptr, 0}
    };
    int c;
    while ((c = getopt_long(argc, argv, "i:p:l:a:b:s:d:L:B:o:", opts, nullptr)) != -1) {
        switch (c) {
            case 'i': a.fpga_ip = optarg; break;
            case 'p': a.fpga_port = static_cast<uint16_t>(std::stoi(optarg)); break;
            case 'l': a.listen_port = static_cast<uint16_t>(std::stoi(optarg)); break;
            case 'a': a.rate_min = std::stod(optarg); break;
            case 'b': a.rate_max = std::stod(optarg); break;
            case 's': a.rate_steps = std::stoi(optarg); break;
            case 'd': a.duration_s = std::stod(optarg); break;
            case 'L': a.abort_loss_pct = std::stod(optarg); break;
            case 'B': a.pkt_bytes = std::stoi(optarg); break;
            case 'o': a.out = optarg; break;
            default: std::cerr << "Unknown opt\n"; return 2;
        }
    }
    if (a.out.empty()) { std::cerr << "--out required\n"; return 2; }

    int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { perror("socket"); return 1; }
    int big = 64 << 20;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &big, sizeof(big));
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &big, sizeof(big));

    sockaddr_in local{}; local.sin_family = AF_INET;
    local.sin_addr.s_addr = INADDR_ANY;
    local.sin_port = htons(a.listen_port);
    if (::bind(sock, reinterpret_cast<sockaddr*>(&local), sizeof(local)) < 0) {
        perror("bind");
        return 1;
    }

    sockaddr_in fpga{}; fpga.sin_family = AF_INET;
    fpga.sin_port = htons(a.fpga_port);
    inet_pton(AF_INET, a.fpga_ip.c_str(), &fpga.sin_addr);

    std::vector<double> rates;
    if (a.rate_steps < 2) {
        rates.push_back(a.rate_min);
    } else {
        double ratio = a.rate_max / a.rate_min;
        for (int i = 0; i < a.rate_steps; ++i) {
            double t = static_cast<double>(i) / (a.rate_steps - 1);
            rates.push_back(a.rate_min * std::pow(ratio, t));
        }
    }

    ::mkdir(a.out.c_str(), 0755);
    std::vector<StepResult> rows;
    rows.reserve(rates.size());
    int high_loss_streak = 0;
    for (double tgt : rates) {
        StepResult r = run_step(sock, fpga, tgt, a.duration_s, a.pkt_bytes);
        rows.push_back(r);
        std::fprintf(stderr,
            "[tp] tgt=%10.0fpps sent=%8lld ack=%8lld loss=%5.1f%% achieved=%10.0fpps  bps=%.1fM\n",
            r.target_pps, r.sent, r.recv, r.loss_pct, r.achieved_pps, r.bps / 1e6);
        if (r.loss_pct > a.abort_loss_pct) {
            ++high_loss_streak;
            if (high_loss_streak >= 2) {
                std::fprintf(stderr, "[tp] high loss two steps in a row -- stopping sweep.\n");
                break;
            }
        } else {
            high_loss_streak = 0;
        }
    }

    {
        std::ofstream f(a.out + "/sweep.csv");
        f << "target_pps,sent,recv_acks,achieved_pps,loss_pct,bps\n";
        for (auto& r : rows) {
            f << r.target_pps << "," << r.sent << "," << r.recv << ","
              << r.achieved_pps << "," << r.loss_pct << "," << r.bps << "\n";
        }
    }
    {
        std::ofstream f(a.out + "/summary.json");
        f << "{\n  \"fpga_ip\": \"" << a.fpga_ip << "\",\n"
          << "  \"fpga_port\": " << a.fpga_port << ",\n"
          << "  \"packet_bytes\": " << a.pkt_bytes << ",\n"
          << "  \"rates\": [\n";
        for (size_t i = 0; i < rows.size(); ++i) {
            auto& r = rows[i];
            f << "    {\"target_pps\": " << r.target_pps
              << ", \"sent\": " << r.sent
              << ", \"recv_acks\": " << r.recv
              << ", \"achieved_pps\": " << r.achieved_pps
              << ", \"loss_pct\": " << r.loss_pct
              << ", \"bps\": " << r.bps << "}";
            if (i + 1 < rows.size()) f << ",";
            f << "\n";
        }
        f << "  ]\n}\n";
    }
    ::close(sock);
    return 0;
}
