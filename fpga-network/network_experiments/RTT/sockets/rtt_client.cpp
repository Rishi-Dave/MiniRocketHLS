// Sender-side RTT (single in-flight). Same wire pattern as the Python
// version: bind to listen_port so src_port matches SocketTable[0].theirPort.
//
// Adapted from pyuvaraj37/nlp:fpga-network/xrt_host_api/network_experiments/
//   RTT/sockets/rtt_client.cpp
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <getopt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Args {
    std::string fpga_ip      = "192.168.40.10";
    uint16_t    fpga_port    = 62177;
    uint16_t    listen_port  = 62178;
    int         packets      = 1000;
    int         warmup       = 50;
    int         pkt_bytes    = 64;
    double      timeout_s    = 1.0;
    std::string out;
};

int main(int argc, char** argv) {
    Args a;
    static option opts[] = {
        {"fpga-ip",      required_argument, nullptr, 'i'},
        {"fpga-port",    required_argument, nullptr, 'p'},
        {"listen-port",  required_argument, nullptr, 'l'},
        {"packets",      required_argument, nullptr, 'n'},
        {"warmup",       required_argument, nullptr, 'w'},
        {"packet-bytes", required_argument, nullptr, 'b'},
        {"timeout-s",    required_argument, nullptr, 't'},
        {"out",          required_argument, nullptr, 'o'},
        {nullptr, 0, nullptr, 0}
    };
    int c;
    while ((c = getopt_long(argc, argv, "i:p:l:n:w:b:t:o:", opts, nullptr)) != -1) {
        switch (c) {
            case 'i': a.fpga_ip = optarg; break;
            case 'p': a.fpga_port = static_cast<uint16_t>(std::stoi(optarg)); break;
            case 'l': a.listen_port = static_cast<uint16_t>(std::stoi(optarg)); break;
            case 'n': a.packets = std::stoi(optarg); break;
            case 'w': a.warmup = std::stoi(optarg); break;
            case 'b': a.pkt_bytes = std::stoi(optarg); break;
            case 't': a.timeout_s = std::stod(optarg); break;
            case 'o': a.out = optarg; break;
            default:
                std::cerr << "Unknown option\n";
                return 2;
        }
    }
    if (a.out.empty()) {
        std::cerr << "--out required\n";
        return 2;
    }

    int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { perror("socket"); return 1; }
    int bufsz = 16 << 20;
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &bufsz, sizeof(bufsz));
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &bufsz, sizeof(bufsz));

    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_addr.s_addr = INADDR_ANY;
    local.sin_port = htons(a.listen_port);
    if (::bind(sock, reinterpret_cast<sockaddr*>(&local), sizeof(local)) < 0) {
        perror("bind");
        return 1;
    }

    timeval tv{};
    tv.tv_sec  = static_cast<long>(a.timeout_s);
    tv.tv_usec = static_cast<long>((a.timeout_s - tv.tv_sec) * 1e6);
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    sockaddr_in fpga{};
    fpga.sin_family = AF_INET;
    fpga.sin_port = htons(a.fpga_port);
    inet_pton(AF_INET, a.fpga_ip.c_str(), &fpga.sin_addr);

    std::vector<char> payload(a.pkt_bytes, 0);
    std::vector<char> rxbuf(2048);

    struct Row {
        int seq;
        long long t_send_ns;
        long long t_recv_ns;
        double rtt_us;
    };
    std::vector<Row> rows;
    rows.reserve(a.packets);

    using clk = std::chrono::high_resolution_clock;
    int losses = 0;
    for (int i = 0; i < a.packets + a.warmup; ++i) {
        std::memcpy(payload.data(), &i, sizeof(int));
        auto t0 = clk::now();
        ::sendto(sock, payload.data(), a.pkt_bytes, 0,
                 reinterpret_cast<sockaddr*>(&fpga), sizeof(fpga));
        sockaddr_in src{};
        socklen_t sl = sizeof(src);
        ssize_t n = ::recvfrom(sock, rxbuf.data(), rxbuf.size(), 0,
                               reinterpret_cast<sockaddr*>(&src), &sl);
        auto t1 = clk::now();
        if (i < a.warmup) continue;
        long long tns0 = std::chrono::duration_cast<std::chrono::nanoseconds>(
            t0.time_since_epoch()).count();
        long long tns1 = (n > 0)
            ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                  t1.time_since_epoch()).count()
            : -1;
        double rtt = (n > 0) ? (tns1 - tns0) / 1000.0 : -1.0;
        if (n <= 0) ++losses;
        rows.push_back({i - a.warmup, tns0, tns1, rtt});
    }
    ::close(sock);

    ::mkdir(a.out.c_str(), 0755);  // ignore EEXIST
    {
        std::ofstream f(a.out + "/raw.csv");
        f << "seq,t_send_ns,t_recv_ns,rtt_us\n";
        for (const auto& r : rows) {
            f << r.seq << "," << r.t_send_ns << "," << r.t_recv_ns << ","
              << r.rtt_us << "\n";
        }
    }
    std::vector<double> rtts;
    for (const auto& r : rows) if (r.rtt_us >= 0) rtts.push_back(r.rtt_us);
    std::sort(rtts.begin(), rtts.end());
    auto pct = [&](double p) -> double {
        if (rtts.empty()) return -1.0;
        size_t k = std::min(rtts.size() - 1,
            static_cast<size_t>(std::round(p / 100.0 * (rtts.size() - 1))));
        return rtts[k];
    };
    {
        std::ofstream f(a.out + "/summary.json");
        f << "{\n"
          << "  \"fpga_ip\": \"" << a.fpga_ip << "\",\n"
          << "  \"fpga_port\": " << a.fpga_port << ",\n"
          << "  \"listen_port\": " << a.listen_port << ",\n"
          << "  \"packets\": " << rows.size() << ",\n"
          << "  \"received\": " << rtts.size() << ",\n"
          << "  \"drop_count\": " << losses << ",\n"
          << "  \"drop_pct\": "
          << (100.0 * losses / std::max(static_cast<size_t>(1), rows.size())) << ",\n"
          << "  \"rtt_us_min\": " << (rtts.empty() ? -1 : rtts.front()) << ",\n"
          << "  \"rtt_us_p50\": " << pct(50) << ",\n"
          << "  \"rtt_us_p95\": " << pct(95) << ",\n"
          << "  \"rtt_us_p99\": " << pct(99) << ",\n"
          << "  \"rtt_us_max\": " << (rtts.empty() ? -1 : rtts.back()) << "\n"
          << "}\n";
    }
    std::cerr << "[rtt] sent=" << rows.size() << " recv=" << rtts.size()
              << " p50=" << pct(50) << "us p99=" << pct(99) << "us\n";
    return 0;
}
