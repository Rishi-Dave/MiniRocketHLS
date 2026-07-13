// udp_echo.cpp — C++ UDP echo server. High-rate stand-in for the FPGA NetLayer.
//
// Build:
//   g++ -std=c++17 -O2 -o udp_echo udp_echo.cpp

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <getopt.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string listen_ip = "0.0.0.0";
    uint16_t listen_port = 62177;
    int rcvbuf = 64 << 20;
    int sndbuf = 64 << 20;

    static option opts[] = {
        {"listen-ip",   required_argument, nullptr, 'i'},
        {"listen-port", required_argument, nullptr, 'p'},
        {"rcvbuf",      required_argument, nullptr, 'R'},
        {"sndbuf",      required_argument, nullptr, 'S'},
        {nullptr, 0, nullptr, 0}
    };
    int c;
    while ((c = getopt_long(argc, argv, "i:p:R:S:", opts, nullptr)) != -1) {
        switch (c) {
            case 'i': listen_ip = optarg; break;
            case 'p': listen_port = static_cast<uint16_t>(std::stoi(optarg)); break;
            case 'R': rcvbuf = std::stoi(optarg); break;
            case 'S': sndbuf = std::stoi(optarg); break;
            default: std::cerr << "unknown opt\n"; return 2;
        }
    }

    int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { perror("socket"); return 1; }
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

    sockaddr_in local{}; local.sin_family = AF_INET;
    inet_pton(AF_INET, listen_ip.c_str(), &local.sin_addr);
    local.sin_port = htons(listen_port);
    if (::bind(sock, reinterpret_cast<sockaddr*>(&local), sizeof(local)) < 0) {
        perror("bind");
        return 1;
    }
    std::fprintf(stderr, "[echo] listening on %s:%u\n", listen_ip.c_str(), listen_port);

    char buf[2048];
    sockaddr_in src{};
    socklen_t sl = sizeof(src);
    long long n = 0;
    while (true) {
        ssize_t r = ::recvfrom(sock, buf, sizeof(buf), 0,
                               reinterpret_cast<sockaddr*>(&src), &sl);
        if (r <= 0) continue;
        ::sendto(sock, buf, r, 0,
                 reinterpret_cast<sockaddr*>(&src), sl);
        if ((++n & 0xFFFFF) == 0) {
            std::fprintf(stderr, "[echo] n=%lld\n", n);
        }
    }
    return 0;
}
