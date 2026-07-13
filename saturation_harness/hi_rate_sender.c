/*
 * hi_rate_sender.c
 *
 * High-rate UDP packet sender for driving the MiniRocketHLS FPGA NetLayer
 * ingestion path far past Python throughput.py's ~250-311k pps ceiling.
 *
 * Two modes:
 *   - Paced mode (--rate PPS given): crafts one Ethernet/IPv4/UDP frame and
 *     sends it via a plain AF_PACKET SOCK_RAW socket in a busy-wait paced
 *     loop (mirrors throughput.py's pacing model but in C, no interpreter
 *     overhead). Good for hitting a specific target rate (500k, 1M, 2M...).
 *   - Max-blast mode (no --rate, or --rate 0): uses AF_PACKET PACKET_MMAP
 *     TX_RING. All ring frames are pre-filled with the same 64-byte-payload
 *     frame; each round re-arms every frame to TP_STATUS_SEND_REQUEST and
 *     calls send(fd, NULL, 0, 0) to flush the whole ring in one syscall.
 *     This is the standard packet_mmap.txt TX flood pattern and is the
 *     fastest practical userspace path to multi-Mpps without DPDK/XDP.
 *
 * Wire format: standard Ethernet + IPv4 + UDP frame. UDP payload is fixed
 * at 64 bytes; the low 4 bytes optionally carry a float32 sample value
 * (--sample), the rest are zero. Content doesn't matter for ingestion-rate
 * testing -- the FPGA NetLayer counts one ingested sample per UDP packet.
 *
 * Requires CAP_NET_RAW (run under sudo, or setcap cap_net_raw+ep on the
 * binary).
 *
 * Note on --count in blast mode: the whole TX_RING (4096 frames) is armed
 * and flushed per round, so --count can overshoot by up to one ring's
 * worth of frames. --count is exact in paced mode (one sendto() per
 * packet). For blast mode, prefer --duration for precise run length.
 *
 * Build:
 *   gcc -O2 -Wall -o hi_rate_sender hi_rate_sender.c
 *
 * Example (DO NOT point --dst-ip at a live FPGA without confirming it is
 * free -- see project safety rules):
 *   sudo ./hi_rate_sender --iface enp135s0f0 --rate 1000000 --duration 2
 *   sudo ./hi_rate_sender --iface enp135s0f0 --duration 2      # max blast
 */

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <getopt.h>
#include <ifaddrs.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define ETH_HDR_LEN   14
#define IP_HDR_LEN    20
#define UDP_HDR_LEN   8
#define PAYLOAD_LEN   64
#define FRAME_LEN     (ETH_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN + PAYLOAD_LEN) /* 106 */

#define DEFAULT_DST_MAC "02:00:00:00:0a:0a"
#define DEFAULT_DST_IP  "192.168.40.10"
#define DEFAULT_DST_PORT 62177
#define DEFAULT_SRC_PORT 40000

/* ---- ring geometry for TX_RING blast mode ---- */
#define RING_FRAME_SIZE 2048   /* must be TPACKET_ALIGNMENT (16) multiple, >= FRAME_LEN + hdr */
#define RING_BLOCK_SIZE 4096   /* must be a multiple of page size */
#define RING_BLOCK_NR   2048   /* -> 2 frames/block -> 4096 total frames, 8MB ring */

struct args {
    const char *iface;
    char dst_mac_s[32];
    char dst_ip_s[32];
    int  dst_port;
    char src_mac_s[32];
    char src_ip_s[32];
    int  src_port;
    int  have_src_mac;
    int  have_src_ip;
    long long count;      /* 0 = unbounded (use duration) */
    double duration;      /* seconds; 0 = unbounded (use count) */
    double rate;          /* pps; 0 = max-blast (TX_RING) */
    float sample;
    int verbose;
};

static void die(const char *msg) {
    perror(msg);
    exit(1);
}

static uint16_t ip_checksum(const void *data, size_t len) {
    const uint16_t *buf = data;
    uint32_t sum = 0;
    while (len > 1) {
        sum += *buf++;
        len -= 2;
    }
    if (len == 1)
        sum += *(const uint8_t *)buf;
    while (sum >> 16)
        sum = (sum & 0xFFFF) + (sum >> 16);
    return (uint16_t)~sum;
}

static void parse_mac(const char *s, uint8_t out[6]) {
    unsigned int b[6];
    if (sscanf(s, "%x:%x:%x:%x:%x:%x", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6) {
        fprintf(stderr, "bad MAC address: %s\n", s);
        exit(1);
    }
    for (int i = 0; i < 6; i++)
        out[i] = (uint8_t)b[i];
}

static void get_iface_mac(const char *iface, uint8_t out[6]) {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) die("socket(get_iface_mac)");
    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, iface, IFNAMSIZ - 1);
    if (ioctl(fd, SIOCGIFHWADDR, &ifr) < 0) die("ioctl(SIOCGIFHWADDR)");
    memcpy(out, ifr.ifr_hwaddr.sa_data, 6);
    close(fd);
}

static void get_iface_ip(const char *iface, char *out, size_t outlen) {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) die("socket(get_iface_ip)");
    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, iface, IFNAMSIZ - 1);
    ifr.ifr_addr.sa_family = AF_INET;
    if (ioctl(fd, SIOCGIFADDR, &ifr) < 0) die("ioctl(SIOCGIFADDR) -- pass --src-ip explicitly if iface has no IPv4 address");
    struct sockaddr_in *sin = (struct sockaddr_in *)&ifr.ifr_addr;
    if (!inet_ntop(AF_INET, &sin->sin_addr, out, outlen)) die("inet_ntop");
    close(fd);
}

static int get_iface_index(const char *iface) {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) die("socket(get_iface_index)");
    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, iface, IFNAMSIZ - 1);
    if (ioctl(fd, SIOCGIFINDEX, &ifr) < 0) die("ioctl(SIOCGIFINDEX)");
    close(fd);
    return ifr.ifr_ifindex;
}

/* Build one Ethernet+IPv4+UDP frame with a 64-byte UDP payload into buf.
 * Returns the total frame length (should equal FRAME_LEN). */
static size_t build_frame(uint8_t *buf,
                           const uint8_t src_mac[6], const uint8_t dst_mac[6],
                           uint32_t src_ip, uint32_t dst_ip,
                           uint16_t src_port, uint16_t dst_port,
                           float sample) {
    uint8_t *p = buf;

    /* Ethernet header */
    memcpy(p, dst_mac, 6);
    memcpy(p + 6, src_mac, 6);
    p[12] = 0x08; p[13] = 0x00; /* ETH_P_IP */
    p += ETH_HDR_LEN;

    /* IPv4 header */
    uint8_t *iph = p;
    iph[0] = 0x45;                 /* version 4, IHL 5 */
    iph[1] = 0x00;                 /* DSCP/ECN */
    uint16_t tot_len = IP_HDR_LEN + UDP_HDR_LEN + PAYLOAD_LEN;
    uint16_t tot_len_n = htons(tot_len);
    memcpy(iph + 2, &tot_len_n, 2);
    uint16_t id_n = 0;
    memcpy(iph + 4, &id_n, 2);
    uint16_t flags_frag = htons(0x4000); /* DF, no fragmentation */
    memcpy(iph + 6, &flags_frag, 2);
    iph[8] = 64;                   /* TTL */
    iph[9] = 17;                   /* protocol UDP */
    iph[10] = 0; iph[11] = 0;      /* checksum placeholder */
    memcpy(iph + 12, &src_ip, 4);
    memcpy(iph + 16, &dst_ip, 4);
    uint16_t csum = ip_checksum(iph, IP_HDR_LEN);
    memcpy(iph + 10, &csum, 2);
    p += IP_HDR_LEN;

    /* UDP header */
    uint8_t *udph = p;
    uint16_t sport_n = htons(src_port);
    uint16_t dport_n = htons(dst_port);
    uint16_t ulen_n = htons(UDP_HDR_LEN + PAYLOAD_LEN);
    memcpy(udph + 0, &sport_n, 2);
    memcpy(udph + 2, &dport_n, 2);
    memcpy(udph + 4, &ulen_n, 2);
    udph[6] = 0; udph[7] = 0; /* checksum = 0: optional for IPv4/UDP */
    p += UDP_HDR_LEN;

    /* Payload: 64 bytes, low 4 bytes = float32 sample, rest zero */
    memset(p, 0, PAYLOAD_LEN);
    memcpy(p, &sample, sizeof(float));
    p += PAYLOAD_LEN;

    return (size_t)(p - buf);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* ---------------- paced mode: plain AF_PACKET SOCK_RAW ---------------- */
static void run_paced(struct args *a, uint8_t *frame, size_t frame_len, int ifindex) {
    int fd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_IP));
    if (fd < 0) die("socket(AF_PACKET, SOCK_RAW) -- need CAP_NET_RAW (run with sudo)");

    struct sockaddr_ll sll;
    memset(&sll, 0, sizeof(sll));
    sll.sll_family = AF_PACKET;
    sll.sll_protocol = htons(ETH_P_IP);
    sll.sll_ifindex = ifindex;
    sll.sll_halen = ETH_ALEN;
    memcpy(sll.sll_addr, frame, 6); /* dst mac, informational for SOCK_RAW */

    if (bind(fd, (struct sockaddr *)&sll, sizeof(sll)) < 0)
        die("bind(AF_PACKET)");

    double interval_s = 1.0 / a->rate;
    long long sent = 0;
    double t_start = now_s();
    double t_end = a->duration > 0 ? t_start + a->duration : 0;
    double next_t = t_start;

    for (;;) {
        if (a->count > 0 && sent >= a->count) break;
        if (t_end > 0 && now_s() >= t_end) break;

        double t = now_s();
        if (t < next_t) {
            double slack = next_t - t;
            if (slack > 50e-6) {
                struct timespec ts = { .tv_sec = (time_t)slack, .tv_nsec = (long)((slack - (time_t)slack) * 1e9) };
                nanosleep(&ts, NULL);
            }
            while (now_s() < next_t) { /* busy spin for final precision */ }
        }

        ssize_t r = sendto(fd, frame, frame_len, 0, (struct sockaddr *)&sll, sizeof(sll));
        if (r < 0) {
            if (errno == ENOBUFS) { next_t += interval_s; continue; }
            die("sendto");
        }
        sent++;
        next_t += interval_s;
    }

    double elapsed = now_s() - t_start;
    fprintf(stderr, "[hi_rate_sender] paced mode: target=%.0f pps sent=%lld elapsed=%.3fs achieved=%.0f pps\n",
            a->rate, sent, elapsed, elapsed > 0 ? sent / elapsed : 0.0);
    close(fd);
}

/* ---------------- max-blast mode: AF_PACKET PACKET_MMAP TX_RING ---------------- */
static void run_blast(struct args *a, uint8_t *frame, size_t frame_len, int ifindex) {
    int fd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_IP));
    if (fd < 0) die("socket(AF_PACKET, SOCK_RAW) -- need CAP_NET_RAW (run with sudo)");

    int version = TPACKET_V2;
    if (setsockopt(fd, SOL_PACKET, PACKET_VERSION, &version, sizeof(version)) < 0)
        die("setsockopt(PACKET_VERSION)");

    struct tpacket_req req;
    memset(&req, 0, sizeof(req));
    req.tp_block_size = RING_BLOCK_SIZE;
    req.tp_block_nr   = RING_BLOCK_NR;
    req.tp_frame_size = RING_FRAME_SIZE;
    req.tp_frame_nr   = (req.tp_block_size * req.tp_block_nr) / req.tp_frame_size;

    if (setsockopt(fd, SOL_PACKET, PACKET_TX_RING, &req, sizeof(req)) < 0)
        die("setsockopt(PACKET_TX_RING)");

    size_t ring_size = (size_t)req.tp_block_size * req.tp_block_nr;
    uint8_t *ring = mmap(NULL, ring_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ring == MAP_FAILED) die("mmap(TX_RING)");

    int qdisc_bypass = 1;
    setsockopt(fd, SOL_PACKET, PACKET_QDISC_BYPASS, &qdisc_bypass, sizeof(qdisc_bypass)); /* best effort, ignore failure */

    struct sockaddr_ll sll;
    memset(&sll, 0, sizeof(sll));
    sll.sll_family = AF_PACKET;
    sll.sll_protocol = htons(ETH_P_IP);
    sll.sll_ifindex = ifindex;
    sll.sll_halen = ETH_ALEN;
    memcpy(sll.sll_addr, frame, 6);
    if (bind(fd, (struct sockaddr *)&sll, sizeof(sll)) < 0)
        die("bind(AF_PACKET)");

    unsigned nframes = req.tp_frame_nr;
    if (a->verbose)
        fprintf(stderr, "[hi_rate_sender] TX_RING: frame_size=%u block_size=%u block_nr=%u total_frames=%u ring_bytes=%zu\n",
                req.tp_frame_size, req.tp_block_size, req.tp_block_nr, nframes, ring_size);

    /* Pre-fill every ring frame with the same crafted packet; leave status
     * TP_STATUS_AVAILABLE for now, we arm to SEND_REQUEST in the send loop. */
    for (unsigned i = 0; i < nframes; i++) {
        struct tpacket2_hdr *hdr = (struct tpacket2_hdr *)(ring + (size_t)i * req.tp_frame_size);
        uint8_t *data = (uint8_t *)hdr + TPACKET_ALIGN(sizeof(struct tpacket2_hdr));
        memcpy(data, frame, frame_len);
        hdr->tp_len = (uint32_t)frame_len;
        hdr->tp_status = TP_STATUS_AVAILABLE;
    }

    long long sent = 0;
    long long errors = 0;
    double t_start = now_s();
    double t_end = a->duration > 0 ? t_start + a->duration : 0;

    for (;;) {
        if (a->count > 0 && sent >= a->count) break;
        if (t_end > 0 && now_s() >= t_end) break;

        for (unsigned i = 0; i < nframes; i++) {
            struct tpacket2_hdr *hdr = (struct tpacket2_hdr *)(ring + (size_t)i * req.tp_frame_size);
            /* Frames should already be AVAILABLE (sent) or still AVAILABLE
             * from init; if the kernel hasn't drained one yet (SENDING),
             * skip it this round rather than spin -- it'll be picked up
             * next round. */
            if (hdr->tp_status == TP_STATUS_AVAILABLE) {
                hdr->tp_len = (uint32_t)frame_len;
                hdr->tp_status = TP_STATUS_SEND_REQUEST;
            }
        }

        ssize_t r = send(fd, NULL, 0, 0);
        if (r < 0) {
            if (errno == EAGAIN || errno == ENOBUFS) continue;
            die("send(TX_RING flush)");
        }

        /* Count how many frames actually left SEND_REQUEST state this round. */
        for (unsigned i = 0; i < nframes; i++) {
            struct tpacket2_hdr *hdr = (struct tpacket2_hdr *)(ring + (size_t)i * req.tp_frame_size);
            if (hdr->tp_status == TP_STATUS_AVAILABLE) {
                sent++;
            } else if (hdr->tp_status == TP_STATUS_WRONG_FORMAT) {
                errors++;
                hdr->tp_status = TP_STATUS_AVAILABLE;
            }
            /* TP_STATUS_SEND_REQUEST / TP_STATUS_SENDING left as-is: still in flight,
             * will be re-checked next round. */
        }
    }

    double elapsed = now_s() - t_start;
    fprintf(stderr, "[hi_rate_sender] blast mode: sent=%lld errors=%lld elapsed=%.3fs achieved=%.0f pps\n",
            sent, errors, elapsed, elapsed > 0 ? sent / elapsed : 0.0);

    munmap(ring, ring_size);
    close(fd);
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --iface IFACE [options]\n"
        "  --iface IFACE        interface to send on (required)\n"
        "  --count N            stop after N packets (0 = unbounded, default)\n"
        "  --duration SECONDS   stop after SECONDS (default 2.0 if --count not given)\n"
        "  --rate PPS           paced mode target rate; omit or 0 for max-blast (TX_RING)\n"
        "  --dst-mac MAC        default %s\n"
        "  --dst-ip IP          default %s\n"
        "  --dst-port PORT      default %d\n"
        "  --src-mac MAC        default: auto-detect from iface\n"
        "  --src-ip IP          default: auto-detect from iface\n"
        "  --src-port PORT      default %d\n"
        "  --sample FLOAT       float32 sample value placed in payload[0:4] (default 0.0)\n"
        "  --verbose            print ring geometry / debug info\n",
        prog, DEFAULT_DST_MAC, DEFAULT_DST_IP, DEFAULT_DST_PORT, DEFAULT_SRC_PORT);
}

int main(int argc, char **argv) {
    struct args a;
    memset(&a, 0, sizeof(a));
    strncpy(a.dst_mac_s, DEFAULT_DST_MAC, sizeof(a.dst_mac_s) - 1);
    strncpy(a.dst_ip_s, DEFAULT_DST_IP, sizeof(a.dst_ip_s) - 1);
    a.dst_port = DEFAULT_DST_PORT;
    a.src_port = DEFAULT_SRC_PORT;
    a.count = 0;
    a.duration = 0.0;
    a.rate = 0.0;
    a.sample = 0.0f;

    static struct option long_opts[] = {
        {"iface",     required_argument, 0, 'i'},
        {"count",     required_argument, 0, 'c'},
        {"duration",  required_argument, 0, 'd'},
        {"rate",      required_argument, 0, 'r'},
        {"dst-mac",   required_argument, 0, 'M'},
        {"dst-ip",    required_argument, 0, 'I'},
        {"dst-port",  required_argument, 0, 'P'},
        {"src-mac",   required_argument, 0, 'm'},
        {"src-ip",    required_argument, 0, 'n'},
        {"src-port",  required_argument, 0, 'p'},
        {"sample",    required_argument, 0, 's'},
        {"verbose",   no_argument,       0, 'v'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'i': a.iface = optarg; break;
            case 'c': a.count = atoll(optarg); break;
            case 'd': a.duration = atof(optarg); break;
            case 'r': a.rate = atof(optarg); break;
            case 'M': strncpy(a.dst_mac_s, optarg, sizeof(a.dst_mac_s) - 1); break;
            case 'I': strncpy(a.dst_ip_s, optarg, sizeof(a.dst_ip_s) - 1); break;
            case 'P': a.dst_port = atoi(optarg); break;
            case 'm': strncpy(a.src_mac_s, optarg, sizeof(a.src_mac_s) - 1); a.have_src_mac = 1; break;
            case 'n': strncpy(a.src_ip_s, optarg, sizeof(a.src_ip_s) - 1); a.have_src_ip = 1; break;
            case 'p': a.src_port = atoi(optarg); break;
            case 's': a.sample = strtof(optarg, NULL); break;
            case 'v': a.verbose = 1; break;
            case 'h':
            default:
                usage(argv[0]);
                return opt == 'h' ? 0 : 1;
        }
    }

    if (!a.iface) {
        usage(argv[0]);
        return 1;
    }
    if (a.count == 0 && a.duration == 0.0)
        a.duration = 2.0; /* default self-limiting run length */

    uint8_t dst_mac[6], src_mac[6];
    parse_mac(a.dst_mac_s, dst_mac);
    if (a.have_src_mac) {
        parse_mac(a.src_mac_s, src_mac);
    } else {
        get_iface_mac(a.iface, src_mac);
    }

    if (!a.have_src_ip)
        get_iface_ip(a.iface, a.src_ip_s, sizeof(a.src_ip_s));

    struct in_addr src_in, dst_in;
    if (inet_pton(AF_INET, a.src_ip_s, &src_in) != 1) { fprintf(stderr, "bad src-ip\n"); return 1; }
    if (inet_pton(AF_INET, a.dst_ip_s, &dst_in) != 1) { fprintf(stderr, "bad dst-ip\n"); return 1; }

    int ifindex = get_iface_index(a.iface);

    uint8_t frame[FRAME_LEN];
    size_t frame_len = build_frame(frame, src_mac, dst_mac,
                                    src_in.s_addr, dst_in.s_addr,
                                    (uint16_t)a.src_port, (uint16_t)a.dst_port,
                                    a.sample);

    if (a.verbose) {
        fprintf(stderr, "[hi_rate_sender] iface=%s ifindex=%d frame_len=%zu\n", a.iface, ifindex, frame_len);
        fprintf(stderr, "[hi_rate_sender] src_ip=%s dst_ip=%s dst_port=%d src_port=%d\n",
                a.src_ip_s, a.dst_ip_s, a.dst_port, a.src_port);
        fprintf(stderr, "[hi_rate_sender] mode=%s\n", a.rate > 0 ? "paced" : "max-blast (TX_RING)");
    }

    if (a.rate > 0)
        run_paced(&a, frame, frame_len, ifindex);
    else
        run_blast(&a, frame, frame_len, ifindex);

    return 0;
}
