#include <iostream>
#include <cstring>

extern "C" {
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_udp.h>
}

#include <fcntl.h>
#include <sys/mman.h>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <sys/types.h>

#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024
#define NUM_MBUFS 8192
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32

#define LISTEN_PORT 9000

// CHANGE THIS
#define DPDK_IP RTE_IPV4(192,168,40,31)
#define PYTHON_IP RTE_IPV4(192,168,40,30)




struct SharedPacket {
    float from_python;
    int ready_to_dpdk;
    float from_dpdk;
    int ready_to_python;
};

SharedPacket* buf;

void init_shared_memory() {
    int fd = shm_open("/dpdk_shm", O_CREAT | O_RDWR, 0666);
    ftruncate(fd, sizeof(SharedPacket));
    buf = (SharedPacket*) mmap(nullptr, sizeof(SharedPacket),
                               PROT_READ | PROT_WRITE,
                               MAP_SHARED, fd, 0);
    buf->ready_to_dpdk = 0;
    buf->ready_to_python = 0;
    close(fd);
}

float wait_for_python_reply() {
    // Busy wait until ready_to_python becomes 1
    while (buf->ready_to_python == 0) {
        // Optionally, yield CPU to avoid 100% spinning
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    float val = buf->from_dpdk;
    buf->ready_to_python = 0; // mark read
    return val;
}

void send_reply_to_python(float val) {
    if (buf->ready_to_python == 0) {
        buf->from_dpdk = val;
        buf->ready_to_python = 1;
    }
}





static struct rte_mempool* mbuf_pool;

void send_udp_reply(uint16_t port_id,
                    struct rte_ether_addr* src_mac,
                    struct rte_ether_addr* dst_mac,
                    float value)
{
    struct rte_mbuf* mbuf = rte_pktmbuf_alloc(mbuf_pool);

    uint16_t pkt_size =
        sizeof(rte_ether_hdr) +
        sizeof(rte_ipv4_hdr) +
        sizeof(rte_udp_hdr) +
        sizeof(float);

    rte_pktmbuf_append(mbuf, pkt_size);

    auto* eth = rte_pktmbuf_mtod(mbuf, rte_ether_hdr*);
    auto* ip  = (rte_ipv4_hdr*)(eth + 1);
    auto* udp = (rte_udp_hdr*)(ip + 1);
    float* payload = (float*)(udp + 1);

    *payload = value;

    rte_ether_addr_copy(dst_mac, &eth->dst_addr);
    rte_ether_addr_copy(src_mac, &eth->src_addr);
    eth->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);

    ip->version_ihl = 0x45;
    ip->total_length = rte_cpu_to_be_16(
        sizeof(rte_ipv4_hdr) +
        sizeof(rte_udp_hdr) +
        sizeof(float)
    );
    ip->time_to_live = 64;
    ip->next_proto_id = IPPROTO_UDP;
    ip->src_addr = rte_cpu_to_be_32(DPDK_IP);
    ip->dst_addr = rte_cpu_to_be_32(PYTHON_IP);
    ip->hdr_checksum = 0;
    ip->hdr_checksum = rte_ipv4_cksum(ip);

    udp->src_port = rte_cpu_to_be_16(LISTEN_PORT);
    udp->dst_port = rte_cpu_to_be_16(9001);
    udp->dgram_len = rte_cpu_to_be_16(
        sizeof(rte_udp_hdr) + sizeof(float)
    );
    udp->dgram_cksum = 0;

    rte_eth_tx_burst(port_id, 0, &mbuf, 1);
}

int main(int argc, char* argv[])
{
    rte_eal_init(argc, argv);

    uint16_t port_id = 0;

    rte_eth_conf port_conf{};
    port_conf.rxmode.mq_mode = ETH_MQ_RX_NONE;
    port_conf.txmode.mq_mode = ETH_MQ_TX_NONE;

    rte_eth_dev_configure(port_id, 1, 1, &port_conf);

    mbuf_pool = rte_pktmbuf_pool_create(
        "MBUF_POOL", NUM_MBUFS,
        MBUF_CACHE_SIZE, 0,
        RTE_MBUF_DEFAULT_BUF_SIZE,
        rte_socket_id());

    rte_eth_rx_queue_setup(port_id, 0,
                           RX_RING_SIZE,
                           rte_eth_dev_socket_id(port_id),
                           nullptr, mbuf_pool);

    rte_eth_tx_queue_setup(port_id, 0,
                           TX_RING_SIZE,
                           rte_eth_dev_socket_id(port_id),
                           nullptr);

    rte_eth_dev_start(port_id);
    rte_eth_promiscuous_enable(port_id);

    struct rte_ether_addr src_mac;
    rte_eth_macaddr_get(port_id, &src_mac);

    struct rte_mbuf* bufs[BURST_SIZE];

    printf("DPDK MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",
            src_mac.addr_bytes[0], src_mac.addr_bytes[1],
            src_mac.addr_bytes[2], src_mac.addr_bytes[3],
            src_mac.addr_bytes[4], src_mac.addr_bytes[5]);

    init_shared_memory();

    while (true)
    {
        uint16_t nb_rx = rte_eth_rx_burst(port_id, 0, bufs, BURST_SIZE);

        for (int i = 0; i < nb_rx; i++)
        {
            auto* mbuf = bufs[i];
            auto* eth = rte_pktmbuf_mtod(mbuf, rte_ether_hdr*);

            if (eth->ether_type != rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4)) {
                std::cout << "Not an Ethernet IPv4 packet" << std::endl;
                //std::cout << "Ether type: " << std::hex << rte_be_to_cpu_16(eth->ether_type) << std::dec << std::endl;
                continue;
            }

            auto* ip = (rte_ipv4_hdr*)(eth + 1);

            if (ip->next_proto_id != IPPROTO_UDP) {
                std::cout << "Not an UDP packet" << std::endl;
                continue;
            }

            auto* udp = (rte_udp_hdr*)(ip + 1);

            if (rte_be_to_cpu_16( udp->dst_port) != LISTEN_PORT) {
                std::cout << "Not the expected destination port" << std::endl;
                continue;
            }

            float* payload = (float*)(udp + 1);
            float received = *payload;

            send_reply_to_python(received);
            float python_reply = wait_for_python_reply();

            std::cout << "Received: " << received << " Python reply: " << python_reply << std::endl;

            send_udp_reply(
                port_id,
                &src_mac,
                &eth->src_addr,
                python_reply
            );

            rte_pktmbuf_free(mbuf);
        }
    }
}