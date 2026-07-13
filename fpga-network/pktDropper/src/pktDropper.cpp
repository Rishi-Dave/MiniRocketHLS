// pktDropper — graceful UDP backpressure relief.
//
// Sits between NetLayer's M_AXIS_nl2sk (UDP payload outbound to user kernel)
// and the user kernel's input stream. When the kernel can't drain fast enough,
// pktDropper silently drops new packets instead of backpressuring the entire
// CMAC RX path (which otherwise causes the 100G GT to wedge — see prior
// CMAC RX wedge investigation, saturation_pivot 2026-05-06).
//
// Verbatim copy of pyuvaraj37/nlp:fpga-network/pktDropper/src/pktDropper.cpp,
// 2026-05-26. Original author: prithviraj.yuvaraj.

#include "../include/pktDropper.hpp"


extern "C" void pktDropper(
    hls::stream<pkt> &input,
    hls::stream<pkt> &output
) {

    #pragma HLS INTERFACE mode=axis port=input
    #pragma HLS INTERFACE mode=axis port=output
    #pragma HLS interface ap_ctrl_none port=return

    #pragma HLS PIPELINE II=1

    enum State {IDLE, READ, WRITE};
    static State state = IDLE;
    static pkt pkt_in;
    switch(state) {
        case IDLE:
            if (!input.empty()) {
                state = READ;
            }
            break;
        case READ:
            pkt_in = input.read();
            state = WRITE;
            break;
        case WRITE:
            if (!output.full()) {
                output.write(pkt_in);
            }
            // else: silently drop pkt_in (the whole point of this module)
            state = IDLE;
            break;
    }
}
