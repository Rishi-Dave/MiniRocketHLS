#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include "../include/pass_through.hpp"

int main(int argc, char* argv[]) {
    
    hls::stream<pkt> input_timeseries;
    hls::stream<pkt> output_predictions;   
    ap_uint<8> dest = 0; // Example destination ID


    // Create a dummy input packet
    pkt input_pkt;
    ap_uint< DWIDTH > input_data = 0x1234567890ABCDEF; // Example input data
    input_pkt.data = input_data;
    input_pkt.keep = -1; // All bytes are valid
    input_pkt.dest = dest;
    input_pkt.last = 1; // Last packet in the stream


    // Write the input packet to the stream
    input_timeseries.write(input_pkt);
    // Call the HLS function
    pass_through(input_timeseries, output_predictions, dest);

    // Read the output packet from the stream
    if (!output_predictions.empty()) {
        pkt output_pkt = output_predictions.read();
        std::cout << "Output Packet Data: 0x" << std::hex << output_pkt.data << std::dec << std::endl;
        std::cout << "Output Packet Dest: " << output_pkt.dest << std::endl;
        std::cout << "Output Packet Last: " << output_pkt.last << std::endl;
    } else {
        std::cout << "No output packet received." << std::endl;
    }

    return 0;
}