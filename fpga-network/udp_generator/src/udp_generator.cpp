/*
 * udp_generator — free-running UDP line-rate packet generator, AXIS wrapper
 * for NetLayer's S_AXIS_sk2nl / M_AXIS_nl2sk app-side ports.
 *
 * PURPOSE (2026-07-07 scoping task): pc161's U280 currently has no on-chip
 * traffic source; the ~700k-pps ceiling measured for the fused MiniRocket
 * DUT (see memory network_bandwidth_2026_07_04.md) is sender-limited by a
 * host-NIC C sender (hi_rate_sender, AF_PACKET, single-thread i40e cap).
 * This kernel is meant to remove that host bottleneck entirely by emitting
 * packets from fabric logic once an FPGA-to-FPGA CMAC link exists between
 * pc161 and pc160 (currently NOT cabled — pc161's CMAC QSFP reads
 * stat_rx_status=0 / no peer, per the same memory note).
 *
 * DESIGN — deliberately mirrors minirocket_fused.cpp's PROVEN free-run
 * pattern instead of inventing a new one:
 *   - `udp_generator_emit`: the only process that writes to NetLayer.
 *     GUARDED, non-blocking `if (!output.full()) write()` — the exact fix
 *     that stopped the free-run NetLayer-TX deadlock in minirocket_fused v4
 *     (unguarded / multi-packet-per-event emit wedged hardware twice in
 *     that history; ONE packed single-beat write per iteration is what
 *     works). Each packet here is already a single packed beat (64B fits
 *     exactly in one 512-bit AXIS beat), so there is no multi-packet-per-
 *     event risk to begin with.
 *   - `udp_generator_drain`: mirrors minirocket_fused_ingest's non-blocking
 *     drain shape. NetLayers/kernel.xml declares M_AXIS_nl2sk as a
 *     mandatory kernel port (v++ requires it connected), so even though
 *     this generator expects no inbound application traffic, the port must
 *     be terminated somewhere. Draining it here removes the need to link
 *     in pktDropper as a separate kernel.
 *   - Same `#if defined(COSIM_MAX_WINDOWS) ... #elif BUILD == 1
 *     while(true) ... #endif` gate as minirocket_fused: default (csim, no
 *     macro) = exactly one iteration per kernel call (deterministic,
 *     fast); COSIM_MAX_WINDOWS = bounded free-run liveness check; BUILD=1
 *     (synthesis/ip export only, via -DBUILD=1 in make.tcl.in, same as
 *     minirocket_fused) = TRUE on-chip free-run for hardware.
 *
 * PACKET FORMAT: single AXIS beat, `pkt.data` low 4 bytes = float32 sample
 * (a free-running counter cast to float — a real signal is not needed for
 * a pure rate/ceiling test; matches minirocket_fused_ingest's decode of
 * "low-order reinterpret of the 512-bit beat" so a fused-style DUT would
 * decode these as legitimate, if meaningless, samples). `keep`/`strb` =
 * all-ones (all 64 bytes marked valid, matching the task's fixed 64-byte
 * payload). `last` = 1 (one beat == one packet). `dest` = 0 selects
 * NetLayer UDP socket-table slot 0 (see saturation_harness/setup_netlayer.py
 * REG_THEIR_IP_BASE/REG_THEIR_PORT_BASE/REG_MY_PORT_BASE slot 0 — program
 * that slot with theirIP=192.168.40.10, theirPort=62177, myPort=62178 on
 * pc161 to get dst 192.168.40.10:62177 / src :62178 framing, matching
 * pc160's expected socket table entry).
 *
 * RATE CONTROL: `rate_divider` AXI-Lite register. 0 (default) = emit every
 * cycle the guard allows (max fabric rate, i.e. "as close to CMAC line
 * rate as possible"). N>0 = emit once every (N+1) cycles, for controlled
 * sub-line-rate sweeps (mirrors the pacing saturation_harness/rate_sweep.py
 * already does on the host side, but in fabric so it isn't capped by a
 * host NIC).
 *
 * NOT YET DONE / left for the actual build: no hw or even csim/cosim run
 * has been executed for this file (scoping task explicitly excluded hw
 * builds and hardware access). Before any hw build: run csim (`vitis_hls
 * -f make.tcl csim`, or `make csim` via CMakeLists.txt) and ideally a
 * bounded COSIM_MAX_WINDOWS cosim liveness check, exactly as
 * minirocket_fused's make.tcl.in does, before spending a ~4h v++ hw link.
 */

#include "../include/udp_generator.hpp"

// ---------------------------------------------------------------------------
// Drain-and-discard whatever NetLayer forwards back to the app on
// M_AXIS_nl2sk. Never blocks, never touches the emit path — structurally
// independent, same reasoning as minirocket_fused_ingest.
// ---------------------------------------------------------------------------
void udp_generator_drain(
    hls::stream<pkt> &input_drain
) {
#pragma HLS INLINE off

#if defined(COSIM_MAX_WINDOWS)
    for (int __it = 0; __it < (COSIM_MAX_WINDOWS) * 4 + 64; __it++) {
#elif BUILD == 1
    while (true) {
#endif
        if (!input_drain.empty()) {
            pkt discard = input_drain.read();
            (void)discard;
        }
#if defined(COSIM_MAX_WINDOWS) || BUILD == 1
    }
#endif
}

// ---------------------------------------------------------------------------
// Free-running packet emitter. GUARDED, non-blocking write — can never
// stall forever waiting on NetLayer TX backpressure (the proven
// anti-deadlock idiom from minirocket_fused v4 / pktDropper.cpp).
// ---------------------------------------------------------------------------
void udp_generator_emit(
    hls::stream<pkt> &output_stream,
    ap_uint<32> rate_divider
) {
#pragma HLS INLINE off

    static ap_uint<32> sample_counter = 0;
    static ap_uint<32> divider_count  = 0;
#pragma HLS RESET variable=sample_counter
#pragma HLS RESET variable=divider_count

#if defined(COSIM_MAX_WINDOWS)
    for (int __it = 0; __it < COSIM_MAX_WINDOWS; __it++) {
#elif BUILD == 1
    while (true) {
#endif
        bool fire = true;
        if (rate_divider != 0) {
            if (divider_count < rate_divider) {
                divider_count++;
                fire = false;
            } else {
                divider_count = 0;
            }
        }

        if (fire) {
            pkt out_pkt;
            ap_uint<GEN_DWIDTH> out_data = 0;

            // Low 4 bytes: float32 sample (free-running counter, no real
            // signal needed for a rate/ceiling test).
            data_t sample = (data_t)sample_counter;
            ap_uint<32> bits = *((ap_uint<32>*)&sample);
            out_data.range(31, 0) = bits;

            out_pkt.data = out_data;
            out_pkt.keep = -1;   // all 64 bytes valid (fixed 64B payload)
            out_pkt.strb = -1;
            out_pkt.last = 1;    // one beat == one packet
            out_pkt.dest = 0;    // NetLayer UDP socket-table slot 0
            out_pkt.user = 0;
            out_pkt.id   = 0;

            if (!output_stream.full()) {
                output_stream.write(out_pkt);
                sample_counter++;
            }
            // else: NetLayer TX is backpressuring -- drop this packet
            // rather than ever block the free-run loop (same idiom as
            // minirocket_fused_compute_emit / pktDropper.cpp).
        }
#if defined(COSIM_MAX_WINDOWS) || BUILD == 1
    }
#endif
}

// ---------------------------------------------------------------------------
// Top-level network AXIS kernel. Two DATAFLOW processes, no shared state
// between them (canonical form, same as minirocket_fused's ingest/
// compute_emit split) -- drain and emit can each free-run independently.
// ---------------------------------------------------------------------------
extern "C" void udp_generator(
    hls::stream<pkt> &input_drain,
    hls::stream<pkt> &output_stream,
    ap_uint<32> rate_divider
) {
    #pragma HLS INTERFACE axis port = input_drain
    #pragma HLS INTERFACE axis port = output_stream

    #pragma HLS INTERFACE s_axilite port = rate_divider bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    #pragma HLS DATAFLOW

    udp_generator_drain(input_drain);
    udp_generator_emit(output_stream, rate_divider);
}
