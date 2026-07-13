#ifndef MULTIROCKET_AXIS_INFERENCE_HLS_H
#define MULTIROCKET_AXIS_INFERENCE_HLS_H

/*
 * MultiRocket AXI-Stream variant — Phase 5 of network-saturation pivot.
 *
 * Same compute path as multirocket_optimized/multirocket/src/multirocket.cpp
 * (84 fixed kernels, 4 pooling operators × 2 representations
 * (original + first-order-difference), StandardScaler, ridge classifier)
 * but the time-series input arrives one float per AXI-Stream packet from
 * the NetLayer rather than via PCIe DMA into HBM. Predictions return the
 * same way.
 *
 * Wire format (matches fpga-network/minirocket and hydra_axis):
 *   - 512-bit AXI-Stream packet (ap_axiu<512, 1, 1, 16>)
 *   - Only the low 32 bits carry a float32 sample
 *   - One incoming pkt = one new sample point. Once the shift register has
 *     time_series_length valid samples, the kernel runs one full inference
 *     (both representations) and emits num_classes floats.
 *
 * Note: kernel weights are STATIC (compile-time) and live in
 *   ../../multirocket/include/weights.txt
 * — they are baked into the bitstream, not loaded from HBM. Same as the
 * fpga-network MiniRocket variant. Only learned model parameters
 * (classifier coefficients, scaler, biases per representation, dilation
 * tables) come from HBM via m_axi.
 */

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include <stdint.h>

// Data types — match the m_axi MultiRocket variant for csim parity.
typedef float data_t;
typedef ap_int<32> int_t;
typedef ap_uint<8> idx_t;

// MultiRocket constants — must match multirocket_optimized/multirocket/include/multirocket.hpp.
#define MAX_TIME_SERIES_LENGTH 8192
#define MAX_FEATURES 50000      // 4 pooling ops × 2 representations × dilations × 84 kernels
#define NUM_KERNELS 84
#define KERNEL_SIZE 9
#define MAX_DILATIONS 32
#define MAX_CLASSES 16
#define NUM_POOLING_OPERATORS 4 // PPV, MPV, MIPV, LSPV
#define NUM_REPRESENTATIONS 2   // original + first-order difference

// AXI-Stream packet definition (matches fpga-network minirocket variant).
#define DWIDTH 512
#define TDWIDTH 16
typedef ap_axiu<DWIDTH, 1, 1, TDWIDTH> pkt;

struct PoolingStats_axis {
    data_t ppv;
    data_t mpv;
    data_t mipv;
    data_t lspv;
};

// Top-level streaming inference kernel.
extern "C" void multirocket_axis_inference(
    hls::stream<pkt> &input_timeseries,
    hls::stream<pkt> &output_predictions,
    data_t* coefficients,           // gmem2  — [num_classes][num_features]
    data_t* intercept,              // gmem3  — [num_classes]
    data_t* scaler_mean,            // gmem4  — [num_features]
    data_t* scaler_scale,           // gmem5  — [num_features]
    int_t*  dilations_0,            // gmem6  — original-rep dilations
    int_t*  num_features_per_dilation_0, // gmem7
    data_t* biases_0,               // gmem8  — original-rep biases
    int_t   num_dilations_0,
    int_t   num_features_0,
    int_t*  dilations_1,            // gmem9  — diff-rep dilations
    int_t*  num_features_per_dilation_1, // gmem10
    data_t* biases_1,               // gmem11 — diff-rep biases
    int_t   num_dilations_1,
    int_t   num_features_1,
    int_t   time_series_length,
    int_t   num_features,
    int_t   num_classes,
    int_t   n_feature_per_kernel
);

#endif // MULTIROCKET_AXIS_INFERENCE_HLS_H
