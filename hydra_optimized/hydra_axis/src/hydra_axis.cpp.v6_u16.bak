/*
 * HYDRA AXI-Stream Inference Kernel — v6 "v2_fixed port" (2026-07-10)
 *
 * See include/hydra_axis.hpp for the full design note. Summary:
 *   compute = hydra v2_fixed core (ap_fixed<32,16>, UNROLL=16 dilation-
 *             sorted batches, fused conv+pool, II=1) — ~12x the old float
 *             core whose CONV_LOOP was stuck at II=5.
 *   wrapper = minirocket_fused v4/dropfull DATAFLOW split: always-drain
 *             ingest with DROP-ON-FULL internal FIFO, buffer-then-classify
 *             compute/emit with packed single-beat GUARDED emit.
 *
 * HISTORY (why the wrapper looks like this — inherited from the
 * minirocket_fused saga, do not "simplify"):
 *   - Unguarded AXIS reads gated on internal state wedge the NetLayer RX
 *     path at high rate (observed >2.24M pps F2F). Ingest must ALWAYS
 *     drain when a packet is present.
 *   - A monolithic loop stalls ingestion during the multi-ms classify,
 *     backpressuring NetLayer the same way. DATAFLOW ingest || compute.
 *   - Per-class multi-packet emit and unguarded emit writes deadlocked
 *     hardware free-run (Bug B family). ONE packed pkt, guarded write,
 *     sideband dest=0/user=0/id=0/strb=-1 explicitly set.
 *   - BUILD=1 compiles a true on-chip while(true) free-run;
 *     COSIM_MAX_WINDOWS bounds the loops for csim/cosim only.
 */

#include "../include/hydra_axis.hpp"

// ---------------------------------------------------------------------------
// v2_fixed batch-parallel feature extraction (verbatim port; types fx_t).
// Host contract: kernels sorted by dilation → uniform dilation per batch.
// 5 duplicate time-series BRAMs give the 9 conflict-free taps per cycle.
// ---------------------------------------------------------------------------
static void hydra_feature_extraction_fx(
    fx_t ts0[MAX_TIME_SERIES_LENGTH],
    fx_t ts1[MAX_TIME_SERIES_LENGTH],
    fx_t ts2[MAX_TIME_SERIES_LENGTH],
    fx_t ts3[MAX_TIME_SERIES_LENGTH],
    fx_t ts4[MAX_TIME_SERIES_LENGTH],
    fx_t features[MAX_FEATURES],
    fx_t local_weights[NUM_KERNELS][KERNEL_SIZE],
    fx_t biases[NUM_KERNELS],
    int_t local_dilations[NUM_KERNELS],
    int_t time_series_length
) {
    #pragma HLS INLINE off

    BATCH_LOOP: for (int_t batch = 0; batch < NUM_BATCHES; batch++) {
        #pragma HLS LOOP_TRIPCOUNT min=32 max=32

        int_t kg = batch * UNROLL_FACTOR;  // base kernel index for this batch

        // Uniform dilation per batch (host sorted by dilation).
        int_t dilation = local_dilations[kg];
        int_t kernel_span = (KERNEL_SIZE - 1) * dilation + 1;
        int_t conv_length = time_series_length - kernel_span + 1;

        // Preload this batch's weights+biases into registers. Pipelined
        // (16 cycles/batch) rather than fully unrolled so local_weights
        // only needs a dim-2 partition (9 BRAM banks), never a static+
        // complete dim-1 partition (register-inference gotcha).
        fx_t group_biases[UNROLL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=group_biases complete
        fx_t reg_weights[UNROLL_FACTOR][KERNEL_SIZE];
        #pragma HLS ARRAY_PARTITION variable=reg_weights complete dim=0

        PRELOAD: for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
            #pragma HLS PIPELINE II=1
            group_biases[ki] = biases[kg + ki];
            PRELOAD_W: for (int_t w = 0; w < KERNEL_SIZE; w++) {
                #pragma HLS UNROLL
                reg_weights[ki][w] = local_weights[kg + ki][w];
            }
        }

        // Accumulators fully in registers.
        acc_t running_sum[UNROLL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=running_sum complete
        fx_t running_max[UNROLL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=running_max complete

        INIT_ACC: for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
            #pragma HLS UNROLL
            running_max[ki] = (fx_t)(-32000);
            running_sum[ki] = 0;
        }

        // Fused conv + pooling: single pass, 16 kernels per cycle.
        CONV_POOL_LOOP: for (int_t t = 0; t < conv_length; t++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=6 max=5112

            fx_t sw[KERNEL_SIZE];
            #pragma HLS ARRAY_PARTITION variable=sw complete

            sw[0] = ts0[t];
            sw[1] = ts0[t + dilation];
            sw[2] = ts1[t + 2 * dilation];
            sw[3] = ts1[t + 3 * dilation];
            sw[4] = ts2[t + 4 * dilation];
            sw[5] = ts2[t + 5 * dilation];
            sw[6] = ts3[t + 6 * dilation];
            sw[7] = ts3[t + 7 * dilation];
            sw[8] = ts4[t + 8 * dilation];

            PARALLEL_KERNELS: for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
                #pragma HLS UNROLL
                fx_t value = group_biases[ki];
                DOT_PRODUCT: for (int_t w = 0; w < KERNEL_SIZE; w++) {
                    #pragma HLS UNROLL
                    value += sw[w] * reg_weights[ki][w];
                }
                if (value > running_max[ki]) {
                    running_max[ki] = value;
                }
                running_sum[ki] += (acc_t)value;
            }
        }

        // Mean via one reciprocal, then write 2 features per kernel.
        fx_t mean_vals[UNROLL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=mean_vals complete

        if (conv_length > 0) {
            acc_t inv_conv_length = (acc_t)1 / (acc_t)conv_length;
            COMPUTE_MEAN: for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
                #pragma HLS UNROLL
                mean_vals[ki] = (fx_t)(running_sum[ki] * inv_conv_length);
            }
        } else {
            ZERO_MEAN: for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
                #pragma HLS UNROLL
                mean_vals[ki] = (fx_t)0;
            }
        }

        WRITE_FEATURES: for (int_t ki = 0; ki < UNROLL_FACTOR; ki++) {
            #pragma HLS PIPELINE II=1
            int_t abs_k = kg + ki;
            int_t feat_idx = abs_k * 2;
            features[feat_idx]     = (conv_length > 0) ? running_max[ki] : (fx_t)0;
            features[feat_idx + 1] = mean_vals[ki];
        }
    }
}

// ---------------------------------------------------------------------------
// StandardScaler (v2_fixed): multiply by 1/scale, division done once per
// feature in acc_t.
// ---------------------------------------------------------------------------
static void apply_scaler_fx(
    fx_t features[MAX_FEATURES],
    fx_t scaler_mean[MAX_FEATURES],
    fx_t scaler_scale[MAX_FEATURES],
    int_t num_features
) {
    #pragma HLS INLINE off
    SCALE_LOOP: for (int_t i = 0; i < num_features; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1024 max=2048
        fx_t scale = scaler_scale[i];
        fx_t inv_s = (scale != (fx_t)0) ? (fx_t)((acc_t)1 / (acc_t)scale) : (fx_t)0;
        features[i] = (features[i] - scaler_mean[i]) * inv_s;
    }
}

// ---------------------------------------------------------------------------
// Ridge classifier. Coefficients kept in the model JSON's native
// FEATURE-major layout ([f * num_classes + c]) — matches load_hydra_hbm and
// the CPU responder; BRAM access order doesn't affect II. acc_t score
// gives the 1-cycle accumulate that keeps FEATURE_LOOP at II=1.
// ---------------------------------------------------------------------------
static void linear_classifier_fx(
    fx_t features[MAX_FEATURES],
    fx_t coefficients[MAX_FEATURES * MAX_CLASSES],
    fx_t intercept[MAX_CLASSES],
    int_t num_features,
    int_t num_classes,
    fx_t prediction[MAX_CLASSES]
) {
    #pragma HLS INLINE off
    CLASS_LOOP: for (int_t c = 0; c < num_classes; c++) {
        #pragma HLS LOOP_TRIPCOUNT min=2 max=10
        acc_t score = (acc_t)intercept[c];
        FEATURE_LOOP: for (int_t f = 0; f < num_features; f++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1024 max=2048
            score += (acc_t)features[f] * (acc_t)coefficients[f * num_classes + c];
        }
        prediction[c] = (fx_t)score;
    }
}

// ---------------------------------------------------------------------------
// DATAFLOW producer: packet -> float -> `raw_samples` ONLY. Never touches
// the compute chain or `output_predictions`, so nothing downstream can ever
// block this process. Non-blocking: only acts when a packet is available.
// (Verbatim minirocket_fused_ingest idiom.)
// ---------------------------------------------------------------------------
static void hydra_axis_ingest(
    hls::stream<pkt> &input_timeseries,
    hls::stream<data_t> &raw_samples,
    int_t time_series_length
) {
    #pragma HLS INLINE off

#if defined(COSIM_MAX_WINDOWS)
    // TEST-ONLY bounded free-run loop; never active for synthesis/hw.
    for (int __it = 0; __it < (COSIM_MAX_WINDOWS) * (int)time_series_length + 64; __it++) {
#elif BUILD == 1
    while (true) {
#endif
        if (!input_timeseries.empty()) {
            // ALWAYS drain the NetLayer app-in stream when a packet is
            // available -- this read must never depend on raw_samples
            // state, or the AXI-stream handshake with NetLayer stalls and
            // the whole RX path wedges.
            pkt v = input_timeseries.read();
            ap_uint<DWIDTH> tmp = v.data;
            data_t new_val = *((data_t*) &tmp);
            // DROP-ON-FULL: forward only if there is room; otherwise
            // discard this sample rather than block the read above.
            if (!raw_samples.full()) {
                raw_samples.write(new_val);
            }
        }
#if defined(COSIM_MAX_WINDOWS) || BUILD == 1
    }
#endif
}

// ---------------------------------------------------------------------------
// DATAFLOW consumer: buffer raw samples into 5 duplicate window BRAMs, run
// the v2_fixed compute chain once per full window, emit ONE packed
// prediction packet with a GUARDED non-blocking write.
// ---------------------------------------------------------------------------
static void hydra_axis_compute_emit(
    hls::stream<data_t> &raw_samples,
    hls::stream<pkt> &output_predictions,
    data_t* coefficients,
    data_t* intercept,
    data_t* scaler_mean,
    data_t* scaler_scale,
    data_t* kernel_weights,
    data_t* biases,
    int_t*  dilations,
    int_t time_series_length,
    int_t num_features,
    int_t num_classes
) {
    #pragma HLS INLINE off

    // 5 duplicate copies of the buffered series (conflict-free 9-tap
    // reads), persisted across per-packet calls. No ARRAY_PARTITION (the
    // bandwidth comes from the 5x duplication), so `static` is safe here.
    static fx_t local_ts0[MAX_TIME_SERIES_LENGTH];
    static fx_t local_ts1[MAX_TIME_SERIES_LENGTH];
    static fx_t local_ts2[MAX_TIME_SERIES_LENGTH];
    static fx_t local_ts3[MAX_TIME_SERIES_LENGTH];
    static fx_t local_ts4[MAX_TIME_SERIES_LENGTH];
    static int_t sample_count = 0;

    // Model parameters cached once. Only the dim-2 (9-bank BRAM) partition
    // is applied to a static array -- no static+complete-dim1 partitions
    // (register-inference gotcha).
    static bool params_loaded = false;
    static fx_t local_weights[NUM_KERNELS][KERNEL_SIZE];
    #pragma HLS ARRAY_PARTITION variable=local_weights complete dim=2
    static fx_t local_biases[NUM_KERNELS];
    static int_t local_dilations[NUM_KERNELS];
    static fx_t local_scaler_mean[MAX_FEATURES];
    static fx_t local_scaler_scale[MAX_FEATURES];
    static fx_t local_coefficients[MAX_FEATURES * MAX_CLASSES];

    // Small, reloaded every window (cheap; avoids static+complete risk).
    fx_t local_intercept[MAX_CLASSES];
    #pragma HLS ARRAY_PARTITION variable=local_intercept complete

    // Transient per-series scratch.
    fx_t local_features[MAX_FEATURES];
    fx_t local_prediction[MAX_CLASSES];

    // One-time copy of the bulk model parameters (float -> fx_t).
    if (!params_loaded) {
        LOAD_KW: for (int_t i = 0; i < NUM_KERNELS * KERNEL_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            local_weights[i / KERNEL_SIZE][i % KERNEL_SIZE] = (fx_t)kernel_weights[i];
        }
        LOAD_BIAS: for (int_t i = 0; i < NUM_KERNELS; i++) {
            #pragma HLS PIPELINE II=1
            local_biases[i] = (fx_t)biases[i];
            local_dilations[i] = dilations[i];
        }
        LOAD_SCALER: for (int_t i = 0; i < num_features; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1024 max=2048
            local_scaler_mean[i]  = (fx_t)scaler_mean[i];
            local_scaler_scale[i] = (fx_t)scaler_scale[i];
        }
        LOAD_COEF: for (int_t i = 0; i < num_features * num_classes; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=2048 max=20480
            local_coefficients[i] = (fx_t)coefficients[i];
        }
        params_loaded = true;
    }

#if defined(COSIM_MAX_WINDOWS)
    // TEST-ONLY bounded free-run loop; never active for synthesis/hw.
    for (int __it = 0; __it < (COSIM_MAX_WINDOWS) * (int)time_series_length + 64; __it++) {
#elif BUILD == 1
    while (true) {
#endif
        if (!raw_samples.empty()) {
            data_t new_val = raw_samples.read();

            if (sample_count >= 0 && sample_count < time_series_length &&
                sample_count < MAX_TIME_SERIES_LENGTH) {
                fx_t v = (fx_t)new_val;
                local_ts0[sample_count] = v;
                local_ts1[sample_count] = v;
                local_ts2[sample_count] = v;
                local_ts3[sample_count] = v;
                local_ts4[sample_count] = v;
                sample_count++;
            }

            // Full window buffered -> one inference + one packed emit.
            if (time_series_length > 0 && sample_count == time_series_length) {

                COPY_INTERCEPT: for (int_t i = 0; i < num_classes; i++) {
                    #pragma HLS PIPELINE II=1
                    local_intercept[i] = (fx_t)intercept[i];
                }

                hydra_feature_extraction_fx(
                    local_ts0, local_ts1, local_ts2, local_ts3, local_ts4,
                    local_features,
                    local_weights,
                    local_biases,
                    local_dilations,
                    time_series_length
                );

                apply_scaler_fx(
                    local_features,
                    local_scaler_mean,
                    local_scaler_scale,
                    num_features
                );

                linear_classifier_fx(
                    local_features,
                    local_coefficients,
                    local_intercept,
                    num_features,
                    num_classes,
                    local_prediction
                );

                // ONE packed packet: num_classes float32 logits in the low
                // num_classes*4 bytes. GUARDED write + Bug-B sideband.
                {
                    pkt out_pkt;
                    ap_uint<DWIDTH> out_data = 0;
                    PACK_PRED: for (int_t c = 0; c < MAX_CLASSES; c++) {
                        #pragma HLS UNROLL
                        if (c < num_classes) {
                            float logit = (float)local_prediction[c];
                            ap_uint<32> bits = *((ap_uint<32>*)&logit);
                            out_data.range(32 * c + 31, 32 * c) = bits;
                        }
                    }
                    out_pkt.data = out_data;
                    out_pkt.keep = (ap_uint<DWIDTH/8>(1) << (num_classes * 4)) - 1;
                    out_pkt.strb = -1;
                    out_pkt.last = 1;
                    out_pkt.dest = 0;
                    if (!output_predictions.full()) {
                        output_predictions.write(out_pkt);
                    }
                    // else: downstream backpressuring -- drop this
                    // prediction rather than stall the compute process.
                }

                sample_count = 0;
            }
        }
#if defined(COSIM_MAX_WINDOWS) || BUILD == 1
    }
#endif
}

// ---------------------------------------------------------------------------
// Top-level network AXIS kernel: DATAFLOW split, one internal
// single-producer/single-consumer stream. External interface (top name,
// arg order, AXIS ports) unchanged from v5 — load_hydra_hbm arg indices
// and config_axis.cfg stream names stay valid.
// ---------------------------------------------------------------------------
extern "C" void hydra_axis_inference(
    hls::stream<pkt> &input_timeseries,
    hls::stream<pkt> &output_predictions,
    data_t* coefficients,
    data_t* intercept,
    data_t* scaler_mean,
    data_t* scaler_scale,
    data_t* kernel_weights,
    data_t* biases,
    int_t*  dilations,
    int_t time_series_length,
    int_t num_features,
    int_t num_classes,
    int_t num_groups
) {
    #pragma HLS INTERFACE axis port=input_timeseries
    #pragma HLS INTERFACE axis port=output_predictions
    // NOTE: do NOT specify offset=slave here — for streaming kernels with
    // axis ports it causes HLS to drop the m_axi pointer slave registers
    // from the control bundle, making the kernel unprogrammable via XRT
    // (saturation_pivot 2026-05-04). Explicit s_axilite per pointer instead.
    #pragma HLS INTERFACE m_axi port=coefficients   bundle=gmem2 depth=20480
    #pragma HLS INTERFACE m_axi port=intercept      bundle=gmem3 depth=10
    #pragma HLS INTERFACE m_axi port=scaler_mean    bundle=gmem4 depth=2048
    #pragma HLS INTERFACE m_axi port=scaler_scale   bundle=gmem5 depth=2048
    #pragma HLS INTERFACE m_axi port=kernel_weights bundle=gmem6 depth=4608
    #pragma HLS INTERFACE m_axi port=biases         bundle=gmem7 depth=512
    #pragma HLS INTERFACE m_axi port=dilations      bundle=gmem8 depth=512
    #pragma HLS INTERFACE s_axilite port=coefficients   bundle=control
    #pragma HLS INTERFACE s_axilite port=intercept      bundle=control
    #pragma HLS INTERFACE s_axilite port=scaler_mean    bundle=control
    #pragma HLS INTERFACE s_axilite port=scaler_scale   bundle=control
    #pragma HLS INTERFACE s_axilite port=kernel_weights bundle=control
    #pragma HLS INTERFACE s_axilite port=biases         bundle=control
    #pragma HLS INTERFACE s_axilite port=dilations      bundle=control

    #pragma HLS INTERFACE s_axilite port=time_series_length bundle=control
    #pragma HLS INTERFACE s_axilite port=num_features       bundle=control
    #pragma HLS INTERFACE s_axilite port=num_classes        bundle=control
    #pragma HLS INTERFACE s_axilite port=num_groups         bundle=control
    #pragma HLS INTERFACE s_axilite port=return             bundle=control

    #pragma HLS DATAFLOW

    // Ingest -> compute channel. Depth = a full max window so one-window
    // producer/consumer skew never blocks ingest.
    hls::stream<data_t> raw_samples("raw_samples");
    #pragma HLS STREAM variable=raw_samples depth=MAX_TIME_SERIES_LENGTH

    hydra_axis_ingest(input_timeseries, raw_samples, time_series_length);

    hydra_axis_compute_emit(
        raw_samples, output_predictions,
        coefficients, intercept, scaler_mean, scaler_scale,
        kernel_weights, biases, dilations,
        time_series_length, num_features, num_classes
    );
    // num_groups is vestigial — kept in the signature only for ABI
    // compatibility with load_hydra_hbm arg indices.
}
