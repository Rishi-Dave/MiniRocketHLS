// Standalone correctness harness for Prith's DP-Reuse MiniRocket convolution.
// Goal (systematic-debugging Phase 1/4): PROVE or refute the in-place ascending
// shift bug in (a) time-series ingestion and (b) conv center carry-forward, by
// comparing the incrementally-maintained conv map against a brute-force golden
// recompute of the current window, after each streamed sample.
//
// No HLS types needed: data_t=float, weights are 1-bit, bug is pure array movement.
// bFP multiply (optimized_fp_multiply) replicated exactly via a float bitfield union.
//
// Build:  g++ -O2 -std=c++14 validate_dpreuse.cpp -o validate_dpreuse && ./validate_dpreuse

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>

typedef float data_t;

typedef union {
    data_t   fp_num;
    uint32_t raw;
    struct { uint32_t mant : 23; uint32_t bexp : 8; uint32_t sign : 1; };
} float_num_t;

// Exact replica of Prith's optimized_fp_multiply (bFP, weight bit -> {-1,+2})
static data_t bfp_mul(uint8_t x, data_t y) {
    float_num_t yb; yb.fp_num = y;
    yb.sign = (~(yb.sign ^ x)) & 1u;
    yb.bexp = yb.bexp + x;
    return yb.fp_num;
}

// ---- test configuration ----
static const int KS = 9;          // KERNEL_SIZE
static const int HK = KS / 2;     // 4
static const int MAXLEN = 160;
static const int NK = 3;          // a few kernels
static uint8_t W[NK][KS];         // 1-bit weights

// single convolution centered at center_idx over time_series[] (matches single_convolution)
static data_t single_conv(const data_t* ts, int dilation, int len, int k, int center) {
    data_t v = 0.0f;
    for (int kk = -4; kk <= 4; kk++) {
        int idx = center + kk * dilation;
        data_t s = (idx < 0 || idx >= len) ? 0.0f : ts[idx];
        v += bfp_mul(W[k][kk + 4], s);
    }
    return v;
}

// GOLDEN: full recompute of the conv map for the current (correct) window.
static void golden_conv(const data_t* ts, int dilation, int len, int k, data_t* out) {
    for (int j = 0; j < len; j++) out[j] = single_conv(ts, dilation, len, k, j);
}

// ---- ingestion variants ----
// Prith's (buggy ascending right-shift): newest at index 0
static void ingest_prith(data_t* ts, int len, data_t nv) {
    for (int i = 1; i < len; i++) ts[i] = ts[i - 1];   // ascending in-place -> plateau
    ts[0] = nv;
}
// Correct right-shift (descending): newest at index 0
static void ingest_fixed(data_t* ts, int len, data_t nv) {
    for (int i = len - 1; i >= 1; i--) ts[i] = ts[i - 1];
    ts[0] = nv;
}

// ---- Prith's conv-map incremental update (exact replica of lines 146-166) ----
static void update_prith(const data_t* ts, int dilation, int len, int k, data_t conv[]) {
    int shift = dilation - 1;
    // beginning boundary
    for (int i = 0; i <= HK; i++) {
        for (int j = 1; j <= shift; j++)
            conv[i * dilation + j] = conv[i * dilation + j - 1];
        conv[i * dilation] = single_conv(ts, dilation, len, k, i * dilation);
    }
    // end boundary
    for (int i = 0; i <= HK; i++) {
        for (int j = 1; j <= shift; j++)
            conv[len - i * dilation - j] = conv[len - i * dilation - j + 1];
        if (len - i * dilation < len && len - i * dilation >= 0)
            conv[len - i * dilation] = single_conv(ts, dilation, len, k, len - i * dilation);
    }
    // center carry-forward (ascending in-place -> plateau bug)
    for (int i = dilation * HK + 1; i < len - dilation * HK; i++)
        conv[i] = conv[i - 1];
}

// ---- Candidate FIX: shift interior FIRST (descending), then recompute boundaries ----
static void update_fixed(const data_t* ts, int dilation, int len, int k, data_t conv[]) {
    // 1. shift whole interior right by one (descending, reads old before overwrite)
    for (int i = len - 1; i >= 1; i--) conv[i] = conv[i - 1];
    // 2. recompute every position whose window includes the new sample (idx 0)
    //    or the dropped sample (idx len-1): centers 0..4*dil and (len-1-4*dil)..(len-1)
    for (int c = 0; c <= HK * dilation && c < len; c++)
        conv[c] = single_conv(ts, dilation, len, k, c);
    for (int c = len - 1; c >= len - HK * dilation && c >= 0; c--)
        conv[c] = single_conv(ts, dilation, len, k, c);
}

static int run_case(int len, int dilation, int nsamples, const char* label,
                    void (*ingest)(data_t*, int, data_t),
                    void (*update)(const data_t*, int, int, int, data_t*)) {
    data_t ts_inc[MAXLEN]; memset(ts_inc, 0, sizeof(ts_inc));
    data_t ts_gold[MAXLEN]; memset(ts_gold, 0, sizeof(ts_gold));
    data_t conv_inc[NK][MAXLEN]; memset(conv_inc, 0, sizeof(conv_inc));
    data_t conv_gold[MAXLEN];

    // warm up incremental cache to the all-zero state's correct conv (all zero) -> ok

    double worst = 0.0; int worst_pos = -1, worst_k = -1, worst_sample = -1;
    for (int s = 0; s < nsamples; s++) {
        data_t nv = (data_t)((s * 37 % 19) - 9) * 0.5f;   // deterministic pseudo-data
        ingest(ts_inc, len, nv);
        ingest_fixed(ts_gold, len, nv);                    // golden window always correct
        for (int k = 0; k < NK; k++) {
            update(ts_inc, dilation, len, k, conv_inc[k]);
            golden_conv(ts_gold, dilation, len, k, conv_gold);
            for (int j = 0; j < len; j++) {
                double d = fabs((double)conv_inc[k][j] - (double)conv_gold[j]);
                if (d > worst) { worst = d; worst_pos = j; worst_k = k; worst_sample = s; }
            }
        }
    }
    printf("%-40s len=%d dil=%d : worst |inc-gold| = %.6g  (sample=%d k=%d pos=%d)\n",
           label, len, dilation, worst, worst_sample, worst_k, worst_pos);
    return worst > 1e-4 ? 1 : 0;
}

int main() {
    // deterministic 1-bit weights
    for (int k = 0; k < NK; k++)
        for (int j = 0; j < KS; j++)
            W[k][j] = (uint8_t)(((k * 7 + j * 3 + 1) % 3) == 0 ? 1 : 0);

    printf("=== bFP multiply sanity ===\n");
    printf("  bfp_mul(0, 3.0) = %g  (expect -3)\n", bfp_mul(0, 3.0f));
    printf("  bfp_mul(1, 3.0) = %g  (expect +6)\n", bfp_mul(1, 3.0f));
    printf("\n=== Prith's code as-is (expect large error = bug) ===\n");
    int bad = 0;
    bad += run_case(16, 1, 25, "PRITH ingest+update", ingest_prith, update_prith);
    bad += run_case(16, 2, 25, "PRITH ingest+update", ingest_prith, update_prith);

    printf("\n=== Isolate: correct TS ingest, Prith conv update ===\n");
    run_case(16, 1, 25, "fixed-ingest + PRITH-update", ingest_fixed, update_prith);

    printf("\n=== Candidate FIX (correct ingest + correct update) ===\n");
    int fixfail = 0;
    fixfail += run_case(16, 1, 25, "FIXED ingest+update", ingest_fixed, update_fixed);
    fixfail += run_case(16, 2, 25, "FIXED ingest+update", ingest_fixed, update_fixed);
    fixfail += run_case(32, 1, 40, "FIXED ingest+update", ingest_fixed, update_fixed);
    fixfail += run_case(32, 3, 40, "FIXED ingest+update", ingest_fixed, update_fixed);

    printf("\n=== Candidate FIX at the REAL model: len=150, dilations [1,2,3,5,7,9,13,18] ===\n");
    int dils[] = {1, 2, 3, 5, 7, 9, 13, 18};
    for (int di = 0; di < 8; di++)
        fixfail += run_case(150, dils[di], 200, "FIXED len150", ingest_fixed, update_fixed);

    printf("\n=== VERDICT ===\n");
    printf("Prith-as-is buggy (expected >0): %d case(s) diverged\n", bad);
    printf("Candidate fix failing (want 0):  %d case(s) diverged\n", fixfail);
    return fixfail;   // nonzero exit => fix not yet correct
}
