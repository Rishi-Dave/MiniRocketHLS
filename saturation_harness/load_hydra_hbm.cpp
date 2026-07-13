// load_hydra_hbm.cpp — HBM model loader for hydra_axis_inference.
//
// Companion to setup_netlayer.cpp. Run AFTER `xbutil program <xclbin>` and
// after setup_netlayer has programmed the NetLayer registers. This binary:
//   1. Opens the device + xclbin via xrt::device + register_xclbin.
//      register_xclbin on an already-loaded image is idempotent and does
//      NOT clobber NetLayer state (those AXI-Lite writes persist).
//   2. Opens xrt::kernel("hydra_axis_inference:{hydra_axis_inference_1}").
//   3. Allocates an xrt::bo per HBM-mapped arg using kernel.group_id(idx)
//      so each BO lands in the bank wired up at link time (HBM[2..8]).
//   4. Parses the model JSON (HYDRA schema: kernel_weights, biases,
//      dilations, scaler_*, coefficients, intercept, num_*).
//   5. Copies floats/ints into BOs, syncs H2D, sets kernel args incl.
//      scalars, and issues kernel.start() — kernel was synthesized with
//      BUILD=1 so it loops forever once started.
//
// Build:
//   g++ -std=c++17 -O2 -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
//       -o load_hydra_hbm load_hydra_hbm.cpp -lxrt_coreutil -luuid
//
// Usage:
//   ./load_hydra_hbm <xclbin> <bdf> <model_json>
// Example:
//   ./load_hydra_hbm hydra_axis/build_dir.hw/krnl.xclbin 0000:3b:00.1 \
//                    hydra_insectsound_model.json

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <experimental/xrt_xclbin.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------- Minimal JSON value extractor ----------
// Pulls a top-level field out of a (well-formed) HYDRA model JSON.
// We don't validate the JSON structure broadly — these files are produced
// by our own training scripts so the schema is fixed.

static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

// Find `"key":` and return the offset just past the colon.
static size_t find_key(const std::string& s, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t p = 0;
    while (true) {
        p = s.find(needle, p);
        if (p == std::string::npos)
            throw std::runtime_error("missing key: " + key);
        size_t after = p + needle.size();
        // skip whitespace, expect ':'
        while (after < s.size() && std::isspace((unsigned char)s[after])) after++;
        if (after < s.size() && s[after] == ':') return after + 1;
        p = after;  // matched a value-string elsewhere; keep searching
    }
}

// Read a scalar number after the colon. Trailing garbage ignored.
static double get_scalar(const std::string& s, const std::string& key) {
    size_t p = find_key(s, key);
    while (p < s.size() && std::isspace((unsigned char)s[p])) p++;
    char* end = nullptr;
    double v = std::strtod(s.c_str() + p, &end);
    if (end == s.c_str() + p)
        throw std::runtime_error("bad scalar for key: " + key);
    return v;
}

// Read a flat array of numbers. Handles nested arrays (recurses into [[]]
// by treating outer/inner '[' / ']' uniformly — this gives row-major flat).
static std::vector<double> get_array(const std::string& s, const std::string& key) {
    size_t p = find_key(s, key);
    while (p < s.size() && std::isspace((unsigned char)s[p])) p++;
    if (p >= s.size() || s[p] != '[')
        throw std::runtime_error("expected '[' for key: " + key);
    int depth = 0;
    std::vector<double> out;
    while (p < s.size()) {
        char c = s[p];
        if (c == '[') { depth++; p++; continue; }
        if (c == ']') { depth--; p++; if (depth == 0) return out; continue; }
        if (c == ',' || std::isspace((unsigned char)c)) { p++; continue; }
        // Try to parse a number
        char* end = nullptr;
        double v = std::strtod(s.c_str() + p, &end);
        if (end == s.c_str() + p)
            throw std::runtime_error("bad number in array: " + key);
        out.push_back(v);
        p = (size_t)(end - s.c_str());
    }
    throw std::runtime_error("unterminated array: " + key);
}

// ---------- HBM helper ----------

template <typename T>
static xrt::bo make_bo(xrt::device& dev, xrt::kernel& krnl, int arg_idx,
                       const std::vector<T>& src, const char* name) {
    if (src.empty())
        throw std::runtime_error(std::string("empty buffer for arg ") + name);
    size_t bytes = src.size() * sizeof(T);
    auto grp = krnl.group_id(arg_idx);
    xrt::bo bo(dev, bytes, grp);
    auto* p = bo.map<T*>();
    std::memcpy(p, src.data(), bytes);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::printf("[hbm ] arg%d %-20s size=%-10zu group=%u\n",
                arg_idx, name, bytes, grp);
    return bo;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
            "usage: %s <xclbin> <bdf> <model_json>\n", argv[0]);
        return 2;
    }
    const std::string xclbin_path = argv[1];
    const std::string bdf         = argv[2];
    const std::string model_path  = argv[3];

    std::printf("[info] loading model %s\n", model_path.c_str());
    std::string js = slurp(model_path);

    // --- Pull HYDRA schema fields ---
    int  num_kernels         = (int)get_scalar(js, "num_kernels");
    int  num_groups          = (int)get_scalar(js, "num_groups");
    int  num_features        = (int)get_scalar(js, "num_features");
    int  num_classes         = (int)get_scalar(js, "num_classes");
    int  time_series_length  = (int)get_scalar(js, "time_series_length");

    auto coefs_d   = get_array(js, "coefficients");
    auto intercept = get_array(js, "intercept");
    auto sm        = get_array(js, "scaler_mean");
    auto ss        = get_array(js, "scaler_scale");
    auto kw        = get_array(js, "kernel_weights");
    auto bi        = get_array(js, "biases");
    auto dil_d     = get_array(js, "dilations");

    // Sanity vs kernel signature.
    if ((int)coefs_d.size()   != num_classes * num_features) throw std::runtime_error("coefficients size mismatch");
    if ((int)intercept.size() != num_classes)                throw std::runtime_error("intercept size mismatch");
    if ((int)sm.size()        != num_features)               throw std::runtime_error("scaler_mean size mismatch");
    if ((int)ss.size()        != num_features)               throw std::runtime_error("scaler_scale size mismatch");
    if ((int)kw.size()        != num_kernels * 9)            throw std::runtime_error("kernel_weights size mismatch (expect num_kernels*9)");
    if ((int)bi.size()        != num_kernels)                throw std::runtime_error("biases size mismatch (expect num_kernels)");
    if ((int)dil_d.size()     != num_kernels)                throw std::runtime_error("dilations size mismatch (expect num_kernels)");

    // Convert to on-device types.
    std::vector<float>   coefs_f(coefs_d.begin(),   coefs_d.end());
    std::vector<float>   inter_f(intercept.begin(), intercept.end());
    std::vector<float>   sm_f   (sm.begin(),        sm.end());
    std::vector<float>   ss_f   (ss.begin(),        ss.end());
    std::vector<float>   kw_f   (kw.begin(),        kw.end());
    std::vector<float>   bi_f   (bi.begin(),        bi.end());
    std::vector<int32_t> dil_i  (dil_d.begin(),     dil_d.end());

    // --- v2_fixed HOST CONTRACT: sort kernels by dilation ---------------
    // The v6 hydra_axis kernel processes kernels in UNROLL=16 batches that
    // share ONE dilation (read from the batch's first kernel). A stable
    // sort by dilation guarantees uniform batches (512 kernels / 8 dilation
    // groups = 64 per dilation, a multiple of 16). Features are emitted in
    // kernel order (2 per kernel: max, mean), so the per-feature arrays
    // (scaler_mean, scaler_scale, coefficients rows — FEATURE-major
    // [f*num_classes+c]) must be permuted identically.
    {
        std::vector<int> order(num_kernels);
        for (int i = 0; i < num_kernels; i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return dil_i[a] < dil_i[b]; });

        std::vector<float>   kw_s(kw_f.size());
        std::vector<float>   bi_s(bi_f.size());
        std::vector<int32_t> dil_s(dil_i.size());
        std::vector<float>   sm_s(sm_f.size());
        std::vector<float>   ss_s(ss_f.size());
        std::vector<float>   coefs_s(coefs_f.size());

        for (int i = 0; i < num_kernels; i++) {
            int src = order[i];
            for (int w = 0; w < 9; w++) kw_s[i * 9 + w] = kw_f[src * 9 + w];
            bi_s[i]  = bi_f[src];
            dil_s[i] = dil_i[src];
            for (int p = 0; p < 2; p++) {           // 2 features per kernel
                int fd = i * 2 + p, fs = src * 2 + p;
                sm_s[fd] = sm_f[fs];
                ss_s[fd] = ss_f[fs];
                for (int c = 0; c < num_classes; c++)
                    coefs_s[fd * num_classes + c] = coefs_f[fs * num_classes + c];
            }
        }
        kw_f.swap(kw_s); bi_f.swap(bi_s); dil_i.swap(dil_s);
        sm_f.swap(sm_s); ss_f.swap(ss_s); coefs_f.swap(coefs_s);

        // Verify batch uniformity (16 kernels per batch, one dilation).
        for (int b = 0; b < num_kernels / 16; b++) {
            for (int k = 1; k < 16; k++) {
                if (dil_i[b * 16 + k] != dil_i[b * 16]) {
                    std::fprintf(stderr,
                        "FATAL: batch %d has mixed dilations (%d vs %d) after "
                        "sort — kernel math would be silently wrong\n",
                        b, (int)dil_i[b * 16 + k], (int)dil_i[b * 16]);
                    return 3;
                }
            }
        }
        std::printf("[sort] kernels dilation-sorted; %d batches of 16 verified uniform\n",
                    num_kernels / 16);
    }

    std::printf("[info] dataset shape: ts_len=%d num_kernels=%d num_groups=%d "
                "num_features=%d num_classes=%d\n",
                time_series_length, num_kernels, num_groups,
                num_features, num_classes);

    // --- XRT device + kernel ---
    std::printf("[info] opening device %s\n", bdf.c_str());
    auto dev = xrt::device(bdf);
    std::printf("[info] registering xclbin %s\n", xclbin_path.c_str());
    auto xb  = xrt::xclbin(xclbin_path);
    auto uuid = dev.register_xclbin(xb);

    const std::string krnl_inst = "hydra_axis_inference:{hydra_axis_inference_1}";
    std::printf("[info] opening kernel %s\n", krnl_inst.c_str());
    auto krnl = xrt::kernel(dev, uuid, krnl_inst,
                            xrt::kernel::cu_access_mode::exclusive);

    // --- HBM allocations ---
    // C++ arg positions (incl. AXIS streams which take indices but are not
    // settable). hydra_axis_inference signature:
    //   (in_strm[0], out_strm[1], coefficients[2], intercept[3],
    //    scaler_mean[4], scaler_scale[5], kernel_weights[6], biases[7],
    //    dilations[8], time_series_length[9], num_features[10],
    //    num_classes[11], num_groups[12])
    constexpr int A_COEFS  = 2;
    constexpr int A_INTER  = 3;
    constexpr int A_SMEAN  = 4;
    constexpr int A_SSCALE = 5;
    constexpr int A_KW     = 6;
    constexpr int A_BIAS   = 7;
    constexpr int A_DIL    = 8;
    constexpr int A_TSLEN  = 9;
    constexpr int A_NFEAT  = 10;
    constexpr int A_NCLS   = 11;
    constexpr int A_NGRP   = 12;

    auto bo_coefs  = make_bo(dev, krnl, A_COEFS,  coefs_f, "coefficients");
    auto bo_inter  = make_bo(dev, krnl, A_INTER,  inter_f, "intercept");
    auto bo_smean  = make_bo(dev, krnl, A_SMEAN,  sm_f,    "scaler_mean");
    auto bo_sscale = make_bo(dev, krnl, A_SSCALE, ss_f,    "scaler_scale");
    auto bo_kw     = make_bo(dev, krnl, A_KW,     kw_f,    "kernel_weights");
    auto bo_bias   = make_bo(dev, krnl, A_BIAS,   bi_f,    "biases");
    auto bo_dil    = make_bo(dev, krnl, A_DIL,    dil_i,   "dilations");

    // --- Bind all args, including scalars ---
    auto run = xrt::run(krnl);
    run.set_arg(A_COEFS,  bo_coefs);
    run.set_arg(A_INTER,  bo_inter);
    run.set_arg(A_SMEAN,  bo_smean);
    run.set_arg(A_SSCALE, bo_sscale);
    run.set_arg(A_KW,     bo_kw);
    run.set_arg(A_BIAS,   bo_bias);
    run.set_arg(A_DIL,    bo_dil);
    run.set_arg(A_TSLEN,  (int32_t)time_series_length);
    run.set_arg(A_NFEAT,  (int32_t)num_features);
    run.set_arg(A_NCLS,   (int32_t)num_classes);
    run.set_arg(A_NGRP,   (int32_t)num_groups);

    std::printf("[hold] HBM loaded, kernel entering host-side spin "
                "(start/wait per packet). SIGKILL to release.\n");
    std::fflush(stdout);
    // Host-side auto-restart spin. Same rationale as load_minirocket_hbm.cpp.
    // Pattern is start→wait per iteration; never call start() twice without
    // an intervening wait() (XRT throws "bad command state, can't launch").
    // Once the patched HYDRA-axis xclbin (with -DBUILD=1) is built, the
    // synthesized kernel will have its own while(true) loop and a single
    // run.start() suffices — but this spin is still safe.
    while (true) {
        run.start();
        run.wait();
    }
    return 0;  // unreachable
}
