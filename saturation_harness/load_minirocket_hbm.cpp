// load_minirocket_hbm.cpp — HBM model loader for fpga-network minirocket_inference.
//
// Companion to setup_netlayer.cpp. Run AFTER `xbutil program <xclbin>` and
// after setup_netlayer has programmed the NetLayer registers. Same pattern
// as load_hydra_hbm.cpp but targets the MiniRocket kernel:
//   - 7 m_axi pointer args at HBM[2..8] (coefficients, intercept,
//     scaler_mean, scaler_scale, dilations, num_features_per_dilation,
//     biases) — see fpga-network/config.cfg.
//   - 4 scalar args (time_series_length, num_features, num_classes,
//     num_dilations).
// JSON schema keys (from minirocket_modular/<DS>_minirocket_model.json):
//   classifier_coef [[num_classes][num_features]], classifier_intercept,
//   scaler_mean, scaler_scale, dilations, num_features_per_dilation,
//   biases — plus scalars.
//
// IMPORTANT: the fpga-network minirocket_fused kernel (include/minirocket_fused.hpp)
// was synthesized with MAX_CLASSES=16, MAX_FEATURES=1024, MAX_DILATIONS=16,
// MAX_TIME_SERIES_LENGTH=8192 — FruitFlies (3 classes), MosquitoSound (6),
// and InsectSound (10) all fit. (Older fpga-network minirocket kernel builds
// were MAX_CLASSES=4 and rejected MosquitoSound/InsectSound; that limit no
// longer applies once the fused xclbin is deployed.)
//
// Build:
//   g++ -std=c++17 -O2 -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
//       -o load_minirocket_hbm load_minirocket_hbm.cpp -lxrt_coreutil -luuid
//
// Usage:
//   ./load_minirocket_hbm <xclbin> <bdf> <model_json>

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <experimental/xrt_xclbin.h>

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

static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

static size_t find_key(const std::string& s, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t p = 0;
    while (true) {
        p = s.find(needle, p);
        if (p == std::string::npos)
            throw std::runtime_error("missing key: " + key);
        size_t after = p + needle.size();
        while (after < s.size() && std::isspace((unsigned char)s[after])) after++;
        if (after < s.size() && s[after] == ':') return after + 1;
        p = after;
    }
}

static double get_scalar(const std::string& s, const std::string& key) {
    size_t p = find_key(s, key);
    while (p < s.size() && std::isspace((unsigned char)s[p])) p++;
    char* end = nullptr;
    double v = std::strtod(s.c_str() + p, &end);
    if (end == s.c_str() + p)
        throw std::runtime_error("bad scalar for key: " + key);
    return v;
}

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
        char* end = nullptr;
        double v = std::strtod(s.c_str() + p, &end);
        if (end == s.c_str() + p)
            throw std::runtime_error("bad number in array: " + key);
        out.push_back(v);
        p = (size_t)(end - s.c_str());
    }
    throw std::runtime_error("unterminated array: " + key);
}

// Kernel-side capacity limits from fpga-network/minirocket_fused/include/
// minirocket_fused.hpp — keep in sync with that header if the kernel is
// re-synthesized with different MAX_* values.
constexpr int MAX_CLASSES            = 16;
constexpr int MAX_FEATURES           = 1024;
constexpr int MAX_DILATIONS          = 16;
constexpr int MAX_TIME_SERIES_LENGTH = 8192;

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
    std::printf("[hbm ] arg%d %-26s size=%-10zu group=%u\n",
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

    int  num_dilations       = (int)get_scalar(js, "num_dilations");
    int  num_features        = (int)get_scalar(js, "num_features");
    int  num_classes         = (int)get_scalar(js, "num_classes");
    int  time_series_length  = (int)get_scalar(js, "time_series_length");

    if (num_classes > MAX_CLASSES) {
        std::fprintf(stderr,
            "[FATAL] num_classes=%d > MAX_CLASSES=%d of the fpga-network "
            "minirocket_fused kernel.\n", num_classes, MAX_CLASSES);
        return 3;
    }
    if (num_features > MAX_FEATURES) {
        std::fprintf(stderr,
            "[FATAL] num_features=%d > MAX_FEATURES=%d of the fpga-network "
            "minirocket_fused kernel.\n", num_features, MAX_FEATURES);
        return 3;
    }
    if (num_dilations > MAX_DILATIONS) {
        std::fprintf(stderr,
            "[FATAL] num_dilations=%d > MAX_DILATIONS=%d of the fpga-network "
            "minirocket_fused kernel.\n", num_dilations, MAX_DILATIONS);
        return 3;
    }
    if (time_series_length > MAX_TIME_SERIES_LENGTH) {
        std::fprintf(stderr,
            "[FATAL] time_series_length=%d > MAX_TIME_SERIES_LENGTH=%d of the "
            "fpga-network minirocket_fused kernel.\n",
            time_series_length, MAX_TIME_SERIES_LENGTH);
        return 3;
    }

    auto coefs_d   = get_array(js, "classifier_coef");
    auto intercept = get_array(js, "classifier_intercept");
    auto sm        = get_array(js, "scaler_mean");
    auto ss        = get_array(js, "scaler_scale");
    auto dil_d     = get_array(js, "dilations");
    auto nfpd_d    = get_array(js, "num_features_per_dilation");
    auto bi        = get_array(js, "biases");

    if ((int)coefs_d.size()   != num_classes * num_features) throw std::runtime_error("classifier_coef size mismatch");
    if ((int)intercept.size() != num_classes)                throw std::runtime_error("classifier_intercept size mismatch");
    if ((int)sm.size()        != num_features)               throw std::runtime_error("scaler_mean size mismatch");
    if ((int)ss.size()        != num_features)               throw std::runtime_error("scaler_scale size mismatch");
    if ((int)dil_d.size()     != num_dilations)              throw std::runtime_error("dilations size mismatch");
    if ((int)nfpd_d.size()    != num_dilations)              throw std::runtime_error("num_features_per_dilation size mismatch");
    if ((int)bi.size()        != num_features)               throw std::runtime_error("biases size mismatch");

    std::vector<float>   coefs_f(coefs_d.begin(),   coefs_d.end());
    std::vector<float>   inter_f(intercept.begin(), intercept.end());
    std::vector<float>   sm_f   (sm.begin(),        sm.end());
    std::vector<float>   ss_f   (ss.begin(),        ss.end());
    std::vector<int32_t> dil_i  (dil_d.begin(),     dil_d.end());
    std::vector<int32_t> nfpd_i (nfpd_d.begin(),    nfpd_d.end());
    std::vector<float>   bi_f   (bi.begin(),        bi.end());

    std::printf("[info] dataset shape: ts_len=%d num_dilations=%d "
                "num_features=%d num_classes=%d\n",
                time_series_length, num_dilations,
                num_features, num_classes);

    std::printf("[info] opening device %s\n", bdf.c_str());
    auto dev = xrt::device(bdf);
    std::printf("[info] registering xclbin %s\n", xclbin_path.c_str());
    auto xb  = xrt::xclbin(xclbin_path);
    auto uuid = dev.register_xclbin(xb);

    // CU instance name. Default = fused kernel; override via optional argv[4]
    // (e.g. "minirocket_inference:{minirocket_inference_1}" for the v3-dpr xclbin).
    const std::string krnl_inst = (argc > 4)
        ? std::string(argv[4])
        : "minirocket_fused_inference:{minirocket_fused_inference_1}";
    std::printf("[info] opening kernel %s\n", krnl_inst.c_str());
    auto krnl = xrt::kernel(dev, uuid, krnl_inst,
                            xrt::kernel::cu_access_mode::exclusive);

    // C++ arg positions (incl. AXIS streams). minirocket_inference signature:
    //   (in_strm[0], out_strm[1], coefficients[2], intercept[3],
    //    scaler_mean[4], scaler_scale[5], dilations[6],
    //    num_features_per_dilation[7], biases[8],
    //    time_series_length[9], num_features[10], num_classes[11],
    //    num_dilations[12])
    constexpr int A_COEFS  = 2;
    constexpr int A_INTER  = 3;
    constexpr int A_SMEAN  = 4;
    constexpr int A_SSCALE = 5;
    constexpr int A_DIL    = 6;
    constexpr int A_NFPD   = 7;
    constexpr int A_BIAS   = 8;
    constexpr int A_TSLEN  = 9;
    constexpr int A_NFEAT  = 10;
    constexpr int A_NCLS   = 11;
    constexpr int A_NDIL   = 12;

    auto bo_coefs  = make_bo(dev, krnl, A_COEFS,  coefs_f, "coefficients");
    auto bo_inter  = make_bo(dev, krnl, A_INTER,  inter_f, "intercept");
    auto bo_smean  = make_bo(dev, krnl, A_SMEAN,  sm_f,    "scaler_mean");
    auto bo_sscale = make_bo(dev, krnl, A_SSCALE, ss_f,    "scaler_scale");
    auto bo_dil    = make_bo(dev, krnl, A_DIL,    dil_i,   "dilations");
    auto bo_nfpd   = make_bo(dev, krnl, A_NFPD,   nfpd_i,  "num_features_per_dilation");
    auto bo_bias   = make_bo(dev, krnl, A_BIAS,   bi_f,    "biases");

    auto run = xrt::run(krnl);
    run.set_arg(A_COEFS,  bo_coefs);
    run.set_arg(A_INTER,  bo_inter);
    run.set_arg(A_SMEAN,  bo_smean);
    run.set_arg(A_SSCALE, bo_sscale);
    run.set_arg(A_DIL,    bo_dil);
    run.set_arg(A_NFPD,   bo_nfpd);
    run.set_arg(A_BIAS,   bo_bias);
    run.set_arg(A_TSLEN,  (int32_t)time_series_length);
    run.set_arg(A_NFEAT,  (int32_t)num_features);
    run.set_arg(A_NCLS,   (int32_t)num_classes);
    run.set_arg(A_NDIL,   (int32_t)num_dilations);

    std::printf("[hold] HBM loaded, kernel entering host-side spin "
                "(start/wait per packet). SIGKILL to release.\n");
    std::fflush(stdout);
    // Host-side auto-restart spin. The kernel HLS source has the streaming
    // while(true) gated by `#if BUILD == 1`, but the make.tcl that built the
    // existing fpga-network MiniRocket xclbin did NOT define -DBUILD=1, so
    // the synthesized kernel processes ONE input packet per ap_start and
    // ap_done's. XRT 2.16's run.start() has no autostart parameter, so we
    // re-trigger ap_start each time the kernel completes. Pattern is
    // start→wait→start→wait... — note: do NOT call start() twice without an
    // intervening wait(), XRT throws "bad command state, can't launch".
    // Throughput cap ~5-10k inf/s due to host XRT IPC overhead — sufficient
    // for FruitFlies smoke + low-rate sweeps. The new (post-pragma-fix +
    // -DBUILD=1) axis xclbins will not need this loop.
    while (true) {
        run.start();
        run.wait();
    }
    return 0;  // unreachable
}
