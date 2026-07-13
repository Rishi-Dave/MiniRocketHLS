/*
 * hydra_stream_host.cpp
 *
 * Host application for the 3-kernel streaming HYDRA pipeline:
 *   load_1 (HBM[0] -> AXI-Stream)
 *     -> hydra_inference_1 (AXI-Stream -> inference -> AXI-Stream, weights on HBM[2-8])
 *       -> store_1 (AXI-Stream -> HBM[1])
 *
 * Kernel argument layout (from xclbin EMBEDDED_METADATA):
 *   load:            arg0=time_series_input(buf), arg1=output_r(stream,internal), arg2=num_values
 *   hydra_inference: arg0=time_series_input(stream,internal), arg1=prediction_output(stream,internal),
 *                    arg2=coefficients, arg3=intercept, arg4=scaler_mean, arg5=scaler_scale,
 *                    arg6=kernel_weights, arg7=biases, arg8=dilations,
 *                    arg9=time_series_length, arg10=num_features, arg11=num_classes, arg12=num_groups
 *   store:           arg0=prediction_input(stream,internal), arg1=prediction_output(buf),
 *                    arg2=num_values, arg3=num_classes
 *
 * Stream args are wired internally (via sc= in config.cfg) and NOT set by the host.
 * Host only touches the m_axi buffer args and s_axilite scalar args.
 *
 * Model data reuses the proven HydraJSONLoader from hydra_optimized (including
 * kernel sorting by dilation, scaler/coefficient reordering).
 *
 * Usage: ./hydra_stream_host <xclbin> <model.json> <test.json>
 */

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <map>
#include <chrono>

// ─── Types (must match hydra.hpp in hydra_stream) ───────────────────────────
typedef float   data_t;
typedef int     int_t;

// ─── HYDRA Constants ─────────────────────────────────────────────────────────
#define NUM_KERNELS   512
#define KERNEL_SIZE   9
#define MAX_FEATURES  2048
#define MAX_CLASSES   10

// ─── OCL helper ──────────────────────────────────────────────────────────────
#define OCL_CHECK(error, call)                                             \
    call;                                                                  \
    if (error != CL_SUCCESS) {                                             \
        std::cerr << "OpenCL Error: " << error << " in " << #call         \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
        exit(EXIT_FAILURE);                                                \
    }

template <typename T>
struct aligned_allocator {
    using value_type = T;
    aligned_allocator() {}
    aligned_allocator(const aligned_allocator&) {}
    template <typename U> aligned_allocator(const aligned_allocator<U>&) {}
    T* allocate(std::size_t num) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 4096, num * sizeof(T)))
            throw std::bad_alloc();
        return reinterpret_cast<T*>(ptr);
    }
    void deallocate(T* p, std::size_t) { free(p); }
};

// ─── Minimal JSON loader (copied from hydra_loader_v2.cpp logic) ─────────────
static std::string read_file(const std::string& fn) {
    std::ifstream f(fn);
    if (!f.is_open()) { std::cerr << "Cannot open: " << fn << std::endl; return ""; }
    std::string c, l;
    while (std::getline(f, l)) c += l;
    return c;
}
static void trim(std::string& s) {
    s.erase(0, s.find_first_not_of(" \t\n\r\f\v"));
    s.erase(s.find_last_not_of(" \t\n\r\f\v") + 1);
}
static int parse_int(const std::string& c, const std::string& k) {
    size_t p = c.find("\"" + k + "\""); if (p == std::string::npos) return -1;
    size_t q = c.find(":", p) + 1, r = c.find_first_of(",}", q);
    std::string v = c.substr(q, r - q); trim(v); return std::stoi(v);
}
static std::vector<float> parse_float_arr(const std::string& c, const std::string& k) {
    std::vector<float> res;
    size_t p = c.find("\"" + k + "\""); if (p == std::string::npos) return res;
    size_t a = c.find("[", p), b = c.find("]", a);
    std::string ac = c.substr(a + 1, b - a - 1);
    std::stringstream ss(ac); std::string it;
    while (std::getline(ss, it, ',')) { trim(it); if (!it.empty()) res.push_back(std::stof(it)); }
    return res;
}
static std::vector<int> parse_int_arr(const std::string& c, const std::string& k) {
    std::vector<int> res;
    size_t p = c.find("\"" + k + "\""); if (p == std::string::npos) return res;
    size_t a = c.find("[", p), b = c.find("]", a);
    std::string ac = c.substr(a + 1, b - a - 1);
    std::stringstream ss(ac); std::string it;
    while (std::getline(ss, it, ',')) {
        trim(it);
        if (!it.empty()) {
            if (it.find('"') != std::string::npos) return std::vector<int>();
            try { res.push_back(std::stoi(it)); } catch (...) { return std::vector<int>(); }
        }
    }
    return res;
}
static std::vector<std::string> parse_str_arr(const std::string& c, const std::string& k) {
    std::vector<std::string> res;
    size_t p = c.find("\"" + k + "\""); if (p == std::string::npos) return res;
    size_t a = c.find("[", p), b = c.find("]", a);
    std::string ac = c.substr(a + 1, b - a - 1);
    size_t pos = 0;
    while (pos < ac.size()) {
        size_t qs = ac.find("\"", pos); if (qs == std::string::npos) break;
        size_t qe = ac.find("\"", qs + 1); if (qe == std::string::npos) break;
        res.push_back(ac.substr(qs + 1, qe - qs - 1)); pos = qe + 1;
    }
    return res;
}
static std::vector<std::vector<float>> parse_2d_float(const std::string& c, const std::string& k) {
    std::vector<std::vector<float>> res;
    size_t p = c.find("\"" + k + "\""); if (p == std::string::npos) return res;
    size_t a = c.find("[", p), ae = a;
    int bc = 0;
    for (size_t i = a; i < c.size(); i++) {
        if (c[i] == '[') bc++; if (c[i] == ']') bc--;
        if (bc == 0) { ae = i; break; }
    }
    std::string outer = c.substr(a + 1, ae - a - 1);
    size_t pos = 0;
    while (pos < outer.size()) {
        size_t ss = outer.find("[", pos); if (ss == std::string::npos) break;
        size_t se = outer.find("]", ss); if (se == std::string::npos) break;
        std::string sub = outer.substr(ss + 1, se - ss - 1);
        std::vector<float> row;
        std::stringstream sss(sub); std::string it;
        while (std::getline(sss, it, ',')) { trim(it); if (!it.empty()) row.push_back(std::stof(it)); }
        res.push_back(row); pos = se + 1;
    }
    return res;
}
static int map_insectsound(const std::string& l) {
    static const std::map<std::string,int> m = {
        {"aedes_female",0},{"aedes_male",1},{"fruit_flies",2},{"house_flies",3},
        {"quinx_female",4},{"quinx_male",5},{"stigma_female",6},{"stigma_male",7},
        {"tarsalis_female",8},{"tarsalis_male",9}
    };
    auto it = m.find(l);
    if (it != m.end()) return it->second;
    std::cerr << "Unknown label: " << l << std::endl; return -1;
}

// ─── xclbin loader ───────────────────────────────────────────────────────────
static std::vector<char> load_xclbin(const std::string& fn) {
    std::ifstream f(fn, std::ios::binary | std::ios::ate);
    if (!f.is_open()) { std::cerr << "Cannot open xclbin: " << fn << std::endl; exit(1); }
    std::streamsize sz = f.tellg(); f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz); return buf;
}

// ─── main ────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::cout << "\n========================================\n";
    std::cout << "  HYDRA Streaming FPGA Host\n";
    std::cout << "========================================\n\n";

    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <xclbin> <model.json> <test.json>\n";
        return EXIT_FAILURE;
    }
    std::string xclbin_file = argv[1];
    std::string model_file  = argv[2];
    std::string test_file   = argv[3];

    // ── Load model ───────────────────────────────────────────────────────────
    std::cout << "Loading model: " << model_file << "\n";
    std::string mc = read_file(model_file);
    if (mc.empty()) { std::cerr << "Failed to read model\n"; return 1; }

    int num_kernels        = parse_int(mc, "num_kernels");
    int num_groups         = parse_int(mc, "num_groups");
    int num_features       = parse_int(mc, "num_features");
    int num_classes        = parse_int(mc, "num_classes");
    int time_series_length = parse_int(mc, "time_series_length");

    std::cout << "  kernels=" << num_kernels << " groups=" << num_groups
              << " features=" << num_features << " classes=" << num_classes
              << " ts_len=" << time_series_length << "\n";

    if (num_kernels > NUM_KERNELS || num_features > MAX_FEATURES || num_classes > MAX_CLASSES) {
        std::cerr << "Model exceeds HLS limits\n"; return 1;
    }

    // Static arrays matching HLS kernel dimensions
    data_t kernel_weights[NUM_KERNELS][KERNEL_SIZE] = {};
    data_t biases[NUM_KERNELS]                      = {};
    int_t  dilations[NUM_KERNELS]                   = {};
    data_t scaler_mean[MAX_FEATURES]                = {};
    data_t scaler_scale[MAX_FEATURES]               = {};
    data_t coefficients[MAX_CLASSES][MAX_FEATURES]  = {};
    data_t intercept[MAX_CLASSES]                   = {};

    auto kw_flat = parse_float_arr(mc, "kernel_weights");
    auto bv      = parse_float_arr(mc, "biases");
    auto dv      = parse_int_arr(mc,   "dilations");
    auto sm      = parse_float_arr(mc, "scaler_mean");
    auto ss      = parse_float_arr(mc, "scaler_scale");
    auto cf_flat = parse_float_arr(mc, "coefficients");
    auto iv      = parse_float_arr(mc, "intercept");

    if ((int)kw_flat.size() != num_kernels * KERNEL_SIZE ||
        (int)bv.size()  != num_kernels ||
        (int)dv.size()  != num_kernels ||
        (int)sm.size()  != num_features ||
        (int)ss.size()  != num_features ||
        (int)cf_flat.size() != num_classes * num_features ||
        (int)iv.size()  != num_classes) {
        std::cerr << "Model size mismatch\n"; return 1;
    }

    for (int i = 0; i < num_kernels; i++) {
        for (int w = 0; w < KERNEL_SIZE; w++)
            kernel_weights[i][w] = kw_flat[i * KERNEL_SIZE + w];
        biases[i]    = bv[i];
        dilations[i] = dv[i];
    }
    for (int i = 0; i < num_features; i++) {
        scaler_mean[i]  = sm[i];
        scaler_scale[i] = ss[i];
    }
    // coefficients: JSON stores coef_.T.flatten() (feature-major):
    // coef_flat[f * num_classes + c] = coeff for class c, feature f
    for (int c = 0; c < num_classes; c++)
        for (int f = 0; f < num_features; f++)
            coefficients[c][f] = cf_flat[f * num_classes + c];
    for (int i = 0; i < num_classes; i++) intercept[i] = iv[i];

    // Sort kernels by dilation (same as hydra_host.cpp HYDRA_NO_SORT path)
    {
        std::vector<int> perm(num_kernels);
        std::iota(perm.begin(), perm.end(), 0);
        std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) {
            return dilations[a] < dilations[b];
        });
        data_t tmp_kw[NUM_KERNELS][KERNEL_SIZE];
        data_t tmp_b[NUM_KERNELS];
        int_t  tmp_d[NUM_KERNELS];
        for (int i = 0; i < num_kernels; i++) {
            for (int w = 0; w < KERNEL_SIZE; w++) tmp_kw[i][w] = kernel_weights[perm[i]][w];
            tmp_b[i] = biases[perm[i]];
            tmp_d[i] = dilations[perm[i]];
        }
        memcpy(kernel_weights, tmp_kw, sizeof(tmp_kw));
        memcpy(biases, tmp_b, sizeof(tmp_b));
        memcpy(dilations, tmp_d, sizeof(tmp_d));

        // Reorder scaler/coefficients to match new feature order
        std::vector<int> fp(num_features);
        for (int i = 0; i < num_kernels; i++) { fp[2*i] = 2*perm[i]; fp[2*i+1] = 2*perm[i]+1; }
        data_t tm[MAX_FEATURES], ts[MAX_FEATURES];
        for (int i = 0; i < num_features; i++) { tm[i] = scaler_mean[fp[i]]; ts[i] = scaler_scale[fp[i]]; }
        memcpy(scaler_mean, tm, num_features * sizeof(data_t));
        memcpy(scaler_scale, ts, num_features * sizeof(data_t));
        data_t tc[MAX_CLASSES][MAX_FEATURES];
        for (int c = 0; c < num_classes; c++)
            for (int f = 0; f < num_features; f++) tc[c][f] = coefficients[c][fp[f]];
        memcpy(coefficients, tc, sizeof(tc));
        std::cout << "  Kernels sorted by dilation\n";
    }

    // ── Load test data ────────────────────────────────────────────────────────
    std::cout << "Loading test data: " << test_file << "\n";
    std::string tc = read_file(test_file);
    if (tc.empty()) { std::cerr << "Failed to read test data\n"; return 1; }

    int test_num_samples = parse_int(tc, "num_samples");
    int test_ts_len      = parse_int(tc, "time_series_length");
    int test_num_classes = parse_int(tc, "num_classes");
    std::cout << "  samples=" << test_num_samples << " ts_len=" << test_ts_len
              << " classes=" << test_num_classes << "\n\n";

    auto test_inputs = parse_2d_float(tc, "time_series");
    if ((int)test_inputs.size() != test_num_samples) {
        std::cerr << "time_series size mismatch\n"; return 1;
    }

    std::vector<int> test_labels = parse_int_arr(tc, "labels");
    if (test_labels.empty()) {
        auto sl = parse_str_arr(tc, "labels");
        if (sl.empty()) { std::cerr << "No labels found\n"; return 1; }
        for (auto& l : sl) { int x = map_insectsound(l); if (x < 0) return 1; test_labels.push_back(x); }
    }
    if ((int)test_labels.size() != test_num_samples) { std::cerr << "Labels size mismatch\n"; return 1; }

    // ── OpenCL setup ──────────────────────────────────────────────────────────
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform plat;
    for (auto& p : platforms)
        if (p.getInfo<CL_PLATFORM_NAME>() == "Xilinx") { plat = p; break; }

    std::vector<cl::Device> devices;
    plat.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
    if (devices.empty()) { std::cerr << "No Xilinx devices\n"; return 1; }
    cl::Device device = devices[0];
    std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

    cl_int err;
    OCL_CHECK(err, cl::Context ctx(device, nullptr, nullptr, nullptr, &err));
    OCL_CHECK(err, cl::CommandQueue q(ctx, device,
        CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

    auto xclbin_data = load_xclbin(xclbin_file);
    cl::Program::Binaries bins{{xclbin_data.data(), xclbin_data.size()}};
    std::vector<cl::Device> dlist = {device};
    OCL_CHECK(err, cl::Program program(ctx, dlist, bins, nullptr, &err));
    std::cout << "XClbin loaded\n";

    // Create the three kernels
    OCL_CHECK(err, cl::Kernel krnl_load      (program, "load",            &err));
    OCL_CHECK(err, cl::Kernel krnl_inference  (program, "hydra_inference", &err));
    OCL_CHECK(err, cl::Kernel krnl_store      (program, "store",           &err));
    std::cout << "Kernels created (load, hydra_inference, store)\n";

    // ── Allocate device buffers for model weights (persistent across samples) ─
    std::vector<data_t, aligned_allocator<data_t>> h_kernel_weights(num_kernels * KERNEL_SIZE);
    std::vector<data_t, aligned_allocator<data_t>> h_biases(num_kernels);
    std::vector<int_t,  aligned_allocator<int_t>>  h_dilations(num_kernels);
    std::vector<data_t, aligned_allocator<data_t>> h_scaler_mean(num_features);
    std::vector<data_t, aligned_allocator<data_t>> h_scaler_scale(num_features);
    std::vector<data_t, aligned_allocator<data_t>> h_coefficients(num_classes * num_features);
    std::vector<data_t, aligned_allocator<data_t>> h_intercept(num_classes);

    for (int i = 0; i < num_kernels; i++) {
        for (int w = 0; w < KERNEL_SIZE; w++)
            h_kernel_weights[i * KERNEL_SIZE + w] = kernel_weights[i][w];
        h_biases[i]    = biases[i];
        h_dilations[i] = dilations[i];
    }
    for (int i = 0; i < num_features; i++) {
        h_scaler_mean[i]  = scaler_mean[i];
        h_scaler_scale[i] = scaler_scale[i];
    }
    // Flatten coefficients for kernel: row-major [num_features][num_classes]
    // (kernel indexes as coefficients[f * num_classes + c])
    for (int c = 0; c < num_classes; c++)
        for (int f = 0; f < num_features; f++)
            h_coefficients[f * num_classes + c] = coefficients[c][f];
    for (int i = 0; i < num_classes; i++) h_intercept[i] = intercept[i];

    // Per-sample I/O buffers
    std::vector<data_t, aligned_allocator<data_t>> h_ts(test_ts_len);
    std::vector<data_t, aligned_allocator<data_t>> h_preds(num_classes);

    // Create device buffers using kernel-associated memory.
    // For multi-kernel streaming xclbins, buffers must be created in the context
    // of the kernel that owns the m_axi port. We do this by setting the kernel arg
    // first, then letting XRT auto-assign the bank. Use COPY_HOST_PTR for
    // read-only weight buffers (no per-sample transfer needed after init).
    // Use plain READ_WRITE for I/O buffers (written per sample via enqueueWriteBuffer).
    OCL_CHECK(err, cl::Buffer buf_ts(ctx,
        CL_MEM_READ_ONLY,
        test_ts_len * sizeof(data_t), nullptr, &err));

    OCL_CHECK(err, cl::Buffer buf_pred(ctx,
        CL_MEM_WRITE_ONLY,
        num_classes * sizeof(data_t), nullptr, &err));

    OCL_CHECK(err, cl::Buffer buf_coef(ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        num_classes * num_features * sizeof(data_t), h_coefficients.data(), &err));

    OCL_CHECK(err, cl::Buffer buf_intercept(ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        num_classes * sizeof(data_t), h_intercept.data(), &err));

    OCL_CHECK(err, cl::Buffer buf_sm(ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        num_features * sizeof(data_t), h_scaler_mean.data(), &err));

    OCL_CHECK(err, cl::Buffer buf_ss(ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        num_features * sizeof(data_t), h_scaler_scale.data(), &err));

    OCL_CHECK(err, cl::Buffer buf_kw(ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        num_kernels * KERNEL_SIZE * sizeof(data_t), h_kernel_weights.data(), &err));

    OCL_CHECK(err, cl::Buffer buf_b(ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        num_kernels * sizeof(data_t), h_biases.data(), &err));

    OCL_CHECK(err, cl::Buffer buf_d(ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        num_kernels * sizeof(int_t), h_dilations.data(), &err));

    // Model weights are already on device via COPY_HOST_PTR at buffer creation.
    OCL_CHECK(err, err = q.finish());
    std::cout << "Model weights ready on device\n\n";

    // ── Set static kernel arguments ───────────────────────────────────────────
    // load: arg0=buf_ts (set per sample), arg2=num_values
    // (arg1 is the internal stream - NOT set by host)
    OCL_CHECK(err, err = krnl_load.setArg(0, buf_ts));
    OCL_CHECK(err, err = krnl_load.setArg(2, static_cast<unsigned int>(test_ts_len)));

    // hydra_inference: stream args 0,1 are internal; set weight buffers and scalars
    // arg order: 0=ts_stream(internal), 1=pred_stream(internal),
    //            2=coefficients, 3=intercept, 4=scaler_mean, 5=scaler_scale,
    //            6=kernel_weights, 7=biases, 8=dilations,
    //            9=time_series_length, 10=num_features, 11=num_classes, 12=num_groups
    OCL_CHECK(err, err = krnl_inference.setArg(2,  buf_coef));
    OCL_CHECK(err, err = krnl_inference.setArg(3,  buf_intercept));
    OCL_CHECK(err, err = krnl_inference.setArg(4,  buf_sm));
    OCL_CHECK(err, err = krnl_inference.setArg(5,  buf_ss));
    OCL_CHECK(err, err = krnl_inference.setArg(6,  buf_kw));
    OCL_CHECK(err, err = krnl_inference.setArg(7,  buf_b));
    OCL_CHECK(err, err = krnl_inference.setArg(8,  buf_d));
    OCL_CHECK(err, err = krnl_inference.setArg(9,  static_cast<unsigned int>(test_ts_len)));
    OCL_CHECK(err, err = krnl_inference.setArg(10, static_cast<unsigned int>(num_features)));
    OCL_CHECK(err, err = krnl_inference.setArg(11, static_cast<unsigned int>(num_classes)));
    OCL_CHECK(err, err = krnl_inference.setArg(12, static_cast<unsigned int>(num_groups)));

    // store: arg0=pred_stream(internal), arg1=buf_pred, arg2=num_values(1 per run), arg3=num_classes
    OCL_CHECK(err, err = krnl_store.setArg(1, buf_pred));
    OCL_CHECK(err, err = krnl_store.setArg(2, static_cast<unsigned int>(1)));
    OCL_CHECK(err, err = krnl_store.setArg(3, static_cast<unsigned int>(num_classes)));

    // ── Inference loop ────────────────────────────────────────────────────────
    std::cout << "========================================\n";
    std::cout << "Running HYDRA Streaming Inference\n";
    std::cout << "========================================\n\n";

    int correct = 0;
    double total_kernel_time = 0.0;

    for (int s = 0; s < test_num_samples; s++) {
        // Write time series to device buffer
        OCL_CHECK(err, err = q.enqueueWriteBuffer(buf_ts, CL_TRUE, 0,
            test_ts_len * sizeof(data_t), test_inputs[s].data()));

        // Enqueue all three kernels; measure load-to-store wall time
        cl::Event ev_load, ev_infer, ev_store;
        OCL_CHECK(err, err = q.enqueueTask(krnl_load,       nullptr, &ev_load));
        OCL_CHECK(err, err = q.enqueueTask(krnl_inference,  nullptr, &ev_infer));
        OCL_CHECK(err, err = q.enqueueTask(krnl_store,       nullptr, &ev_store));
        OCL_CHECK(err, err = q.finish());

        // Measure wall-time from load start to store end
        cl_ulong t0, t1;
        ev_load.getProfilingInfo(CL_PROFILING_COMMAND_START, &t0);
        ev_store.getProfilingInfo(CL_PROFILING_COMMAND_END,  &t1);
        double ms = (t1 - t0) / 1e6;
        total_kernel_time += ms;

        // Read back predictions
        OCL_CHECK(err, err = q.enqueueReadBuffer(buf_pred, CL_TRUE, 0,
            num_classes * sizeof(data_t), h_preds.data()));
        OCL_CHECK(err, err = q.finish());

        // argmax
        int pred = 0;
        data_t best = h_preds[0];
        for (int c = 1; c < num_classes; c++)
            if (h_preds[c] > best) { best = h_preds[c]; pred = c; }

        bool ok = (pred == test_labels[s]);
        if (ok) correct++;

        if (s < 10 || s % 5000 == 0)
            std::cout << "Sample " << std::setw(6) << s+1 << "/" << test_num_samples
                      << ": pred=" << pred << " true=" << test_labels[s]
                      << " time=" << std::fixed << std::setprecision(3) << ms << "ms"
                      << " [" << (ok ? "OK" : "FAIL") << "]\n";
    }

    double accuracy   = (double)correct / test_num_samples * 100.0;
    double avg_lat_ms = total_kernel_time / test_num_samples;
    double throughput = 1000.0 / avg_lat_ms;

    std::cout << "\n========================================\n";
    std::cout << "Results\n";
    std::cout << "========================================\n";
    std::cout << "Accuracy  : " << correct << "/" << test_num_samples
              << " = " << std::fixed << std::setprecision(2) << accuracy << "%\n";
    std::cout << "Avg latency: " << std::fixed << std::setprecision(3) << avg_lat_ms << " ms\n";
    std::cout << "Throughput : " << std::fixed << std::setprecision(0) << throughput << " inf/s\n";
    std::cout << "Total time : " << std::fixed << std::setprecision(2)
              << total_kernel_time / 1000.0 << " s\n";
    std::cout << "========================================\n\n";

    return (accuracy > 30.0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
