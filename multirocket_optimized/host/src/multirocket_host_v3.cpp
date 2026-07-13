// multirocket_host_v3.cpp — Self-contained FPGA host for Round 3 xclbin.
// Parses the real MultiRocket84 model JSON schema (num_dilations_orig, etc.)
// and dispatches to the multirocket_inference kernel via OpenCL.

#include "../include/multirocket_host.h"

#include "ap_fixed.h"
#include "ap_int.h"

typedef ap_fixed<32,16> data_t;
typedef ap_int<32>      int_t;

#define MAX_TIME_SERIES_LENGTH 8192
#define MAX_FEATURES           50000
#define NUM_KERNELS            84
#define MAX_DILATIONS          32
#define MAX_CLASSES            16

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>

// ---------- Minimal JSON parser (copied from csim_v3.cpp) ----------
struct JsonVal;
using JsonArr = std::vector<JsonVal>;
using JsonObj = std::map<std::string, JsonVal>;
struct JsonVal {
    enum Type { Null, Bool, Int, Float, Str, Array, Object } type;
    bool b; long long i; double f; std::string s;
    JsonArr arr; JsonObj obj;
    JsonVal() : type(Null), b(false), i(0), f(0) {}
};
static void skip_ws(const char*& p) {
    while (*p==' '||*p=='\t'||*p=='\n'||*p=='\r') p++;
}
static JsonVal parse_val(const char*& p);
static std::string parse_string(const char*& p) {
    assert(*p=='"'); p++;
    std::string s;
    while (*p!='"') { if (*p=='\\') { p++; } s += *p++; }
    p++; return s;
}
static JsonVal parse_array(const char*& p) {
    assert(*p=='['); p++;
    JsonVal v; v.type=JsonVal::Array;
    skip_ws(p); if (*p==']') { p++; return v; }
    while (true) {
        skip_ws(p); v.arr.push_back(parse_val(p)); skip_ws(p);
        if (*p==']') { p++; break; }
        if (*p==',') p++;
    }
    return v;
}
static JsonVal parse_object(const char*& p) {
    assert(*p=='{'); p++;
    JsonVal v; v.type=JsonVal::Object;
    skip_ws(p); if (*p=='}') { p++; return v; }
    while (true) {
        skip_ws(p); std::string key=parse_string(p);
        skip_ws(p); assert(*p==':'); p++; skip_ws(p);
        v.obj[key]=parse_val(p); skip_ws(p);
        if (*p=='}') { p++; break; }
        if (*p==',') p++;
    }
    return v;
}
static JsonVal parse_val(const char*& p) {
    skip_ws(p);
    if (*p=='"') { JsonVal v; v.type=JsonVal::Str; v.s=parse_string(p); return v; }
    if (*p=='[') return parse_array(p);
    if (*p=='{') return parse_object(p);
    if (*p=='t') { p+=4; JsonVal v; v.type=JsonVal::Bool; v.b=true; return v; }
    if (*p=='f') { p+=5; JsonVal v; v.type=JsonVal::Bool; v.b=false; return v; }
    if (*p=='n') { p+=4; return JsonVal(); }
    char* end; double d = strtod(p, &end);
    JsonVal v; bool is_int=true;
    for (const char* q=p; q<end; q++) if (*q=='.'||*q=='e'||*q=='E') { is_int=false; break; }
    if (is_int) { v.type=JsonVal::Int; v.i=(long long)d; }
    else { v.type=JsonVal::Float; v.f=d; }
    p=end; return v;
}
static JsonVal parse_json_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) { fprintf(stderr,"Cannot open %s\n", path.c_str()); exit(1); }
    std::string s((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    const char* p = s.c_str();
    return parse_val(p);
}
static double get_num(const JsonVal& v) {
    if (v.type==JsonVal::Int) return (double)v.i;
    if (v.type==JsonVal::Float) return v.f;
    return 0.0;
}
static std::vector<double> get_arr_double(const JsonVal& v) {
    std::vector<double> out; out.reserve(v.arr.size());
    for (auto& x : v.arr) out.push_back(get_num(x));
    return out;
}
static std::vector<int> get_arr_int(const JsonVal& v) {
    std::vector<int> out; out.reserve(v.arr.size());
    for (auto& x : v.arr) out.push_back((int)get_num(x));
    return out;
}

// ---------- Main ----------
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <xclbin> <model.json> <test.json>\n";
        return 1;
    }
    std::string binaryFile = argv[1];
    std::string model_file = argv[2];
    std::string test_file  = argv[3];

    // ---- OpenCL setup ----
    cl_int err;
    cl::Context context;
    cl::Kernel  krnl;
    cl::CommandQueue q;
    auto devices = get_xil_devices();
    auto fileBuf = read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned i=0; i<devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device,
            CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Programming device[" << i << "]: "
                  << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) continue;
        OCL_CHECK(err, krnl = cl::Kernel(program, "multirocket_inference", &err));
        valid_device = true;
        break;
    }
    if (!valid_device) { std::cerr << "No device programmed\n"; return 1; }

    // ---- Parse model ----
    std::cout << "Parsing model: " << model_file << std::endl;
    JsonVal model = parse_json_file(model_file);
    auto& mobj = model.obj;

    int num_features  = (int)get_num(mobj["num_features"]);
    int num_classes   = (int)get_num(mobj["num_classes"]);
    int ts_length     = (int)get_num(mobj["time_series_length"]);
    int num_dil_0     = (int)get_num(mobj["num_dilations_orig"]);
    int num_dil_1     = (int)get_num(mobj["num_dilations_diff"]);
    int n_feature_per_kernel = 4;

    auto dilations_0_v = get_arr_int(mobj["dilations_orig"]);
    auto dilations_1_v = get_arr_int(mobj["dilations_diff"]);
    auto biases_0_v    = get_arr_double(mobj["biases_orig"]);
    auto biases_1_v    = get_arr_double(mobj["biases_diff"]);
    auto scaler_mean_v = get_arr_double(mobj["scaler_mean"]);
    auto scaler_scale_v= get_arr_double(mobj["scaler_scale"]);
    auto intercept_v   = get_arr_double(mobj["intercept"]);

    int num_features_0 = (int)biases_0_v.size();
    int num_features_1 = (int)biases_1_v.size();
    int slots_orig = num_features_0 / (num_dil_0 * NUM_KERNELS);
    int slots_diff = num_features_1 / (num_dil_1 * NUM_KERNELS);

    std::cout << "  num_features=" << num_features
              << " num_classes=" << num_classes
              << " ts_length=" << ts_length << std::endl;
    std::cout << "  num_features_0=" << num_features_0 << " (dil=" << num_dil_0 << ", slots=" << slots_orig << ")"
              << "  num_features_1=" << num_features_1 << " (dil=" << num_dil_1 << ", slots=" << slots_diff << ")"
              << std::endl;

    // Allocate host buffers
    std::vector<data_t, aligned_allocator<data_t>> time_series_input(MAX_TIME_SERIES_LENGTH, (data_t)0);
    std::vector<data_t, aligned_allocator<data_t>> prediction_output(MAX_CLASSES, (data_t)0);
    std::vector<data_t, aligned_allocator<data_t>> flat_coefficients(MAX_CLASSES * MAX_FEATURES, (data_t)0);
    std::vector<data_t, aligned_allocator<data_t>> intercept(MAX_CLASSES, (data_t)0);
    std::vector<data_t, aligned_allocator<data_t>> scaler_mean(MAX_FEATURES, (data_t)0);
    std::vector<data_t, aligned_allocator<data_t>> scaler_scale(MAX_FEATURES, (data_t)0);
    std::vector<data_t, aligned_allocator<data_t>> biases_0(MAX_FEATURES, (data_t)0);
    std::vector<data_t, aligned_allocator<data_t>> biases_1(MAX_FEATURES, (data_t)0);
    std::vector<int_t,  aligned_allocator<int_t>>  dilations_0(MAX_DILATIONS, (int_t)0);
    std::vector<int_t,  aligned_allocator<int_t>>  dilations_1(MAX_DILATIONS, (int_t)0);
    std::vector<int_t,  aligned_allocator<int_t>>  nfpd_0(MAX_DILATIONS, (int_t)0);
    std::vector<int_t,  aligned_allocator<int_t>>  nfpd_1(MAX_DILATIONS, (int_t)0);

    // Populate
    auto& coef_arr = mobj["coefficients"].arr;
    for (int c=0; c<num_classes; c++) {
        auto row = get_arr_double(coef_arr[c]);
        for (int f=0; f<num_features; f++)
            flat_coefficients[c*num_features + f] = (data_t)row[f];
    }
    for (int c=0; c<num_classes; c++) intercept[c] = (data_t)intercept_v[c];
    for (int f=0; f<num_features; f++) {
        scaler_mean[f]  = (data_t)scaler_mean_v[f];
        scaler_scale[f] = (data_t)scaler_scale_v[f];
    }
    for (int i=0; i<num_dil_0; i++) { dilations_0[i]=(int_t)dilations_0_v[i]; nfpd_0[i]=(int_t)slots_orig; }
    for (int i=0; i<num_dil_1; i++) { dilations_1[i]=(int_t)dilations_1_v[i]; nfpd_1[i]=(int_t)slots_diff; }
    for (int i=0; i<num_features_0; i++) biases_0[i] = (data_t)biases_0_v[i];
    for (int i=0; i<num_features_1; i++) biases_1[i] = (data_t)biases_1_v[i];

    // classes mapping (string labels)
    std::map<std::string,int> class_to_idx;
    if (mobj.count("classes")) {
        auto& ca = mobj["classes"].arr;
        for (int c=0; c<(int)ca.size(); c++) class_to_idx[ca[c].s] = c;
    }

    // ---- Parse test data ----
    std::cout << "Parsing test data: " << test_file << std::endl;
    JsonVal tv = parse_json_file(test_file);
    auto& tobj = tv.obj;
    auto& ts_arr  = tobj["time_series"].arr;
    auto& lbl_arr = tobj["labels"].arr;
    int num_samples = (int)ts_arr.size();
    std::cout << "  samples=" << num_samples << std::endl;

    // ---- Buffers ----
    size_t coef_bytes = sizeof(data_t) * (size_t)num_classes * (size_t)num_features;
    OCL_CHECK(err, cl::Buffer b_ts(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
        sizeof(data_t)*ts_length, time_series_input.data(), &err));
    OCL_CHECK(err, cl::Buffer b_coef(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
        coef_bytes, flat_coefficients.data(), &err));
    OCL_CHECK(err, cl::Buffer b_inter(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
        sizeof(data_t)*num_classes, intercept.data(), &err));
    OCL_CHECK(err, cl::Buffer b_smean(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
        sizeof(data_t)*num_features, scaler_mean.data(), &err));
    OCL_CHECK(err, cl::Buffer b_sscale(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
        sizeof(data_t)*num_features, scaler_scale.data(), &err));
    OCL_CHECK(err, cl::Buffer b_d0(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
        sizeof(int_t)*MAX_DILATIONS, dilations_0.data(), &err));
    OCL_CHECK(err, cl::Buffer b_n0(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
        sizeof(int_t)*MAX_DILATIONS, nfpd_0.data(), &err));
    OCL_CHECK(err, cl::Buffer b_b0(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
        sizeof(data_t)*num_features_0, biases_0.data(), &err));
    OCL_CHECK(err, cl::Buffer b_d1(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
        sizeof(int_t)*MAX_DILATIONS, dilations_1.data(), &err));
    OCL_CHECK(err, cl::Buffer b_n1(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
        sizeof(int_t)*MAX_DILATIONS, nfpd_1.data(), &err));
    OCL_CHECK(err, cl::Buffer b_b1(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
        sizeof(data_t)*num_features_1, biases_1.data(), &err));
    OCL_CHECK(err, cl::Buffer b_pred(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY,
        sizeof(data_t)*num_classes, prediction_output.data(), &err));

    int_t k_num_dil_0 = (int_t)num_dil_0;
    int_t k_num_feat_0 = (int_t)num_features_0;
    int_t k_num_dil_1 = (int_t)num_dil_1;
    int_t k_num_feat_1 = (int_t)num_features_1;
    int_t k_ts_len = (int_t)ts_length;
    int_t k_num_feat = (int_t)num_features;
    int_t k_num_cls = (int_t)num_classes;
    int_t k_nfpk = (int_t)n_feature_per_kernel;

    OCL_CHECK(err, err = krnl.setArg(0, b_ts));
    OCL_CHECK(err, err = krnl.setArg(1, b_pred));
    OCL_CHECK(err, err = krnl.setArg(2, b_coef));
    OCL_CHECK(err, err = krnl.setArg(3, b_inter));
    OCL_CHECK(err, err = krnl.setArg(4, b_smean));
    OCL_CHECK(err, err = krnl.setArg(5, b_sscale));
    OCL_CHECK(err, err = krnl.setArg(6, b_d0));
    OCL_CHECK(err, err = krnl.setArg(7, b_n0));
    OCL_CHECK(err, err = krnl.setArg(8, b_b0));
    OCL_CHECK(err, err = krnl.setArg(9,  k_num_dil_0));
    OCL_CHECK(err, err = krnl.setArg(10, k_num_feat_0));
    OCL_CHECK(err, err = krnl.setArg(11, b_d1));
    OCL_CHECK(err, err = krnl.setArg(12, b_n1));
    OCL_CHECK(err, err = krnl.setArg(13, b_b1));
    OCL_CHECK(err, err = krnl.setArg(14, k_num_dil_1));
    OCL_CHECK(err, err = krnl.setArg(15, k_num_feat_1));
    OCL_CHECK(err, err = krnl.setArg(16, k_ts_len));
    OCL_CHECK(err, err = krnl.setArg(17, k_num_feat));
    OCL_CHECK(err, err = krnl.setArg(18, k_num_cls));
    OCL_CHECK(err, err = krnl.setArg(19, k_nfpk));

    std::cout << "Migrating weights to device..." << std::endl;
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
        {b_coef,b_inter,b_smean,b_sscale,b_d0,b_n0,b_b0,b_d1,b_n1,b_b1}, 0));
    q.finish();

    // ---- Inference loop ----
    int correct = 0;
    double total_kernel_ms = 0.0;
    auto wall_t0 = std::chrono::steady_clock::now();

    for (int si=0; si<num_samples; si++) {
        auto& row = ts_arr[si].arr;
        int L = (int)row.size();
        if (L > ts_length) L = ts_length;
        for (int t=0; t<L; t++) time_series_input[t] = (data_t)get_num(row[t]);

        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({b_ts}, 0));
        auto ks = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueTask(krnl));
        q.finish();
        auto ke = std::chrono::high_resolution_clock::now();
        total_kernel_ms += std::chrono::duration<double,std::milli>(ke-ks).count();

        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({b_pred}, CL_MIGRATE_MEM_OBJECT_HOST));
        q.finish();

        int pred = 0;
        data_t mx = prediction_output[0];
        for (int c=1; c<num_classes; c++) {
            if (prediction_output[c] > mx) { mx = prediction_output[c]; pred = c; }
        }
        int truth;
        auto& lv = lbl_arr[si];
        if (lv.type == JsonVal::Str) truth = class_to_idx[lv.s];
        else truth = (int)get_num(lv);
        if (pred == truth) correct++;

        if ((si+1) % 50 == 0 || si == num_samples-1) {
            auto tn = std::chrono::steady_clock::now();
            double el = std::chrono::duration<double>(tn-wall_t0).count();
            fprintf(stderr, "\r  [%d/%d] acc=%.2f%% elapsed=%.1fs", si+1, num_samples,
                100.0*correct/(si+1), el);
        }
    }
    fprintf(stderr, "\n");
    auto wall_t1 = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(wall_t1-wall_t0).count();

    double acc = 100.0 * correct / num_samples;
    double kernel_throughput = num_samples / (total_kernel_ms/1000.0);
    double wall_throughput   = num_samples / wall_s;

    std::cout << "\n========== FINAL RESULTS ==========" << std::endl;
    std::cout << "Samples:           " << num_samples << std::endl;
    std::cout << "Correct:           " << correct << std::endl;
    std::cout << "Accuracy:          " << std::fixed << std::setprecision(2) << acc << " %" << std::endl;
    std::cout << "Kernel total:      " << std::fixed << std::setprecision(3) << total_kernel_ms << " ms" << std::endl;
    std::cout << "Kernel throughput: " << std::fixed << std::setprecision(1) << kernel_throughput << " inf/s" << std::endl;
    std::cout << "Wall throughput:   " << std::fixed << std::setprecision(1) << wall_throughput << " inf/s" << std::endl;
    std::cout << "====================================" << std::endl;
    return 0;
}
