/*
 * MultiRocket FPGA Host Application
 * Optimized for Xilinx Alveo U280
 *
 * Features:
 * - Dual representation processing (original + first-order difference)
 * - 4 pooling operators (PPV, MPV, MIPV, LSPV)
 * - Efficient HBM buffer management
 * - Supports batch inference
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <map>
#include <string>

// JSON parsing
#include "json.hpp"
using json = nlohmann::json;

// XRT includes
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h>

// Constants (must match kernel m_axi depth pragmas in multirocket.cpp)
const int MAX_TIME_SERIES_LENGTH = 8192;
const int MAX_FEATURES = 50000;
const int MAX_CLASSES = 16;
const int MAX_DILATIONS = 32;

// Helper function to read binary file
std::vector<char> read_binary_file(const std::string& xclbin_file_name) {
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    if (!bin_file.good()) {
        std::cerr << "Failed to open xclbin file: " << xclbin_file_name << std::endl;
        exit(EXIT_FAILURE);
    }

    bin_file.seekg(0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    std::vector<char> buf(nb);
    bin_file.read(buf.data(), nb);
    return buf;
}

class MultiRocketFPGA {
private:
    cl::Context context;
    cl::CommandQueue q;
    cl::Program program;
    cl::Kernel kernel;

    // Model parameters
    int num_features;
    int num_classes;
    int num_dilations_orig;
    int num_dilations_diff;
    int time_series_length;

    // Device buffers
    cl::Buffer input_buf;
    cl::Buffer output_buf;
    cl::Buffer coefficients_buf;
    cl::Buffer intercept_buf;
    cl::Buffer scaler_mean_buf;
    cl::Buffer scaler_scale_buf;
    cl::Buffer dilations_orig_buf;
    cl::Buffer nfpd_orig_buf;
    cl::Buffer biases_orig_buf;
    cl::Buffer dilations_diff_buf;
    cl::Buffer nfpd_diff_buf;
    cl::Buffer biases_diff_buf;
    std::vector<int> nfpd_orig_vec;
    std::vector<int> nfpd_diff_vec;
    int num_features_0 = 0;
    int num_features_1 = 0;
    int n_feature_per_kernel = 4;

    // Model parameters (host side)
    std::vector<float> coefficients_flat;
    std::vector<float> intercept_vec;
    std::vector<float> scaler_mean_vec;
    std::vector<float> scaler_scale_vec;
    std::vector<int> dilations_orig_vec;
    std::vector<float> biases_orig_vec;
    std::vector<int> dilations_diff_vec;
    std::vector<float> biases_diff_vec;

public:
    MultiRocketFPGA(const std::string& xclbin_path) {
        std::cout << "================================================================================\n";
        std::cout << "MultiRocket FPGA Accelerator Initialization\n";
        std::cout << "================================================================================\n\n";

        // Get platform and device
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found");
        }

        cl::Platform platform = platforms[0];
        std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No accelerator devices found");
        }

        cl::Device device = devices[0];
        std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // Create context and command queue
        context = cl::Context(device);
        q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

        // Load xclbin
        std::cout << "\nLoading bitstream: " << xclbin_path << std::endl;
        auto fileBuf = read_binary_file(xclbin_path);
        cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

        std::vector<cl::Device> devices_vec = {device};
        program = cl::Program(context, devices_vec, bins);

        // Create kernel
        std::cout << "Creating kernel..." << std::endl;
        kernel = cl::Kernel(program, "multirocket_inference");

        std::cout << "\n✓ FPGA initialization complete!\n" << std::endl;
    }

    bool load_model(const std::string& model_path) {
        std::cout << "================================================================================\n";
        std::cout << "Loading MultiRocket Model\n";
        std::cout << "================================================================================\n\n";
        std::cout << "Model file: " << model_path << std::endl;

        std::ifstream file(model_path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open model file" << std::endl;
            return false;
        }

        json j;
        try {
            file >> j;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing JSON: " << e.what() << std::endl;
            return false;
        }

        // Load basic parameters
        num_features = j["num_features"];
        num_classes = j["num_classes"];
        time_series_length = j["time_series_length"];

        std::cout << "\nModel Configuration:" << std::endl;
        std::cout << "  Features: " << num_features << std::endl;
        std::cout << "  Classes: " << num_classes << std::endl;
        std::cout << "  Time series length: " << time_series_length << std::endl;

        // Load dilations and biases for original representation
        auto dilations_orig_json = j["dilations_orig"];
        num_dilations_orig = dilations_orig_json.size();
        dilations_orig_vec.resize(num_dilations_orig);
        for (size_t i = 0; i < dilations_orig_json.size(); i++) {
            dilations_orig_vec[i] = dilations_orig_json[i];
        }

        auto biases_orig_json = j["biases_orig"];
        biases_orig_vec.resize(biases_orig_json.size());
        for (size_t i = 0; i < biases_orig_json.size(); i++) {
            biases_orig_vec[i] = biases_orig_json[i];
        }

        // Load dilations and biases for diff representation
        auto dilations_diff_json = j["dilations_diff"];
        num_dilations_diff = dilations_diff_json.size();
        dilations_diff_vec.resize(num_dilations_diff);
        for (size_t i = 0; i < dilations_diff_json.size(); i++) {
            dilations_diff_vec[i] = dilations_diff_json[i];
        }

        auto biases_diff_json = j["biases_diff"];
        biases_diff_vec.resize(biases_diff_json.size());
        for (size_t i = 0; i < biases_diff_json.size(); i++) {
            biases_diff_vec[i] = biases_diff_json[i];
        }

        std::cout << "  Dilations (orig): " << num_dilations_orig << std::endl;
        std::cout << "  Dilations (diff): " << num_dilations_diff << std::endl;
        std::cout << "  Biases (orig): " << biases_orig_vec.size() << std::endl;
        std::cout << "  Biases (diff): " << biases_diff_vec.size() << std::endl;

        // Load scaler parameters
        auto scaler_mean_json = j["scaler_mean"];
        auto scaler_scale_json = j["scaler_scale"];
        scaler_mean_vec.resize(scaler_mean_json.size());
        scaler_scale_vec.resize(scaler_scale_json.size());
        for (size_t i = 0; i < scaler_mean_json.size(); i++) {
            scaler_mean_vec[i] = scaler_mean_json[i];
            scaler_scale_vec[i] = scaler_scale_json[i];
        }

        // Load classifier parameters
        auto coef_json = j["coefficients"];
        auto intercept_json = j["intercept"];

        coefficients_flat.resize(num_classes * num_features);
        intercept_vec.resize(num_classes);

        for (int c = 0; c < num_classes; c++) {
            for (int f = 0; f < num_features; f++) {
                coefficients_flat[c * num_features + f] = coef_json[c][f];
            }
            intercept_vec[c] = intercept_json[c];
        }

        std::cout << "  Scaler parameters: " << scaler_mean_vec.size() << std::endl;
        std::cout << "  Classifier coefficients: " << coefficients_flat.size() << std::endl;

        // Create device buffers
        std::cout << "\nAllocating device buffers..." << std::endl;

        // Pad all buffers to MAX sizes declared in kernel m_axi depth pragmas
        // (XRT reports "offset+size > mem size" when buffer < kernel declared depth).
        coefficients_flat.resize(MAX_CLASSES * MAX_FEATURES, 0.0f);
        intercept_vec.resize(MAX_CLASSES, 0.0f);
        scaler_mean_vec.resize(MAX_FEATURES, 0.0f);
        scaler_scale_vec.resize(MAX_FEATURES, 1.0f);

        input_buf = cl::Buffer(context, CL_MEM_READ_ONLY, MAX_TIME_SERIES_LENGTH * sizeof(float));
        output_buf = cl::Buffer(context, CL_MEM_WRITE_ONLY, MAX_CLASSES * sizeof(float));

        coefficients_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     coefficients_flat.size() * sizeof(float),
                                     coefficients_flat.data());

        intercept_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  intercept_vec.size() * sizeof(float),
                                  intercept_vec.data());

        scaler_mean_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    scaler_mean_vec.size() * sizeof(float),
                                    scaler_mean_vec.data());

        scaler_scale_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     scaler_scale_vec.size() * sizeof(float),
                                     scaler_scale_vec.data());

        // Build num_features_per_dilation arrays. custom_multirocket84 lays out biases as
        // one per (kernel, dilation) => features_per_dilation = biases.size() / num_kernels / num_dilations
        // For MultiRocket84 this is 1 bias slot per kernel per dilation.
        int num_kernels = 84;  // MultiRocket84
        int slots_orig = (int)biases_orig_vec.size() / (num_kernels * num_dilations_orig);
        int slots_diff = (int)biases_diff_vec.size() / (num_kernels * num_dilations_diff);
        if (slots_orig < 1) slots_orig = 1;
        if (slots_diff < 1) slots_diff = 1;
        nfpd_orig_vec.assign(num_dilations_orig, slots_orig);
        nfpd_diff_vec.assign(num_dilations_diff, slots_diff);
        num_features_0 = (int)biases_orig_vec.size();   // 84 * num_dilations_orig * slots
        num_features_1 = (int)biases_diff_vec.size();

        // Pad dilation/nfpd/biases to MAX sizes to satisfy kernel m_axi depth pragmas
        dilations_orig_vec.resize(MAX_DILATIONS, 0);
        dilations_diff_vec.resize(MAX_DILATIONS, 0);
        nfpd_orig_vec.resize(MAX_DILATIONS, 0);
        nfpd_diff_vec.resize(MAX_DILATIONS, 0);
        biases_orig_vec.resize(MAX_FEATURES, 0.0f);
        biases_diff_vec.resize(MAX_FEATURES, 0.0f);

        dilations_orig_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       dilations_orig_vec.size() * sizeof(int),
                                       dilations_orig_vec.data());

        nfpd_orig_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   nfpd_orig_vec.size() * sizeof(int),
                                   nfpd_orig_vec.data());

        biases_orig_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    biases_orig_vec.size() * sizeof(float),
                                    biases_orig_vec.data());

        dilations_diff_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       dilations_diff_vec.size() * sizeof(int),
                                       dilations_diff_vec.data());

        nfpd_diff_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   nfpd_diff_vec.size() * sizeof(int),
                                   nfpd_diff_vec.data());

        biases_diff_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    biases_diff_vec.size() * sizeof(float),
                                    biases_diff_vec.data());

        // Set kernel arguments — must match kernel signature exactly (20 args).
        // Kernel uses int_t (int32) for all scalar ints.
        int arg = 0;
        kernel.setArg(arg++, input_buf);                          // 0 time_series_input
        kernel.setArg(arg++, output_buf);                         // 1 prediction_output
        kernel.setArg(arg++, coefficients_buf);                   // 2 coefficients
        kernel.setArg(arg++, intercept_buf);                      // 3 intercept
        kernel.setArg(arg++, scaler_mean_buf);                    // 4 scaler_mean
        kernel.setArg(arg++, scaler_scale_buf);                   // 5 scaler_scale
        kernel.setArg(arg++, dilations_orig_buf);                 // 6 dilations_0
        kernel.setArg(arg++, nfpd_orig_buf);                      // 7 num_features_per_dilation_0
        kernel.setArg(arg++, biases_orig_buf);                    // 8 biases_0
        kernel.setArg(arg++, (int)num_dilations_orig);            // 9 num_dilations_0
        kernel.setArg(arg++, (int)num_features_0);                // 10 num_features_0
        kernel.setArg(arg++, dilations_diff_buf);                 // 11 dilations_1
        kernel.setArg(arg++, nfpd_diff_buf);                      // 12 num_features_per_dilation_1
        kernel.setArg(arg++, biases_diff_buf);                    // 13 biases_1
        kernel.setArg(arg++, (int)num_dilations_diff);            // 14 num_dilations_1
        kernel.setArg(arg++, (int)num_features_1);                // 15 num_features_1
        kernel.setArg(arg++, (int)time_series_length);            // 16 time_series_length
        kernel.setArg(arg++, (int)num_features);                  // 17 num_features
        kernel.setArg(arg++, (int)num_classes);                   // 18 num_classes
        kernel.setArg(arg++, (int)n_feature_per_kernel);          // 19 n_feature_per_kernel

        std::cout << "✓ Model loaded and device buffers allocated\n" << std::endl;
        return true;
    }

    std::vector<float> predict(const std::vector<float>& time_series) {
        if (time_series.size() != (size_t)time_series_length) {
            throw std::runtime_error("Input time series length mismatch");
        }

        // Transfer input to device
        q.enqueueWriteBuffer(input_buf, CL_TRUE, 0,
                            time_series.size() * sizeof(float),
                            time_series.data());

        // Execute kernel
        cl::Event event;
        q.enqueueTask(kernel, nullptr, &event);
        q.finish();

        // Get execution time
        cl_ulong start, end;
        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        double exec_time_ms = (end - start) / 1000000.0;

        // Read results
        std::vector<float> output(num_classes);
        q.enqueueReadBuffer(output_buf, CL_TRUE, 0,
                           num_classes * sizeof(float),
                           output.data());

        return output;
    }

    double get_last_exec_time_ms() {
        return last_exec_time_ms;
    }

private:
    double last_exec_time_ms = 0.0;
};

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <xclbin> <model.json> <test_data.json>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string xclbin_path = argv[1];
    std::string model_path = argv[2];
    std::string test_data_path = argv[3];

    try {
        // Initialize FPGA
        MultiRocketFPGA fpga(xclbin_path);

        // Load model
        if (!fpga.load_model(model_path)) {
            return EXIT_FAILURE;
        }

        // Load test data
        std::cout << "================================================================================\n";
        std::cout << "Loading Test Data\n";
        std::cout << "================================================================================\n\n";

        std::ifstream test_file(test_data_path);
        if (!test_file.is_open()) {
            std::cerr << "Error: Could not open test data file" << std::endl;
            return EXIT_FAILURE;
        }

        json test_json;
        test_file >> test_json;

        int num_samples = test_json["num_samples"];
        std::cout << "Loaded " << num_samples << " test samples\n" << std::endl;

        // Build label->index map from model's class list (labels may be strings like "aedes_female")
        std::ifstream model_file_for_classes(model_path);
        json model_json_for_classes;
        model_file_for_classes >> model_json_for_classes;
        std::map<std::string, int> label_to_idx;
        if (model_json_for_classes.contains("classes")) {
            const auto& classes_arr = model_json_for_classes["classes"];
            for (size_t ci = 0; ci < classes_arr.size(); ci++) {
                label_to_idx[classes_arr[ci].get<std::string>()] = (int)ci;
            }
        }

        // Run predictions
        std::cout << "================================================================================\n";
        std::cout << "Running FPGA Inference\n";
        std::cout << "================================================================================\n\n";

        int correct = 0;
        double total_time_ms = 0.0;

        for (int i = 0; i < num_samples; i++) {
            std::string label_str = test_json["labels"][i];
            int true_label;
            auto it = label_to_idx.find(label_str);
            if (it != label_to_idx.end()) {
                true_label = it->second;
            } else {
                try { true_label = std::stoi(label_str) - 1; }
                catch (...) { true_label = -1; }
            }

            std::vector<float> time_series = test_json["time_series"][i];

            auto start = std::chrono::high_resolution_clock::now();
            auto predictions = fpga.predict(time_series);
            auto end = std::chrono::high_resolution_clock::now();

            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            total_time_ms += time_ms;

            int predicted_label = std::max_element(predictions.begin(), predictions.end()) - predictions.begin();

            bool match = (predicted_label == true_label);
            if (match) correct++;

            std::cout << "Sample " << i << ": Label=" << true_label
                     << ", Predicted=" << predicted_label
                     << (match ? " ✓" : " ✗")
                     << " (" << std::fixed << std::setprecision(3) << time_ms << " ms)"
                     << std::endl;
        }

        // Print summary
        std::cout << "\n================================================================================\n";
        std::cout << "Results Summary\n";
        std::cout << "================================================================================\n\n";

        std::cout << "Accuracy: " << std::fixed << std::setprecision(2)
                 << (100.0 * correct / num_samples) << "% ("
                 << correct << "/" << num_samples << ")\n";

        std::cout << "Average latency: " << std::fixed << std::setprecision(3)
                 << (total_time_ms / num_samples) << " ms\n";

        std::cout << "Throughput: " << std::fixed << std::setprecision(1)
                 << (1000.0 * num_samples / total_time_ms) << " inferences/second\n";

        std::cout << "\n================================================================================\n";

        return (correct == num_samples) ? EXIT_SUCCESS : EXIT_FAILURE;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
