#include "../include/minirocket_host.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <sstream>

#include "../include/minirocket_loader.h"

// Redefine types for host side (must match kernel common.hpp)
#ifndef MINIROCKET_MODULAR_COMMON_HPP
typedef float data_t;
typedef int int_t;
#define MAX_TIME_SERIES_LENGTH 8192
#define MAX_FEATURES 1024
#define NUM_KERNELS 84
#define KERNEL_SIZE 9
#define MAX_DILATIONS 16
#define MAX_CLASSES 16
#endif

int main(int argc, char** argv) {

    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File> <Model JSON File> <Test Data JSON File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    std::string model_file = argv[2];
    std::string test_file = argv[3];

    /*====================================================CL===============================================================*/

    cl_int err;
    cl::Context context;
    cl::Kernel krnl_feature_extraction;
    cl::Kernel krnl_scaler;
    cl::Kernel krnl_classifier;
    cl::CommandQueue q;  // Will be created with profiling enabled

    auto devices = get_xil_devices();
    auto fileBuf = read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;

    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            std::cout << "Setting up 3 kernel CUs..." << std::endl;
            OCL_CHECK(err, krnl_feature_extraction = cl::Kernel(program, "feature_extraction", &err));
            OCL_CHECK(err, krnl_scaler = cl::Kernel(program, "scaler", &err));
            OCL_CHECK(err, krnl_classifier = cl::Kernel(program, "classifier", &err));
            valid_device = true;
            break;
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    /*====================================================LOAD MODEL===============================================================*/

    std::cout << "Loading MiniRocket model and test data" << std::endl;
    std::cout << "Model file: " << model_file << std::endl;
    std::cout << "Test file: " << test_file << std::endl;

    MiniRocketTestbenchLoader loader;

    data_t (*coefficients)[MAX_FEATURES] = new data_t[MAX_CLASSES][MAX_FEATURES];

    std::vector<data_t, aligned_allocator<data_t>> time_series_input(MAX_TIME_SERIES_LENGTH);
    std::vector<data_t, aligned_allocator<data_t>> prediction_output(MAX_CLASSES);
    std::vector<data_t, aligned_allocator<data_t>> flattened_coefficients(MAX_CLASSES * MAX_FEATURES);

    std::vector<data_t, aligned_allocator<data_t>> intercept_vec(MAX_CLASSES);
    std::vector<data_t, aligned_allocator<data_t>> scaler_mean(MAX_FEATURES);
    std::vector<data_t, aligned_allocator<data_t>> scaler_scale(MAX_FEATURES);
    std::vector<data_t, aligned_allocator<data_t>> inv_scale(MAX_FEATURES);
    std::vector<data_t, aligned_allocator<data_t>> biases(MAX_FEATURES);
    std::vector<int_t, aligned_allocator<int_t>> dilations_vec(MAX_DILATIONS);
    std::vector<int_t, aligned_allocator<int_t>> num_features_per_dilation(MAX_DILATIONS);
    int_t num_dilations, num_features, num_classes, time_series_length;

    std::cout << "Loading model..." << std::endl;
    if (!loader.load_model_to_hls_arrays(model_file, coefficients, intercept_vec.data(),
                                        scaler_mean.data(), scaler_scale.data(), dilations_vec.data(),
                                        num_features_per_dilation.data(), biases.data(),
                                        num_dilations, num_features, num_classes,
                                        time_series_length)) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }

    // Precompute inv_scale = 1.0 / scale (eliminates division on FPGA)
    std::cout << "Precomputing inverse scale values..." << std::endl;
    for (int i = 0; i < num_features; i++) {
        if (scaler_scale[i] != 0.0f) {
            inv_scale[i] = 1.0f / scaler_scale[i];
        } else {
            inv_scale[i] = 0.0f;
        }
    }

    time_series_input.resize(time_series_length);
    prediction_output.resize(num_classes);

    for (int i = 0; i < num_classes * num_features; i++) {
        int row = i / num_features;
        int col = i % num_features;
        flattened_coefficients[i] = coefficients[row][col];
    }

    // Load test data
    std::cout << "Loading test data..." << std::endl;
    std::vector<std::vector<float>> test_inputs, expected_outputs;
    if (!loader.load_test_data(test_file, test_inputs, expected_outputs)) {
        std::cerr << "Failed to load test data!" << std::endl;
        return 1;
    }

    // Try to load Python baseline accuracy
    float python_baseline = 0.0f;
    std::ifstream test_file_stream(test_file);
    if (test_file_stream.is_open()) {
        std::string content((std::istreambuf_iterator<char>(test_file_stream)),
                           std::istreambuf_iterator<char>());
        size_t accuracy_pos = content.find("\"test_accuracy\":");
        if (accuracy_pos != std::string::npos) {
            size_t start = content.find(":", accuracy_pos) + 1;
            size_t end = content.find_first_of(",}", start);
            std::string accuracy_str = content.substr(start, end - start);
            python_baseline = std::stof(accuracy_str) * 100.0f;
        }
    }

    std::vector<int> expected_classes;
    for (const auto& output : expected_outputs) {
        if (!output.empty()) {
            expected_classes.push_back((int)output[0]);
        }
    }

    /*====================================================KERNEL I/O===============================================================*/

    // K1: Feature Extraction buffers
    OCL_CHECK(err, cl::Buffer buffer_time_series(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        sizeof(data_t) * time_series_length, time_series_input.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_dilations(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        sizeof(int_t) * dilations_vec.size(), dilations_vec.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_num_features_per_dilation(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        sizeof(int_t) * num_features_per_dilation.size(), num_features_per_dilation.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_biases(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        sizeof(data_t) * biases.size(), biases.data(), &err));

    // K2: Scaler buffers
    OCL_CHECK(err, cl::Buffer buffer_scaler_mean(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        sizeof(data_t) * scaler_mean.size(), scaler_mean.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_inv_scale(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        sizeof(data_t) * inv_scale.size(), inv_scale.data(), &err));

    // K3: Classifier buffers
    OCL_CHECK(err, cl::Buffer buffer_coefficients(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        sizeof(data_t) * flattened_coefficients.size(), flattened_coefficients.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_intercept(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        sizeof(data_t) * intercept_vec.size(), intercept_vec.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_predictions(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
        sizeof(data_t) * num_classes, prediction_output.data(), &err));

    // Set K1 kernel arguments (HBM ports + scalars; stream is implicit)
    OCL_CHECK(err, err = krnl_feature_extraction.setArg(0, buffer_time_series));
    OCL_CHECK(err, err = krnl_feature_extraction.setArg(1, buffer_dilations));
    OCL_CHECK(err, err = krnl_feature_extraction.setArg(2, buffer_num_features_per_dilation));
    OCL_CHECK(err, err = krnl_feature_extraction.setArg(3, buffer_biases));
    // arg 4 = features_out stream (implicit, not set via host)
    OCL_CHECK(err, err = krnl_feature_extraction.setArg(5, time_series_length));
    OCL_CHECK(err, err = krnl_feature_extraction.setArg(6, num_features));
    OCL_CHECK(err, err = krnl_feature_extraction.setArg(7, num_dilations));

    // Set K2 kernel arguments
    OCL_CHECK(err, err = krnl_scaler.setArg(0, buffer_scaler_mean));
    OCL_CHECK(err, err = krnl_scaler.setArg(1, buffer_inv_scale));
    // arg 2 = features_in stream (implicit)
    // arg 3 = scaled_features_out stream (implicit)
    OCL_CHECK(err, err = krnl_scaler.setArg(4, num_features));

    // Set K3 kernel arguments
    OCL_CHECK(err, err = krnl_classifier.setArg(0, buffer_coefficients));
    OCL_CHECK(err, err = krnl_classifier.setArg(1, buffer_intercept));
    OCL_CHECK(err, err = krnl_classifier.setArg(2, buffer_predictions));
    // arg 3 = scaled_features_in stream (implicit)
    OCL_CHECK(err, err = krnl_classifier.setArg(4, num_features));
    OCL_CHECK(err, err = krnl_classifier.setArg(5, num_classes));

    // Migrate model parameters to FPGA (one-time cost)
    std::cout << "Loading model weights to FPGA..." << std::endl;
    std::vector<cl::Memory> model_buffers = {
        buffer_dilations, buffer_num_features_per_dilation, buffer_biases,
        buffer_scaler_mean, buffer_inv_scale,
        buffer_coefficients, buffer_intercept
    };
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(model_buffers, 0));
    q.finish();

    /*====================================================INFERENCE LOOP===============================================================*/

    double total_h2d_ms = 0, total_kernel_ms = 0, total_d2h_ms = 0;
    double total_k1_ms = 0, total_k2_ms = 0, total_k3_ms = 0;  // Per-kernel profiling
    int correct_predictions = 0;

    for (size_t test_idx = 0; test_idx < test_inputs.size(); test_idx++) {
        // Prepare input
        for (int j = 0; j < time_series_length; j++) {
            time_series_input[j] = test_inputs[test_idx][j];
        }

        // H2D: send time series to FPGA
        std::cout << "Loading time series data[" << test_idx << "] to FPGA" << std::endl;
        auto h2d_start = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_time_series}, 0));
        q.finish();
        auto h2d_end = std::chrono::high_resolution_clock::now();
        double h2d_ms = std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
        total_h2d_ms += h2d_ms;

        // Launch all 3 kernels (AXI-Stream connections handle data flow)
        std::cout << "STARTING pipeline for input[" << test_idx << "]" << std::endl;
        auto kernel_start = std::chrono::high_resolution_clock::now();

        cl::Event event_k1, event_k2, event_k3;
        OCL_CHECK(err, err = q.enqueueTask(krnl_feature_extraction, nullptr, &event_k1));
        OCL_CHECK(err, err = q.enqueueTask(krnl_scaler, nullptr, &event_k2));
        OCL_CHECK(err, err = q.enqueueTask(krnl_classifier, nullptr, &event_k3));
        q.finish();

        auto kernel_end = std::chrono::high_resolution_clock::now();
        double kernel_ms = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();
        total_kernel_ms += kernel_ms;

        // Extract per-kernel timing from OpenCL events
        cl_ulong k1_start, k1_end, k2_start, k2_end, k3_start, k3_end;
        event_k1.getProfilingInfo(CL_PROFILING_COMMAND_START, &k1_start);
        event_k1.getProfilingInfo(CL_PROFILING_COMMAND_END, &k1_end);
        event_k2.getProfilingInfo(CL_PROFILING_COMMAND_START, &k2_start);
        event_k2.getProfilingInfo(CL_PROFILING_COMMAND_END, &k2_end);
        event_k3.getProfilingInfo(CL_PROFILING_COMMAND_START, &k3_start);
        event_k3.getProfilingInfo(CL_PROFILING_COMMAND_END, &k3_end);

        double k1_ms = (k1_end - k1_start) / 1e6;  // Convert ns to ms
        double k2_ms = (k2_end - k2_start) / 1e6;
        double k3_ms = (k3_end - k3_start) / 1e6;
        total_k1_ms += k1_ms;
        total_k2_ms += k2_ms;
        total_k3_ms += k3_ms;

        // D2H: read predictions from FPGA
        std::cout << "Transferring results for input[" << test_idx << "] from FPGA to host" << std::endl;
        auto d2h_start = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_predictions}, CL_MIGRATE_MEM_OBJECT_HOST));
        q.finish();
        auto d2h_end = std::chrono::high_resolution_clock::now();
        double d2h_ms = std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();
        total_d2h_ms += d2h_ms;

        // Find predicted class
        int predicted_class = 0;
        float max_score = prediction_output[0];

        std::cout << "Classification scores: ";
        for (int i = 0; i < num_classes; i++) {
            std::cout << "Class" << i << "=" << std::fixed << std::setprecision(4) << prediction_output[i] << " ";
            if (prediction_output[i] > max_score) {
                max_score = prediction_output[i];
                predicted_class = i;
            }
        }
        std::cout << std::endl;

        std::cout << "Predicted class: " << predicted_class << " (score: " << std::fixed << std::setprecision(4) << max_score << ")" << std::endl;

        int expected_class = (test_idx < expected_classes.size()) ? expected_classes[test_idx] : 0;
        std::cout << "Expected class: " << expected_class << std::endl;

        bool correct = (predicted_class == expected_class);
        if (correct) {
            correct_predictions++;
            std::cout << "Prediction correct!" << std::endl;
        } else {
            std::cout << "Prediction incorrect!" << std::endl;
        }

        double single_ms = h2d_ms + kernel_ms + d2h_ms;
        std::cout << "\n========== TIMING RESULTS ==========" << std::endl;
        std::cout << "H2D transfer:        " << std::fixed << std::setprecision(3) << h2d_ms << " ms" << std::endl;
        std::cout << "Kernel pipeline:     " << std::fixed << std::setprecision(3) << kernel_ms << " ms" << std::endl;
        std::cout << "  K1 (FeatureExt):   " << std::fixed << std::setprecision(3) << k1_ms << " ms" << std::endl;
        std::cout << "  K2 (Scaler):       " << std::fixed << std::setprecision(3) << k2_ms << " ms" << std::endl;
        std::cout << "  K3 (Classifier):   " << std::fixed << std::setprecision(3) << k3_ms << " ms" << std::endl;
        std::cout << "D2H transfer:        " << std::fixed << std::setprecision(3) << d2h_ms << " ms" << std::endl;
        std::cout << "Total latency:       " << std::fixed << std::setprecision(3) << single_ms << " ms" << std::endl;
        std::cout << "====================================" << std::endl;
    }

    /*====================================================RESULTS===============================================================*/

    double total_ms = total_h2d_ms + total_kernel_ms + total_d2h_ms;
    double correct_percentage = ((double)correct_predictions / test_inputs.size()) * 100.0;
    std::cout << "\n========== FINAL RESULTS ==========" << std::endl;
    std::cout << "Total correct predictions: " << correct_predictions << " / " << test_inputs.size() << std::endl;
    std::cout << "Overall accuracy: " << std::fixed << std::setprecision(2) << correct_percentage << " %" << std::endl;
    if (python_baseline > 0.0f) {
        std::cout << "Python baseline accuracy: " << std::fixed << std::setprecision(2) << python_baseline << " %" << std::endl;
    }

    std::cout << "\n========== TIMING RESULTS ==========" << std::endl;
    std::cout << "H2D transfer:        " << std::fixed << std::setprecision(3) << total_h2d_ms << " ms" << std::endl;
    std::cout << "Kernel pipeline:     " << std::fixed << std::setprecision(3) << total_kernel_ms << " ms" << std::endl;
    std::cout << "  K1 (FeatureExt):   " << std::fixed << std::setprecision(3) << total_k1_ms << " ms ("
              << std::fixed << std::setprecision(1) << (total_k1_ms/total_kernel_ms*100) << "%)" << std::endl;
    std::cout << "  K2 (Scaler):       " << std::fixed << std::setprecision(3) << total_k2_ms << " ms ("
              << std::fixed << std::setprecision(1) << (total_k2_ms/total_kernel_ms*100) << "%)" << std::endl;
    std::cout << "  K3 (Classifier):   " << std::fixed << std::setprecision(3) << total_k3_ms << " ms ("
              << std::fixed << std::setprecision(1) << (total_k3_ms/total_kernel_ms*100) << "%)" << std::endl;
    std::cout << "D2H transfer:        " << std::fixed << std::setprecision(3) << total_d2h_ms << " ms" << std::endl;
    std::cout << "Total latency:       " << std::fixed << std::setprecision(3) << total_ms << " ms" << std::endl;
    std::cout << "Throughput:          " << std::fixed << std::setprecision(1) << (test_inputs.size() / (total_kernel_ms / 1000.0)) << " inferences/sec" << std::endl;
    std::cout << "====================================" << std::endl;

    std::cout << "\nMiniRocket Modular FPGA inference complete!" << std::endl;

    delete[] coefficients;
    return EXIT_SUCCESS;
}
