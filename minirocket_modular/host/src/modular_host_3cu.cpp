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

static const int NUM_CUS = 3;

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
    cl::CommandQueue q[NUM_CUS];
    cl::Kernel krnl_feature_extraction[NUM_CUS];
    cl::Kernel krnl_scaler[NUM_CUS];
    cl::Kernel krnl_classifier[NUM_CUS];

    auto devices = get_xil_devices();
    auto fileBuf = read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;

    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        for (int cu = 0; cu < NUM_CUS; cu++) {
            OCL_CHECK(err, q[cu] = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        }
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            std::cout << "Setting up " << NUM_CUS << " pipeline CUs..." << std::endl;
            for (int cu = 0; cu < NUM_CUS; cu++) {
                std::string fe_name = "feature_extraction:{feature_extraction_" + std::to_string(cu + 1) + "}";
                std::string sc_name = "scaler:{scaler_" + std::to_string(cu + 1) + "}";
                std::string cl_name = "classifier:{classifier_" + std::to_string(cu + 1) + "}";
                OCL_CHECK(err, krnl_feature_extraction[cu] = cl::Kernel(program, fe_name.c_str(), &err));
                OCL_CHECK(err, krnl_scaler[cu] = cl::Kernel(program, sc_name.c_str(), &err));
                OCL_CHECK(err, krnl_classifier[cu] = cl::Kernel(program, cl_name.c_str(), &err));
                std::cout << "  CU" << cu << ": " << fe_name << " + " << sc_name << " + " << cl_name << std::endl;
            }
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

    MiniRocketTestbenchLoader loader;

    data_t (*coefficients)[MAX_FEATURES] = new data_t[MAX_CLASSES][MAX_FEATURES];

    std::vector<data_t, aligned_allocator<data_t>> flattened_coefficients(MAX_CLASSES * MAX_FEATURES);
    std::vector<data_t, aligned_allocator<data_t>> intercept_vec(MAX_CLASSES);
    std::vector<data_t, aligned_allocator<data_t>> scaler_mean(MAX_FEATURES);
    std::vector<data_t, aligned_allocator<data_t>> scaler_scale(MAX_FEATURES);
    std::vector<data_t, aligned_allocator<data_t>> inv_scale(MAX_FEATURES);
    std::vector<data_t, aligned_allocator<data_t>> biases(MAX_FEATURES);
    std::vector<int_t, aligned_allocator<int_t>> dilations_vec(MAX_DILATIONS);
    std::vector<int_t, aligned_allocator<int_t>> num_features_per_dilation(MAX_DILATIONS);
    int_t num_dilations, num_features, num_classes, time_series_length;

    if (!loader.load_model_to_hls_arrays(model_file, coefficients, intercept_vec.data(),
                                        scaler_mean.data(), scaler_scale.data(), dilations_vec.data(),
                                        num_features_per_dilation.data(), biases.data(),
                                        num_dilations, num_features, num_classes,
                                        time_series_length)) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }

    // Precompute inv_scale = 1.0 / scale
    for (int i = 0; i < num_features; i++) {
        inv_scale[i] = (scaler_scale[i] != 0.0f) ? (1.0f / scaler_scale[i]) : 0.0f;
    }

    for (int i = 0; i < num_classes * num_features; i++) {
        flattened_coefficients[i] = coefficients[i / num_features][i % num_features];
    }

    // Load test data
    std::vector<std::vector<float>> test_inputs, expected_outputs;
    if (!loader.load_test_data(test_file, test_inputs, expected_outputs)) {
        std::cerr << "Failed to load test data!" << std::endl;
        return 1;
    }

    // Try to load Python baseline accuracy
    float python_baseline = 0.0f;
    {
        std::ifstream test_file_stream(test_file);
        if (test_file_stream.is_open()) {
            std::string content((std::istreambuf_iterator<char>(test_file_stream)),
                               std::istreambuf_iterator<char>());
            size_t accuracy_pos = content.find("\"test_accuracy\":");
            if (accuracy_pos != std::string::npos) {
                size_t start = content.find(":", accuracy_pos) + 1;
                size_t end = content.find_first_of(",}", start);
                python_baseline = std::stof(content.substr(start, end - start)) * 100.0f;
            }
        }
    }

    std::vector<int> expected_classes;
    for (const auto& output : expected_outputs) {
        if (!output.empty()) {
            expected_classes.push_back((int)output[0]);
        }
    }

    /*====================================================KERNEL I/O (per-CU buffers)===============================================*/

    std::vector<data_t, aligned_allocator<data_t>> time_series_input[NUM_CUS];
    std::vector<data_t, aligned_allocator<data_t>> prediction_output[NUM_CUS];
    cl::Buffer buffer_time_series[NUM_CUS];
    cl::Buffer buffer_predictions[NUM_CUS];

    cl::Buffer buffer_dilations[NUM_CUS];
    cl::Buffer buffer_num_features_per_dilation[NUM_CUS];
    cl::Buffer buffer_biases[NUM_CUS];
    cl::Buffer buffer_scaler_mean[NUM_CUS];
    cl::Buffer buffer_inv_scale[NUM_CUS];
    cl::Buffer buffer_coefficients[NUM_CUS];
    cl::Buffer buffer_intercept[NUM_CUS];

    for (int cu = 0; cu < NUM_CUS; cu++) {
        time_series_input[cu].resize(time_series_length);
        prediction_output[cu].resize(num_classes);

        OCL_CHECK(err, buffer_time_series[cu] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            sizeof(data_t) * time_series_length, time_series_input[cu].data(), &err));
        OCL_CHECK(err, buffer_predictions[cu] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
            sizeof(data_t) * num_classes, prediction_output[cu].data(), &err));

        OCL_CHECK(err, buffer_dilations[cu] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            sizeof(int_t) * dilations_vec.size(), dilations_vec.data(), &err));
        OCL_CHECK(err, buffer_num_features_per_dilation[cu] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            sizeof(int_t) * num_features_per_dilation.size(), num_features_per_dilation.data(), &err));
        OCL_CHECK(err, buffer_biases[cu] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            sizeof(data_t) * biases.size(), biases.data(), &err));
        OCL_CHECK(err, buffer_scaler_mean[cu] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            sizeof(data_t) * scaler_mean.size(), scaler_mean.data(), &err));
        OCL_CHECK(err, buffer_inv_scale[cu] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            sizeof(data_t) * inv_scale.size(), inv_scale.data(), &err));
        OCL_CHECK(err, buffer_coefficients[cu] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            sizeof(data_t) * flattened_coefficients.size(), flattened_coefficients.data(), &err));
        OCL_CHECK(err, buffer_intercept[cu] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            sizeof(data_t) * intercept_vec.size(), intercept_vec.data(), &err));

        // Set K1 kernel arguments
        OCL_CHECK(err, err = krnl_feature_extraction[cu].setArg(0, buffer_time_series[cu]));
        OCL_CHECK(err, err = krnl_feature_extraction[cu].setArg(1, buffer_dilations[cu]));
        OCL_CHECK(err, err = krnl_feature_extraction[cu].setArg(2, buffer_num_features_per_dilation[cu]));
        OCL_CHECK(err, err = krnl_feature_extraction[cu].setArg(3, buffer_biases[cu]));
        // arg 4 = features_out stream (implicit)
        OCL_CHECK(err, err = krnl_feature_extraction[cu].setArg(5, time_series_length));
        OCL_CHECK(err, err = krnl_feature_extraction[cu].setArg(6, num_features));
        OCL_CHECK(err, err = krnl_feature_extraction[cu].setArg(7, num_dilations));

        // Set K2 kernel arguments
        OCL_CHECK(err, err = krnl_scaler[cu].setArg(0, buffer_scaler_mean[cu]));
        OCL_CHECK(err, err = krnl_scaler[cu].setArg(1, buffer_inv_scale[cu]));
        // arg 2 = features_in stream (implicit)
        // arg 3 = scaled_features_out stream (implicit)
        OCL_CHECK(err, err = krnl_scaler[cu].setArg(4, num_features));

        // Set K3 kernel arguments
        OCL_CHECK(err, err = krnl_classifier[cu].setArg(0, buffer_coefficients[cu]));
        OCL_CHECK(err, err = krnl_classifier[cu].setArg(1, buffer_intercept[cu]));
        OCL_CHECK(err, err = krnl_classifier[cu].setArg(2, buffer_predictions[cu]));
        // arg 3 = scaled_features_in stream (implicit)
        OCL_CHECK(err, err = krnl_classifier[cu].setArg(4, num_features));
        OCL_CHECK(err, err = krnl_classifier[cu].setArg(5, num_classes));

        // Migrate model parameters to FPGA for this CU
        std::vector<cl::Memory> model_buffers = {
            buffer_dilations[cu], buffer_num_features_per_dilation[cu], buffer_biases[cu],
            buffer_scaler_mean[cu], buffer_inv_scale[cu],
            buffer_coefficients[cu], buffer_intercept[cu]
        };
        OCL_CHECK(err, err = q[cu].enqueueMigrateMemObjects(model_buffers, 0));
    }
    for (int cu = 0; cu < NUM_CUS; cu++) {
        q[cu].finish();
    }
    std::cout << "Model weights loaded to FPGA (" << NUM_CUS << " CU pipelines)" << std::endl;

    /*====================================================INFERENCE LOOP (overlapped 3-CU)==========================================*/

    int correct_predictions = 0;
    size_t num_samples = test_inputs.size();

    cl::Event event_k3[NUM_CUS];
    bool cu_busy[NUM_CUS] = {false, false, false};
    int cu_sample[NUM_CUS] = {-1, -1, -1};

    auto total_start = std::chrono::high_resolution_clock::now();

    size_t next_sample = 0;
    size_t completed_samples = 0;

    std::vector<int> predicted_classes(num_samples);

    while (completed_samples < num_samples) {
        // Try to dispatch work to any idle CU
        for (int cu = 0; cu < NUM_CUS && next_sample < num_samples; cu++) {
            if (cu_busy[cu]) continue;

            size_t s = next_sample++;
            cu_sample[cu] = s;

            for (int j = 0; j < time_series_length; j++) {
                time_series_input[cu][j] = test_inputs[s][j];
            }

            // H2D transfer
            OCL_CHECK(err, err = q[cu].enqueueMigrateMemObjects({buffer_time_series[cu]}, 0));

            // Launch pipeline (all 3 kernels on same queue — they connect via AXI-Stream)
            cl::Event ev_k1, ev_k2, ev_k3_local;
            OCL_CHECK(err, err = q[cu].enqueueTask(krnl_feature_extraction[cu], nullptr, &ev_k1));
            OCL_CHECK(err, err = q[cu].enqueueTask(krnl_scaler[cu], nullptr, &ev_k2));
            OCL_CHECK(err, err = q[cu].enqueueTask(krnl_classifier[cu], nullptr, &ev_k3_local));

            // D2H transfer (enqueue after kernels)
            OCL_CHECK(err, err = q[cu].enqueueMigrateMemObjects({buffer_predictions[cu]}, CL_MIGRATE_MEM_OBJECT_HOST));

            event_k3[cu] = ev_k3_local;
            cu_busy[cu] = true;

            if (s % 100 == 0 || s < 5) {
                std::cout << "Dispatched sample " << s << " to CU" << cu << std::endl;
            }
        }

        // Collect results from any completed CU
        for (int cu = 0; cu < NUM_CUS; cu++) {
            if (!cu_busy[cu]) continue;

            q[cu].finish();

            size_t s = cu_sample[cu];

            int pred_class = 0;
            float max_score = prediction_output[cu][0];
            for (int i = 1; i < num_classes; i++) {
                if (prediction_output[cu][i] > max_score) {
                    max_score = prediction_output[cu][i];
                    pred_class = i;
                }
            }
            predicted_classes[s] = pred_class;

            int expected_class = (s < expected_classes.size()) ? expected_classes[s] : 0;
            if (pred_class == expected_class) correct_predictions++;

            cu_busy[cu] = false;
            completed_samples++;
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_wall_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    /*====================================================RESULTS===============================================================*/

    double correct_percentage = ((double)correct_predictions / num_samples) * 100.0;
    double throughput = num_samples / (total_wall_ms / 1000.0);

    std::cout << "\n========== FINAL RESULTS (3-CU) ==========" << std::endl;
    std::cout << "CU pipelines:         " << NUM_CUS << std::endl;
    std::cout << "Total samples:        " << num_samples << std::endl;
    std::cout << "Correct predictions:  " << correct_predictions << " / " << num_samples << std::endl;
    std::cout << "Overall accuracy:     " << std::fixed << std::setprecision(2) << correct_percentage << " %" << std::endl;
    if (python_baseline > 0.0f) {
        std::cout << "Python baseline:      " << std::fixed << std::setprecision(2) << python_baseline << " %" << std::endl;
    }
    std::cout << "\n========== TIMING RESULTS ==========" << std::endl;
    std::cout << "Total wall time:      " << std::fixed << std::setprecision(3) << total_wall_ms << " ms" << std::endl;
    std::cout << "Throughput:           " << std::fixed << std::setprecision(1) << throughput << " inferences/sec" << std::endl;
    std::cout << "Avg latency/sample:   " << std::fixed << std::setprecision(3) << (total_wall_ms / num_samples) << " ms" << std::endl;
    std::cout << "====================================" << std::endl;

    std::cout << "\nMiniRocket Modular 3-CU FPGA inference complete!" << std::endl;

    delete[] coefficients;
    return EXIT_SUCCESS;
}
