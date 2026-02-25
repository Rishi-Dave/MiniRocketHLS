#include "../include/minirocket_host.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <cmath>
#include <chrono>

// Include JSON parsing capability (simplified)
#include <sstream>

// Include MiniRocket loader
#include "../include/minirocket_loader.h"


int main(int argc, char** argv) {

    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File> <Model JSON File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    std::string model_file = argv[2];
    std::string test_file = argv[3];


    /*====================================================CL===============================================================*/

    cl_int err;
    cl::Context context;
    cl::Kernel load, compute, store;
    cl::CommandQueue q_load, q_compute, q_store;
    
    auto devices = get_xil_devices();
    auto fileBuf = read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q_load = cl::CommandQueue(context, device, 0, &err));
        OCL_CHECK(err, q_compute = cl::CommandQueue(context, device, 0, &err));
        OCL_CHECK(err, q_store = cl::CommandQueue(context, device, 0, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            std::cout << "Setting CU(s) up..." << std::endl; 
            OCL_CHECK(err, load = cl::Kernel(program, "load", &err));  // Note: function name
            OCL_CHECK(err, compute = cl::Kernel(program, "minirocket_inference", &err));  // Note: function name
            OCL_CHECK(err, store = cl::Kernel(program, "store", &err));  // Note: function name
            valid_device = true;
            break;
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    /*====================================================CL===============================================================*/

    std::cout << "Loading MiniRocket model and test data" << std::endl;
    std::cout << "Model file: " << model_file << std::endl;
    std::cout << "Test file: " << test_file << std::endl;


    // Initialize loader
    MiniRocketTestbenchLoader loader;
    
    // HLS arrays (heap allocated for testbench to avoid stack overflow)
    data_t (*coefficients)[MAX_FEATURES] = new data_t[MAX_CLASSES][MAX_FEATURES];

    std::vector<data_t, aligned_allocator<data_t>> time_series_input(MAX_TIME_SERIES_LENGTH);
    std::vector<data_t, aligned_allocator<data_t>> prediction_output(MAX_CLASSES);
    std::vector<data_t, aligned_allocator<data_t>> flattened_coefficients(MAX_CLASSES * MAX_FEATURES);

    std::vector<data_t, aligned_allocator<data_t>> intercept(MAX_CLASSES);
    std::vector<data_t, aligned_allocator<data_t>> scaler_mean(MAX_FEATURES);
    std::vector<data_t, aligned_allocator<data_t>> scaler_scale(MAX_FEATURES);
    std::vector<data_t, aligned_allocator<data_t>> biases(MAX_FEATURES);
    std::vector<int_t, aligned_allocator<int_t>> dilations(MAX_DILATIONS);
    std::vector<int_t, aligned_allocator<int_t>> num_features_per_dilation(MAX_DILATIONS);
    int_t num_dilations, num_features, num_classes, num_samples;
    
    // Load model into HLS arrays
    std::cout << "Loading model..." << std::endl;
    if (!loader.load_model_to_hls_arrays(model_file, coefficients, intercept.data(), 
                                        scaler_mean.data(), scaler_scale.data(), dilations.data(),
                                        num_features_per_dilation.data(), biases.data(),
                                        num_dilations, num_features, num_classes,
                                        num_samples)) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }



    for (int i = 0; i < num_classes * num_features; i++) {
        int row = i / num_features;
        int col = i % num_features;
        flattened_coefficients[i] = coefficients[row][col];
    }
    
    // Load test data
    std::cout << "Loading test data..." << std::endl;
    std::vector<std::vector<data_t>> expected_outputs;
    std::vector<data_t> test_inputs;
    if (!loader.load_test_data(test_file, test_inputs, expected_outputs)) {
        std::cerr << "Failed to load test data!" << std::endl;
        return 1;
    }
    
    // Try to load Python baseline accuracy from test data
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
            python_baseline = std::stof(accuracy_str) * 100.0f; // Convert to percentage
        }
    }
    
    // Convert y_test (single values) to expected class indices
    std::vector<int> expected_classes;
    for (const auto& output : expected_outputs) {
        if (!output.empty()) {
            expected_classes.push_back((int)output[0]);
        }
    }

    time_series_input.resize(time_series_input.size());
    prediction_output.resize(time_series_input.size() * num_classes);

    for (size_t i = 0; i < time_series_input.size(); i++) {
        time_series_input[i] = test_inputs[i];
    }

    /*====================================================Setting up kernel I/O===============================================================*/

    std::cout << "Setting up LOAD I/O buffers" << std::endl;

    // Load buffer (input time series)
    OCL_CHECK(err, cl::Buffer buffer_time_series(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * time_series_input.size(), time_series_input.data(), &err));
    
    std::cout << "Setting up COMPUTE I/O buffers" << std::endl;
    // Compute buffers (model parameters)
    OCL_CHECK(err, cl::Buffer buffer_coefficients(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * flattened_coefficients.size(), flattened_coefficients.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_intercept(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * intercept.size(), intercept.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_scaler_mean(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * scaler_mean.size(), scaler_mean.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_scaler_scale(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * scaler_scale.size(), scaler_scale.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_dilations(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int_t) * dilations.size(), dilations.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_num_features_per_dilation(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int_t) * num_features_per_dilation.size(), num_features_per_dilation.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_biases(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * biases.size(), biases.data(), &err));

    // Store buffer (output predictions)
    std::cout << "Setting up STORE I/O buffers" << std::endl;
    OCL_CHECK(err, cl::Buffer buffer_predictions(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(data_t) * time_series_input.size() * num_classes, prediction_output.data(), &err));

    std::cout << "Setting up kernel args" << std::endl;
    // Set kernel arguments
    OCL_CHECK(err, err = load.setArg(0, buffer_time_series));
    OCL_CHECK(err, err = load.setArg(2, (int_t) time_series_input.size()));


    OCL_CHECK(err, err = store.setArg(1, buffer_predictions));
    OCL_CHECK(err, err = store.setArg(2, (int_t) time_series_input.size())); 
    OCL_CHECK(err, err = store.setArg(3, (int_t) num_classes));

    OCL_CHECK(err, err = compute.setArg(2, buffer_coefficients));
    OCL_CHECK(err, err = compute.setArg(3, buffer_intercept));
    OCL_CHECK(err, err = compute.setArg(4, buffer_scaler_mean));
    OCL_CHECK(err, err = compute.setArg(5, buffer_scaler_scale));
    OCL_CHECK(err, err = compute.setArg(6, buffer_dilations));
    OCL_CHECK(err, err = compute.setArg(7, buffer_num_features_per_dilation));
    OCL_CHECK(err, err = compute.setArg(8, buffer_biases));
    OCL_CHECK(err, err = compute.setArg(9, 128));
    OCL_CHECK(err, err = compute.setArg(10, num_features));
    OCL_CHECK(err, err = compute.setArg(11, num_classes));
    OCL_CHECK(err, err = compute.setArg(12, num_dilations));

    std::cout << "Loading Weights to FPGA" << std::endl;
    std::vector<cl::Memory> input_buffers = {
        buffer_coefficients, buffer_intercept,
        buffer_scaler_mean, buffer_scaler_scale, buffer_dilations,
        buffer_num_features_per_dilation, buffer_biases
    };
    OCL_CHECK(err, err = q_compute.enqueueMigrateMemObjects(input_buffers, 0));
    q_compute.finish();

    OCL_CHECK(err, err = q_load.enqueueMigrateMemObjects({buffer_time_series}, 0));
    q_load.finish();

    /*====================================================KERNEL===============================================================*/
    std::cout << "Starting inference on FPGA" << std::endl;
    q_compute.enqueueTask(compute);


    auto start_time = std::chrono::high_resolution_clock::now();
    q_load.enqueueTask(load);
    q_store.enqueueTask(store);
    // q_compute.finish();
    q_load.finish();
    q_store.finish();
    auto end_time = std::chrono::high_resolution_clock::now();
    /*====================================================RESULTS===============================================================*/

    auto duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "Total FPGA execution time: " << std::fixed << std::setprecision(3) << duration_ms << " ms" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(1) << (test_inputs.size() / (duration_ms / 1000.0)) << " inferences/sec" << std::endl;



    // // Print timing results
    // double total_ms = total_h2d_ms + total_kernel_ms + total_d2h_ms;
    // double correct_percentage = ((double)correct_predictions / test_inputs.size()) * 100.0;
    // std::cout << "\n========== FINAL RESULTS ==========" << std::endl;
    // std::cout << "Model: " << model_file << std::endl;
    // std::cout << "Test data: " << test_file << std::endl;
    // std::cout << "Total correct predictions: " << correct_predictions << " / " << test_inputs.size() << std::endl;
    // std::cout << "Overall accuracy: " << std::fixed << std::setprecision(2) << correct_percentage << " %" << std::endl;
    // if (python_baseline > 0.0f) {
    //     std::cout << "Python baseline accuracy: " << std::fixed << std::setprecision(2) << python_baseline << " %" << std::endl;
    // }

    // std::cout << "\n========== TIMING RESULTS ==========" << std::endl;
    // std::cout << "H2D transfer:      " << std::fixed << std::setprecision(3) << total_h2d_ms << " ms" << std::endl;
    // std::cout << "Kernel execution:  " << std::fixed << std::setprecision(3) << total_kernel_ms << " ms" << std::endl;
    // std::cout << "D2H transfer:      " << std::fixed << std::setprecision(3) << total_d2h_ms << " ms" << std::endl;
    // std::cout << "Total latency:     " << std::fixed << std::setprecision(3) << total_ms << " ms" << std::endl;
    // std::cout << "Throughput:        " << std::fixed << std::setprecision(1) << (test_inputs.size() / total_kernel_ms) << " inferences/sec" << std::endl;
    // std::cout << "====================================" << std::endl;

    // std::cout << "\nMiniRocket FPGA inference complete!" << std::endl;

    return EXIT_SUCCESS;
}