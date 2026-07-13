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
    cl::Kernel krnl;
    cl::CommandQueue q;
    
    auto devices = get_xil_devices();
    auto fileBuf = read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, 0, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            std::cout << "Setting CU(s) up..." << std::endl; 
            OCL_CHECK(err, krnl = cl::Kernel(program, "minirocket_inference", &err));  // Note: function name
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
    int_t num_dilations, num_features, num_classes, time_series_length;
    
    // Load model into HLS arrays
    std::cout << "Loading model..." << std::endl;
    if (!loader.load_model_to_hls_arrays(model_file, coefficients, intercept.data(), 
                                        scaler_mean.data(), scaler_scale.data(), dilations.data(),
                                        num_features_per_dilation.data(), biases.data(),
                                        num_dilations, num_features, num_classes,
                                        time_series_length)) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
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


    /*====================================================Setting up kernel I/O===============================================================*/

    // Input buffers
    OCL_CHECK(err, cl::Buffer buffer_time_series(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
        sizeof(data_t) * time_series_length, time_series_input.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_coefficients(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
        sizeof(data_t) * flattened_coefficients.size(), flattened_coefficients.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_intercept(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
        sizeof(data_t) * intercept.size(), intercept.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_scaler_mean(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
        sizeof(data_t) * scaler_mean.size(), scaler_mean.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_scaler_scale(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
        sizeof(data_t) * scaler_scale.size(), scaler_scale.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_dilations(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
        sizeof(int_t) * dilations.size(), dilations.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_num_features_per_dilation(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
        sizeof(int_t) * num_features_per_dilation.size(), num_features_per_dilation.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_biases(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
        sizeof(data_t) * biases.size(), biases.data(), &err));

    // Output buffer
    OCL_CHECK(err, cl::Buffer buffer_predictions(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
        sizeof(data_t) * num_classes, prediction_output.data(), &err));

    // Set kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, buffer_time_series));
    OCL_CHECK(err, err = krnl.setArg(1, buffer_predictions));
    OCL_CHECK(err, err = krnl.setArg(2, buffer_coefficients));
    OCL_CHECK(err, err = krnl.setArg(3, buffer_intercept));
    OCL_CHECK(err, err = krnl.setArg(4, buffer_scaler_mean));
    OCL_CHECK(err, err = krnl.setArg(5, buffer_scaler_scale));
    OCL_CHECK(err, err = krnl.setArg(6, buffer_dilations));
    OCL_CHECK(err, err = krnl.setArg(7, buffer_num_features_per_dilation));
    OCL_CHECK(err, err = krnl.setArg(8, buffer_biases));
    OCL_CHECK(err, err = krnl.setArg(9, time_series_length));
    OCL_CHECK(err, err = krnl.setArg(10, num_features));
    OCL_CHECK(err, err = krnl.setArg(11, num_classes));
    OCL_CHECK(err, err = krnl.setArg(12, num_dilations));

    std::cout << "Loading Weights to FPGA" << std::endl;
    std::vector<cl::Memory> input_buffers = {
        buffer_coefficients, buffer_intercept,
        buffer_scaler_mean, buffer_scaler_scale, buffer_dilations,
        buffer_num_features_per_dilation, buffer_biases
    };
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(input_buffers, 0));
    q.finish();

    /*====================================================KERNEL===============================================================*/

    // Timing variables
    double h2d_ms = 0, kernel_ms = 0, d2h_ms = 0;
    double total_h2d_ms = 0, total_kernel_ms = 0, total_d2h_ms = 0;

    int correct_predictions = 0;

    for (int test_inputs_idx = 0; test_inputs_idx < test_inputs.size(); test_inputs_idx++) {
        // Prepare input time series
        for (int j = 0; j < time_series_length; j++) {
            time_series_input[j] = test_inputs[test_inputs_idx][j];
        }

        // Host to device data transfer
        //std::cout << "Loading time series data[" << test_inputs_idx << "] to FPGA" << std::endl;
        auto h2d_start = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_time_series}, 0));
        q.finish();
        auto h2d_end = std::chrono::high_resolution_clock::now();
        h2d_ms = std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
        total_h2d_ms += h2d_ms;

        // Execute kernel
        //std::cout << "STARTING kernel for input[" << test_inputs_idx << "]" << std::endl;
        auto kernel_start = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueTask(krnl));
        q.finish();
        auto kernel_end = std::chrono::high_resolution_clock::now();
        kernel_ms = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();
        total_kernel_ms += kernel_ms;

        // Device to host data transfer
        //std::cout << "Transferring results for input[" << test_inputs_idx << "] from FPGA to host" << std::endl;
        auto d2h_start = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_predictions}, CL_MIGRATE_MEM_OBJECT_HOST));
        q.finish();
        auto d2h_end = std::chrono::high_resolution_clock::now();
        d2h_ms = std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();
        total_d2h_ms += d2h_ms;
    
        /*====================================================PRINTING OUTPUTS===============================================================*/


        // // Find predicted class
        // int predicted_class = 0;
        // float max_score = prediction_output[0];
        
        // std::cout << "Classification scores: ";
        // for (int i = 0; i < num_classes; i++) {
        //     std::cout << "Class" << i << "=" << std::fixed << std::setprecision(4) << prediction_output[i] << " ";
        //     if (prediction_output[i] > max_score) {
        //         max_score = prediction_output[i];
        //         predicted_class = i;
        //     }
        // }
        // std::cout << std::endl;
        
        // std::cout << "Predicted class: " << predicted_class << " (score: " << std::fixed << std::setprecision(4) << max_score << ")" << std::endl;
        
        // // Get expected class directly from labels
        // int expected_class = (test_inputs_idx < expected_classes.size()) ? expected_classes[test_inputs_idx] : 0;
        // std::cout << "Expected class: " << expected_class << std::endl;

        // bool correct = (predicted_class == expected_class);
        // if (correct) {
        //     correct_predictions++;
        //     std::cout << "Prediction correct!" << std::endl;
        // } else {
        //     std::cout << "Prediction incorrect!" << std::endl;
        // }

        // // Print timing results
        // double single_ms = h2d_ms + kernel_ms + d2h_ms;
        // std::cout << "\n========== TIMING RESULTS ==========" << std::endl;
        // std::cout << "H2D transfer:      " << std::fixed << std::setprecision(3) << h2d_ms << " ms" << std::endl;
        // std::cout << "Kernel execution:  " << std::fixed << std::setprecision(3) << kernel_ms << " ms" << std::endl;
        // std::cout << "D2H transfer:      " << std::fixed << std::setprecision(3) << d2h_ms << " ms" << std::endl;
        // std::cout << "Total latency:     " << std::fixed << std::setprecision(3) << single_ms << " ms" << std::endl;
        // std::cout << "====================================" << std::endl;

    }
    /*====================================================RESULTS===============================================================*/

    // Print timing results
    double total_ms = total_h2d_ms + total_kernel_ms + total_d2h_ms;
    double correct_percentage = ((double)correct_predictions / test_inputs.size()) * 100.0;
    std::cout << "\n========== FINAL RESULTS ==========" << std::endl;
    std::cout << "Model: " << model_file << std::endl;
    std::cout << "Test data: " << test_file << std::endl;
    std::cout << "Total correct predictions: " << correct_predictions << " / " << test_inputs.size() << std::endl;
    std::cout << "Overall accuracy: " << std::fixed << std::setprecision(2) << correct_percentage << " %" << std::endl;
    if (python_baseline > 0.0f) {
        std::cout << "Python baseline accuracy: " << std::fixed << std::setprecision(2) << python_baseline << " %" << std::endl;
    }

    std::cout << "\n========== TIMING RESULTS ==========" << std::endl;
    std::cout << "H2D transfer:      " << std::fixed << std::setprecision(3) << total_h2d_ms << " ms" << std::endl;
    std::cout << "Kernel execution:  " << std::fixed << std::setprecision(3) << total_kernel_ms << " ms" << std::endl;
    std::cout << "D2H transfer:      " << std::fixed << std::setprecision(3) << total_d2h_ms << " ms" << std::endl;
    std::cout << "Total latency:     " << std::fixed << std::setprecision(3) << total_ms << " ms" << std::endl;
    std::cout << "Throughput:        " << std::fixed << std::setprecision(1) << (test_inputs.size() / total_kernel_ms) << " inferences/sec" << std::endl;
    std::cout << "====================================" << std::endl;

    std::cout << "\nMiniRocket FPGA inference complete!" << std::endl;

    return EXIT_SUCCESS;
}