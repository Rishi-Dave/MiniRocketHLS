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
    // Three separate kernels for 3-stage streaming pipeline
    cl::Kernel load_krnl, minirocket_krnl, store_krnl;
    // THREE SEPARATE QUEUES: inference kernel runs on q (never waited), load/store on separate queues
    cl::CommandQueue q, q_load, q_store;

    auto devices = get_xil_devices();
    auto fileBuf = read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    cl::Program program;

    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, 0, &err));
        OCL_CHECK(err, q_load = cl::CommandQueue(context, device, 0, &err));
        OCL_CHECK(err, q_store = cl::CommandQueue(context, device, 0, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        program = cl::Program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            std::cout << "Setting up 3-stage streaming pipeline (load → inference → store)..." << std::endl;

            // Create all three kernel objects
            OCL_CHECK(err, load_krnl = cl::Kernel(program, "load_kernel", &err));
            if (err != CL_SUCCESS) {
                std::cout << "Failed to create load_kernel!\n";
                continue;
            }

            OCL_CHECK(err, minirocket_krnl = cl::Kernel(program, "minirocket_inference", &err));
            if (err != CL_SUCCESS) {
                std::cout << "Failed to create minirocket_inference kernel!\n";
                continue;
            }

            OCL_CHECK(err, store_krnl = cl::Kernel(program, "store_kernel", &err));
            if (err != CL_SUCCESS) {
                std::cout << "Failed to create store_kernel!\n";
                continue;
            }

            std::cout << "All three kernels created successfully!" << std::endl;
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

    // BATCH MODE: Allocate for ALL samples at once (mentor's design)
    // Will be resized after loading test data
    std::vector<data_t, aligned_allocator<data_t>> time_series_input;
    std::vector<data_t, aligned_allocator<data_t>> prediction_output;
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

    // BATCH MODE: Allocate buffers for ALL samples at once
    // Limit samples if buffer would be too large (>100MB to avoid allocation issues)
    int num_samples = test_inputs.size();
    size_t max_buffer_bytes = 100 * 1024 * 1024;  // 100 MB limit
    int max_samples = max_buffer_bytes / (time_series_length * sizeof(data_t));
    if (num_samples > max_samples) {
        std::cout << "WARNING: Limiting samples from " << num_samples << " to " << max_samples
                  << " due to buffer size limit" << std::endl;
        num_samples = max_samples;
    }
    int total_input_values = num_samples * time_series_length;
    int total_output_values = num_samples * num_classes;

    time_series_input.resize(total_input_values);
    prediction_output.resize(total_output_values);

    // Concatenate all test samples into one contiguous buffer
    std::cout << "Preparing batch input: " << num_samples << " samples x "
              << time_series_length << " values = " << total_input_values << " total values" << std::endl;
    for (int s = 0; s < num_samples; s++) {
        for (int t = 0; t < time_series_length; t++) {
            time_series_input[s * time_series_length + t] = test_inputs[s][t];
        }
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

    // Input buffers - BATCH MODE: buffer holds ALL samples
    // Use plain READ_ONLY and write data separately (avoids allocation issues with large buffers)
    OCL_CHECK(err, cl::Buffer buffer_time_series(context, CL_MEM_READ_ONLY,
        sizeof(data_t) * total_input_values, nullptr, &err));
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

    // Output buffer - BATCH MODE: holds predictions for ALL samples
    // Use READ_WRITE for output since we need to read it back
    OCL_CHECK(err, cl::Buffer buffer_predictions(context, CL_MEM_READ_WRITE,
        sizeof(data_t) * total_output_values, nullptr, &err));

    // Set kernel arguments for each of the three kernels
    std::cout << "Setting kernel arguments for 3-stage pipeline..." << std::endl;

    // load_kernel arguments: only set m_axi ports, streams are auto-connected
    // Arg 0: time_series_input (m_axi buffer)
    // Arg 1: output stream (NOT set by host - auto-connected)
    // Arg 2: num_values (s_axilite scalar) - BATCH MODE: ALL values from ALL samples
    OCL_CHECK(err, err = load_krnl.setArg(0, buffer_time_series));
    OCL_CHECK(err, err = load_krnl.setArg(2, total_input_values));

    // minirocket_inference arguments: skip stream ports (0,1), set m_axi and scalars
    // Arg 0-1: input/output streams (NOT set by host - auto-connected)
    // Arg 2-8: m_axi buffers (coefficients, intercept, scaler_mean, scaler_scale, dilations, num_features_per_dilation, biases)
    // Arg 9-12: s_axilite scalars (time_series_length, num_features, num_classes, num_dilations)
    OCL_CHECK(err, err = minirocket_krnl.setArg(2, buffer_coefficients));
    OCL_CHECK(err, err = minirocket_krnl.setArg(3, buffer_intercept));
    OCL_CHECK(err, err = minirocket_krnl.setArg(4, buffer_scaler_mean));
    OCL_CHECK(err, err = minirocket_krnl.setArg(5, buffer_scaler_scale));
    OCL_CHECK(err, err = minirocket_krnl.setArg(6, buffer_dilations));
    OCL_CHECK(err, err = minirocket_krnl.setArg(7, buffer_num_features_per_dilation));
    OCL_CHECK(err, err = minirocket_krnl.setArg(8, buffer_biases));
    OCL_CHECK(err, err = minirocket_krnl.setArg(9, time_series_length));
    OCL_CHECK(err, err = minirocket_krnl.setArg(10, num_features));
    OCL_CHECK(err, err = minirocket_krnl.setArg(11, num_classes));
    OCL_CHECK(err, err = minirocket_krnl.setArg(12, num_dilations));

    // store_kernel arguments: skip input stream, set m_axi buffer and scalar
    // Arg 0: input stream (NOT set by host - auto-connected)
    // Arg 1: predictions_output (m_axi buffer)
    // Arg 2: num_predictions (s_axilite scalar) - BATCH MODE: ALL predictions from ALL samples
    OCL_CHECK(err, err = store_krnl.setArg(1, buffer_predictions));
    OCL_CHECK(err, err = store_krnl.setArg(2, total_output_values));

    std::cout << "Loading model weights to FPGA..." << std::endl;
    std::vector<cl::Memory> model_buffers = {
        buffer_coefficients, buffer_intercept,
        buffer_scaler_mean, buffer_scaler_scale, buffer_dilations,
        buffer_num_features_per_dilation, buffer_biases
    };
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(model_buffers, 0));
    q.finish();
    std::cout << "Model weights loaded successfully!" << std::endl;

    /*====================================================KERNEL===============================================================*/
    // BATCH MODE (mentor's design): Load ALL data, run kernels ONCE, read ALL results

    std::cout << "\n========== BATCH STREAMING MODE ==========" << std::endl;
    std::cout << "Total samples: " << num_samples << std::endl;
    std::cout << "Time series length: " << time_series_length << std::endl;
    std::cout << "Total input values: " << total_input_values << std::endl;
    std::cout << "Total output values: " << total_output_values << std::endl;
    std::cout << "==========================================\n" << std::endl;

    // Timing variables
    double h2d_ms = 0, kernel_ms = 0, d2h_ms = 0;

    // Step 1: H2D - Transfer ALL input data at once
    std::cout << "Step 1: H2D transfer (all " << total_input_values << " values)..." << std::flush;
    auto h2d_start = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q_load.enqueueWriteBuffer(buffer_time_series, CL_TRUE, 0,
        sizeof(data_t) * total_input_values, time_series_input.data()));
    auto h2d_end = std::chrono::high_resolution_clock::now();
    h2d_ms = std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
    std::cout << " Done! (" << h2d_ms << " ms)" << std::endl;

    // Step 2: Launch all three kernels ONCE
    std::cout << "Step 2: Launching 3-stage pipeline (load → inference → store)..." << std::flush;
    auto kernel_start = std::chrono::high_resolution_clock::now();

    // Launch inference kernel first (it runs in while(true) loop for BUILD=1)
    OCL_CHECK(err, err = q.enqueueTask(minirocket_krnl));

    // Launch load kernel to stream all input values
    OCL_CHECK(err, err = q_load.enqueueTask(load_krnl));

    // Launch store kernel to collect all output predictions
    OCL_CHECK(err, err = q_store.enqueueTask(store_krnl));

    // Wait for store to complete (indicates all data processed)
    q_store.finish();

    auto kernel_end = std::chrono::high_resolution_clock::now();
    kernel_ms = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();
    std::cout << " Done! (" << kernel_ms << " ms)" << std::endl;

    // Step 3: D2H - Transfer ALL results at once
    std::cout << "Step 3: D2H transfer (all " << total_output_values << " predictions)..." << std::flush;
    auto d2h_start = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q_store.enqueueReadBuffer(buffer_predictions, CL_TRUE, 0,
        sizeof(data_t) * total_output_values, prediction_output.data()));
    auto d2h_end = std::chrono::high_resolution_clock::now();
    d2h_ms = std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();
    std::cout << " Done! (" << d2h_ms << " ms)" << std::endl;

    // Step 4: Post-process results - extract predictions for each sample
    std::cout << "\nProcessing results..." << std::endl;
    int correct_predictions = 0;

    for (int s = 0; s < num_samples; s++) {
        // Each sample's predictions start at offset s * num_classes
        int pred_offset = s * num_classes;

        // Find predicted class (argmax)
        int predicted_class = 0;
        float max_score = prediction_output[pred_offset];
        for (int c = 1; c < num_classes; c++) {
            if (prediction_output[pred_offset + c] > max_score) {
                max_score = prediction_output[pred_offset + c];
                predicted_class = c;
            }
        }

        // Get expected class
        int expected_class = (s < expected_classes.size()) ? expected_classes[s] : 0;

        if (predicted_class == expected_class) {
            correct_predictions++;
        }

        // Print first few and last sample results
        if (s < 5 || s == num_samples - 1) {
            std::cout << "Sample " << s << ": Predicted=" << predicted_class
                      << " Expected=" << expected_class
                      << " (" << (predicted_class == expected_class ? "CORRECT" : "WRONG") << ")" << std::endl;
        }
    }
    /*====================================================RESULTS===============================================================*/

    // Print timing results
    double total_ms = h2d_ms + kernel_ms + d2h_ms;
    double correct_percentage = ((double)correct_predictions / num_samples) * 100.0;
    double throughput = (num_samples * 1000.0) / kernel_ms;  // inferences per second

    std::cout << "\n========== FINAL RESULTS ==========" << std::endl;
    std::cout << "Model: " << model_file << std::endl;
    std::cout << "Test data: " << test_file << std::endl;
    std::cout << "Total correct predictions: " << correct_predictions << " / " << num_samples << std::endl;
    std::cout << "Overall accuracy: " << std::fixed << std::setprecision(2) << correct_percentage << " %" << std::endl;
    if (python_baseline > 0.0f) {
        std::cout << "Python baseline accuracy: " << std::fixed << std::setprecision(2) << python_baseline << " %" << std::endl;
    }

    std::cout << "\n========== TIMING RESULTS ==========" << std::endl;
    std::cout << "H2D transfer:      " << std::fixed << std::setprecision(3) << h2d_ms << " ms" << std::endl;
    std::cout << "Kernel execution:  " << std::fixed << std::setprecision(3) << kernel_ms << " ms" << std::endl;
    std::cout << "D2H transfer:      " << std::fixed << std::setprecision(3) << d2h_ms << " ms" << std::endl;
    std::cout << "Total latency:     " << std::fixed << std::setprecision(3) << total_ms << " ms" << std::endl;
    std::cout << "Throughput:        " << std::fixed << std::setprecision(1) << throughput << " inferences/sec" << std::endl;
    std::cout << "Per-sample latency: " << std::fixed << std::setprecision(3) << (kernel_ms / num_samples) << " ms" << std::endl;
    std::cout << "====================================" << std::endl;

    std::cout << "\nMiniRocket FPGA inference complete!" << std::endl;

    return EXIT_SUCCESS;
}