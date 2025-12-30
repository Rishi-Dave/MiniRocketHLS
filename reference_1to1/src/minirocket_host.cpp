/*
 * MiniRocket FPGA Host Application
 * Optimized for Xilinx Alveo U280 with multiple compute units
 *
 * Features:
 * - Multi-CU parallel execution
 * - Efficient buffer management
 * - Pipelined data transfer and compute
 * - Supports batch inference
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>

// XRT includes
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h>

// Local includes
#include "minirocket_hls_testbench_loader.h"
#include "krnl.hpp"

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

class MiniRocketFPGA {
private:
    cl::Context context;
    cl::CommandQueue q;
    cl::Program program;
    std::vector<cl::Kernel> kernels;
    int num_compute_units;

    // Model parameters
    int num_features;
    int num_classes;
    int num_dilations;
    int time_series_length;

    // Device buffers (one set per CU for pipelining)
    struct CUBuffers {
        cl::Buffer input_buf;
        cl::Buffer output_buf;
        cl::Buffer coefficients_buf;
        cl::Buffer intercept_buf;
        cl::Buffer scaler_mean_buf;
        cl::Buffer scaler_scale_buf;
        cl::Buffer dilations_buf;
        cl::Buffer num_features_per_dilation_buf;
        cl::Buffer biases_buf;
    };
    std::vector<CUBuffers> cu_buffers;

    // Model parameters (shared across CUs)
    std::vector<float> coefficients_flat;
    std::vector<float> intercept_vec;
    std::vector<float> scaler_mean_vec;
    std::vector<float> scaler_scale_vec;
    std::vector<int> dilations_vec;
    std::vector<int> num_features_per_dilation_vec;
    std::vector<float> biases_vec;

public:
    MiniRocketFPGA(const std::string& xclbin_path, int num_cus = 1)
        : num_compute_units(num_cus) {

        std::cout << "Initializing MiniRocket FPGA accelerator..." << std::endl;
        std::cout << "Number of compute units: " << num_cus << std::endl;

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
        q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

        // Load xclbin
        std::cout << "Loading xclbin: " << xclbin_path << std::endl;
        auto fileBuf = read_binary_file(xclbin_path);
        cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

        std::vector<cl::Device> devices_vec = {device};
        program = cl::Program(context, devices_vec, bins);

        // Create kernels for each CU
        std::cout << "Creating kernels..." << std::endl;
        for (int i = 0; i < num_compute_units; i++) {
            kernels.push_back(cl::Kernel(program, "krnl_top"));
        }

        std::cout << "FPGA initialization complete!" << std::endl;
    }

    void load_model(const std::string& model_path) {
        std::cout << "\nLoading model: " << model_path << std::endl;

        MiniRocketTestbenchLoader loader;

        // Allocate arrays for model loading
        data_t (*coefficients)[MAX_FEATURES] = new data_t[MAX_CLASSES][MAX_FEATURES];
        data_t *intercept = new data_t[MAX_CLASSES];
        data_t *scaler_mean = new data_t[MAX_FEATURES];
        data_t *scaler_scale = new data_t[MAX_FEATURES];
        int_t *dilations = new int_t[MAX_DILATIONS];
        int_t *num_features_per_dilation = new int_t[MAX_DILATIONS];
        data_t *biases = new data_t[MAX_FEATURES];

        int_t num_dilations_out, num_features_out, num_classes_out, time_series_length_out;

        if (!loader.load_model_to_hls_arrays(model_path, coefficients, intercept,
                                             scaler_mean, scaler_scale, dilations,
                                             num_features_per_dilation, biases,
                                             num_dilations_out, num_features_out,
                                             num_classes_out, time_series_length_out)) {
            throw std::runtime_error("Failed to load model");
        }

        num_features = num_features_out;
        num_classes = num_classes_out;
        num_dilations = num_dilations_out;
        time_series_length = time_series_length_out;

        std::cout << "Model loaded successfully into HLS arrays:" << std::endl;
        std::cout << "  Features: " << num_features << std::endl;
        std::cout << "  Classes: " << num_classes << std::endl;
        std::cout << "  Dilations: " << num_dilations << std::endl;
        std::cout << "Model loaded: " << num_features << " features, "
                  << num_classes << " classes, "
                  << num_dilations << " dilations" << std::endl;

        // Separate model parameters into individual vectors
        coefficients_flat.resize(num_classes * num_features);
        intercept_vec.resize(num_classes);
        scaler_mean_vec.resize(num_features);
        scaler_scale_vec.resize(num_features);
        dilations_vec.resize(num_dilations);
        num_features_per_dilation_vec.resize(num_dilations);
        biases_vec.resize(num_features);

        for (int i = 0; i < num_dilations; i++) dilations_vec[i] = dilations[i];
        for (int i = 0; i < num_dilations; i++) num_features_per_dilation_vec[i] = num_features_per_dilation[i];
        for (int i = 0; i < num_features; i++) biases_vec[i] = biases[i];
        for (int i = 0; i < num_features; i++) scaler_mean_vec[i] = scaler_mean[i];
        for (int i = 0; i < num_features; i++) scaler_scale_vec[i] = scaler_scale[i];
        for (int c = 0; c < num_classes; c++)
            for (int f = 0; f < num_features; f++)
                coefficients_flat[c * num_features + f] = coefficients[c][f];
        for (int i = 0; i < num_classes; i++) intercept_vec[i] = intercept[i];

        // Create device buffers for each CU
        for (int cu = 0; cu < num_compute_units; cu++) {
            CUBuffers bufs;

            bufs.input_buf = cl::Buffer(context, CL_MEM_READ_ONLY, MAX_TIME_SERIES_LENGTH * sizeof(float));
            bufs.output_buf = cl::Buffer(context, CL_MEM_WRITE_ONLY, MAX_CLASSES * sizeof(float));
            bufs.coefficients_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              coefficients_flat.size() * sizeof(float),
                                              coefficients_flat.data());
            bufs.intercept_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           intercept_vec.size() * sizeof(float),
                                           intercept_vec.data());
            bufs.scaler_mean_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             scaler_mean_vec.size() * sizeof(float),
                                             scaler_mean_vec.data());
            bufs.scaler_scale_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              scaler_scale_vec.size() * sizeof(float),
                                              scaler_scale_vec.data());
            bufs.dilations_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           dilations_vec.size() * sizeof(int),
                                           dilations_vec.data());
            bufs.num_features_per_dilation_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                           num_features_per_dilation_vec.size() * sizeof(int),
                                                           num_features_per_dilation_vec.data());
            bufs.biases_buf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        biases_vec.size() * sizeof(float),
                                        biases_vec.data());

            cu_buffers.push_back(bufs);
        }

        // Cleanup
        delete[] coefficients;
        delete[] intercept;
        delete[] scaler_mean;
        delete[] scaler_scale;
        delete[] dilations;
        delete[] num_features_per_dilation;
        delete[] biases;

        std::cout << "Model loaded to device memory" << std::endl;
    }

    std::vector<int> predict_batch(const std::vector<std::vector<float>>& inputs) {
        int batch_size = inputs.size();
        std::vector<int> predictions(batch_size);

        auto start_time = std::chrono::high_resolution_clock::now();

        // Process in parallel across CUs
        std::vector<cl::Event> events;

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            int cu_idx = batch_idx % num_compute_units;

            // Prepare input
            std::vector<float> input_padded(MAX_TIME_SERIES_LENGTH, 0.0f);
            std::copy(inputs[batch_idx].begin(), inputs[batch_idx].end(), input_padded.begin());

            // Transfer input to device
            cl::Event write_event;
            q.enqueueWriteBuffer(cu_buffers[cu_idx].input_buf, CL_FALSE, 0,
                               MAX_TIME_SERIES_LENGTH * sizeof(float),
                               input_padded.data(), nullptr, &write_event);

            // Set kernel arguments (must match krnl_top signature)
            kernels[cu_idx].setArg(0, cu_buffers[cu_idx].input_buf);               // time_series_input
            kernels[cu_idx].setArg(1, cu_buffers[cu_idx].output_buf);              // prediction_output
            kernels[cu_idx].setArg(2, cu_buffers[cu_idx].coefficients_buf);        // coefficients
            kernels[cu_idx].setArg(3, cu_buffers[cu_idx].intercept_buf);           // intercept
            kernels[cu_idx].setArg(4, cu_buffers[cu_idx].scaler_mean_buf);         // scaler_mean
            kernels[cu_idx].setArg(5, cu_buffers[cu_idx].scaler_scale_buf);        // scaler_scale
            kernels[cu_idx].setArg(6, cu_buffers[cu_idx].dilations_buf);           // dilations
            kernels[cu_idx].setArg(7, cu_buffers[cu_idx].num_features_per_dilation_buf);  // num_features_per_dilation
            kernels[cu_idx].setArg(8, cu_buffers[cu_idx].biases_buf);              // biases
            kernels[cu_idx].setArg(9, (int)inputs[batch_idx].size());              // time_series_length
            kernels[cu_idx].setArg(10, num_features);                              // num_features
            kernels[cu_idx].setArg(11, num_classes);                               // num_classes
            kernels[cu_idx].setArg(12, num_dilations);                             // num_dilations

            // Execute kernel
            cl::Event exec_event;
            std::vector<cl::Event> wait_events = {write_event};
            q.enqueueTask(kernels[cu_idx], &wait_events, &exec_event);

            // Read output
            std::vector<float> output(MAX_CLASSES);
            cl::Event read_event;
            std::vector<cl::Event> exec_events = {exec_event};
            q.enqueueReadBuffer(cu_buffers[cu_idx].output_buf, CL_FALSE, 0,
                              MAX_CLASSES * sizeof(float),
                              output.data(), &exec_events, &read_event);

            events.push_back(read_event);

            // Get prediction (wait for this specific task to complete)
            read_event.wait();

            int pred_class = 0;
            float max_score = output[0];
            for (int c = 1; c < num_classes; c++) {
                if (output[c] > max_score) {
                    max_score = output[c];
                    pred_class = c;
                }
            }
            predictions[batch_idx] = pred_class;
        }

        // Wait for all operations to complete
        cl::Event::waitForEvents(events);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        std::cout << "Batch inference (" << batch_size << " samples): "
                  << duration.count() / 1000.0 << " ms" << std::endl;
        std::cout << "Throughput: " << (batch_size * 1000000.0 / duration.count())
                  << " inferences/sec" << std::endl;

        return predictions;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <xclbin> <model.json> <test_data.json>" << std::endl;
        std::cout << std::endl;
        std::cout << "Example:" << std::endl;
        std::cout << "  " << argv[0] << " krnl.xclbin minirocket_model.json minirocket_model_test_data.json" << std::endl;
        return 1;
    }

    std::string xclbin_path = argv[1];
    std::string model_path = argv[2];
    std::string test_path = argv[3];

    try {
        // Single compute unit for 1:1 paper-faithful reference
        int num_cus = 1;  // 1 CU configured in config.cfg

        // Initialize FPGA
        MiniRocketFPGA fpga(xclbin_path, num_cus);

        // Load model
        fpga.load_model(model_path);

        // Load test data
        MiniRocketTestbenchLoader loader;
        std::vector<std::vector<float>> test_inputs, expected_outputs;

        if (!loader.load_test_data(test_path, test_inputs, expected_outputs)) {
            std::cerr << "Failed to load test data" << std::endl;
            return 1;
        }

        std::cout << "\nRunning inference on " << test_inputs.size() << " samples..." << std::endl;

        // Run predictions
        auto predictions = fpga.predict_batch(test_inputs);

        // Calculate accuracy
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); i++) {
            int expected = (int)expected_outputs[i][0];
            if (predictions[i] == expected) {
                correct++;
            }
        }

        float accuracy = (float)correct / predictions.size() * 100.0f;

        std::cout << "\n=== RESULTS ===" << std::endl;
        std::cout << "Accuracy: " << correct << "/" << predictions.size()
                  << " (" << std::fixed << std::setprecision(2) << accuracy << "%)" << std::endl;

        return (accuracy >= 90.0f) ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
