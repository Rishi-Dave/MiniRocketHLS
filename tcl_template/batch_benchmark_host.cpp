#include "host.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <sstream>

// MiniRocket data types for host
typedef float data_t;
typedef int int_t;

// Constants
const int MAX_TIME_SERIES_LENGTH = 512;
const int MAX_FEATURES = 10000;
const int MAX_CLASSES = 4;
const int MAX_DILATIONS = 8;

class SimpleMiniRocketLoader {
private:
    std::vector<float> parse_json_array(const std::string& content, const std::string& key) {
        std::vector<float> result;

        size_t key_pos = content.find("\"" + key + "\"");
        if (key_pos == std::string::npos) return result;

        size_t array_start = content.find("[", key_pos);
        size_t array_end = content.find("]", array_start);

        if (array_start == std::string::npos || array_end == std::string::npos) return result;

        std::string array_content = content.substr(array_start + 1, array_end - array_start - 1);
        std::stringstream ss(array_content);
        std::string item;

        while (std::getline(ss, item, ',')) {
            item.erase(0, item.find_first_not_of(" \t\n\r\f\v"));
            item.erase(item.find_last_not_of(" \t\n\r\f\v") + 1);
            if (!item.empty()) {
                try { result.push_back(std::stof(item)); } catch (...) {}
            }
        }
        return result;
    }

    // Parse a scalar (non-array) JSON value
    int parse_json_scalar(const std::string& content, const std::string& key, int default_val) {
        size_t key_pos = content.find("\"" + key + "\"");
        if (key_pos == std::string::npos) return default_val;

        size_t colon_pos = content.find(":", key_pos);
        if (colon_pos == std::string::npos) return default_val;

        size_t value_start = colon_pos + 1;
        while (value_start < content.size() && (content[value_start] == ' ' || content[value_start] == '\t'))
            value_start++;

        size_t value_end = value_start;
        while (value_end < content.size() && (isdigit(content[value_end]) || content[value_end] == '-' || content[value_end] == '.'))
            value_end++;

        if (value_end > value_start) {
            try {
                return std::stoi(content.substr(value_start, value_end - value_start));
            } catch (...) {}
        }
        return default_val;
    }

public:
    struct ModelParams {
        int num_features;
        int num_classes;
        int num_dilations;
        std::vector<int> dilations;
        std::vector<int> num_features_per_dilation;
        std::vector<float> biases;
        std::vector<float> scaler_mean;
        std::vector<float> scaler_scale;
        std::vector<float> classifier_coef;
        std::vector<float> classifier_intercept;
    };

    bool load_model(const std::string& filename, ModelParams& params) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        // Use scalar parser for single values
        params.num_features = parse_json_scalar(content, "num_features", 420);
        params.num_classes = parse_json_scalar(content, "num_classes", 4);
        params.num_dilations = parse_json_scalar(content, "num_dilations", 5);

        auto dil_floats = parse_json_array(content, "dilations");
        for (float f : dil_floats) params.dilations.push_back((int)f);

        auto feat_per_dil_floats = parse_json_array(content, "num_features_per_dilation");
        for (float f : feat_per_dil_floats) params.num_features_per_dilation.push_back((int)f);

        params.biases = parse_json_array(content, "biases");
        params.scaler_mean = parse_json_array(content, "scaler_mean");
        params.scaler_scale = parse_json_array(content, "scaler_scale");
        params.classifier_coef = parse_json_array(content, "classifier_coef");
        params.classifier_intercept = parse_json_array(content, "classifier_intercept");

        return true;
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File> <Model JSON File> [batch_size=100]" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    std::string modelFile = argv[2];
    int batch_size = (argc > 3) ? std::stoi(argv[3]) : 100;

    std::cout << "\n========== BATCH INFERENCE BENCHMARK ==========" << std::endl;
    std::cout << "Batch size: " << batch_size << " samples" << std::endl;

    // Load model
    SimpleMiniRocketLoader loader;
    SimpleMiniRocketLoader::ModelParams model_params;
    if (!loader.load_model(modelFile, model_params)) {
        std::cerr << "Error: Cannot load model" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Model: " << model_params.num_features << " features, "
              << model_params.num_classes << " classes" << std::endl;

    // OpenCL setup
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
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err == CL_SUCCESS) {
            OCL_CHECK(err, krnl = cl::Kernel(program, "krnl_top", &err));
            valid_device = true;
            std::cout << "FPGA programmed: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            break;
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program FPGA!" << std::endl;
        return EXIT_FAILURE;
    }

    // Create batch of test time series with different patterns
    int time_series_length = 128;
    std::vector<std::vector<data_t, aligned_allocator<data_t>>> batch_inputs(batch_size);
    std::vector<std::vector<data_t, aligned_allocator<data_t>>> batch_outputs(batch_size);

    for (int b = 0; b < batch_size; b++) {
        batch_inputs[b].resize(time_series_length);
        batch_outputs[b].resize(model_params.num_classes);

        // Generate different patterns for each sample
        double freq = 1.0 + (b % 10) * 0.5;  // Different frequencies
        double phase = (b * 0.1);
        for (int i = 0; i < time_series_length; i++) {
            batch_inputs[b][i] = sin(2.0 * M_PI * freq * i / 32.0 + phase)
                                + 0.1 * ((i + b) % 10 - 5) / 5.0;
        }
    }

    // Prepare model parameter arrays (these stay constant for all batches)
    std::vector<data_t, aligned_allocator<data_t>> coefficients(model_params.classifier_coef.begin(), model_params.classifier_coef.end());
    std::vector<data_t, aligned_allocator<data_t>> intercept(model_params.classifier_intercept.begin(), model_params.classifier_intercept.end());
    std::vector<data_t, aligned_allocator<data_t>> scaler_mean(model_params.scaler_mean.begin(), model_params.scaler_mean.end());
    std::vector<data_t, aligned_allocator<data_t>> scaler_scale(model_params.scaler_scale.begin(), model_params.scaler_scale.end());
    std::vector<data_t, aligned_allocator<data_t>> biases(model_params.biases.begin(), model_params.biases.end());
    std::vector<int_t, aligned_allocator<int_t>> dilations(model_params.dilations.begin(), model_params.dilations.end());
    std::vector<int_t, aligned_allocator<int_t>> num_features_per_dilation(model_params.num_features_per_dilation.begin(), model_params.num_features_per_dilation.end());

    // Create buffers for model parameters (transferred once)
    OCL_CHECK(err, cl::Buffer buffer_coefficients(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        sizeof(data_t) * coefficients.size(), coefficients.data(), &err));
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

    // Set constant kernel arguments (model parameters)
    OCL_CHECK(err, err = krnl.setArg(2, buffer_coefficients));
    OCL_CHECK(err, err = krnl.setArg(3, buffer_intercept));
    OCL_CHECK(err, err = krnl.setArg(4, buffer_scaler_mean));
    OCL_CHECK(err, err = krnl.setArg(5, buffer_scaler_scale));
    OCL_CHECK(err, err = krnl.setArg(6, buffer_dilations));
    OCL_CHECK(err, err = krnl.setArg(7, buffer_num_features_per_dilation));
    OCL_CHECK(err, err = krnl.setArg(8, buffer_biases));
    OCL_CHECK(err, err = krnl.setArg(9, time_series_length));
    OCL_CHECK(err, err = krnl.setArg(10, model_params.num_features));
    OCL_CHECK(err, err = krnl.setArg(11, model_params.num_classes));
    OCL_CHECK(err, err = krnl.setArg(12, model_params.num_dilations));

    // Transfer model parameters to device ONCE
    std::cout << "\nTransferring model parameters to FPGA..." << std::endl;
    std::vector<cl::Memory> model_buffers = {
        buffer_coefficients, buffer_intercept, buffer_scaler_mean,
        buffer_scaler_scale, buffer_dilations, buffer_num_features_per_dilation, buffer_biases
    };

    auto model_h2d_start = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(model_buffers, 0));
    q.finish();
    auto model_h2d_end = std::chrono::high_resolution_clock::now();
    double model_h2d_ms = std::chrono::duration<double, std::milli>(model_h2d_end - model_h2d_start).count();
    std::cout << "Model transfer time: " << std::fixed << std::setprecision(3) << model_h2d_ms << " ms" << std::endl;

    // Timing accumulators
    double total_input_h2d_ms = 0;
    double total_kernel_ms = 0;
    double total_output_d2h_ms = 0;

    std::vector<int> predictions(batch_size);

    std::cout << "\nRunning batch inference..." << std::endl;
    auto batch_start = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < batch_size; b++) {
        // Create buffers for this sample
        cl::Buffer buffer_time_series(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            sizeof(data_t) * time_series_length, batch_inputs[b].data(), &err);
        cl::Buffer buffer_predictions(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
            sizeof(data_t) * model_params.num_classes, batch_outputs[b].data(), &err);

        // Set input/output arguments
        OCL_CHECK(err, err = krnl.setArg(0, buffer_time_series));
        OCL_CHECK(err, err = krnl.setArg(1, buffer_predictions));

        // Transfer input
        auto h2d_start = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_time_series}, 0));
        q.finish();
        auto h2d_end = std::chrono::high_resolution_clock::now();
        total_input_h2d_ms += std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();

        // Run kernel
        auto kernel_start = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueTask(krnl));
        q.finish();
        auto kernel_end = std::chrono::high_resolution_clock::now();
        total_kernel_ms += std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();

        // Transfer output
        auto d2h_start = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_predictions}, CL_MIGRATE_MEM_OBJECT_HOST));
        q.finish();
        auto d2h_end = std::chrono::high_resolution_clock::now();
        total_output_d2h_ms += std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();

        // Find predicted class
        int pred = 0;
        float max_score = batch_outputs[b][0];
        for (int c = 1; c < model_params.num_classes; c++) {
            if (batch_outputs[b][c] > max_score) {
                max_score = batch_outputs[b][c];
                pred = c;
            }
        }
        predictions[b] = pred;

        if ((b + 1) % 25 == 0 || b == batch_size - 1) {
            std::cout << "  Completed " << (b + 1) << "/" << batch_size << " samples" << std::endl;
        }
    }

    auto batch_end = std::chrono::high_resolution_clock::now();
    double total_batch_ms = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();

    // Calculate statistics
    double avg_input_h2d = total_input_h2d_ms / batch_size;
    double avg_kernel = total_kernel_ms / batch_size;
    double avg_output_d2h = total_output_d2h_ms / batch_size;
    double avg_per_sample = total_batch_ms / batch_size;
    double throughput = 1000.0 * batch_size / total_batch_ms;

    // Count class distribution
    std::vector<int> class_counts(model_params.num_classes, 0);
    for (int p : predictions) class_counts[p]++;

    std::cout << "\n============== BATCH TIMING RESULTS ==============" << std::endl;
    std::cout << "Batch size:             " << batch_size << " samples" << std::endl;
    std::cout << "Model transfer (once):  " << std::fixed << std::setprecision(3) << model_h2d_ms << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "Per-sample averages:" << std::endl;
    std::cout << "  Input H2D transfer:   " << std::fixed << std::setprecision(3) << avg_input_h2d << " ms" << std::endl;
    std::cout << "  Kernel execution:     " << std::fixed << std::setprecision(3) << avg_kernel << " ms" << std::endl;
    std::cout << "  Output D2H transfer:  " << std::fixed << std::setprecision(3) << avg_output_d2h << " ms" << std::endl;
    std::cout << "  Total per sample:     " << std::fixed << std::setprecision(3) << avg_per_sample << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "Batch totals:" << std::endl;
    std::cout << "  Total input H2D:      " << std::fixed << std::setprecision(3) << total_input_h2d_ms << " ms" << std::endl;
    std::cout << "  Total kernel time:    " << std::fixed << std::setprecision(3) << total_kernel_ms << " ms" << std::endl;
    std::cout << "  Total output D2H:     " << std::fixed << std::setprecision(3) << total_output_d2h_ms << " ms" << std::endl;
    std::cout << "  Total batch time:     " << std::fixed << std::setprecision(3) << total_batch_ms << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "THROUGHPUT:             " << std::fixed << std::setprecision(1) << throughput << " inferences/sec" << std::endl;
    std::cout << "Amortized model H2D:    " << std::fixed << std::setprecision(3) << (model_h2d_ms / batch_size) << " ms/sample" << std::endl;
    std::cout << std::endl;
    std::cout << "Class distribution:" << std::endl;
    for (int c = 0; c < model_params.num_classes; c++) {
        std::cout << "  Class " << c << ": " << class_counts[c] << " ("
                  << std::fixed << std::setprecision(1) << (100.0 * class_counts[c] / batch_size) << "%)" << std::endl;
    }
    std::cout << "==================================================" << std::endl;

    return EXIT_SUCCESS;
}
