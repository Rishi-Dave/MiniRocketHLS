#include "host.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <cmath>
#include <chrono>

// Include JSON parsing capability (simplified)
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
        
        // Find the key
        size_t key_pos = content.find("\"" + key + "\"");
        if (key_pos == std::string::npos) {
            std::cerr << "Warning: Key " << key << " not found!" << std::endl;
            return result;
        }
        
        // Find array start
        size_t array_start = content.find("[", key_pos);
        size_t array_end = content.find("]", array_start);
        
        if (array_start == std::string::npos || array_end == std::string::npos) {
            std::cerr << "Warning: Array for " << key << " not found!" << std::endl;
            return result;
        }
        
        std::string array_content = content.substr(array_start + 1, array_end - array_start - 1);
        
        std::stringstream ss(array_content);
        std::string item;
        
        while (std::getline(ss, item, ',')) {
            // Clean whitespace
            item.erase(0, item.find_first_not_of(" \t\n\r\f\v"));
            item.erase(item.find_last_not_of(" \t\n\r\f\v") + 1);
            
            if (!item.empty()) {
                try {
                    result.push_back(std::stof(item));
                } catch (...) {
                    // Skip invalid values
                }
            }
        }
        
        return result;
    }

public:
    struct ModelParams {
        int num_features;
        int num_classes;
        int num_dilations;
        int time_series_length;
        
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
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open model file " << filename << std::endl;
            return false;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        
        // Parse basic parameters
        auto temp = parse_json_array(content, "num_features");
        params.num_features = temp.empty() ? 420 : (int)temp[0];
        
        temp = parse_json_array(content, "num_classes");
        params.num_classes = temp.empty() ? 4 : (int)temp[0];
        
        temp = parse_json_array(content, "num_dilations");
        params.num_dilations = temp.empty() ? 5 : (int)temp[0];
        
        // Parse arrays
        auto dil_floats = parse_json_array(content, "dilations");
        params.dilations.clear();
        for (float f : dil_floats) {
            params.dilations.push_back((int)f);
        }
        
        auto feat_per_dil_floats = parse_json_array(content, "num_features_per_dilation");
        params.num_features_per_dilation.clear();
        for (float f : feat_per_dil_floats) {
            params.num_features_per_dilation.push_back((int)f);
        }
        
        params.biases = parse_json_array(content, "biases");
        params.scaler_mean = parse_json_array(content, "scaler_mean");
        params.scaler_scale = parse_json_array(content, "scaler_scale");
        params.classifier_coef = parse_json_array(content, "classifier_coef");
        params.classifier_intercept = parse_json_array(content, "classifier_intercept");
        
        std::cout << "Model loaded: " << params.num_features << " features, " 
                  << params.num_classes << " classes, " 
                  << params.num_dilations << " dilations" << std::endl;
        
        return true;
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File> <Model JSON File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    std::string modelFile = argv[2];
    
    // Load model parameters
    SimpleMiniRocketLoader loader;
    SimpleMiniRocketLoader::ModelParams model_params;
    
    if (!loader.load_model(modelFile, model_params)) {
        return EXIT_FAILURE;
    }

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
            OCL_CHECK(err, krnl = cl::Kernel(program, "krnl_top", &err));  // Note: function name
            valid_device = true;
            break;
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    /*====================================================INIT INPUT/OUTPUT VECTORS===============================================================*/

    // Create test time series (sine wave pattern)
    int time_series_length = 128;
    std::vector<data_t, aligned_allocator<data_t>> time_series_input(time_series_length);
    std::vector<data_t, aligned_allocator<data_t>> prediction_output(model_params.num_classes);
    
    // Generate test sine wave
    for (int i = 0; i < time_series_length; i++) {
        time_series_input[i] = sin(2.0 * M_PI * i / 32.0) + 0.1 * ((i % 10) - 5) / 5.0;
    }
    
    std::cout << "Input time series (first 10 values): ";
    for (int i = 0; i < 10; i++) {
        std::cout << std::fixed << std::setprecision(3) << time_series_input[i] << " ";
    }
    std::cout << std::endl;

    // Prepare model parameter arrays for device
    std::vector<data_t, aligned_allocator<data_t>> coefficients(model_params.classifier_coef.begin(), model_params.classifier_coef.end());
    std::vector<data_t, aligned_allocator<data_t>> intercept(model_params.classifier_intercept.begin(), model_params.classifier_intercept.end());
    std::vector<data_t, aligned_allocator<data_t>> scaler_mean(model_params.scaler_mean.begin(), model_params.scaler_mean.end());
    std::vector<data_t, aligned_allocator<data_t>> scaler_scale(model_params.scaler_scale.begin(), model_params.scaler_scale.end());
    std::vector<data_t, aligned_allocator<data_t>> biases(model_params.biases.begin(), model_params.biases.end());

    std::vector<int_t, aligned_allocator<int_t>> dilations(model_params.dilations.begin(), model_params.dilations.end());
    std::vector<int_t, aligned_allocator<int_t>> num_features_per_dilation(model_params.num_features_per_dilation.begin(), model_params.num_features_per_dilation.end());

    /*====================================================Setting up kernel I/O===============================================================*/

    // Input buffers
    OCL_CHECK(err, cl::Buffer buffer_time_series(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
        sizeof(data_t) * time_series_length, time_series_input.data(), &err));
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

    // Output buffer
    OCL_CHECK(err, cl::Buffer buffer_predictions(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
        sizeof(data_t) * model_params.num_classes, prediction_output.data(), &err));

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
    OCL_CHECK(err, err = krnl.setArg(10, model_params.num_features));
    OCL_CHECK(err, err = krnl.setArg(11, model_params.num_classes));
    OCL_CHECK(err, err = krnl.setArg(12, model_params.num_dilations));

    /*====================================================KERNEL===============================================================*/

    // Timing variables
    double h2d_ms = 0, kernel_ms = 0, d2h_ms = 0;

    // Host to device data transfer
    std::cout << "HOST -> DEVICE" << std::endl;
    std::vector<cl::Memory> input_buffers = {
        buffer_time_series, buffer_coefficients, buffer_intercept,
        buffer_scaler_mean, buffer_scaler_scale, buffer_dilations,
        buffer_num_features_per_dilation, buffer_biases
    };
    auto h2d_start = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(input_buffers, 0));
    q.finish();
    auto h2d_end = std::chrono::high_resolution_clock::now();
    h2d_ms = std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();

    // Execute kernel
    std::cout << "STARTING KERNEL" << std::endl;
    auto kernel_start = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(krnl));
    q.finish();
    auto kernel_end = std::chrono::high_resolution_clock::now();
    kernel_ms = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();
    std::cout << "KERNEL FINISHED" << std::endl;

    // Device to host data transfer
    std::cout << "HOST <- DEVICE" << std::endl;
    auto d2h_start = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_predictions}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    auto d2h_end = std::chrono::high_resolution_clock::now();
    d2h_ms = std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();

    /*====================================================RESULTS===============================================================*/

    // Find predicted class
    int predicted_class = 0;
    float max_score = prediction_output[0];
    
    std::cout << "Classification scores: ";
    for (int i = 0; i < model_params.num_classes; i++) {
        std::cout << "Class" << i << "=" << std::fixed << std::setprecision(4) << prediction_output[i] << " ";
        if (prediction_output[i] > max_score) {
            max_score = prediction_output[i];
            predicted_class = i;
        }
    }
    std::cout << std::endl;
    
    std::cout << "Predicted class: " << predicted_class
              << " (score: " << std::fixed << std::setprecision(4) << max_score << ")" << std::endl;

    // Print timing results
    double total_ms = h2d_ms + kernel_ms + d2h_ms;
    std::cout << "\n========== TIMING RESULTS ==========" << std::endl;
    std::cout << "H2D transfer:      " << std::fixed << std::setprecision(3) << h2d_ms << " ms" << std::endl;
    std::cout << "Kernel execution:  " << std::fixed << std::setprecision(3) << kernel_ms << " ms" << std::endl;
    std::cout << "D2H transfer:      " << std::fixed << std::setprecision(3) << d2h_ms << " ms" << std::endl;
    std::cout << "Total latency:     " << std::fixed << std::setprecision(3) << total_ms << " ms" << std::endl;
    std::cout << "Throughput:        " << std::fixed << std::setprecision(1) << (1000.0 / total_ms) << " inferences/sec" << std::endl;
    std::cout << "====================================" << std::endl;

    std::cout << "\nMiniRocket FPGA inference complete!" << std::endl;

    return EXIT_SUCCESS;
}