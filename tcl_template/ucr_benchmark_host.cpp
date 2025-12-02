#include "host.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <sstream>

typedef float data_t;
typedef int int_t;

const int MAX_TIME_SERIES_LENGTH = 512;
const int MAX_FEATURES = 10000;
const int MAX_CLASSES = 4;
const int MAX_DILATIONS = 8;

// Simple JSON parsing for model parameters
class ModelLoader {
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

    static int parse_scalar(const std::string& content, const std::string& key, int default_val) {
        size_t key_pos = content.find("\"" + key + "\"");
        if (key_pos == std::string::npos) return default_val;
        size_t colon_pos = content.find(":", key_pos);
        if (colon_pos == std::string::npos) return default_val;
        size_t value_start = colon_pos + 1;
        while (value_start < content.size() && (content[value_start] == ' ' || content[value_start] == '\t'))
            value_start++;
        size_t value_end = value_start;
        while (value_end < content.size() && (isdigit(content[value_end]) || content[value_end] == '-'))
            value_end++;
        if (value_end > value_start) {
            try { return std::stoi(content.substr(value_start, value_end - value_start)); }
            catch (...) {}
        }
        return default_val;
    }

    static std::vector<float> parse_array(const std::string& content, const std::string& key) {
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
                try { result.push_back(std::stof(item)); }
                catch (...) {}
            }
        }
        return result;
    }

    static bool load_model(const std::string& filename, ModelParams& params) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        params.num_features = parse_scalar(content, "num_features", 420);
        params.num_classes = parse_scalar(content, "num_classes", 4);
        params.num_dilations = parse_scalar(content, "num_dilations", 5);
        auto dil_floats = parse_array(content, "dilations");
        for (float f : dil_floats) params.dilations.push_back((int)f);
        auto feat_per_dil_floats = parse_array(content, "num_features_per_dilation");
        for (float f : feat_per_dil_floats) params.num_features_per_dilation.push_back((int)f);
        params.biases = parse_array(content, "biases");
        params.scaler_mean = parse_array(content, "scaler_mean");
        params.scaler_scale = parse_array(content, "scaler_scale");
        params.classifier_coef = parse_array(content, "classifier_coef");
        params.classifier_intercept = parse_array(content, "classifier_intercept");
        return true;
    }
};

// UCR dataset loader
struct UCRDataset {
    std::string name;
    int num_samples;
    int time_series_length;
    int num_classes;
    std::vector<std::vector<float>> samples;
    std::vector<int> labels;

    static bool load(const std::string& filename, UCRDataset& dataset) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open dataset file: " << filename << std::endl;
            return false;
        }
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        dataset.name = ModelLoader::parse_scalar(content, "dataset_name", 0) == 0 ? "unknown" : "dataset";
        dataset.num_samples = ModelLoader::parse_scalar(content, "num_samples", 0);
        dataset.time_series_length = ModelLoader::parse_scalar(content, "time_series_length", 0);
        dataset.num_classes = ModelLoader::parse_scalar(content, "num_classes", 0);

        // Parse labels
        auto labels_float = ModelLoader::parse_array(content, "labels");
        for (float f : labels_float) dataset.labels.push_back((int)f);

        // Parse samples (nested array - need special handling)
        size_t samples_pos = content.find("\"samples\"");
        if (samples_pos != std::string::npos) {
            size_t outer_start = content.find("[[", samples_pos);
            if (outer_start != std::string::npos) {
                // Find each inner array
                size_t pos = outer_start + 1;
                while (pos < content.size()) {
                    size_t inner_start = content.find("[", pos);
                    if (inner_start == std::string::npos || inner_start > content.find("]]", outer_start)) break;
                    size_t inner_end = content.find("]", inner_start);
                    if (inner_end == std::string::npos) break;

                    std::string inner = content.substr(inner_start + 1, inner_end - inner_start - 1);
                    std::vector<float> sample;
                    std::stringstream ss(inner);
                    std::string item;
                    while (std::getline(ss, item, ',')) {
                        item.erase(0, item.find_first_not_of(" \t\n\r\f\v"));
                        item.erase(item.find_last_not_of(" \t\n\r\f\v") + 1);
                        if (!item.empty()) {
                            try { sample.push_back(std::stof(item)); }
                            catch (...) {}
                        }
                    }
                    if (!sample.empty()) dataset.samples.push_back(sample);
                    pos = inner_end + 1;
                }
            }
        }

        std::cout << "Loaded dataset: " << dataset.num_samples << " samples, "
                  << dataset.time_series_length << " length, "
                  << dataset.num_classes << " classes" << std::endl;
        return dataset.samples.size() > 0;
    }
};

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN> <Model JSON> <UCR Dataset JSON> [max_samples]" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    std::string modelFile = argv[2];
    std::string datasetFile = argv[3];
    int max_samples = (argc > 4) ? std::stoi(argv[4]) : -1;

    // Load model
    ModelLoader::ModelParams model_params;
    if (!ModelLoader::load_model(modelFile, model_params)) {
        std::cerr << "Error loading model" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Model: " << model_params.num_features << " features, "
              << model_params.num_classes << " classes" << std::endl;

    // Load UCR dataset
    UCRDataset dataset;
    if (!UCRDataset::load(datasetFile, dataset)) {
        std::cerr << "Error loading dataset" << std::endl;
        return EXIT_FAILURE;
    }

    int num_samples = (max_samples > 0 && max_samples < dataset.num_samples) ? max_samples : dataset.num_samples;

    // OpenCL setup
    cl_int err;
    cl::Context context;
    cl::Kernel krnl;
    cl::CommandQueue q;

    auto devices = get_xil_devices();
    auto fileBuf = read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, 0, &err));
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err == CL_SUCCESS) {
            OCL_CHECK(err, krnl = cl::Kernel(program, "krnl_top", &err));
            std::cout << "FPGA programmed successfully" << std::endl;
            break;
        }
    }

    // Prepare model buffers (transferred once)
    std::vector<data_t, aligned_allocator<data_t>> coefficients(model_params.classifier_coef.begin(), model_params.classifier_coef.end());
    std::vector<data_t, aligned_allocator<data_t>> intercept(model_params.classifier_intercept.begin(), model_params.classifier_intercept.end());
    std::vector<data_t, aligned_allocator<data_t>> scaler_mean(model_params.scaler_mean.begin(), model_params.scaler_mean.end());
    std::vector<data_t, aligned_allocator<data_t>> scaler_scale(model_params.scaler_scale.begin(), model_params.scaler_scale.end());
    std::vector<data_t, aligned_allocator<data_t>> biases(model_params.biases.begin(), model_params.biases.end());
    std::vector<int_t, aligned_allocator<int_t>> dilations(model_params.dilations.begin(), model_params.dilations.end());
    std::vector<int_t, aligned_allocator<int_t>> num_features_per_dilation(model_params.num_features_per_dilation.begin(), model_params.num_features_per_dilation.end());

    // Create and transfer model buffers
    cl::Buffer buffer_coefficients(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * coefficients.size(), coefficients.data(), &err);
    cl::Buffer buffer_intercept(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * intercept.size(), intercept.data(), &err);
    cl::Buffer buffer_scaler_mean(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * scaler_mean.size(), scaler_mean.data(), &err);
    cl::Buffer buffer_scaler_scale(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * scaler_scale.size(), scaler_scale.data(), &err);
    cl::Buffer buffer_dilations(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int_t) * dilations.size(), dilations.data(), &err);
    cl::Buffer buffer_num_features_per_dilation(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int_t) * num_features_per_dilation.size(), num_features_per_dilation.data(), &err);
    cl::Buffer buffer_biases(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * biases.size(), biases.data(), &err);

    // Set constant kernel arguments
    krnl.setArg(2, buffer_coefficients);
    krnl.setArg(3, buffer_intercept);
    krnl.setArg(4, buffer_scaler_mean);
    krnl.setArg(5, buffer_scaler_scale);
    krnl.setArg(6, buffer_dilations);
    krnl.setArg(7, buffer_num_features_per_dilation);
    krnl.setArg(8, buffer_biases);
    krnl.setArg(10, model_params.num_features);
    krnl.setArg(11, model_params.num_classes);
    krnl.setArg(12, model_params.num_dilations);

    // Transfer model to device
    std::vector<cl::Memory> model_buffers = {buffer_coefficients, buffer_intercept, buffer_scaler_mean, buffer_scaler_scale, buffer_dilations, buffer_num_features_per_dilation, buffer_biases};
    q.enqueueMigrateMemObjects(model_buffers, 0);
    q.finish();

    // Run inference on dataset
    std::cout << "\nRunning FPGA inference on " << num_samples << " samples..." << std::endl;
    std::vector<int> predictions(num_samples);
    int correct = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < num_samples; s++) {
        // Prepare input
        int ts_len = dataset.samples[s].size();
        std::vector<data_t, aligned_allocator<data_t>> input(dataset.samples[s].begin(), dataset.samples[s].end());
        std::vector<data_t, aligned_allocator<data_t>> output(model_params.num_classes);

        cl::Buffer buffer_input(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * ts_len, input.data(), &err);
        cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(data_t) * model_params.num_classes, output.data(), &err);

        krnl.setArg(0, buffer_input);
        krnl.setArg(1, buffer_output);
        krnl.setArg(9, ts_len);

        q.enqueueMigrateMemObjects({buffer_input}, 0);
        q.enqueueTask(krnl);
        q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST);
        q.finish();

        // Find prediction
        int pred = 0;
        float max_score = output[0];
        for (int c = 1; c < model_params.num_classes; c++) {
            if (output[c] > max_score) {
                max_score = output[c];
                pred = c;
            }
        }
        predictions[s] = pred;
        if (pred == dataset.labels[s]) correct++;

        if ((s + 1) % 100 == 0 || s == num_samples - 1) {
            std::cout << "  " << (s + 1) << "/" << num_samples << " processed..." << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double accuracy = 100.0 * correct / num_samples;

    std::cout << "\n================ RESULTS ================" << std::endl;
    std::cout << "Samples:     " << num_samples << std::endl;
    std::cout << "Correct:     " << correct << std::endl;
    std::cout << "Accuracy:    " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    std::cout << "Total time:  " << std::fixed << std::setprecision(1) << total_ms << " ms" << std::endl;
    std::cout << "Per sample:  " << std::fixed << std::setprecision(3) << (total_ms / num_samples) << " ms" << std::endl;
    std::cout << "Throughput:  " << std::fixed << std::setprecision(1) << (1000.0 * num_samples / total_ms) << " inf/sec" << std::endl;
    std::cout << "=========================================" << std::endl;

    return EXIT_SUCCESS;
}
