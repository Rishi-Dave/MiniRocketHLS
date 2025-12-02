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

const int NUM_CU = 4;  // Number of compute units

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

        auto labels_float = ModelLoader::parse_array(content, "labels");
        for (float f : labels_float) dataset.labels.push_back((int)f);

        size_t samples_pos = content.find("\"samples\"");
        if (samples_pos != std::string::npos) {
            size_t outer_start = content.find("[[", samples_pos);
            if (outer_start != std::string::npos) {
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

        std::cout << "Loaded dataset: " << dataset.samples.size() << " samples, "
                  << dataset.time_series_length << " length" << std::endl;
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

    int num_samples = (max_samples > 0 && max_samples < (int)dataset.samples.size()) ? max_samples : dataset.samples.size();

    // OpenCL setup
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;

    auto devices = get_xil_devices();
    auto fileBuf = read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

    cl::Device device;
    cl::Program program;
    for (unsigned int i = 0; i < devices.size(); i++) {
        device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
        program = cl::Program(context, {device}, bins, nullptr, &err);
        if (err == CL_SUCCESS) {
            std::cout << "FPGA programmed successfully" << std::endl;
            break;
        }
    }

    // Create kernels for each compute unit
    std::vector<cl::Kernel> kernels(NUM_CU);
    for (int cu = 0; cu < NUM_CU; cu++) {
        std::string cu_name = "krnl_top:{krnl_top_" + std::to_string(cu + 1) + "}";
        OCL_CHECK(err, kernels[cu] = cl::Kernel(program, cu_name.c_str(), &err));
        std::cout << "Created kernel for CU " << (cu + 1) << std::endl;
    }

    // Prepare model buffers (one set per CU for independent HBM banks)
    std::vector<std::vector<data_t, aligned_allocator<data_t>>> coefficients(NUM_CU);
    std::vector<std::vector<data_t, aligned_allocator<data_t>>> intercept(NUM_CU);
    std::vector<std::vector<data_t, aligned_allocator<data_t>>> scaler_mean(NUM_CU);
    std::vector<std::vector<data_t, aligned_allocator<data_t>>> scaler_scale(NUM_CU);
    std::vector<std::vector<data_t, aligned_allocator<data_t>>> biases(NUM_CU);
    std::vector<std::vector<int_t, aligned_allocator<int_t>>> dilations(NUM_CU);
    std::vector<std::vector<int_t, aligned_allocator<int_t>>> num_features_per_dilation(NUM_CU);

    std::vector<cl::Buffer> buffer_coefficients(NUM_CU);
    std::vector<cl::Buffer> buffer_intercept(NUM_CU);
    std::vector<cl::Buffer> buffer_scaler_mean(NUM_CU);
    std::vector<cl::Buffer> buffer_scaler_scale(NUM_CU);
    std::vector<cl::Buffer> buffer_dilations(NUM_CU);
    std::vector<cl::Buffer> buffer_num_features_per_dilation(NUM_CU);
    std::vector<cl::Buffer> buffer_biases(NUM_CU);

    for (int cu = 0; cu < NUM_CU; cu++) {
        coefficients[cu] = std::vector<data_t, aligned_allocator<data_t>>(model_params.classifier_coef.begin(), model_params.classifier_coef.end());
        intercept[cu] = std::vector<data_t, aligned_allocator<data_t>>(model_params.classifier_intercept.begin(), model_params.classifier_intercept.end());
        scaler_mean[cu] = std::vector<data_t, aligned_allocator<data_t>>(model_params.scaler_mean.begin(), model_params.scaler_mean.end());
        scaler_scale[cu] = std::vector<data_t, aligned_allocator<data_t>>(model_params.scaler_scale.begin(), model_params.scaler_scale.end());
        biases[cu] = std::vector<data_t, aligned_allocator<data_t>>(model_params.biases.begin(), model_params.biases.end());
        dilations[cu] = std::vector<int_t, aligned_allocator<int_t>>(model_params.dilations.begin(), model_params.dilations.end());
        num_features_per_dilation[cu] = std::vector<int_t, aligned_allocator<int_t>>(model_params.num_features_per_dilation.begin(), model_params.num_features_per_dilation.end());

        buffer_coefficients[cu] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * coefficients[cu].size(), coefficients[cu].data(), &err);
        buffer_intercept[cu] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * intercept[cu].size(), intercept[cu].data(), &err);
        buffer_scaler_mean[cu] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * scaler_mean[cu].size(), scaler_mean[cu].data(), &err);
        buffer_scaler_scale[cu] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * scaler_scale[cu].size(), scaler_scale[cu].data(), &err);
        buffer_dilations[cu] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int_t) * dilations[cu].size(), dilations[cu].data(), &err);
        buffer_num_features_per_dilation[cu] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int_t) * num_features_per_dilation[cu].size(), num_features_per_dilation[cu].data(), &err);
        buffer_biases[cu] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * biases[cu].size(), biases[cu].data(), &err);

        // Set constant kernel arguments
        kernels[cu].setArg(2, buffer_coefficients[cu]);
        kernels[cu].setArg(3, buffer_intercept[cu]);
        kernels[cu].setArg(4, buffer_scaler_mean[cu]);
        kernels[cu].setArg(5, buffer_scaler_scale[cu]);
        kernels[cu].setArg(6, buffer_dilations[cu]);
        kernels[cu].setArg(7, buffer_num_features_per_dilation[cu]);
        kernels[cu].setArg(8, buffer_biases[cu]);
        kernels[cu].setArg(10, model_params.num_features);
        kernels[cu].setArg(11, model_params.num_classes);
        kernels[cu].setArg(12, model_params.num_dilations);

        // Transfer model to device
        std::vector<cl::Memory> model_buffers = {buffer_coefficients[cu], buffer_intercept[cu], buffer_scaler_mean[cu],
                                                  buffer_scaler_scale[cu], buffer_dilations[cu], buffer_num_features_per_dilation[cu], buffer_biases[cu]};
        q.enqueueMigrateMemObjects(model_buffers, 0);
    }
    q.finish();
    std::cout << "Model parameters loaded to all " << NUM_CU << " compute units" << std::endl;

    // Run inference on dataset using all CUs in parallel
    std::cout << "\nRunning FPGA inference on " << num_samples << " samples using " << NUM_CU << " CUs..." << std::endl;
    std::vector<int> predictions(num_samples);
    int correct = 0;

    auto start = std::chrono::high_resolution_clock::now();

    // Process samples in batches of NUM_CU
    for (int batch_start = 0; batch_start < num_samples; batch_start += NUM_CU) {
        int batch_size = std::min(NUM_CU, num_samples - batch_start);

        // Prepare buffers for this batch
        std::vector<std::vector<data_t, aligned_allocator<data_t>>> inputs(batch_size);
        std::vector<std::vector<data_t, aligned_allocator<data_t>>> outputs(batch_size);
        std::vector<cl::Buffer> buffer_inputs(batch_size);
        std::vector<cl::Buffer> buffer_outputs(batch_size);
        std::vector<cl::Event> kernel_events(batch_size);
        std::vector<cl::Event> read_events(batch_size);

        // Launch all CUs in parallel
        for (int i = 0; i < batch_size; i++) {
            int sample_idx = batch_start + i;
            int ts_len = dataset.samples[sample_idx].size();

            inputs[i] = std::vector<data_t, aligned_allocator<data_t>>(dataset.samples[sample_idx].begin(), dataset.samples[sample_idx].end());
            outputs[i] = std::vector<data_t, aligned_allocator<data_t>>(model_params.num_classes);

            buffer_inputs[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * ts_len, inputs[i].data(), &err);
            buffer_outputs[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(data_t) * model_params.num_classes, outputs[i].data(), &err);

            kernels[i].setArg(0, buffer_inputs[i]);
            kernels[i].setArg(1, buffer_outputs[i]);
            kernels[i].setArg(9, ts_len);

            // Enqueue input transfer and kernel execution
            q.enqueueMigrateMemObjects({buffer_inputs[i]}, 0);
            q.enqueueTask(kernels[i], nullptr, &kernel_events[i]);
        }

        // Wait for all kernels and read outputs
        for (int i = 0; i < batch_size; i++) {
            std::vector<cl::Event> wait_list = {kernel_events[i]};
            q.enqueueMigrateMemObjects({buffer_outputs[i]}, CL_MIGRATE_MEM_OBJECT_HOST, &wait_list, &read_events[i]);
        }

        // Wait for all reads to complete
        cl::Event::waitForEvents(read_events);

        // Process predictions
        for (int i = 0; i < batch_size; i++) {
            int sample_idx = batch_start + i;
            int pred = 0;
            float max_score = outputs[i][0];
            for (int c = 1; c < model_params.num_classes; c++) {
                if (outputs[i][c] > max_score) {
                    max_score = outputs[i][c];
                    pred = c;
                }
            }
            predictions[sample_idx] = pred;
            if (pred == dataset.labels[sample_idx]) correct++;
        }

        if ((batch_start + batch_size) % 100 == 0 || batch_start + batch_size >= num_samples) {
            std::cout << "  " << (batch_start + batch_size) << "/" << num_samples << " processed..." << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double accuracy = 100.0 * correct / num_samples;

    std::cout << "\n================ RESULTS (" << NUM_CU << " CUs) ================" << std::endl;
    std::cout << "Samples:     " << num_samples << std::endl;
    std::cout << "Correct:     " << correct << std::endl;
    std::cout << "Accuracy:    " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    std::cout << "Total time:  " << std::fixed << std::setprecision(1) << total_ms << " ms" << std::endl;
    std::cout << "Per sample:  " << std::fixed << std::setprecision(3) << (total_ms / num_samples) << " ms" << std::endl;
    std::cout << "Throughput:  " << std::fixed << std::setprecision(1) << (1000.0 * num_samples / total_ms) << " inf/sec" << std::endl;
    std::cout << "=================================================" << std::endl;

    return EXIT_SUCCESS;
}
