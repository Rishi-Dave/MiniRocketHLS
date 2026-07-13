#include "../include/hydra_host.h"
#include "../include/hydra_loader_v2.h"
#include <iomanip>
#include <numeric>
#include <algorithm>

std::vector<char> load_file_to_memory(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "ERROR: Cannot read file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    return buffer;
}

cl::Platform find_xilinx_platform() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cl::Platform xilinx_platform;
    bool found = false;

    for (auto& platform : platforms) {
        std::string platform_name = platform.getInfo<CL_PLATFORM_NAME>();
        if (platform_name == "Xilinx") {
            xilinx_platform = platform;
            found = true;
            std::cout << "Found Platform: " << platform_name << std::endl;
            break;
        }
    }

    if (!found) {
        std::cerr << "ERROR: Failed to find Xilinx platform" << std::endl;
        exit(EXIT_FAILURE);
    }

    return xilinx_platform;
}

void print_device_info(const cl::Device& device) {
    std::cout << "\nDevice Info:" << std::endl;
    std::cout << "  Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "  Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
    std::cout << "  Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
    std::cout << "  Global Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024 * 1024) << " GB" << std::endl;
    std::cout << std::endl;
}

bool has_hbm(const cl::Device& device) {
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    return (device_name.find("u280") != std::string::npos ||
            device_name.find("u50") != std::string::npos ||
            device_name.find("u55") != std::string::npos);
}

int main(int argc, char* argv[]) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  HYDRA FPGA Host Application" << std::endl;
    std::cout << "========================================\n" << std::endl;

    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <xclbin> <model.json> <test.json>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string xclbin_file = argv[1];
    std::string model_file = argv[2];
    std::string test_file = argv[3];

    std::cout << "Configuration:" << std::endl;
    std::cout << "  XClbin: " << xclbin_file << std::endl;
    std::cout << "  Model: " << model_file << std::endl;
    std::cout << "  Test data: " << test_file << std::endl;
    std::cout << std::endl;

    // Load model and test data using new JSON loader
    HydraJSONLoader loader;

    // Allocate HLS-style arrays for model parameters
    data_t kernel_weights[NUM_KERNELS][KERNEL_SIZE];
    data_t biases[NUM_KERNELS];
    int_t dilations[NUM_KERNELS];
    int_t group_assignments[NUM_KERNELS];
    data_t scaler_mean[MAX_FEATURES];
    data_t scaler_scale[MAX_FEATURES];
    data_t coefficients[MAX_CLASSES][MAX_FEATURES];
    data_t intercept[MAX_CLASSES];

    int_t num_kernels, num_groups, num_features, num_classes, time_series_length;

    if (!loader.load_hydra_model(
        model_file,
        kernel_weights,
        biases,
        dilations,
        group_assignments,
        scaler_mean,
        scaler_scale,
        coefficients,
        intercept,
        num_kernels,
        num_groups,
        num_features,
        num_classes,
        time_series_length
    )) {
        std::cerr << "ERROR: Failed to load model" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\n=== HYDRA Model Info ===" << std::endl;
    std::cout << "Kernels: " << num_kernels << std::endl;
    std::cout << "Groups: " << num_groups << std::endl;
    std::cout << "Features: " << num_features << std::endl;
    std::cout << "Classes: " << num_classes << std::endl;
    std::cout << "Time series length: " << time_series_length << std::endl;
    std::cout << "========================\n" << std::endl;

    // Sort kernels by dilation so each UNROLL batch has uniform dilation
    // Required by v2 kernel's shared sliding window (BATCH_LOOP structure)
    // Set HYDRA_NO_SORT=1 to skip sorting for old kernel builds (u32_fixed, u16_fixed)
    bool do_sort = (getenv("HYDRA_NO_SORT") == nullptr);

    if (do_sort) {
        std::vector<int> perm(num_kernels);
        std::iota(perm.begin(), perm.end(), 0);
        std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) {
            return dilations[a] < dilations[b];
        });

        // Apply permutation to kernel data
        {
            data_t tmp_weights[NUM_KERNELS][KERNEL_SIZE];
            data_t tmp_biases[NUM_KERNELS];
            int_t tmp_dilations[NUM_KERNELS];
            for (int i = 0; i < num_kernels; i++) {
                for (int w = 0; w < KERNEL_SIZE; w++)
                    tmp_weights[i][w] = kernel_weights[perm[i]][w];
                tmp_biases[i] = biases[perm[i]];
                tmp_dilations[i] = dilations[perm[i]];
            }
            memcpy(kernel_weights, tmp_weights, sizeof(tmp_weights));
            memcpy(biases, tmp_biases, sizeof(tmp_biases));
            memcpy(dilations, tmp_dilations, sizeof(tmp_dilations));
        }

        // Build feature permutation (2 features per kernel: max at 2k, mean at 2k+1)
        std::vector<int> feat_perm(num_features);
        for (int i = 0; i < num_kernels; i++) {
            feat_perm[2 * i]     = 2 * perm[i];
            feat_perm[2 * i + 1] = 2 * perm[i] + 1;
        }

        // Reorder scaler arrays to match new feature order
        {
            data_t tmp_mean[MAX_FEATURES], tmp_scale[MAX_FEATURES];
            for (int i = 0; i < num_features; i++) {
                tmp_mean[i]  = scaler_mean[feat_perm[i]];
                tmp_scale[i] = scaler_scale[feat_perm[i]];
            }
            memcpy(scaler_mean, tmp_mean, num_features * sizeof(data_t));
            memcpy(scaler_scale, tmp_scale, num_features * sizeof(data_t));
        }

        // Reorder coefficient columns to match new feature order
        {
            data_t tmp_coef[MAX_CLASSES][MAX_FEATURES];
            for (int c = 0; c < num_classes; c++)
                for (int f = 0; f < num_features; f++)
                    tmp_coef[c][f] = coefficients[c][feat_perm[f]];
            memcpy(coefficients, tmp_coef, sizeof(tmp_coef));
        }

        std::cout << "Kernels sorted by dilation for FPGA batch processing" << std::endl;
    } else {
        std::cout << "Dilation sorting DISABLED (HYDRA_NO_SORT=1)" << std::endl;
    }

    std::cout << "  Dilation distribution: ";
    for (int d : {1, 2, 4, 8}) {
        int cnt = std::count(dilations, dilations + num_kernels, d);
        std::cout << "d=" << d << ":" << cnt << " ";
    }
    std::cout << std::endl;

    // Load test data
    std::vector<std::vector<float>> test_inputs;
    std::vector<int> test_labels;
    int test_num_samples, test_time_series_length, test_num_classes;

    if (!loader.load_hydra_test_data(
        test_file,
        test_inputs,
        test_labels,
        test_num_samples,
        test_time_series_length,
        test_num_classes
    )) {
        std::cerr << "ERROR: Failed to load test data" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\n=== Test Data Info ===" << std::endl;
    std::cout << "Samples: " << test_num_samples << std::endl;
    std::cout << "Time series length: " << test_time_series_length << std::endl;
    std::cout << "Classes: " << test_num_classes << std::endl;
    std::cout << "=======================\n" << std::endl;

    // Find Xilinx platform
    cl::Platform platform = find_xilinx_platform();

    // Get devices
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);

    if (devices.empty()) {
        std::cerr << "ERROR: No devices found" << std::endl;
        return EXIT_FAILURE;
    }

    cl::Device device = devices[0];
    print_device_info(device);

    // Create context and command queue
    cl_int err;
    OCL_CHECK(err, cl::Context context(device, nullptr, nullptr, nullptr, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device,
                                      CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                                      &err));

    // Load XClbin
    std::cout << "Loading XClbin: " << xclbin_file << std::endl;
    auto xclbin_data = load_file_to_memory(xclbin_file);

    cl::Program::Binaries bins{{xclbin_data.data(), xclbin_data.size()}};
    std::vector<cl::Device> devices_list = {device};

    OCL_CHECK(err, cl::Program program(context, devices_list, bins, nullptr, &err));

    // Create kernel
    OCL_CHECK(err, cl::Kernel kernel(program, "hydra_inference", &err));
    std::cout << "Kernel created successfully" << std::endl;

    // Prepare model data for transfer - flatten 2D arrays
    std::cout << "\nAllocating device buffers..." << std::endl;

    // Flatten kernel weights: [512][9] -> [4608]
    std::vector<data_t, aligned_allocator<data_t>> h_kernel_weights(num_kernels * KERNEL_SIZE);
    for (int k = 0; k < num_kernels; k++) {
        for (int w = 0; w < KERNEL_SIZE; w++) {
            h_kernel_weights[k * KERNEL_SIZE + w] = kernel_weights[k][w];
        }
    }

    // Flatten coefficients: [10][1024] -> [10240] (row-major)
    std::vector<data_t, aligned_allocator<data_t>> h_coefficients(num_classes * num_features);
    for (int c = 0; c < num_classes; c++) {
        for (int f = 0; f < num_features; f++) {
            h_coefficients[c * num_features + f] = coefficients[c][f];
        }
    }

    // Allocate aligned buffers for other parameters
    std::vector<data_t, aligned_allocator<data_t>> h_time_series(time_series_length);
    std::vector<data_t, aligned_allocator<data_t>> h_predictions(num_classes);
    std::vector<data_t, aligned_allocator<data_t>> h_biases(num_kernels);
    std::vector<int_t, aligned_allocator<int_t>> h_dilations(num_kernels);
    std::vector<data_t, aligned_allocator<data_t>> h_scaler_mean(num_features);
    std::vector<data_t, aligned_allocator<data_t>> h_scaler_scale(num_features);
    std::vector<data_t, aligned_allocator<data_t>> h_intercept(num_classes);

    // Copy biases, dilations, scaler parameters, and intercept
    for (int i = 0; i < num_kernels; i++) {
        h_biases[i] = biases[i];
        h_dilations[i] = dilations[i];
    }
    for (int i = 0; i < num_features; i++) {
        h_scaler_mean[i] = scaler_mean[i];
        h_scaler_scale[i] = scaler_scale[i];
    }
    for (int i = 0; i < num_classes; i++) {
        h_intercept[i] = intercept[i];
    }

    // Create device buffers
    OCL_CHECK(err, cl::Buffer buf_time_series(context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                h_time_series.size() * sizeof(data_t),
                                                h_time_series.data(), &err));

    OCL_CHECK(err, cl::Buffer buf_predictions(context,
                                               CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                               h_predictions.size() * sizeof(data_t),
                                               h_predictions.data(), &err));

    OCL_CHECK(err, cl::Buffer buf_kernel_weights(context,
                                                  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                  h_kernel_weights.size() * sizeof(data_t),
                                                  h_kernel_weights.data(), &err));

    OCL_CHECK(err, cl::Buffer buf_biases(context,
                                          CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          h_biases.size() * sizeof(data_t),
                                          h_biases.data(), &err));

    OCL_CHECK(err, cl::Buffer buf_dilations(context,
                                             CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                             h_dilations.size() * sizeof(int_t),
                                             h_dilations.data(), &err));

    OCL_CHECK(err, cl::Buffer buf_scaler_mean(context,
                                               CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                               h_scaler_mean.size() * sizeof(data_t),
                                               h_scaler_mean.data(), &err));

    OCL_CHECK(err, cl::Buffer buf_scaler_scale(context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                h_scaler_scale.size() * sizeof(data_t),
                                                h_scaler_scale.data(), &err));

    OCL_CHECK(err, cl::Buffer buf_coefficients(context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                h_coefficients.size() * sizeof(data_t),
                                                h_coefficients.data(), &err));

    OCL_CHECK(err, cl::Buffer buf_intercept(context,
                                             CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                             h_intercept.size() * sizeof(data_t),
                                             h_intercept.data(), &err));

    std::cout << "Device buffers created" << std::endl;

    // Set kernel arguments (indices match the interface definition)
    int arg_idx = 0;
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, buf_time_series));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, buf_predictions));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, buf_coefficients));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, buf_intercept));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, buf_scaler_mean));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, buf_scaler_scale));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, buf_kernel_weights));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, buf_biases));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, buf_dilations));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, static_cast<int>(time_series_length)));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, static_cast<int>(num_features)));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, static_cast<int>(num_classes)));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, static_cast<int>(num_groups)));

    std::cout << "Kernel arguments set" << std::endl;

    // Process test samples
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running HYDRA Inference on FPGA" << std::endl;
    std::cout << "========================================\n" << std::endl;

    int correct = 0;
    double total_kernel_time = 0.0;

    for (int sample_idx = 0; sample_idx < test_num_samples; sample_idx++) {
        // Copy time series to host buffer
        for (int t = 0; t < test_time_series_length; t++) {
            h_time_series[t] = test_inputs[sample_idx][t];
        }

        // Transfer data to device
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buf_time_series}, 0));

        // Execute kernel
        cl::Event event;
        OCL_CHECK(err, err = q.enqueueTask(kernel, nullptr, &event));
        OCL_CHECK(err, err = q.finish());

        // Get kernel execution time
        cl_ulong start_time, end_time;
        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
        double kernel_time_ms = (end_time - start_time) / 1e6;
        total_kernel_time += kernel_time_ms;

        // Transfer results back
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buf_predictions}, CL_MIGRATE_MEM_OBJECT_HOST));
        OCL_CHECK(err, err = q.finish());

        // Find predicted class
        int predicted_class = 0;
        data_t max_score = h_predictions[0];
        for (int c = 1; c < num_classes; c++) {
            if (h_predictions[c] > max_score) {
                max_score = h_predictions[c];
                predicted_class = c;
            }
        }

        int true_class = test_labels[sample_idx];
        bool is_correct = (predicted_class == true_class);
        if (is_correct) correct++;

        if (sample_idx < 10 || sample_idx % 10 == 0) {
            std::cout << "Sample " << std::setw(3) << sample_idx + 1 << "/" << test_num_samples
                      << ": predicted=" << predicted_class
                      << ", actual=" << true_class
                      << ", time=" << std::fixed << std::setprecision(3) << kernel_time_ms << " ms"
                      << " [" << (is_correct ? "✓" : "✗") << "]"
                      << std::endl;
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results Summary" << std::endl;
    std::cout << "========================================" << std::endl;

    double accuracy = static_cast<double>(correct) / test_num_samples;
    double avg_latency = total_kernel_time / test_num_samples;
    double throughput = 1000.0 / avg_latency;  // inferences per second

    std::cout << "\nAccuracy: " << correct << "/" << test_num_samples
              << " = " << std::fixed << std::setprecision(2) << (accuracy * 100.0) << "%" << std::endl;

    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Average latency: " << std::fixed << std::setprecision(3) << avg_latency << " ms" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(0) << throughput << " inferences/sec" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << total_kernel_time / 1000.0 << " s" << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST " << (accuracy > 0.3 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "========================================\n" << std::endl;

    return (accuracy > 0.3) ? EXIT_SUCCESS : EXIT_FAILURE;
}
