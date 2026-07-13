#include "../include/hydra_host.h"

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
    for (auto& platform : platforms) {
        std::string name = platform.getInfo<CL_PLATFORM_NAME>();
        if (name == "Xilinx") {
            std::cout << "Found Platform: " << name << std::endl;
            return platform;
        }
    }
    std::cerr << "ERROR: Failed to find Xilinx platform" << std::endl;
    exit(EXIT_FAILURE);
}

void print_device_info(const cl::Device& device) {
    std::cout << "\nDevice Info:" << std::endl;
    std::cout << "  Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "  Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
    std::cout << "  Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
    std::cout << "  Global Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024*1024*1024) << " GB" << std::endl;
    std::cout << std::endl;
}

bool has_hbm(const cl::Device& device) {
    std::string name = device.getInfo<CL_DEVICE_NAME>();
    return (name.find("u280") != std::string::npos ||
            name.find("u50") != std::string::npos ||
            name.find("u55") != std::string::npos);
}
