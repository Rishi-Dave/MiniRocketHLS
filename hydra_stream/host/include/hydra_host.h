#ifndef HYDRA_HOST_H
#define HYDRA_HOST_H

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>

// OpenCL error checking macro
#define OCL_CHECK(error, call)                                                 \
    call;                                                                      \
    if (error != CL_SUCCESS) {                                                 \
        std::cerr << "OpenCL Error: " << error << " in " << #call             \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;      \
        exit(EXIT_FAILURE);                                                    \
    }

// Aligned allocator for 4KB page alignment (required for HBM)
template <typename T>
struct aligned_allocator {
    using value_type = T;

    aligned_allocator() {}

    aligned_allocator(const aligned_allocator&) {}

    template <typename U>
    aligned_allocator(const aligned_allocator<U>&) {}

    T* allocate(std::size_t num) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 4096, num * sizeof(T))) {
            throw std::bad_alloc();
        }
        return reinterpret_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t num) {
        free(p);
    }
};

// Utility functions

/**
 * Load XClbin file into memory
 */
std::vector<char> load_file_to_memory(const std::string& filename);

/**
 * Find Xilinx platform
 */
cl::Platform find_xilinx_platform();

/**
 * Print device information
 */
void print_device_info(const cl::Device& device);

/**
 * Check if device has HBM
 */
bool has_hbm(const cl::Device& device);

#endif // HYDRA_HOST_H
