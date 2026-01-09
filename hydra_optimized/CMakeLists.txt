cmake_minimum_required(VERSION 3.0)

project(hydra-optimized)

# Set CMake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

# Set FPGA part
set(FPGA_PART xcu280-fsvh2892-2L-e)

# Find Vitis HLS
find_package(Vitis REQUIRED)

# Add HLS kernel subdirectory
add_subdirectory(hydra)
