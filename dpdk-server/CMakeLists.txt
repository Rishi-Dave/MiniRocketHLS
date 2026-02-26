cmake_minimum_required(VERSION 3.16)
project(dpdk_udp_float_receiver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pkg-config
find_package(PkgConfig REQUIRED)

# Find DPDK using pkg-config
pkg_check_modules(DPDK REQUIRED libdpdk)

# Include directories
include_directories(${DPDK_INCLUDE_DIRS})

# Link directories
link_directories(${DPDK_LIBRARY_DIRS})

# Add executable
add_executable(server server.cpp)

# Compile flags from DPDK
target_compile_options(server PRIVATE ${DPDK_CFLAGS_OTHER})

# Link DPDK libraries
target_link_libraries(server ${DPDK_LIBRARIES})