# Copyright (C) 2022 Xilinx, Inc
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.12)
project(setup_link CXX)

# Set useful compile warnings
add_compile_options(-Wall -Wextra -Wno-unused-variable
                    -Wno-unused-but-set-variable -Wno-unused-parameter)

add_subdirectory(../../ "CMakeFiles/xrt_host_api")
find_package(jsoncpp REQUIRED)
get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)

add_executable(minirocket minirocket.cpp)
target_compile_features(minirocket PRIVATE cxx_std_17)
target_include_directories(minirocket PUBLIC ${VNX_INCLUDE_PATH} ${JSON_INC_PATH})
target_link_libraries(minirocket PUBLIC jsoncpp_lib vnx)
