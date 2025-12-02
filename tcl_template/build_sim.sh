#!/bin/bash

# MiniRocket HLS Simulation Build Script
# This script builds and runs the HLS simulation for MiniRocket

set -e  # Exit on any error

echo "=== MiniRocket HLS Simulation Build Script ==="
echo "Building HLS simulation environment..."

# Check if required tools are available
if ! command -v vitis_hls &> /dev/null; then
    echo "WARNING: vitis_hls not found. Trying alternative approaches..."
    echo ""
    echo "To run full HLS simulation, you need Xilinx Vitis HLS installed."
    echo "Common installation paths to try:"
    echo "  source /tools/Xilinx/Vitis_HLS/*/settings64.sh"
    echo "  source /opt/Xilinx/Vitis_HLS/*/settings64.sh"
    echo "  source ~/Xilinx/Vitis_HLS/*/settings64.sh"
    echo ""
    echo "For now, running C++ functional simulation only..."
    
    # Run C++ simulation instead
    cd src
    echo "Compiling C++ testbench..."
    g++ -std=c++14 -I. -DSIM_ONLY -o test_sim test_krnl.cpp minirocket_hls_testbench_loader.cpp -ljson-c 2>/dev/null || {
        echo "Attempting compilation without json-c library..."
        g++ -std=c++14 -I. -DSIM_ONLY -o test_sim test_krnl.cpp minirocket_hls_testbench_loader.cpp || {
            echo "ERROR: Cannot compile C++ testbench. Please install g++ and development tools."
            exit 1
        }
    }
    
    echo "Running functional simulation..."
    cp ../minirocket_model.json .
    cp ../minirocket_model_test_data.json .
    ./test_sim
    exit $?
fi

# Create build directory
BUILD_DIR="build_hls_sim"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Copy source files
echo "Copying source files..."
cp -r ../src .
cp ../minirocket_model.json .
cp ../minirocket_model_test_data.json .

# Generate HLS project
echo "Generating HLS project..."
cat > create_project.tcl << 'EOF'
# Create new project
open_project minirocket_hls
set_top krnl_top

# Add source files
add_files src/krnl.cpp
add_files src/krnl.hpp
add_files -tb src/test_krnl.cpp
add_files -tb src/minirocket_hls_testbench_loader.cpp
add_files -tb src/minirocket_hls_testbench_loader.h
add_files -tb minirocket_model.json
add_files -tb minirocket_model_test_data.json

# Create solution
open_solution "solution1"
set_part {xcvu9p-flga2104-2-i}
create_clock -period 10 -name default

# Run C simulation
csim_design

# Run synthesis
csynth_design

# Run cosimulation if C simulation passes
cosim_design -trace_level all

exit
EOF

# Run HLS
echo "Running HLS synthesis and simulation..."
vitis_hls -f create_project.tcl

echo "=== HLS Simulation Complete ==="
echo "Check results in $BUILD_DIR/minirocket_hls/solution1/"
echo "C simulation log: minirocket_hls/solution1/csim/report/"
echo "Synthesis report: minirocket_hls/solution1/syn/report/"
echo "Cosim report: minirocket_hls/solution1/sim/report/"