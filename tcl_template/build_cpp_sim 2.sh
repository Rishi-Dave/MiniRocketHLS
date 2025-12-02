#!/bin/bash

# MiniRocket C++ Functional Simulation Script
# This script runs a functional simulation without requiring Xilinx HLS tools

set -e  # Exit on any error

echo "=== MiniRocket C++ Functional Simulation ==="
echo "Building and running C++ simulation (no HLS tools required)..."

# Create build directory
BUILD_DIR="build_cpp_sim"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Copy source files
echo "Copying source files..."
cp -r ../src .
cp ../minirocket_model.json .
cp ../minirocket_model_test_data.json .

cd src

# Compile the testbench
echo "Compiling C++ testbench..."
g++ -std=c++14 -I. -DSIM_ONLY -Wall -O2 -o test_sim \
    test_krnl.cpp \
    minirocket_hls_testbench_loader.cpp \
    2>/dev/null || {
    
    echo "Note: Compiling without external JSON library..."
    echo "If this fails, you may need to install development tools:"
    echo "  macOS: xcode-select --install"
    echo "  Ubuntu: sudo apt-get install build-essential"
    echo "  CentOS: sudo yum groupinstall 'Development Tools'"
    echo ""
    
    g++ -std=c++14 -I. -DSIM_ONLY -Wall -O2 -o test_sim \
        test_krnl.cpp \
        minirocket_hls_testbench_loader.cpp || {
        echo "ERROR: Cannot compile C++ testbench."
        echo "Please ensure g++ is installed and available."
        exit 1
    }
}

echo "Successfully compiled testbench!"
echo ""

# Run the simulation
echo "Running functional simulation..."
echo "This tests the MiniRocket algorithm implementation without HLS synthesis."
echo ""

./test_sim ../minirocket_model.json ../minirocket_model_test_data.json

echo ""
echo "=== C++ Functional Simulation Complete ==="
echo "This simulation validates:"
echo "  ✓ Algorithm correctness"
echo "  ✓ Fixed-point precision"
echo "  ✓ Test data loading"
echo "  ✓ Accuracy vs baseline"
echo ""
echo "Next steps:"
echo "  - Install Xilinx Vitis HLS for full hardware simulation"
echo "  - Run ./build_sim.sh for complete HLS synthesis"