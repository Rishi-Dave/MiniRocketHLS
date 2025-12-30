#!/usr/bin/env tclsh

# MiniRocket HLS Simulation Script
# Runs C simulation, synthesis, and co-simulation

# Create project
open_project minirocket_hls -reset
set_top krnl_top

# Add design files
add_files src/krnl.cpp -cflags "-I./src -std=c++11"
add_files src/minirocket_inference_hls.cpp -cflags "-I./src -std=c++11"

# Add testbench files
add_files -tb src/test_hls.cpp -cflags "-I./src -std=c++11"
add_files -tb src/minirocket_hls_testbench_loader.cpp -cflags "-I./src -std=c++11"
add_files -tb minirocket_model.json
add_files -tb minirocket_model_test_data.json

# Create solution
open_solution "solution1" -reset
set_part {xcvu9p-flga2104-2-i}
create_clock -period 10 -name default

# Run C simulation
puts "\n=========================================="
puts "Running C Simulation (csim_design)"
puts "=========================================="
csim_design -argv "../../../../minirocket_model.json ../../../../minirocket_model_test_data.json"

# Run synthesis
puts "\n=========================================="
puts "Running HLS Synthesis (csynth_design)"
puts "=========================================="
csynth_design

# Run co-simulation (RTL verification)
puts "\n=========================================="
puts "Running Co-Simulation (cosim_design)"
puts "=========================================="
cosim_design -argv "../../../../minirocket_model.json ../../../../minirocket_model_test_data.json" -trace_level all

# Export RTL
# export_design -format ip_catalog

puts "\n=========================================="
puts "HLS Simulation Complete!"
puts "=========================================="
puts "Results:"
puts "  C Simulation:   minirocket_hls/solution1/csim/report/"
puts "  Synthesis:      minirocket_hls/solution1/syn/report/"
puts "  Co-Simulation:  minirocket_hls/solution1/sim/report/"
puts "=========================================="

exit
