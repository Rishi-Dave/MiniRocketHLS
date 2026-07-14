# Vitis HLS TCL for hydra_axis_inference (Phase 3 of network-saturation pivot).
#
# Mirrors hydra_optimized/hydra/make.tcl.in patterns. Self-contained: no
# CMake configure step needed. Invoke from this directory:
#
#   vitis_hls -f make.tcl -tclargs csim
#   vitis_hls -f make.tcl -tclargs synthesis
#   vitis_hls -f make.tcl -tclargs ip
#
# `ip` produces hydra_axis_inference.xo, which the OCT v++ link step picks
# up via config_axis.cfg + the fpga-network NetLayer/CMAC chain.

set command [lindex $argv end]

open_project -reset hydra_axis_inference

set_top hydra_axis_inference

# BUILD=1 wraps the top function body in `while(true)` for the free-running
# hardware kernel (see src/hydra_axis.cpp `#if BUILD == 1`) — the kernel is
# kicked once via ap_start and then services the AXI-Stream forever without
# further host intervention. That while(true) never returns, so compiling
# the C testbench (test/test_hydra_axis.cpp, which calls hydra_axis_inference
# once per beat and expects each call to return) with BUILD=1 would hang
# csim forever. csim therefore always compiles the single-shot (BUILD
# undefined -> #if BUILD==1 is false) form of the function; synthesis/ip/
# cosim compile the real free-running hardware form.
if {$command == "csim" || $command == "csim_acc"} {
    add_files src/hydra_axis.cpp
} else {
    add_files src/hydra_axis.cpp -cflags "-DBUILD=1"
}
# csim_acc = real-model accuracy gate (env: HYDRA_MODEL/HYDRA_TEST/
# HYDRA_MAXN/HYDRA_GATE); plain csim = synthetic smoke test.
if {$command == "csim_acc"} {
    add_files -tb test/test_hydra_axis_accuracy.cpp -cflags "-std=c++14 -O2"
} else {
    add_files -tb test/test_hydra_axis.cpp -cflags "-std=c++14"
}

open_solution -reset solution

# U280 part, 300 MHz target — match m_axi HYDRA build.
set_part xcu280-fsvh2892-2L-e
create_clock -period 3.333333

config_interface -m_axi_latency=64
config_interface -m_axi_alignment_byte_size=64
config_interface -m_axi_max_widen_bitwidth=512
config_rtl -register_reset_num=3
config_export -format xo -output hydra_axis_inference.xo

if {$command == "csim"} {
    csim_design
} elseif {$command == "csim_acc"} {
    csim_design -O
} elseif {$command == "synthesis"} {
    csynth_design
} elseif {$command == "cosim"} {
    cosim_design
} elseif {$command == "ip"} {
    csynth_design
    export_design -flow impl
} else {
    puts "Usage: vitis_hls -f make.tcl -tclargs <csim|synthesis|cosim|ip>"
    exit 1
}

exit
