# Vitis HLS TCL for multirocket_axis_inference (Phase 5).
# Mirrors hydra_axis/make.tcl pattern. See ../hydra_axis/make.tcl for usage.

open_project -reset multirocket_axis_inference

set_top multirocket_axis_inference

add_files src/multirocket_axis.cpp -cflags "-DBUILD=1"
add_files -tb test/test_multirocket_axis.cpp -cflags "-std=c++14"

open_solution -reset solution

set_part xcu280-fsvh2892-2L-e
create_clock -period 3.333333

config_interface -m_axi_latency=64
config_interface -m_axi_alignment_byte_size=64
config_interface -m_axi_max_widen_bitwidth=512
config_rtl -register_reset_num=3
config_export -format xo -output multirocket_axis_inference.xo

set command [lindex $argv end]

if {$command == "csim"} {
    csim_design
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
