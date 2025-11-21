# Open existing HLS project
open_project build_hls_sim/minirocket_hls
open_solution "solution1"

# Export as Vivado IP
export_design -format ip_catalog -description "MiniRocket Accelerator" -vendor "xilinx.com" -library "user" -version "1.0" -display_name "MiniRocket HLS"

exit
