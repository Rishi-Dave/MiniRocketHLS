# Open existing HLS project
open_project build_hls_sim/minirocket_hls
open_solution "solution1"

# Export as Vitis kernel (.xo)
export_design -format xo -output minirocket_kernel.xo

exit
