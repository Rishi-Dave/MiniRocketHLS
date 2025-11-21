# Package as Vitis kernel (.xo file)
open_project build_hls_sim/minirocket_hls
open_solution "solution1"

# Export as Vitis kernel
export_design -format xo -output minirocket_kernel.xo

exit
