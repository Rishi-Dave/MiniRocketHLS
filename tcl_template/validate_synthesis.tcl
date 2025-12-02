# Quick synthesis validation script
# Tests if the code compiles with the optimizations

open_project -reset minirocket_hls_validate
set_top krnl_top

# Add source files
add_files src/krnl.cpp
add_files src/krnl.hpp

# Create solution
open_solution "solution1"
set_part {xcvu9p-flga2104-2-i}
create_clock -period 10 -name default

# Run synthesis only
csynth_design

exit
