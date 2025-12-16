#!/bin/bash
# Fix lifecycle in extracted IP during build
BUILD_DIR="_x.hw.xilinx_u280_gen3x16_xdma_1_202211_1"

echo "Monitoring for extracted IP and fixing lifecycle..."
while true; do
    if [ -d "$BUILD_DIR/link/sys_link/iprepo" ]; then
        echo "Found extracted IP, applying fix..."
        find $BUILD_DIR -name "component.xml" -path "*krnl_top*" -exec sed -i 's/lifeCycle="Pre-Production"/lifeCycle="Production"/g' {} \;
        echo "Lifecycle fixed in extracted IPs"
        break
    fi
    sleep 2
done
