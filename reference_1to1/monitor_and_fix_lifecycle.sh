#!/bin/bash
# Monitor and fix lifecycle during build
BUILD_DIR="_x.hw.xilinx_u280_gen3x16_xdma_1_202211_1"

echo "Starting continuous lifecycle monitor..."
while true; do
    # Fix all component.xml files with krnl_top in the path
    FIXED_COUNT=$(find $BUILD_DIR -name "component.xml" -path "*krnl_top*" -exec grep -l 'lifeCycle="Pre-Production"' {} \; 2>/dev/null | wc -l)

    if [ "$FIXED_COUNT" -gt 0 ]; then
        echo "[$(date '+%H:%M:%S')] Found $FIXED_COUNT files to fix, applying lifecycle fix..."
        find $BUILD_DIR -name "component.xml" -path "*krnl_top*" -exec sed -i 's/lifeCycle="Pre-Production"/lifeCycle="Production"/g' {} \; 2>/dev/null
        echo "[$(date '+%H:%M:%S')] Fixed lifecycle in $FIXED_COUNT files"
    fi

    # Also check if build is still running
    if ! pgrep -f "make build TARGET=hw" > /dev/null; then
        echo "[$(date '+%H:%M:%S')] Build process not found, exiting monitor"
        break
    fi

    sleep 2
done

echo "Monitor exiting"
