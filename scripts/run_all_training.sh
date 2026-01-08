#!/bin/bash
# Master script to run all training with proper environment setup
# This script trains HYDRA, MiniRocket, and MultiRocket on all three datasets

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "MASTER TRAINING SCRIPT - HYDRA, MiniRocket, MultiRocket"
echo "========================================================================"
echo ""

# Set TMPDIR to avoid /tmp space limitations (8GB limit)
export TMPDIR=/home/rdave009/minirocket-hls/tmp
mkdir -p "$TMPDIR"

echo "Environment Setup:"
echo "  TMPDIR: $TMPDIR"
echo "  Python: $(which python3)"
echo "  Date: $(date)"
echo ""

# Function to run training and check status
run_training() {
    local name=$1
    local script_dir=$2
    local script_name=$3
    local args=$4

    echo "========================================================================"
    echo -e "${YELLOW}Training: $name${NC}"
    echo "========================================================================"
    echo "Directory: $script_dir"
    echo "Command: python3 $script_name $args"
    echo ""

    cd "$script_dir"

    if python3 "$script_name" $args; then
        echo ""
        echo -e "${GREEN}SUCCESS: $name completed${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}FAILED: $name training failed${NC}"
        return 1
    fi
}

# Track results
declare -a results
declare -a names

# HYDRA Training
run_training "HYDRA - All Datasets" \
    "/home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/scripts" \
    "train_and_save_models.py" \
    "--all"
results+=($?)
names+=("HYDRA")

echo ""
echo "========================================================================"
echo ""

# MiniRocket Training
run_training "MiniRocket - All Datasets" \
    "/home/rdave009/minirocket-hls/MiniRocketHLS/reference_1to1/scripts" \
    "train_and_save_models.py" \
    "--all"
results+=($?)
names+=("MiniRocket")

echo ""
echo "========================================================================"
echo ""

# MultiRocket Training
run_training "MultiRocket - All Datasets" \
    "/home/rdave009/minirocket-hls/MiniRocketHLS/multirocket_optimized/scripts" \
    "train_and_save_models.py" \
    "--all"
results+=($?)
names+=("MultiRocket")

# Print summary
echo ""
echo "========================================================================"
echo "TRAINING SUMMARY"
echo "========================================================================"
echo ""

all_success=true
for i in "${!names[@]}"; do
    if [ "${results[$i]}" -eq 0 ]; then
        echo -e "${GREEN}✓ ${names[$i]}: SUCCESS${NC}"
    else
        echo -e "${RED}✗ ${names[$i]}: FAILED${NC}"
        all_success=false
    fi
done

echo ""
echo "========================================================================"

if $all_success; then
    echo -e "${GREEN}All training completed successfully!${NC}"
    echo ""
    echo "Model files saved in:"
    echo "  - /home/rdave009/minirocket-hls/MiniRocketHLS/hydra_optimized/models/"
    echo "  - /home/rdave009/minirocket-hls/MiniRocketHLS/reference_1to1/models/"
    echo "  - /home/rdave009/minirocket-hls/MiniRocketHLS/multirocket_optimized/models/"
    exit 0
else
    echo -e "${RED}Some training runs failed. Check logs above for details.${NC}"
    exit 1
fi
