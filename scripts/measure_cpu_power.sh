#!/bin/bash
# CPU Power Measurement for Paper
# ================================
#
# REQUIRES: sudo access to install linux-tools and read RAPL counters
#
# Ask someone with sudo to run:
#   sudo apt install linux-tools-$(uname -r) linux-tools-generic
#
# Then run this script (may also need sudo for perf):
#   sudo bash scripts/measure_cpu_power.sh
#
# Alternative without perf (no sudo needed):
#   Method 2 below reads RAPL directly from sysfs

set -euo pipefail
cd "$(dirname "$0")/.."

CPU_DIR="cpu"
RESULTS_DIR="results/power"
mkdir -p "$RESULTS_DIR"

echo "=== CPU Power Measurement ==="
echo "CPU: $(cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2 | xargs)"
echo ""

# ============================================================
# Method 1: Using perf (requires sudo + linux-tools)
# ============================================================
measure_with_perf() {
    local binary="$1"
    local args="$2"
    local label="$3"
    local logfile="$RESULTS_DIR/perf_${label}.log"

    echo "--- Measuring: $label ---"
    perf stat -e power/energy-pkg/,power/energy-ram/ -a -- $binary $args 2>&1 | tee "$logfile"

    # Extract energy
    pkg_energy=$(grep "energy-pkg" "$logfile" | awk '{print $1}' | tr -d ',')
    ram_energy=$(grep "energy-ram" "$logfile" | awk '{print $1}' | tr -d ',')
    elapsed=$(grep "seconds time elapsed" "$logfile" | awk '{print $1}')

    if [ -n "$pkg_energy" ] && [ -n "$elapsed" ]; then
        pkg_power=$(echo "scale=2; $pkg_energy / $elapsed" | bc)
        echo "  Package energy: ${pkg_energy} J over ${elapsed} s = ${pkg_power} W"
    fi
    echo ""
}

# ============================================================
# Method 2: Using RAPL sysfs (may work without sudo)
# ============================================================
measure_with_rapl() {
    local binary="$1"
    local args="$2"
    local label="$3"

    RAPL_PATH="/sys/class/powercap/intel-rapl:0/energy_uj"

    if [ ! -f "$RAPL_PATH" ]; then
        echo "RAPL sysfs not available at $RAPL_PATH"
        echo "Try: sudo chmod 444 /sys/class/powercap/intel-rapl:0/energy_uj"
        return 1
    fi

    echo "--- Measuring with RAPL: $label ---"

    # Read energy before
    energy_before=$(cat "$RAPL_PATH")
    time_before=$(date +%s%N)

    # Run the benchmark
    $binary $args > /dev/null 2>&1

    # Read energy after
    energy_after=$(cat "$RAPL_PATH")
    time_after=$(date +%s%N)

    # Calculate
    energy_uj=$((energy_after - energy_before))
    energy_j=$(echo "scale=4; $energy_uj / 1000000" | bc)
    time_ns=$((time_after - time_before))
    time_s=$(echo "scale=4; $time_ns / 1000000000" | bc)
    power_w=$(echo "scale=2; $energy_j / $time_s" | bc)

    echo "  Energy: ${energy_j} J"
    echo "  Time: ${time_s} s"
    echo "  Average power: ${power_w} W"
    echo "$label,$energy_j,$time_s,$power_w" >> "$RESULTS_DIR/rapl_measurements.csv"
    echo ""
}

# ============================================================
# Method 3: turbostat (requires sudo)
# ============================================================
measure_with_turbostat() {
    local binary="$1"
    local args="$2"
    local label="$3"

    echo "--- Measuring with turbostat: $label ---"
    turbostat --Summary --quiet -- $binary $args 2>&1 | tee "$RESULTS_DIR/turbostat_${label}.log"
    echo ""
}

# ============================================================
# Run measurements
# ============================================================

echo "Testing available methods..."
echo ""

# Check which method is available
HAS_PERF=false
HAS_RAPL=false
HAS_TURBOSTAT=false

which perf >/dev/null 2>&1 && HAS_PERF=true
[ -r "/sys/class/powercap/intel-rapl:0/energy_uj" ] && HAS_RAPL=true
which turbostat >/dev/null 2>&1 && HAS_TURBOSTAT=true

echo "perf available: $HAS_PERF"
echo "RAPL sysfs readable: $HAS_RAPL"
echo "turbostat available: $HAS_TURBOSTAT"
echo ""

if ! $HAS_PERF && ! $HAS_RAPL && ! $HAS_TURBOSTAT; then
    echo "ERROR: No power measurement method available!"
    echo ""
    echo "Options:"
    echo "  1. Ask someone with sudo to run:"
    echo "     sudo apt install linux-tools-$(uname -r) linux-tools-generic"
    echo "     Then re-run this script with: sudo bash scripts/measure_cpu_power.sh"
    echo ""
    echo "  2. Ask someone with sudo to make RAPL readable:"
    echo "     sudo chmod 444 /sys/class/powercap/intel-rapl:0/energy_uj"
    echo "     Then re-run without sudo"
    echo ""
    echo "  3. Use the Xeon E5-2640 v3 published TDP of 90W as upper bound."
    echo "     Note in paper: 'CPU TDP of 90W used as conservative upper bound;"
    echo "     actual power under single-threaded inference is lower, making"
    echo "     efficiency ratios conservative estimates favoring the CPU.'"
    exit 1
fi

# Measure idle power first
echo "=== Idle Power (10s) ==="
if $HAS_RAPL; then
    measure_with_rapl "sleep" "10" "idle"
elif $HAS_PERF; then
    measure_with_perf "sleep" "10" "idle"
fi

# Measure each CPU benchmark
echo ""
echo "=== MiniRocket C++ Inference ==="

MINI_HOST="$CPU_DIR/minirocket_cpu"
MINI_MODEL_DIR="minirocket_modular"

for dataset in insectsound mosquitosound fruitflies; do
    model=$(ls ${MINI_MODEL_DIR}/${dataset}_minirocket_model.json 2>/dev/null || echo "")
    test_data=$(ls ${MINI_MODEL_DIR}/${dataset}_minirocket_model_test_data.json 2>/dev/null || echo "")

    if [ -z "$model" ] || [ -z "$test_data" ]; then
        echo "Skipping $dataset (model or test data not found)"
        continue
    fi

    if $HAS_RAPL; then
        measure_with_rapl "$MINI_HOST" "$model $test_data" "minirocket_${dataset}"
    elif $HAS_PERF; then
        measure_with_perf "$MINI_HOST" "$model $test_data" "minirocket_${dataset}"
    fi
done

echo ""
echo "============================================"
echo "  Power measurement complete!"
echo "  Results in: $RESULTS_DIR/"
echo "============================================"

if [ -f "$RESULTS_DIR/rapl_measurements.csv" ]; then
    echo ""
    echo "=== RAPL Summary ==="
    cat "$RESULTS_DIR/rapl_measurements.csv"
fi
