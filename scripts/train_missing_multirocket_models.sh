#!/bin/bash
# Train MultiRocket84 models for datasets that are missing.
# Previous attempt failed with disk-full on MosquitoSound/FruitFlies.
# Now 411GB free — should complete.
#
# Usage: tmux new-session -d -s mr84-train 'bash scripts/train_missing_multirocket_models.sh'
# Check: tmux attach -t mr84-train
#
# Expected runtime: ~30+ hours per dataset (pure Python convolution is slow)

set -euo pipefail
cd "$(dirname "$0")/.."

SCRIPTS_DIR="multirocket_optimized/scripts"
MODELS_DIR="multirocket_optimized/models"

echo "=== MultiRocket84 Training (Missing Datasets) ==="
echo "Date: $(date)"
echo "Disk: $(df -h /home/rdave009 | tail -1)"
echo ""

# Check which models are missing
MISSING=()
for ds in MosquitoSound FruitFlies; do
    model="$MODELS_DIR/multirocket84_$(echo $ds | tr '[:upper:]' '[:lower:]')_model.json"
    if [ ! -f "$model" ]; then
        MISSING+=("$ds")
        echo "MISSING: $model"
    else
        echo "EXISTS:  $model"
    fi
done

if [ ${#MISSING[@]} -eq 0 ]; then
    echo ""
    echo "All models already exist! Nothing to train."
    exit 0
fi

echo ""
echo "Will train: ${MISSING[*]}"
echo ""

cd "$SCRIPTS_DIR"

for ds in "${MISSING[@]}"; do
    echo "============================================"
    echo "  Training $ds"
    echo "  Started: $(date)"
    echo "============================================"

    python3 train_multirocket84_benchmarks.py --dataset "$ds" 2>&1 | tee "$MODELS_DIR/train_${ds,,}.log"

    echo ""
    echo "=== $ds completed at $(date) ==="
    echo ""
done

echo ""
echo "============================================"
echo "  All training complete!"
echo "  Models in: $MODELS_DIR/"
echo "  Now run C++ baseline:"
echo "    cd cpu && ./multirocket_cpu ../multirocket_optimized/models/multirocket84_<dataset>_model.json ../multirocket_optimized/models/multirocket84_<dataset>_test.json"
echo "============================================"
