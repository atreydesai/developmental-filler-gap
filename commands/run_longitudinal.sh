#!/bin/bash
# Longitudinal experiment pipeline for BabyLM filler-gap analysis
# Usage: bash commands/run_longitudinal.sh [steps] [batch_size]

set -e  # Exit on error

# Activate conda environment (adjust name as needed)
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate fillergap 2>/dev/null || echo "Warning: could not activate conda env 'fillergap'; ensure dependencies are installed"

# Configuration
STEPS=${1:-16}
BATCH_SIZE=${2:-25}

echo "============================================="
echo "Longitudinal Filler-Gap Experiment"
echo "============================================="
echo "Steps: $STEPS, Batch Size: $BATCH_SIZE"
echo ""
echo "============================================="
echo ""

# Create log directory and file
LOG_DIR="logs/longitudinal"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/run_$(date +%d-%H-%M).log"

echo "Logging to: $LOG_FILE"
echo ""

python causalgym/run_longitudinal.py \
    --steps $STEPS \
    --batch-size $BATCH_SIZE \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================="
echo "All experiments complete!"
echo "Log saved to: $LOG_FILE"
echo "Run: python results/programs/process_results.py"
echo "============================================="
