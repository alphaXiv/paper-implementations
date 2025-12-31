#!/bin/bash

# Inference and Evaluation Script for Spurious Rewards Paper Implementation
# This script provides commands to run evaluations on multiple trained models.
# Usage: ./inference.sh -c <checkpoint_dir> -s <steps> [-b <base_model>]

# Default values
BASE_MODEL="Qwen/Qwen2.5-Math-7B"

# Parse command line arguments
while getopts "c:s:b:h" opt; do
  case $opt in
    c) CHECKPOINT_DIR="$OPTARG" ;;
    s) STEPS="$OPTARG" ;;
    b) BASE_MODEL="$OPTARG" ;;
    h) echo "Usage: $0 -c <checkpoint_dir> -s <steps> [-b <base_model>]"
       echo "  -c: Path to the DeepSpeed checkpoint directory (required)"
       echo "  -s: Comma-separated list of checkpoint step numbers (required, e.g., 450,500,600,700)"
       echo "  -b: Base model name (default: Qwen/Qwen2.5-Math-7B)"
       exit 0 ;;
    *) echo "Invalid option: -$OPTARG" >&2
       echo "Use -h for help"
       exit 1 ;;
  esac
done

# Check required arguments
if [ -z "$CHECKPOINT_DIR" ] || [ -z "$STEPS" ]; then
  echo "Error: -c (checkpoint_dir) and -s (steps) are required arguments."
  echo "Use -h for help."
  exit 1
fi

# Parse steps
IFS=',' read -ra STEP_ARRAY <<< "$STEPS"

echo "Starting Inference and Evaluation Process..."
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo "Steps: ${STEP_ARRAY[*]}"
echo "Base Model: $BASE_MODEL"

## Evaluations
# To reproduce our evaluation results, use the following commands:

echo "Navigating to code directory..."
cd src/spurious_rewards/code

# Create results directory
mkdir -p results

for step in "${STEP_ARRAY[@]}"; do
    echo "Processing step $step..."

    OUTPUT_DIR="./export-for-eval-step${step}"
    RESULTS_DIR="results/step${step}"

    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$RESULTS_DIR"

    echo "Exporting checkpoint for step $step..."
    python scripts/export_checkpoint.py --checkpoint "$CHECKPOINT_DIR" --step "$step" --base-model "$BASE_MODEL" --output-dir "$OUTPUT_DIR"

    echo "Running evaluation for step $step..."
    python eval_checkpoint.py --model_path "$OUTPUT_DIR" --datasets MATH-500,AIME-2024,AIME-2025,AMC --shards 2 --output_dir "$RESULTS_DIR"

    echo "Completed evaluation for step $step."
done

echo "All evaluations completed. Generating plots..."

# Prepare steps for python
STEPS_PYTHON=$(printf '%s ' "${STEP_ARRAY[@]}")
STEPS_PYTHON=${STEPS_PYTHON% }

# Generate plots
python plot_performance.py $STEPS_PYTHON

echo "Process completed."