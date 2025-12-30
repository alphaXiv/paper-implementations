#!/bin/bash

# Inference and Evaluation Script for Spurious Rewards Paper Implementation
# This script provides commands to run evaluations on the trained models.
# Usage: ./inference.sh -c <checkpoint_dir> -s <step> [-b <base_model>] [-o <output_dir>]

# Default values
BASE_MODEL="Qwen/Qwen2.5-Math-7B"
OUTPUT_DIR="./export-for-eval"

# Parse command line arguments
while getopts "c:s:b:o:h" opt; do
  case $opt in
    c) CHECKPOINT="$OPTARG" ;;
    s) STEP="$OPTARG" ;;
    b) BASE_MODEL="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    h) echo "Usage: $0 -c <checkpoint_dir> -s <step> [-b <base_model>] [-o <output_dir>]"
       echo "  -c: Path to the DeepSpeed checkpoint directory (required)"
       echo "  -s: Checkpoint step number to export (required)"
       echo "  -b: Base model name (default: Qwen/Qwen2.5-Math-7B)"
       echo "  -o: Directory to save the exported model (default: ./export-for-eval)"
       exit 0 ;;
    *) echo "Invalid option: -$OPTARG" >&2
       echo "Use -h for help"
       exit 1 ;;
  esac
done

# Check required arguments
if [ -z "$CHECKPOINT" ] || [ -z "$STEP" ]; then
  echo "Error: -c (checkpoint) and -s (step) are required arguments."
  echo "Use -h for help."
  exit 1
fi

echo "Starting Inference and Evaluation Process..."
echo "Checkpoint: $CHECKPOINT"
echo "Step: $STEP"
echo "Base Model: $BASE_MODEL"
echo "Output Dir: $OUTPUT_DIR"

## Evaluations
# To reproduce our evaluation results, use the following commands:

echo "Navigating to code directory..."
cd code

echo "Exporting checkpoint for evaluation..."
# For MATH-500 evaluation matching our reported scores in wandb using checkpoints 
python export_checkpoint.py --checkpoint "$CHECKPOINT" --step "$STEP" --base-model "$BASE_MODEL" --output-dir "$OUTPUT_DIR"

echo "Running evaluation on exported checkpoint..."
python eval_checkpoint.py --model_path "$OUTPUT_DIR" --datasets MATH-500,AIME-2024,AIME-2025,AMC --shards 2

echo "Evaluation completed."