#!/bin/bash

set -e  # Exit on error

# ============================================================================
# Grassmann Flows - Complete Training & Evaluation Pipeline
# ============================================================================
# This script provides a one-file solution for training and evaluating
# Grassmann Flow models on various datasets.
#
# Usage: ./speedrun.sh [MODEL] [DATASET]
#   MODEL: gpt2 | grassmann | all
#   DATASET: wikitext
#
# Examples:
#   ./speedrun.sh grassmann wikitext    # Train Grassmann on Wikitext-2
#   ./speedrun.sh gpt2 wikitext         # Train GPT-2 baseline on Wikitext-2
#   ./speedrun.sh all wikitext          # Train all models on Wikitext-2
# ============================================================================

# Detect number of GPUs dynamically
if command -v nvidia-smi &> /dev/null; then
    DETECTED_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    DETECTED_GPUS=0
fi

if [ "$DETECTED_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected. This training requires at least 1 GPU."
    echo "Please ensure CUDA and nvidia-smi are properly installed."
    exit 1
fi

# Configuration
MODEL=${1:-"grassmann"}  # Default to Grassmann
DATASET=${2:-"wikitext"}  # Default to Wikitext-2
NUM_GPUS=$DETECTED_GPUS  # Use all available GPUs

echo "=========================================="
echo "Grassmann Flows Training & Evaluation"
echo "=========================================="
echo "Detected GPUs: $DETECTED_GPUS"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Using GPUs: $NUM_GPUS"
echo "=========================================="
echo ""

# ============================================================================
# Step 0: Environment Setup
# ============================================================================

echo "[Step 0/4] Setting up environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -e .

# Login to Weights & Biases if API key is set
if [ -n "$WANDB_API_KEY" ]; then
    echo "Logging in to Weights & Biases..."
    wandb login "$WANDB_API_KEY"
    echo "W&B login successful!"
else
    echo "WANDB_API_KEY not set. Skipping W&B login. Set it to enable logging."
fi

# ============================================================================
# Step 1: Data Preparation
# ============================================================================

echo "[Step 1/4] Preparing data..."

# Data preparation happens automatically in the training scripts
echo "Data will be downloaded automatically during training..."

# ============================================================================
# Step 2: Training
# ============================================================================

echo "[Step 2/4] Starting training..."

# Set training parameters based on GPU count
if [ "$NUM_GPUS" -gt 1 ]; then
    BATCH_SIZE=$((32 * NUM_GPUS))
    GRAD_ACCUM=1
else
    BATCH_SIZE=32
    GRAD_ACCUM=1
fi

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="outputs/${TIMESTAMP}"

# Function to run training for a specific model and dataset
run_training() {
    local model=$1
    local dataset=$2

    echo "Training $model on $dataset..."

    if [ "$dataset" = "wikitext" ]; then
        # Use the exact reproduction script for Wikitext-2
        python scripts/train_wikitext2.py \
            --model "$model" \
            --batch_size "$BATCH_SIZE" \
            --output_dir "${OUTPUT_DIR}/${model}_${dataset}" \
            --epochs 5
    else
        # Use the general training script
        python scripts/train.py \
            --model "$model" \
            --dataset "$dataset" \
            --batch_size "$BATCH_SIZE" \
            --epochs 5 \
            --output_dir "${OUTPUT_DIR}/${model}_${dataset}"
    fi
}

# Determine which models and datasets to run
if [ "$MODEL" = "all" ]; then
    MODELS=("gpt2" "grassmann")
else
    MODELS=("$MODEL")
fi

if [ "$DATASET" = "all" ]; then
    DATASETS=("wikitext")
else
    DATASETS=("$DATASET")
fi

# Run training for all combinations
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        run_training "$model" "$dataset"
    done
done

# ============================================================================
# Step 3: Evaluation
# ============================================================================

echo "[Step 3/4] Running evaluation..."

# Run analysis script
echo "Running performance analysis..."
python scripts/analyze.py --results_dir "$OUTPUT_DIR"

# Run SNLI evaluation for each trained model
echo "Running SNLI evaluation..."
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        MODEL_DIR="${OUTPUT_DIR}/${model}_${dataset}"
        if [ -d "$MODEL_DIR" ]; then
            # Check for checkpoint (different naming conventions)
            CHECKPOINT=""
            if [ -f "${MODEL_DIR}/${model}_best.pt" ]; then
                CHECKPOINT="${MODEL_DIR}/${model}_best.pt"
            elif [ -f "${MODEL_DIR}/best_model.pt" ]; then
                CHECKPOINT="${MODEL_DIR}/best_model.pt"
            fi
            if [ -f "$CHECKPOINT" ]; then
                echo "Evaluating $model trained on $dataset on SNLI..."
                python scripts/eval_snli.py \
                    --model_path "$CHECKPOINT" \
                    --model_type "$model" \
                    --output_file "${MODEL_DIR}/snli_results.json" \
                    --max_samples 1000  # Limit for speed
            else
                echo "Warning: Checkpoint not found for $model on $dataset"
            fi
        fi
    done
done

# ============================================================================
# Step 4: Results Summary
# ============================================================================

echo "[Step 4/4] Training complete!"
echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Check the analysis results in: $OUTPUT_DIR/analysis/"
echo ""
echo "To run inference on trained models:"
echo "  ./speedrun-inference.sh [model_path]"
echo ""
echo "=========================================="