
#!/bin/bash

set -e  # Exit on error

# ============================================================================
# Grassmann Flows - Complete Training & Evaluation Pipeline
# ============================================================================
# This script provides a one-file solution for training and evaluating
# Attention Is Not All You Need models on various datasets.
#
# Usage: ./speedrun.sh [MODEL] [DATASET] [MODE]
#   MODEL: transformer | grassmann | all
#   DATASET: wikitext | snli | all
#   MODE: train | eval | both (default: both)
#
# Paper Specifications:
#   - Wikitext-2: Trains with BOTH L=128 and L=256 block sizes (paper spec)
#   - Default: 6-layer models (N=6)
#   - For 12-layer models: LAYER_DEPTHS_OVERRIDE=6,12 ./speedrun.sh all wikitext
#
# Examples:
#   ./speedrun.sh grassmann wikitext     # Train & eval Grassmann on Wikitext (L=128 & L=256)
#   ./speedrun.sh transformer snli       # Train & eval Transformer on SNLI
#   ./speedrun.sh all wikitext           # Train & eval both models on Wikitext (L=128 & L=256)
#   ./speedrun.sh all snli               # Train & eval both models on SNLI
#   ./speedrun.sh grassmann all          # Train & eval Grassmann on all datasets
#   ./speedrun.sh all all                # Train & eval all models on all datasets
#   
#   # Eval only mode (use existing checkpoints):
#   ./speedrun.sh all wikitext eval      # Eval only - all models on Wikitext
#   ./speedrun.sh grassmann snli eval    # Eval only - Grassmann on SNLI
#   
#   # Train only (skip eval):
#   ./speedrun.sh all wikitext train     # Train only - all models on Wikitext
#   
#   # For 12-layer models:
#   LAYER_DEPTHS_OVERRIDE=12 ./speedrun.sh all wikitext
#   
#   # For both 6 and 12 layer models:
#   LAYER_DEPTHS_OVERRIDE=6,12 ./speedrun.sh all wikitext
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
MODE=${3:-"both"}  # Default to both train and eval
NUM_GPUS=$DETECTED_GPUS  # Use all available GPUs

# Validate mode
if [[ "$MODE" != "train" && "$MODE" != "eval" && "$MODE" != "both" ]]; then
    echo "ERROR: Invalid MODE '$MODE'. Must be 'train', 'eval', or 'both'."
    exit 1
fi

echo "=========================================="
echo "Attention Is Not All You Need Training & Evaluation"
echo "=========================================="
echo "Detected GPUs: $DETECTED_GPUS"
echo "Model(s): $MODEL"
echo "Dataset(s): $DATASET"
echo "Mode: $MODE"
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
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment if it doesn't exist
VENV_DIR=".venv-attn-is-not-all-you-need"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    uv venv "$VENV_DIR"
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Install dependencies only on first creation
    echo "Installing dependencies..."
    uv pip install -e .
else
    echo "Virtual environment already exists, skipping dependency installation..."
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
fi

# Login to Weights & Biases - REQUIRED
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is not set!"
    echo "This script requires Weights & Biases for experiment tracking."
    echo "Please set your WANDB_API_KEY environment variable:"
    echo "  export WANDB_API_KEY=your_api_key_here"
    echo ""
    echo "Get your API key from: https://wandb.ai/authorize"
    exit 1
fi

echo "Logging in to Weights & Biases..."
wandb login "$WANDB_API_KEY"
echo "W&B login successful!"

# ============================================================================
# Step 1: Data Preparation
# ============================================================================

echo "[Step 1/4] Preparing data..."

# Data preparation happens automatically in the training scripts
echo "Data will be downloaded automatically during training..."

# ============================================================================
# Step 2: Training
# ============================================================================

if [[ "$MODE" == "eval" ]]; then
    # For eval-only mode, use the most recent output directory
    echo "[Step 2/4] Skipping training (eval-only mode)..."
    
    # Find the most recent output directory
    LATEST_OUTPUT=$(ls -td outputs/*/ 2>/dev/null | head -1)
    if [ -z "$LATEST_OUTPUT" ]; then
        echo "ERROR: No existing output directory found. Please train models first or specify a different mode."
        exit 1
    fi
    OUTPUT_DIR="${LATEST_OUTPUT%/}"  # Remove trailing slash
    echo "Using existing output directory: $OUTPUT_DIR"
    echo ""
    echo "Available model directories:"
    ls -1 "$OUTPUT_DIR" | grep -E "^(transformer|grassmann)_" | head -10
    echo ""
else
    echo "[Step 2/4] Starting training..."
    
    # Create output directory with timestamp
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    OUTPUT_DIR="outputs/${TIMESTAMP}"
fi

# Function to run training for a specific model and dataset
run_training() {
    local model=$1
    local dataset=$2
    local block_size=$3
    local num_layers=$4

    if [ "$dataset" = "wikitext" ]; then
        echo "Training $model on $dataset (L=$block_size, N=$num_layers layers)..."
        # Use the exact reproduction script for Wikitext-2
        python src/attn_is_not_all_you_need/train.py \
            --task wikitext \
            --model "$model" \
            --max-seq-len "$block_size" \
            --num-layers "$num_layers" \
            --epochs 20 \
            --output-dir "${OUTPUT_DIR}/${model}_${dataset}_L${block_size}_N${num_layers}"
    elif [ "$dataset" = "snli" ]; then
        echo "Training $model on $dataset..."
        # Use separate SNLI training script
        python src/attn_is_not_all_you_need/train_snli.py \
            --model "$model" \
            --epochs 20 \
            --output-dir "${OUTPUT_DIR}/${model}_${dataset}"
    fi
}

# Determine which models and datasets to run
if [ "$MODEL" = "all" ]; then
    MODELS=("transformer" "grassmann")
else
    MODELS=("$MODEL")
fi

if [ "$DATASET" = "all" ]; then
    DATASETS=("wikitext" "snli")
else
    DATASETS=("$DATASET")
fi

# Run training for all combinations
# Paper specs: L=128 and L=256 for Wikitext, N=6 or N=12 layers
BLOCK_SIZES=(128 256)  # Both block sizes from paper
LAYER_DEPTHS=(6 12)       # Default to 6 layers (can be overridden with env var)

# Allow user to specify layer depths via environment variable
if [ -n "$LAYER_DEPTHS_OVERRIDE" ]; then
    IFS=',' read -ra LAYER_DEPTHS <<< "$LAYER_DEPTHS_OVERRIDE"
fi

# Run training only if not in eval-only mode
if [[ "$MODE" != "eval" ]]; then
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            if [ "$dataset" = "wikitext" ]; then
                # For Wikitext, run with both block sizes (L=128 and L=256)
                for block_size in "${BLOCK_SIZES[@]}"; do
                    for num_layers in "${LAYER_DEPTHS[@]}"; do
                        run_training "$model" "$dataset" "$block_size" "$num_layers"
                    done
                done
            else
                # For SNLI, block size and layers don't apply
                run_training "$model" "$dataset" "" ""
            fi
        done
    done
fi

# ============================================================================
# Step 3: Evaluation
# ============================================================================

# Skip evaluation if in train-only mode
if [[ "$MODE" == "train" ]]; then
    echo "[Step 3/4] Skipping evaluation (train-only mode)..."
else
    echo "[Step 3/4] Running evaluation..."
    
    # Run analysis script
    echo "Running performance analysis..."
    python scripts/analyze.py --results_dir "$OUTPUT_DIR"
    
    # Evaluate each trained model on appropriate test datasets
    echo "Running test dataset evaluations..."
fi

# Only run evaluation loop if not in train-only mode
if [[ "$MODE" != "train" ]]; then
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        if [ "$dataset" = "wikitext" ]; then
            # Evaluate Wikitext models with both block sizes
            for block_size in "${BLOCK_SIZES[@]}"; do
                for num_layers in "${LAYER_DEPTHS[@]}"; do
                    MODEL_DIR="${OUTPUT_DIR}/${model}_${dataset}_L${block_size}_N${num_layers}"
                    if [ -d "$MODEL_DIR" ]; then
                        CHECKPOINT=""
                        # Only use best.pt checkpoint
                        if [ -f "${MODEL_DIR}/checkpoints/best.pt" ]; then
                            CHECKPOINT="${MODEL_DIR}/checkpoints/best.pt"
                        else
                            echo "WARNING: No best.pt checkpoint found in ${MODEL_DIR}/checkpoints/"
                            echo "         Skipping evaluation. Please ensure training completed successfully."
                            continue
                        fi
                        
                        if [ -n "$CHECKPOINT" ] && [ -f "$CHECKPOINT" ]; then
                            # Run Wikitext test evaluation (single run - eval is deterministic)
                            echo "Evaluating $model on Wikitext-2 test split (L=$block_size, N=$num_layers)..."
                            python src/attn_is_not_all_you_need/eval_wikitext.py \
                                --model_path "$CHECKPOINT" \
                                --model_type "$model" \
                                --max-seq-len "$block_size" \
                                --num-layers "$num_layers" \
                                --num_runs 1 \
                                --output_file "${MODEL_DIR}/wikitext_test_results.json"
                        fi
                    fi
                done
            done
        else
            # Evaluate SNLI models
            MODEL_DIR="${OUTPUT_DIR}/${model}_${dataset}"
            if [ -d "$MODEL_DIR" ]; then
                CHECKPOINT=""
                # Only use best.pt checkpoint
                if [ -f "${MODEL_DIR}/checkpoints/best.pt" ]; then
                    CHECKPOINT="${MODEL_DIR}/checkpoints/best.pt"
                else
                    echo "WARNING: No best.pt checkpoint found in ${MODEL_DIR}/checkpoints/"
                    echo "         Skipping evaluation. Please ensure training completed successfully."
                    continue
                fi
                
                if [ -n "$CHECKPOINT" ] && [ -f "$CHECKPOINT" ]; then
                    # SNLI test evaluation
                    echo "Evaluating $model on SNLI test split..."
                    python src/attn_is_not_all_you_need/eval_snli.py \
                        --model_path "$CHECKPOINT" \
                        --model_type "$model" \
                        --split test \
                        --num_runs 1 \
                        --output_file "${MODEL_DIR}/snli_test_results.json"
                else
                    echo "Warning: Checkpoint not found for $model on $dataset in $MODEL_DIR"
                fi
            else
                echo "Warning: Model directory not found: $MODEL_DIR"
            fi
        fi
    done
done
fi  # End of evaluation mode check

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
echo "Test Dataset Evaluations:"
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        if [ "$dataset" = "wikitext" ]; then
            for block_size in "${BLOCK_SIZES[@]}"; do
                for num_layers in "${LAYER_DEPTHS[@]}"; do
                    MODEL_DIR="${OUTPUT_DIR}/${model}_${dataset}_L${block_size}_N${num_layers}"
                    if [ -d "$MODEL_DIR" ]; then
                        echo "  $model (L=$block_size, N=$num_layers):"
                        if [ -f "${MODEL_DIR}/wikitext_test_results.json" ]; then
                            echo "    - Wikitext-2 test: ${MODEL_DIR}/wikitext_test_results.json"
                        fi
                        if [ -f "${MODEL_DIR}/snli_test_results.json" ]; then
                            echo "    - SNLI test: ${MODEL_DIR}/snli_test_results.json"
                        fi
                    fi
                done
            done
        else
            MODEL_DIR="${OUTPUT_DIR}/${model}_${dataset}"
            if [ -d "$MODEL_DIR" ]; then
                echo "  $model:"
                if [ -f "${MODEL_DIR}/snli_test_results.json" ]; then
                    echo "    - SNLI test: ${MODEL_DIR}/snli_test_results.json"
                fi
                if [ -f "${MODEL_DIR}/snli_val_results.json" ]; then
                    echo "    - SNLI validation: ${MODEL_DIR}/snli_val_results.json"
                fi
            fi
        fi
    done
done
echo ""
echo "To run inference on trained models:"
echo "  ./speedrun-inference.sh [model_path]"
echo ""
echo "=========================================="
