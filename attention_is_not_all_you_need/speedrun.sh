
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
# HuggingFace Integration (Eval Mode):
#   - By default, eval mode downloads models from HuggingFace
#   - Falls back to local models if download fails
#   - Control with environment variables:
#     USE_HF_MODELS=false ./speedrun.sh all wikitext eval  # Force local models
#     HF_MODEL_REPO=username/repo ./speedrun.sh all wikitext eval  # Custom repo
#
# Examples:
#   ./speedrun.sh grassmann wikitext     # Train & eval Grassmann on Wikitext (L=128 & L=256)
#   ./speedrun.sh transformer snli       # Train & eval Transformer on SNLI
#   ./speedrun.sh all wikitext           # Train & eval both models on Wikitext (L=128 & L=256)
#   ./speedrun.sh all snli               # Train & eval both models on SNLI
#   ./speedrun.sh grassmann all          # Train & eval Grassmann on all datasets
#   ./speedrun.sh all all                # Train & eval all models on all datasets
#   
#   # Eval only mode (downloads from HuggingFace by default):
#   ./speedrun.sh all wikitext eval      # Eval only - downloads from HF
#   ./speedrun.sh grassmann snli eval    # Eval only - Grassmann on SNLI from HF
#   
#   # Eval with local models only:
#   USE_HF_MODELS=false ./speedrun.sh all wikitext eval
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
# Configuration for model variants
# ============================================================================

# Paper specs: L=128 and L=256 for Wikitext, N=6 or N=12 layers
BLOCK_SIZES=(128 256)  # Both block sizes from paper
LAYER_DEPTHS=(6 12)    # Default to both 6 and 12 layers

# Allow user to specify layer depths via environment variable
if [ -n "$LAYER_DEPTHS_OVERRIDE" ]; then
    IFS=',' read -ra LAYER_DEPTHS <<< "$LAYER_DEPTHS_OVERRIDE"
fi

# ============================================================================
# Step 2: Training / Model Loading
# ============================================================================

if [[ "$MODE" == "eval" ]]; then
    # For eval-only mode, try to download from HuggingFace first, then fall back to local
    echo "[Step 2/4] Skipping training (eval-only mode)..."
    
    # HuggingFace repository configuration
    HF_REPO="${HF_MODEL_REPO:-alphaXiv/attention-is-not-all-you-need-models}"
    USE_HF_MODELS="${USE_HF_MODELS:-true}"  # Default to using HF models
    
    # Create output directory for downloaded models
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    OUTPUT_DIR="outputs/${TIMESTAMP}_hf_eval"
    
    if [[ "$USE_HF_MODELS" == "true" ]]; then
        echo "Attempting to download models from HuggingFace: $HF_REPO"
        mkdir -p "$OUTPUT_DIR"
        
        # Function to download a model from HuggingFace
        download_hf_model() {
            local model_name=$1
            local target_dir="${OUTPUT_DIR}/${model_name}"
            
            echo "  Downloading $model_name..."
            mkdir -p "$target_dir/checkpoints"
            
            # Download checkpoint
            if python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='$HF_REPO', filename='${model_name}/checkpoints/best.pt', local_dir='$OUTPUT_DIR', local_dir_use_symlinks=False)"; then
                echo "    ✅ Downloaded checkpoint for $model_name"
            else
                echo "    ⚠️  Failed to download $model_name from HuggingFace"
                return 1
            fi
        }
        
        # Download models based on requested configuration
        HF_DOWNLOAD_SUCCESS=true
        
        # Determine which models to download
        if [ "$MODEL" = "all" ]; then
            HF_MODELS=("transformer" "grassmann")
        else
            HF_MODELS=("$MODEL")
        fi
        
        if [ "$DATASET" = "all" ]; then
            HF_DATASETS=("wikitext" "snli")
        else
            HF_DATASETS=("$DATASET")
        fi
        
        # Download all requested models
        for hf_model in "${HF_MODELS[@]}"; do
            for hf_dataset in "${HF_DATASETS[@]}"; do
                if [ "$hf_dataset" = "wikitext" ]; then
                    for block_size in "${BLOCK_SIZES[@]}"; do
                        for num_layers in "${LAYER_DEPTHS[@]}"; do
                            model_name="${hf_model}_${hf_dataset}_L${block_size}_N${num_layers}"
                            if ! download_hf_model "$model_name"; then
                                HF_DOWNLOAD_SUCCESS=false
                            fi
                        done
                    done
                else
                    model_name="${hf_model}_${hf_dataset}"
                    if ! download_hf_model "$model_name"; then
                        HF_DOWNLOAD_SUCCESS=false
                    fi
                fi
            done
        done
        
        if [ "$HF_DOWNLOAD_SUCCESS" = false ]; then
            echo ""
            echo "⚠️  Some models failed to download from HuggingFace"
            echo "Falling back to local models..."
            USE_HF_MODELS=false
        else
            echo "✅ All models downloaded successfully from HuggingFace"
        fi
    fi
    
    # Fall back to local models if HF download failed or disabled
    if [[ "$USE_HF_MODELS" == "false" ]] || [[ "$HF_DOWNLOAD_SUCCESS" == "false" ]]; then
        # Find the most recent output directory
        LATEST_OUTPUT=$(ls -td outputs/*/ 2>/dev/null | head -1)
        if [ -z "$LATEST_OUTPUT" ]; then
            echo "ERROR: No existing output directory found and HuggingFace download failed."
            echo "Please train models first or check your HuggingFace repository."
            echo ""
            echo "To use HuggingFace models, ensure:"
            echo "  1. Models are uploaded: python scripts/upload_to_hf.py"
            echo "  2. Repository is public or you're logged in: huggingface-cli login"
            echo "  3. HF_MODEL_REPO is set correctly (default: alphaXiv/attention-is-not-all-you-need-models)"
            exit 1
        fi
        OUTPUT_DIR="${LATEST_OUTPUT%/}"  # Remove trailing slash
        echo "Using existing local output directory: $OUTPUT_DIR"
    fi
    
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
# Configuration already set above

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
                            # Run Wikitext validation evaluation (single run - eval is deterministic)
                            echo "Evaluating $model on Wikitext-2 validation split (L=$block_size, N=$num_layers)..."
                            python src/attn_is_not_all_you_need/eval_wikitext.py \
                                --model_path "$CHECKPOINT" \
                                --model_type "$model" \
                                --max-seq-len "$block_size" \
                                --num-layers "$num_layers" \
                                --num_runs 1 \
                                --output_file "${MODEL_DIR}/wikitext_validation_results.json"
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
                    # SNLI validation evaluation
                    echo "Evaluating $model on SNLI validation split..."
                    python src/attn_is_not_all_you_need/eval_snli.py \
                        --model_path "$CHECKPOINT" \
                        --model_type "$model" \
                        --split validation \
                        --num_runs 1 \
                        --output_file "${MODEL_DIR}/snli_validation_results.json"
                    
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
echo "Val & Test Dataset Evaluations:"
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        if [ "$dataset" = "wikitext" ]; then
            for block_size in "${BLOCK_SIZES[@]}"; do
                for num_layers in "${LAYER_DEPTHS[@]}"; do
                    MODEL_DIR="${OUTPUT_DIR}/${model}_${dataset}_L${block_size}_N${num_layers}"
                    if [ -d "$MODEL_DIR" ]; then
                        echo "  $model (L=$block_size, N=$num_layers):"
                        if [ -f "${MODEL_DIR}/wikitext_validation_results.json" ]; then
                            echo "    - Wikitext-2 validation: ${MODEL_DIR}/wikitext_validation_results.json"
                        fi
                        if [ -f "${MODEL_DIR}/snli_validation_results.json" ]; then
                            echo "    - SNLI validation: ${MODEL_DIR}/snli_validation_results.json"
                        fi
                    fi
                done
            done
        else
            MODEL_DIR="${OUTPUT_DIR}/${model}_${dataset}"
            if [ -d "$MODEL_DIR" ]; then
                echo "  $model:"
                if [ -f "${MODEL_DIR}/snli_validation_results.json" ]; then
                    echo "    - SNLI validation: ${MODEL_DIR}/snli_validation_results.json"
                fi
                if [ -f "${MODEL_DIR}/snli_test_results.json" ]; then
                    echo "    - SNLI test: ${MODEL_DIR}/snli_test_results.json"
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
