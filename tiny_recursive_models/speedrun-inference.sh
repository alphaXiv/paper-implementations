#!/bin/bash

set -e  # Exit on error

# ============================================================================
# TinyRecursiveModels (TRM) - Inference & Evaluation Script
# ============================================================================
# This script provides a one-file solution for evaluating pre-trained TRM
# models from HuggingFace on ARC-AGI, Sudoku, and Maze tasks.
#
# Usage: ./speedrun-inference.sh [TASK]
#   TASK: arc1 | arc2 | sudoku | maze | all
#
# Examples:
#   ./speedrun-inference.sh arc1    # ARC-AGI-1 full evaluation
#   ./speedrun-inference.sh maze    # Maze-Hard full evaluation
#   ./speedrun-inference.sh all     # All tasks full evaluation
# ============================================================================

# Detect number of GPUs dynamically
if command -v nvidia-smi &> /dev/null; then
    DETECTED_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    DETECTED_GPUS=0
fi

# Configuration
TASK=${1:-"arc1"}  # Default to ARC-AGI-1

# Determine GPU settings
if [ "$DETECTED_GPUS" -gt 1 ]; then
    NUM_GPUS=$DETECTED_GPUS
else
    NUM_GPUS=1
fi

echo "=========================================="
echo "TinyRecursiveModels Inference"
echo "=========================================="
echo "Detected GPUs: $DETECTED_GPUS"
echo "Task: $TASK"
echo "Using GPUs: $NUM_GPUS"
echo "=========================================="
echo ""

# ============================================================================
# Step 0: Environment Setup
# ============================================================================

echo "[Step 0/3] Setting up environment..."

# Check if uv is installed, install if not
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "uv installed successfully!"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment with uv..."
    uv venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install PyTorch with CUDA 12.8 support
echo "Installing PyTorch (CUDA 12.8)..."
uv pip install --pre --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Install project dependencies
echo "Installing project dependencies..."
uv pip install -e .

# Create necessary directories
mkdir -p data checkpoints logs results

# Login to Weights & Biases if API key is set
if [ -n "$WANDB_API_KEY" ]; then
    echo "Logging in to Weights & Biases..."
    wandb login "$WANDB_API_KEY"
    echo "W&B login successful!"
fi

echo "Environment setup complete!"
echo ""

# ============================================================================
# Step 1: Dataset Building
# ============================================================================

build_arc1_dataset() {
    if [ -d "data/arc1concept-aug-1000/test" ]; then
        echo "ARC-AGI-1 dataset already exists, skipping build..."
        return
    fi
    
    echo "[Step 1/3] Building ARC-AGI-1 dataset..."
    python -m trtiny_recursive_modelsm.data.build_arc_dataset \
        --input-file-prefix kaggle/combined/arc-agi \
        --output-dir data/arc1concept-aug-1000 \
        --subsets training evaluation concept \
        --test-set-name evaluation
    echo "ARC-AGI-1 dataset built successfully!"
    echo ""
}

build_arc2_dataset() {
    if [ -d "data/arc2concept-aug-1000/test" ]; then
        echo "ARC-AGI-2 dataset already exists, skipping build..."
        return
    fi
    
    echo "[Step 1/3] Building ARC-AGI-2 dataset..."
    python -m tiny_recursive_models.data.build_arc_dataset \
        --input-file-prefix kaggle/combined/arc-agi \
        --output-dir data/arc2concept-aug-1000 \
        --subsets training2 evaluation2 concept \
        --test-set-name evaluation2
    echo "ARC-AGI-2 dataset built successfully!"
    echo ""
}

build_sudoku_dataset() {
    if [ -d "data/sudoku-extreme-1k-aug-1000/test" ]; then
        echo "Sudoku-Extreme dataset already exists, skipping build..."
        return
    fi
    
    echo "[Step 1/3] Building Sudoku-Extreme dataset..."
    python -m tiny_recursive_models.data.build_sudoku_dataset \
        --output-dir data/sudoku-extreme-1k-aug-1000 \
        --subsample-size 1000 \
        --num-aug 1000
    echo "Sudoku-Extreme dataset built successfully!"
    echo ""
}

build_maze_dataset() {
    if [ -d "data/maze-30x30-hard-1k/test" ]; then
        echo "Maze-Hard dataset already exists, skipping build..."
        return
    fi
    
    echo "[Step 1/3] Building Maze-Hard 30x30 dataset..."
    python -m tiny_recursive_models.data.build_maze_dataset
    echo "Maze-Hard dataset built successfully!"
    echo ""
}

# ============================================================================
# Step 2: Evaluation Functions
# ============================================================================

evaluate_arc1() {
    echo "[Step 2/3] Evaluating ARC-AGI-1 model from HuggingFace..."
    
    local checkpoint="alphaxiv/trm-model-arc-agi-1/step_259320_arc_ag1_attn_type_h3l4"
    local dataset="data/arc1concept-aug-1000"
    local outdir="checkpoints/arc1_eval_inference"
    
    echo "Running full evaluation..."
    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node=$NUM_GPUS scripts/run_eval_only.py \
            --checkpoint $checkpoint \
            --dataset $dataset \
            --outdir $outdir \
            --global-batch-size 1024 \
            --apply-ema
    else
        python scripts/run_eval_only.py \
            --checkpoint $checkpoint \
            --dataset $dataset \
            --outdir $outdir \
            --apply-ema
    fi
    
    echo "ARC-AGI-1 evaluation complete!"
    echo ""
}

evaluate_arc2() {
    echo "[Step 2/3] Note: ARC-AGI-2 pre-trained model not available on HuggingFace."
    echo "Please train your own model using ./speedrun.sh arc2"
    echo ""
}

evaluate_sudoku() {
    echo "[Step 2/3] Evaluating Sudoku model from HuggingFace..."
    
    local checkpoint="alphaxiv/trm-model-sudoku"
    local dataset="data/sudoku-extreme-1k-aug-1000"
    local outdir="checkpoints/sudoku_eval_inference"
    
    echo "Running full evaluation..."
    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node=$NUM_GPUS scripts/run_eval_only.py \
            --checkpoint $checkpoint \
            --dataset $dataset \
            --outdir $outdir \
            --global-batch-size 1536 \
            --apply-ema
    else
        python scripts/run_eval_only.py \
            --checkpoint $checkpoint \
            --dataset $dataset \
            --outdir $outdir \
            --apply-ema
    fi
    
    echo "Sudoku evaluation complete!"
    echo ""
}

evaluate_maze() {
    echo "[Step 2/3] Evaluating Maze-Hard model from HuggingFace..."
    
    local checkpoint="alphaxiv/trm-model-maze/maze_hard_step_32550"
    local dataset="data/maze-30x30-hard-1k"
    local outdir="checkpoints/maze_eval_inference"
    
    echo "Running full evaluation..."
    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node=$NUM_GPUS scripts/run_eval_only.py \
            --checkpoint $checkpoint \
            --dataset $dataset \
            --outdir $outdir \
            --global-batch-size 1536 \
            --apply-ema
    else
        python scripts/run_eval_only.py \
            --checkpoint $checkpoint \
            --dataset $dataset \
            --outdir $outdir \
            --apply-ema
    fi
    
    echo "Maze-Hard evaluation complete!"
    echo ""
}

# ============================================================================
# Step 3: Results Summary
# ============================================================================

show_results() {
    echo "[Step 3/3] Evaluation Summary"
    echo "=========================================="
    echo "Full evaluation completed!"
    echo "Results saved in checkpoints/*_eval_inference/"
    echo ""
    echo "Check the output files for:"
    echo "  - exact_accuracy: Overall accuracy"
    echo "  - per-task metrics and predictions"
    echo "=========================================="
}

# ============================================================================
# Main Execution
# ============================================================================

case "$TASK" in
    arc1)
        build_arc1_dataset
        evaluate_arc1
        show_results
        ;;
    arc2)
        build_arc2_dataset
        evaluate_arc2
        ;;
    sudoku)
        build_sudoku_dataset
        evaluate_sudoku
        show_results
        ;;
    maze)
        build_maze_dataset
        evaluate_maze
        show_results
        ;;
    all)
        echo "Running all evaluations..."
        build_arc1_dataset
        build_sudoku_dataset
        build_maze_dataset
        evaluate_arc1
        evaluate_sudoku
        evaluate_maze
        show_results
        ;;
    *)
        echo "Unknown task: $TASK"
        echo "Valid tasks: arc1, arc2, sudoku, maze, all"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Inference pipeline complete!"
echo "=========================================="
