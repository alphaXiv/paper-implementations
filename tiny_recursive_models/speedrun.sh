#!/bin/bash

set -e  # Exit on error

# ============================================================================
# TinyRecursiveModels (TRM) - Complete Training & Evaluation Pipeline
# ============================================================================
# This script provides a one-file solution for building datasets, training,
# and evaluating TRM models on ARC-AGI, Sudoku, and Maze tasks.
#
# Usage: ./speedrun.sh [TASK]
#   TASK: arc1 | arc2 | sudoku | maze | all
#
# Examples:
#   ./speedrun.sh arc1              # ARC-AGI-1 on all available GPUs
#   ./speedrun.sh sudoku            # Sudoku on all available GPUs
#   ./speedrun.sh all               # All tasks on all available GPUs
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
TASK=${1:-"arc1"}  # Default to ARC-AGI-1
NUM_GPUS=$DETECTED_GPUS  # Use all available GPUs

echo "=========================================="
echo "TinyRecursiveModels Training & Evaluation"
echo "=========================================="
echo "Detected GPUs: $DETECTED_GPUS"
echo "Task: $TASK"
echo "Using GPUs: $NUM_GPUS"
echo "=========================================="
echo ""

# ============================================================================
# Step 0: Environment Setup
# ============================================================================

echo "[Step 0/4] Setting up environment..."

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
mkdir -p data checkpoints logs results wandb

# Login to Weights & Biases
if [ -n "$WANDB_API_KEY" ]; then
    echo "Logging in to Weights & Biases..."
    wandb login "$WANDB_API_KEY"
    echo "W&B login successful!"
else
    echo "Weights & Biases API key not found in environment variable WANDB_API_KEY"
    echo "Would you like to enter your W&B API key now? (y/n)"
    read -p "" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Enter your W&B API key:"
        read -s wandb_key
        echo "Logging in to Weights & Biases..."
        wandb login "$wandb_key"
        echo "W&B login successful!"
        export WANDB_API_KEY="$wandb_key"
    else
        exit 1
    fi
fi

echo "Environment setup complete!"
echo ""

# ============================================================================
# Step 1: Dataset Building
# ============================================================================

build_arc1_dataset() {
    echo "[Step 1/4] Building ARC-AGI-1 dataset..."
    python -m src.tiny_recursive_models.data.build_arc_dataset \
        --input-file-prefix kaggle/combined/arc-agi \
        --output-dir data/arc1concept-aug-1000 \
        --subsets training evaluation concept \
        --test-set-name evaluation
    echo "ARC-AGI-1 dataset built successfully!"
    echo ""
}

build_arc2_dataset() {
    echo "[Step 1/4] Building ARC-AGI-2 dataset..."
    python -m src.tiny_recursive_models.data.build_arc_dataset \
        --input-file-prefix kaggle/combined/arc-agi \
        --output-dir data/arc2concept-aug-1000 \
        --subsets training2 evaluation2 concept \
        --test-set-name evaluation2
    echo "ARC-AGI-2 dataset built successfully!"
    echo ""
}

build_sudoku_dataset() {
    echo "[Step 1/4] Building Sudoku-Extreme dataset..."
    python -m src.tiny_recursive_models.data.build_sudoku_dataset \
        --output-dir data/sudoku-extreme-1k-aug-1000 \
        --subsample-size 1000 \
        --num-aug 1000
    echo "Sudoku-Extreme dataset built successfully!"
    echo ""
}

build_maze_dataset() {
    echo "[Step 1/4] Building Maze-Hard 30x30 dataset..."
    python -m src.tiny_recursive_models.data.build_maze_dataset
    echo "Maze-Hard dataset built successfully!"
    echo ""
}

# ============================================================================
# Step 2: Training Functions
# ============================================================================

train_arc1() {
    local run_name="pretrain_att_arc1concept_$(date +%Y%m%d_%H%M%S)"
    local batch_size=1536
    local nproc=$NUM_GPUS
    
    echo "[Step 2/4] Training ARC-AGI-1 model..."
    echo "Run name: $run_name"
    echo "Batch size: $batch_size"
    echo "GPUs: $nproc"
    echo ""
    
    torchrun --nproc-per-node $nproc --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
        scripts/train.py \
        arch=trm \
        data_paths="[data/arc1concept-aug-1000]" \
        arch.L_layers=2 \
        arch.H_cycles=3 arch.L_cycles=6 \
        lr=2e-4 weight_decay=0.1 puzzle_emb_lr=1e-2 \
        global_batch_size=$batch_size lr_warmup_steps=4000 \
        epochs=100000 eval_interval=5000 checkpoint_every_eval=True \
        +run_name=${run_name} ema=True
    
    echo "ARC-AGI-1 training complete!"
    LAST_CHECKPOINT="checkpoints/TRM/${run_name}"
    LAST_DATASET="data/arc1concept-aug-1000"
    echo ""
}

train_arc2() {
    local run_name="pretrain_att_arc2concept_$(date +%Y%m%d_%H%M%S)"
    local batch_size=1536
    local nproc=$NUM_GPUS
    
    echo "[Step 2/4] Training ARC-AGI-2 model..."
    echo "Run name: $run_name"
    echo "Batch size: $batch_size"
    echo "GPUs: $nproc"
    echo ""
    
    torchrun --nproc-per-node $nproc --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
        scripts/train.py \
        arch=trm \
        data_paths="[data/arc2concept-aug-1000]" \
        arch.L_layers=2 \
        arch.H_cycles=3 arch.L_cycles=6 \
        lr=2e-4 weight_decay=0.1 puzzle_emb_lr=1e-2 \
        global_batch_size=$batch_size lr_warmup_steps=4000 \
        epochs=100000 eval_interval=5000 checkpoint_every_eval=True \
        +run_name=${run_name} ema=True
    
    echo "ARC-AGI-2 training complete!"
    LAST_CHECKPOINT="checkpoints/TRM/${run_name}"
    LAST_DATASET="data/arc2concept-aug-1000"
    echo ""
}

train_sudoku_mlp() {
    local run_name="pretrain_mlp_t_sudoku_$(date +%Y%m%d_%H%M%S)"
    local batch_size=1536
    local nproc=$NUM_GPUS
    
    echo "[Step 2/4] Training Sudoku model (MLP-Tiny variant)..."
    echo "Run name: $run_name"
    echo "Batch size: $batch_size"
    echo "GPUs: $nproc"
    echo ""
    
    torchrun --nproc-per-node $nproc --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
        scripts/train.py \
        arch=trm \
        data_paths="[data/sudoku-extreme-1k-aug-1000]" \
        evaluators="[]" \
        epochs=50000 eval_interval=5000 \
        lr=2e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
        arch.mlp_t=True arch.pos_encodings=none \
        arch.L_layers=2 \
        arch.H_cycles=3 arch.L_cycles=6 \
        lr_warmup_steps=4000 \
        global_batch_size=$batch_size \
        checkpoint_every_eval=True \
        +run_name=${run_name} ema=True
    
    echo "Sudoku training complete!"
    LAST_CHECKPOINT="checkpoints/TRM/${run_name}"
    LAST_DATASET="data/sudoku-extreme-1k-aug-1000"
    echo ""
}

train_sudoku_att() {
    local run_name="pretrain_att_sudoku_$(date +%Y%m%d_%H%M%S)"
    local batch_size=1536
    local nproc=$NUM_GPUS
    
    echo "[Step 2/4] Training Sudoku model (Attention variant)..."
    echo "Run name: $run_name"
    echo "Batch size: $batch_size"
    echo "GPUs: $nproc"
    echo ""
    
    torchrun --nproc-per-node $nproc --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
        scripts/train.py \
        arch=trm \
        data_paths="[data/sudoku-extreme-1k-aug-1000]" \
        evaluators="[]" \
        epochs=50000 eval_interval=5000 \
        lr=2e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
        arch.L_layers=2 \
        arch.H_cycles=3 arch.L_cycles=6 \
        lr_warmup_steps=4000 \
        global_batch_size=$batch_size \
        checkpoint_every_eval=True \
        +run_name=${run_name} ema=True
    
    echo "Sudoku training complete!"
    LAST_CHECKPOINT="checkpoints/TRM/${run_name}"
    LAST_DATASET="data/sudoku-extreme-1k-aug-1000"
    echo ""
}

train_maze() {
    local run_name="pretrain_att_maze30x30_$(date +%Y%m%d_%H%M%S)"
    local batch_size=1536
    local nproc=$NUM_GPUS
    
    echo "[Step 2/4] Training Maze-Hard 30x30 model..."
    echo "Run name: $run_name"
    echo "Batch size: $batch_size"
    echo "GPUs: $nproc"
    echo ""
    
    torchrun --nproc-per-node $nproc --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
        scripts/train.py \
        arch=trm \
        data_paths="[data/maze-30x30-hard-1k]" \
        evaluators="[]" \
        epochs=50000 eval_interval=5000 \
        lr=2e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
        arch.L_layers=2 \
        arch.H_cycles=3 arch.L_cycles=4 \
        global_batch_size=$batch_size lr_warmup_steps=4000 \
        checkpoint_every_eval=True \
        +run_name=${run_name} ema=True
    
    echo "Maze-Hard training complete!"
    LAST_CHECKPOINT="checkpoints/TRM/${run_name}"
    LAST_DATASET="data/maze-30x30-hard-1k"
    echo ""
}

# ============================================================================
# Step 3: Evaluation Function
# ============================================================================

evaluate_model() {
    local checkpoint_path=$1
    local dataset_path=$2
    local eval_name=$(basename $checkpoint_path)
    local nproc=$NUM_GPUS
    local batch_size=1536
    
    echo "[Step 3/4] Evaluating model..."
    echo "Checkpoint: $checkpoint_path"
    echo "Dataset: $dataset_path"
    echo "Output directory: checkpoints/eval_${eval_name}"
    echo ""
    
    # Find the latest checkpoint directory
    if [ -d "$checkpoint_path" ]; then
        # Look for the latest checkpoint subdirectory
        local latest_ckpt=$(find "$checkpoint_path" -type d -name "step_*" | sort -V | tail -1)
        
        if [ -z "$latest_ckpt" ]; then
            echo "WARNING: No checkpoint found in $checkpoint_path, skipping evaluation"
            return
        fi
        
        torchrun --nproc-per-node=$nproc scripts/run_eval_only.py \
            --checkpoint "$latest_ckpt" \
            --dataset "$dataset_path" \
            --outdir "checkpoints/eval_${eval_name}" \
            --eval-save-outputs inputs labels puzzle_identifiers preds \
            --global-batch-size $batch_size \
            --apply-ema
        
        echo "Evaluation complete! Results saved to checkpoints/eval_${eval_name}"
    else
        echo "WARNING: Checkpoint path $checkpoint_path does not exist, skipping evaluation"
    fi
    echo ""
}

# ============================================================================
# Step 4: Smoke Test Function (Quick Verification)
# ============================================================================

smoke_test() {
    echo "[Step 4/4] Running smoke test (one batch) to verify setup..."
    echo ""
    
    # Use pretrained model from HuggingFace for quick test
    echo "Testing with pre-trained Maze model from HuggingFace..."
    python scripts/run_eval_only.py \
        --checkpoint alphaxiv/trm-model-maze/maze_hard_step_32550 \
        --dataset data/maze-30x30-hard-1k \
        --one-batch
    
    echo "Smoke test complete!"
    echo ""
}

# ============================================================================
# Main Execution Logic
# IMPORTANT: when re-running you may comment out the building dataset steps
# ============================================================================

case $TASK in
    arc1)
        build_arc1_dataset
        train_arc1
        evaluate_model "$LAST_CHECKPOINT" "$LAST_DATASET"
        ;;
    
    arc2)
        build_arc2_dataset
        train_arc2
        evaluate_model "$LAST_CHECKPOINT" "$LAST_DATASET"
        ;;
    
    sudoku)
        build_sudoku_dataset
        echo "Choose Sudoku variant: [1] MLP-Tiny  [2] Attention  [3] Both"
        read -p "Enter choice (1/2/3) [default: 2]: " sudoku_variant
        sudoku_variant=${sudoku_variant:-2}
        
        case $sudoku_variant in
            1)
                train_sudoku_mlp
                ;;
            2)
                train_sudoku_att
                ;;
            3)
                train_sudoku_mlp
                train_sudoku_att
                ;;
            *)
                echo "Invalid choice, using Attention variant"
                train_sudoku_att
                ;;
        esac
        evaluate_model "$LAST_CHECKPOINT" "$LAST_DATASET"
        ;;
    
    maze)
        build_maze_dataset
        train_maze
        evaluate_model "$LAST_CHECKPOINT" "$LAST_DATASET"
        ;;
    
    all)
        echo "Running all tasks (this will take a VERY long time)..."
        echo ""
        
        build_arc1_dataset
        build_arc2_dataset
        build_sudoku_dataset
        build_maze_dataset
        
        train_arc1
        evaluate_model "$LAST_CHECKPOINT" "$LAST_DATASET"
        
        train_arc2
        evaluate_model "$LAST_CHECKPOINT" "$LAST_DATASET"
        
        train_sudoku_att
        evaluate_model "$LAST_CHECKPOINT" "$LAST_DATASET"
        
        train_maze
        evaluate_model "$LAST_CHECKPOINT" "$LAST_DATASET"
        ;;
    
    smoke-test)
        build_maze_dataset
        smoke_test
        ;;
    
    *)
        echo "ERROR: Invalid task '$TASK'"
        echo ""
        echo "Usage: $0 [TASK]"
        echo ""
        echo "Available tasks:"
        echo "  arc1        - Train and evaluate on ARC-AGI-1"
        echo "  arc2        - Train and evaluate on ARC-AGI-2"
        echo "  sudoku      - Train and evaluate on Sudoku-Extreme"
        echo "  maze        - Train and evaluate on Maze-Hard 30x30"
        echo "  all         - Run all tasks sequentially"
        echo "  smoke-test  - Quick verification test with pre-trained model"
        echo ""
        echo "Examples:"
        echo "  $0 arc1"
        echo "  $0 sudoku"
        echo "  $0 all"
        echo "  $0 smoke-test"
        exit 1
        ;;
esac

# ============================================================================
# Final Summary
# ============================================================================

echo "=========================================="
echo "Training and Evaluation Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Task: $TASK"
echo "  GPUs used: $NUM_GPUS"
if [ -n "$LAST_CHECKPOINT" ]; then
    echo "  Last checkpoint: $LAST_CHECKPOINT"
    echo "  Evaluation results: checkpoints/eval_$(basename $LAST_CHECKPOINT)"
fi
echo ""
echo "Next steps:"
echo "  - Check training logs in: logs/"
echo "  - View checkpoints in: checkpoints/"
echo "  - Review evaluation results in: checkpoints/eval_*/"
if command -v wandb &> /dev/null; then
    echo "  - View training metrics in W&B dashboard"
fi
echo ""
echo "To evaluate a pretrained model from HuggingFace:"
echo "  python scripts/run_eval_only.py \\"
echo "    --checkpoint alphaxiv/trm-model-arc-agi-1/step_259320_arc_ag1_attn_type_h3l4 \\"
echo "    --dataset data/arc1concept-aug-1000 \\"
echo "    --apply-ema"
echo ""
echo "=========================================="

