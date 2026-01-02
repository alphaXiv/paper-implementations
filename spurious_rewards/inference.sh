#!/bin/bash

# Spurious Rewards Training and Evaluation Speedrun Script

set -e

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN environment variable is not set."
    echo "You need to set it to access gated models like Qwen/Qwen2.5-Math-1.5B"
    echo "Example: export HF_TOKEN='your_token_here'"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    exit 1
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/projects/miniconda/en/latest/"
    exit 1
fi

# Create conda environment if it doesn't exist
ENV_NAME="spurious-rewards-env"
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.10 -y

    # Activate conda environment
    echo "Activating conda environment: $ENV_NAME"
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME

    # Navigate to code directory
    cd src/spurious_rewards/code

    # Install PyTorch and dependencies
    echo "Installing PyTorch and dependencies..."
    pip install -r requirements.txt
    pip uninstall vllm -y
    pip install vllm==0.7.2

    # Modify setup.py to pin vllm version for CUDA 12.8 compatibility
    # Change extras_require vllm from ["vllm"] to ["vllm==0.7.2"]
    sed -i 's/"vllm": \["vllm"\]/"vllm": ["vllm==0.7.2"]/g' setup.py

    # Install flash_attn with proper build isolation handling
    pip install flash_attn==2.7.0.post2 --no-build-isolation

    # Install the package in editable mode
    pip install -e .
    
    # Install matplotlib for plotting
    pip install matplotlib

    # Go back to the project root
    cd ../../../
else
    # Activate conda environment
    echo "Activating existing conda environment: $ENV_NAME"
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
fi

# Check for data directory
if [ -d "src/spurious_rewards/code/data" ]; then
    echo "Data directory already exists. Skipping data download."
else
    echo "Data directory not found. Proceeding to download data..."

    # Navigate to code directory
    cd src/spurious_rewards/code

    # Get the data from RLVR repository
    echo "Cloning RLVR-Fine-Tuning-Spurious-Rewards for data..."

    # Remove any existing clone directory
    if [ -d "RLVR-Fine-Tuning-Spurious-Rewards" ]; then
        echo "Removing existing RLVR-Fine-Tuning-Spurious-Rewards directory..."
        rm -rf RLVR-Fine-Tuning-Spurious-Rewards
    fi

    # Clone the repository
    git clone https://github.com/alphaXiv/RLVR-Fine-Tuning-Spurious-Rewards.git
    cd RLVR-Fine-Tuning-Spurious-Rewards/code

    # Move data folder to parent directory
    echo "Moving data folder..."
    if [ -d "data" ]; then
        mv data ../../data
        echo "Data folder moved successfully"
    else
        echo "ERROR: data folder not found in RLVR-Fine-Tuning-Spurious-Rewards/code"
        exit 1
    fi
    # Go back to project root and remove the cloned repo
    cd ../../..
    rm -rf src/spurious_rewards/code/RLVR-Fine-Tuning-Spurious-Rewards

    # Create symlink for data
    if [ ! -L "data" ]; then
        ln -s src/spurious_rewards/code/data data
    fi

fi

# Default values
BASE_MODEL="Qwen/Qwen2.5-Math-1.5B"
HF_REPO="alphaXiv/spurious-rewards-rlvr-training-qwen-2.5-1.5b-math-ckpt"
DOWNLOAD_FROM_HF=false
HF_DEFAULT_CHECKPOINTS="50,200,400,1000"

# Parse command line arguments
while getopts "c:s:b:h" opt; do
  case $opt in
    c) CHECKPOINT_DIR="$OPTARG" ;;
    s) STEPS="$OPTARG" ;;
    b) BASE_MODEL="$OPTARG" ;;
    h) echo "Usage: $0 [-hf] [-c <checkpoint_dir> -s <steps>] [-b <base_model>]"
       echo ""
       echo "Option 1: Download from Hugging Face Hub"
       echo "  -hf: Download and evaluate default HF checkpoints (50, 200, 400, 1000)"
       echo ""
       echo "Option 2: Use local DeepSpeed checkpoint"
       echo "  -c: Path to the DeepSpeed checkpoint directory (required)"
       echo "  -s: Comma-separated list of checkpoint step numbers (required, e.g., 450,500,600,700)"
       echo ""
       echo "Optional:"
       echo "  -b: Base model name (default: Qwen/Qwen2.5-Math-1.5B)"
       exit 0 ;;
    *) echo "Invalid option: -$OPTARG" >&2
       echo "Use -h for help"
       exit 1 ;;
  esac
done

# Check for HF flag in remaining arguments
for arg in "$@"; do
  if [ "$arg" = "-hf" ]; then
    DOWNLOAD_FROM_HF=true
  fi
done

# Check required arguments
if [ "$DOWNLOAD_FROM_HF" = true ]; then
    HF_CHECKPOINT_NUMS="$HF_DEFAULT_CHECKPOINTS"
else
    if [ -z "$CHECKPOINT_DIR" ] || [ -z "$STEPS" ]; then
        echo "Error: -c (checkpoint_dir) and -s (steps) are required arguments (when not using -hf)."
        echo "Use -h for help."
        exit 1
    fi
    
    # Convert checkpoint path to absolute before changing directories
    if [[ ! "$CHECKPOINT_DIR" = /* ]]; then
        CHECKPOINT_DIR="$(pwd)/$CHECKPOINT_DIR"
    fi
fi

# Navigate to code directory for evaluations
cd src/spurious_rewards/code

# Setup based on download method
if [ "$DOWNLOAD_FROM_HF" = true ]; then
    echo "Setting up for HF Hub downloads..."
    
    # Parse HF checkpoint numbers
    IFS=',' read -ra CHECKPOINT_NUMS <<< "$HF_CHECKPOINT_NUMS"
    STEP_ARRAY=()
    
    # Create models directory
    mkdir -p hf_models
    
    # Download each checkpoint from HF Hub
    echo "=========================================="
    echo "Downloading models from Hugging Face Hub"
    echo "=========================================="
    
    for ckpt_num in "${CHECKPOINT_NUMS[@]}"; do
        ckpt_num=$(echo "$ckpt_num" | xargs)  # Trim whitespace
        HF_REPO_WITH_CKPT="${HF_REPO}-${ckpt_num}"
        MODEL_DIR="hf_models/model-${ckpt_num}"
        
        echo ""
        echo "Downloading from: $HF_REPO_WITH_CKPT"
        echo "To: $MODEL_DIR"
        
        # Use huggingface-cli to download the model
        huggingface-cli download "$HF_REPO_WITH_CKPT" --repo-type model --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
        
        # Extract step number from checkpoint number for consistency
        STEP_ARRAY+=("$ckpt_num")
    done
else
    # Parse steps from local checkpoint
    IFS=',' read -ra STEP_ARRAY <<< "$STEPS"
fi

START_TIME=$(date +%s)
echo "=========================================="
echo "Starting Inference and Evaluation Process..."
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
if [ "$DOWNLOAD_FROM_HF" = true ]; then
    echo "Mode: Hugging Face Hub (default checkpoints)"
    echo "Checkpoints: ${HF_CHECKPOINT_NUMS}"
else
    echo "Mode: Local DeepSpeed checkpoint"
    echo "Checkpoint Dir: $CHECKPOINT_DIR"
    echo "Steps: ${STEP_ARRAY[*]}"
fi
echo "Base Model: $BASE_MODEL"
echo ""

## Evaluations
# To reproduce our evaluation results, use the following commands:

# Create results directory
mkdir -p results

# Phase 0: Evaluate base model
echo "=========================================="
echo "Phase 0: Evaluating base model"
echo "=========================================="
echo "Evaluating base model: $BASE_MODEL"
BASE_RESULTS_DIR="results/base"
mkdir -p "$BASE_RESULTS_DIR"

python eval_checkpoint.py --model_path "$BASE_MODEL" --datasets MATH-500,AIME-2024,AIME-2025,AMC --shards 2 --output_dir "$BASE_RESULTS_DIR"

echo "Base model evaluation completed."
echo ""

# Phase 1: Export or prepare checkpoints
if [ "$DOWNLOAD_FROM_HF" = true ]; then
    echo "=========================================="
    echo "Phase 1: HF models already downloaded"
    echo "=========================================="
    echo "Models are ready in hf_models/ directory"
else
    echo "=========================================="
    echo "Phase 1: Exporting all checkpoints"
    echo "=========================================="
    for step in "${STEP_ARRAY[@]}"; do
        OUTPUT_DIR="./export-for-eval-step${step}"
        mkdir -p "$OUTPUT_DIR"
        
        echo "Exporting checkpoint for step $step..."
        python scripts/export_checkpoint.py --checkpoint "$CHECKPOINT_DIR" --step "$step" --base-model "$BASE_MODEL" --output-dir "$OUTPUT_DIR"
    done
fi

# Phase 2: Evaluate all exported checkpoints
echo ""
echo "=========================================="
echo "Phase 2: Evaluating all checkpoints"
echo "=========================================="
for step in "${STEP_ARRAY[@]}"; do
    STEP_START_TIME=$(date +%s)
    echo ""
    echo "Evaluating checkpoint: $step..."
    echo "Evaluation start time: $(date '+%Y-%m-%d %H:%M:%S')"

    if [ "$DOWNLOAD_FROM_HF" = true ]; then
        MODEL_PATH="hf_models/model-${step}"
    else
        MODEL_PATH="./export-for-eval-step${step}"
    fi
    
    RESULTS_DIR="results/step${step}"
    mkdir -p "$RESULTS_DIR"

    python eval_checkpoint.py --model_path "$MODEL_PATH" --datasets MATH-500,AIME-2024,AIME-2025,AMC --shards 2 --output_dir "$RESULTS_DIR"

    STEP_END_TIME=$(date +%s)
    STEP_ELAPSED=$((STEP_END_TIME - STEP_START_TIME))
    echo "Completed evaluation for step $step."
    echo "Evaluation end time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Evaluation duration: $((STEP_ELAPSED / 60)) minutes $((STEP_ELAPSED % 60)) seconds"
done

echo "All evaluations completed. Generating plots..."

# Prepare steps for python
STEPS_PYTHON=$(printf '%s ' "${STEP_ARRAY[@]}")
STEPS_PYTHON=${STEPS_PYTHON% }

# Generate plots with base model included
python plot_performance.py --base-model $STEPS_PYTHON

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "Process completed successfully!"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total duration: $((TOTAL_ELAPSED / 60)) minutes $((TOTAL_ELAPSED % 60)) seconds"
echo "=========================================="