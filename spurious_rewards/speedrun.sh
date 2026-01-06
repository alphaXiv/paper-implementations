#!/bin/bash

# Spurious Rewards Training and Evaluation Speedrun Script

set -e

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN environment variable is not set."
    echo "You need to set it to access gated models like Qwen/Qwen2.5-Math-7B"
    echo "Example: export HF_TOKEN='your_token_here'"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    exit 1
fi



# Check if python3.10 venv is available
if ! python3.10 -m venv --help &> /dev/null; then
    echo "ERROR: python3.10 venv not available. Please ensure Python 3.10 is installed."
    exit 1
fi

# Create venv environment if it doesn't exist
ENV_NAME="spurious-rewards-env"
if [ ! -d "$ENV_NAME" ]; then
    echo "Creating venv environment: $ENV_NAME"
    python3.10 -m venv $ENV_NAME
fi

# Activate venv environment
echo "Activating venv environment: $ENV_NAME"
source $ENV_NAME/bin/activate

# Install wheel for faster package installations
pip install wheel




# Check if WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY environment variable is not set."
    echo "You need to set it to log to Weights & Biases"
    echo "Example: export WANDB_API_KEY='your_wandb_api_key'"
    echo "Get your API key from: https://wandb.ai/authorize"
    exit 1
fi


# Navigate to code directory
cd src/spurious_rewards/code

# Install PyTorch and dependencies
echo "Installing PyTorch and dependencies..."
pip install -r requirements.txt
pip uninstall vllm -y
pip install vllm==0.7.2

# Set WANDB login
wandb login "$WANDB_API_KEY"


# Modify setup.py to pin vllm version for CUDA 12.8 compatibility
# Change extras_require vllm from ["vllm"] to ["vllm==0.7.2"]
sed -i 's/"vllm": \["vllm"\]/"vllm": ["vllm==0.7.2"]/g' setup.py

# Install flash_attn (pip will use pre-built wheels if available for your CUDA version)
echo "Installing flash-attn..."
pip install flash-attn==2.7.0.post2 --no-build-isolation 2>&1 | grep -v "Preparing metadata" || true

# Install the package in editable mode
pip install -e .

if [ -d "data" ]; then
    echo "Data directory already exists. Skipping data download."

else
    echo "Data directory not found. Proceeding to download data..."

    # Get the data from Hugging Face
    echo "Downloading data from Hugging Face..."

    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='alphaXiv/spurious-rewards-data', repo_type='dataset', local_dir='data')"

    echo "Data downloaded successfully"
fi


# Now we're back in src/spurious_rewards/code with data/ in place
echo ""
echo "=========================================="
echo "Setup Complete! Starting Training..."
echo "=========================================="

# Run the main training script
bash scripts/rlvr_deepscaler_grpo_qwen_1.5b_ground_truth.sh

echo "=========================================="
echo "Training Complete! Starting Evaluation..."
echo "=========================================="

# Stage 1: Export checkpoint
echo ""
echo "Exporting checkpoint..."
python export_checkpoint.py

# Stage 2: Evaluate on benchmarks using exported checkpoint
echo ""
echo "Evaluating on MATH-500, AIME-2024, AIME-2025, AMC..."
python eval_checkpoint.py \
    --model_path "./exported_model" \
    --datasets MATH-500,AIME-2024,AIME-2025,AMC \
    --shards 2

echo "=========================================="
echo "Training and Evaluation Complete!"
echo "=========================================="
echo ""
echo "Check the logs directory for detailed results."
echo "Model outputs saved in outputs/ directory."


