#!/bin/bash

# Spurious Rewards Training and Evaluation Speedrun Script

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN environment variable is not set."
    echo "You need to set it to access gated models like Qwen/Qwen2.5-Math-7B"
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
fi

# Activate conda environment
echo "Activating conda environment: $ENV_NAME"
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME




# Check if WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY environment variable is not set."
    echo "You need to set it to log to Weights & Biases"
    echo "Example: export WANDB_API_KEY='your_wandb_api_key'"
    echo "Get your API key from: https://wandb.ai/authorize"
    exit 1
fi

# Set WANDB login
wandb login "$WANDB_API_KEY"

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

# Go back to src/spurious_rewards/code and remove the cloned repo
cd ../../
rm -rf RLVR-Fine-Tuning-Spurious-Rewards

# Now we're back in src/spurious_rewards/code with data/ in place
echo ""
echo "=========================================="
echo "Setup Complete! Starting Training..."
echo "=========================================="

# Run the main training script
bash scripts/rlvr_deepscaler_grpo_qwen_ground_truth.sh

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


