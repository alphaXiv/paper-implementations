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



# Install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv --python 3.10

# Activate venv
source .venv/bin/activate

# Install all dependencies using uv
uv sync


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

# Install flash_attn with proper build isolation handling
uv pip install flash_attn==2.7.0.post2 --no-build-isolation

# Install the package in editable mode
uv pip install -e .

# Modify setup.py to pin vllm version for CUDA 12.8 compatibility
# Change extras_require vllm from ["vllm"] to ["vllm==0.7.2"]
sed -i 's/"vllm": \["vllm"\]/"vllm": ["vllm==0.7.2"]/g' setup.py

# Install flash_attn with proper build isolation handling
uv pip install flash_attn==2.7.0.post2 --no-build-isolation

# Install the package in editable mode
uv pip install -e .


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


