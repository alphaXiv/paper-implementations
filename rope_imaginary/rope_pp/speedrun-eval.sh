#!/bin/bash

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN environment variable is not set."
    echo "You need to set it to access gated models like meta-llama/Meta-Llama-3-8B"
    echo "Example: export HF_TOKEN='your_token_here'"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    exit 1
fi

# Install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv --python 3.12

# Activate venv first
source .venv/bin/activate

# Install all dependencies including flash-attn
# The extra-build-dependencies config will provide torch, setuptools, etc. during flash-attn build
uv sync --extra gpu

# Venv already activated above

# Create necessary directories
mkdir -p results logs

echo "=========================================="
echo "Starting Model Evaluation with LM Harness"
echo "=========================================="

# Run the standard evaluation script
python eval/eval_lmharness.py

echo "Evaluation (standard) complete!"

echo "=========================================="
echo "Starting Long Context Evaluation with LM Harness"
echo "=========================================="

# Run the long context evaluation script
python eval/eval_lmharness-lctx.py

echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Check the results directory for detailed evaluation outputs."
