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
mkdir -p checkpoints logs results wandb

# Generate random port for deepspeed
port=$(shuf -i25000-30000 -n1)

# Experiment name used for checkpoints and logs
EXPERIMENT_NAME="rope_pp-376m-4k-imag2"

export WANDB_RUN_GROUP="rope_pp-376m"

# Stage 1: Initial training for 100k steps
echo "=========================================="
echo "Starting Stage 1: Initial Training"
echo "=========================================="

deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
  multi-gpu/train_rope_pp.py \
  --config_abbr '376m' \
  --imag \
  --imag_mode 'imag2' \
  --save_abbr "$EXPERIMENT_NAME"

echo "Stage 1 complete! Check logs/${EXPERIMENT_NAME}.log"

# Wait for stage 1 to complete
wait

# Stage 2: Learning rate decay for 10k steps from checkpoint 90000
echo "=========================================="
echo "Starting Stage 2: Learning Rate Decay"
echo "=========================================="

deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
  multi-gpu/train_rope_pp-decay.py \
  --config_abbr '376m' \
  --imag \
  --imag_mode 'imag2' \
  --save_abbr "$EXPERIMENT_NAME" \
  --load_ckpt 90000 \
  --decay_step 10000

echo "Stage 2 complete! Check logs/${EXPERIMENT_NAME}-ckpt90000-decay.log"

# Wait for stage 2 to complete
wait


# Stage 4: Model Evaluation with LM Harness
echo "=========================================="
echo "Starting Model Evaluation with LM Harness"
echo "=========================================="

# Evaluate the checkpoint after decay training
python eval/eval_lmharness.py \
  --local-checkpoint "checkpoints/${EXPERIMENT_NAME}-ckpt90000-decay/checkpoint-10000" \
  --model-name "Local-RoPEPP-376M-After-Decay" \
  --model-type "ropepp" \
  --include-baselines

echo "Evaluation (standard) complete!"

wait

# Stage 3: Long context training from checkpoint 10000
echo "=========================================="
echo "Starting Stage 3: Long Context Training"
echo "=========================================="

deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
  multi-gpu/train_rope_pp-lctx.py \
  --config_abbr '376m' \
  --imag \
  --imag_mode 'imag2' \
  --save_abbr "${EXPERIMENT_NAME}-ckpt90000-decay" \
  --load_ckpt 10000

echo "Stage 3 complete! Check logs/${EXPERIMENT_NAME}-ckpt90000-decay-lctx.log"

# Wait for stage 3 to complete
wait

# Stage 5: Model Evaluation with LM Harness (Long Context)
echo "=========================================="
echo "Starting Long Context Evaluation with LM Harness"
echo "=========================================="

# Evaluate the checkpoint after long context training
python eval/eval_lmharness-lctx.py \
  --local-checkpoint "checkpoints/${EXPERIMENT_NAME}-ckpt90000-decay-lctx/checkpoint-10000" \
  --model-name "Local-RoPEPP-376M-After-LongContext" \
  --model-type "ropepp" \
  --include-baselines

echo "Evaluation (long context) complete!"

echo "=========================================="
echo "Training and Evaluation Complete!"
echo "=========================================="
echo "Final model saved at: checkpoints/${EXPERIMENT_NAME}-ckpt90000-decay-lctx"
echo ""
echo "Check the results directory for detailed evaluation outputs."
