#!/bin/bash

# Inference Script for Agent-R1
# This script automates the inference process based on the current repo structure.
# Usage: bash inference.sh [--checkpoint-dir <dir>] [--hf-model-path <path>] [--target-dir <dir>]

# Source the virtual environment
source ~/verl_env/bin/activate

# Default values
CHECKPOINT_DIR="checkpoints/hotpotqa/ppo-qwen2.5-1.5b-instruct/global_step_1/actor"
HF_MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
TARGET_DIR="./converted_model"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --hf-model-path)
            HF_MODEL_PATH="$2"
            shift 2
            ;;
        --target-dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--checkpoint-dir <dir>] [--hf-model-path <path>] [--target-dir <dir>]"
            echo ""
            echo "Options:"
            echo "  --checkpoint-dir    Path to the checkpoint directory (default: ./checkpoints/step1/actor)"
            echo "  --hf-model-path     Hugging Face model path (default: Qwen/Qwen2.5-3B-Instruct)"
            echo "  --target-dir        Target directory for converted model (default: ./converted_model)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information."
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Starting Agent-R1 Inference Process"
echo "=========================================="

MODEL_NAME="${TARGET_DIR}"

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory '$CHECKPOINT_DIR' does not exist."
    exit 1
else
    echo "Checkpoint directory '$CHECKPOINT_DIR' found."
fi

echo "Using checkpoint dir: ${CHECKPOINT_DIR}"
echo "HF model path: ${HF_MODEL_PATH}"
echo "Target dir: ${TARGET_DIR}"
echo ""

# Step 1: Convert Training Checkpoints to HF Format
echo "Step 1: Converting checkpoints to HF format..."

# Copy and modify the merge script
cp src/scripts/model_merge.sh ./model_merge.sh

# Replace placeholders in the script
sed -i "s|<your_checkpoint_dir>|${CHECKPOINT_DIR}|g" model_merge.sh
sed -i "s|<your_hf_model_path>|${HF_MODEL_PATH}|g" model_merge.sh
sed -i "s|<your_target_dir>|${TARGET_DIR}|g" model_merge.sh

# Run the conversion
bash model_merge.sh

if [ $? -eq 0 ]; then
    echo "Checkpoint conversion completed successfully."
else
    echo "Error: Checkpoint conversion failed."
    exit 1
fi

echo ""

# Step 2: Deploy the vLLM Service
echo "Step 2: Deploying vLLM service..."

# Copy and modify the serve script
cp src/scripts/vllm_serve.sh ./vllm_serve.sh

# Replace placeholder
sed -i "s|<your_model_name>|${MODEL_NAME}|g" vllm_serve.sh

# Shutdown any existing vLLM processes
echo "Shutting down any existing vLLM processes..."
pkill -f vllm || true
sleep 2

# Start the vLLM service in the background
bash vllm_serve.sh &
VLLM_PID=$!

echo "vLLM service started (PID: ${VLLM_PID}). Waiting a few seconds for it to initialize..."
sleep 10

echo ""

# Step 3: Running Inference
echo "Step 3: Running inference..."

echo "Launching interactive chat interface..."
echo "Note: The vLLM service is running in the background."
echo "To stop the service later, run: kill ${VLLM_PID}"
echo ""

# Start the chat interface
python3 -m src.agent_r1.inference.chat

# After chat exits, stop the service
echo "Stopping vLLM service..."
kill ${VLLM_PID}

echo "Inference process completed."