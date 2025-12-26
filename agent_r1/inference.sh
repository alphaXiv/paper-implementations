#!/bin/bash

set -e
# Inference Script for Agent-R1
# This script automates the inference process based on the current repo structure.
# Usage: bash inference.sh [--checkpoint-dir <dir>] [--hf-model-path <path>] [--target-dir <dir>]


### Installing dependencies
if [ ! -d "verl_env" ]; then
    echo "=========================================="
    echo "Agent-R1 / VERL – Libraries Only Installer"
    echo "=========================================="

    echo "=== System update & base tools ==="
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        unzip \
        python3.10 \
        python3.10-venv \
        python3.10-dev

    echo "=== Create Python 3.10 virtual environment ==="
    python3.10 -m venv verl_env
    source verl_env/bin/activate

    echo "=== Upgrade pip tooling ==="
    pip install --upgrade pip setuptools wheel

    echo "=== Remove conflicting NVIDIA PyTorch forks (if any) ==="
    pip uninstall -y \
        torch torchvision torchaudio \
        pytorch-quantization pytorch-triton torch-tensorrt \
        xgboost transformer_engine flash_attn apex megatron-core grpcio \
        || true

    # Installing required libraries for indexing

    pip3 install -e . || {
        echo "Failed to install required libraries for indexing."
        exit 1
    }

    echo "=== Install PyTorch 2.6.0 + CUDA 12.4 ==="
    pip install \
        torch==2.6.0+cu124 \
        torchvision==0.21.0+cu124 \
        torchaudio==2.6.0 \
        tensordict==0.6.2 \
        torchdata \
        --extra-index-url https://download.pytorch.org/whl/cu124

    echo "=== Install vLLM 0.8.3 ==="
    pip install vllm==0.8.3

    echo "=== Install flash-attn 2.7.4.post1 (cxx11abi=False) ==="
    pip install flash_attn==2.7.4.post1 --no-build-isolation

    echo "=== Install flashinfer 0.2.2.post1 (vLLM-compatible) ==="
    pip install flashinfer_python==0.2.2.post1

    echo "=== Install ML / RL / infra libraries ==="
    pip install \
        "transformers[hf_xet]>=4.51.0" \
        accelerate datasets peft hf-transfer \
        "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
        ray[default] \
        codetiming hydra-core pylatexenc qwen-vl-utils wandb \
        dill pybind11 liger-kernel mathruler \
        pytest py-spy pyext pre-commit ruff

    echo "=== Fix known incompatibilities ==="
    pip uninstall -y pynvml nvidia-ml-py || true
    pip install --upgrade \
        "nvidia-ml-py>=12.560.30" \
        "fastapi[standard]>=0.115.0" \
        "optree>=0.13.0" \
        "pydantic>=2.9" \
        "grpcio>=1.62.1"

    echo "=== Install VERL with vLLM support ==="

    # Clone and install a clean verl from official repo
    if [ ! -d "src/verl" ];
    then 
        echo "Cloning verl from official repo and installing verl..."
        git config --global --add safe.directory '*' && cd src && git clone https://github.com/volcengine/verl.git && cd verl && git checkout a43ead6
        pip3 install -e .

    else
        echo "verl exists!"
    fi
     

    echo "=========================================="
    echo " Libraries installation complete!"
    echo "=========================================="

else
    # Source the virtual environment
    source verl_env/bin/activate
fi


# Default values
CHECKPOINT_DIR="checkpoints/hotpotqa/ppo-qwen2.5-1.5b-instruct/global_step_102/actor"
HF_MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
TARGET_DIR="./converted_model"
BACKEND="fsdp"
USE_HF_MODEL=false

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
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --use-hf-model)
            USE_HF_MODEL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--checkpoint-dir <dir>] [--hf-model-path <path>] [--target-dir <dir>] [--backend <backend>] [--use-hf-model]"
            echo ""
            echo "Options:"
            echo "  --checkpoint-dir    Path to the checkpoint directory (default: ./checkpoints/step1/actor)"
            echo "  --hf-model-path     Hugging Face model path (default: Qwen/Qwen2.5-3B-Instruct)"
            echo "  --target-dir        Target directory for converted model (default: ./converted_model)"
            echo "  --backend           Backend type (default: fsdp)"
            echo "  --use-hf-model      Use Hugging Face model directly without conversion (default: false)"
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

if [ "$USE_HF_MODEL" = true ]; then
    MODEL_NAME="${HF_MODEL_PATH}"
    echo "Using Hugging Face model directly: ${MODEL_NAME}"
else
    MODEL_NAME="${TARGET_DIR}"
    # Check if checkpoint directory exists
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "Error: Checkpoint directory '$CHECKPOINT_DIR' does not exist."
        exit 1
    else
        echo "Checkpoint directory '$CHECKPOINT_DIR' found."
    fi
fi

echo "Using checkpoint dir: ${CHECKPOINT_DIR}"
echo "HF model path: ${HF_MODEL_PATH}"
echo "Target dir: ${TARGET_DIR}"
echo "Backend: ${BACKEND}"
echo "Use HF model: ${USE_HF_MODEL}"
echo ""

if [ "$USE_HF_MODEL" = false ]; then
    # Step 1: Convert Training Checkpoints to HF Format
    echo "Step 1: Converting checkpoints to HF format..."

    # Run the conversion
    python3 src/verl/scripts/model_merger.py --backend $BACKEND --hf_model_path $HF_MODEL_PATH --local_dir $CHECKPOINT_DIR --target_dir $TARGET_DIR

    if [ $? -eq 0 ]; then
        echo "Checkpoint conversion completed successfully."
    else
        echo "Error: Checkpoint conversion failed."
        exit 1
    fi

    echo ""
fi

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