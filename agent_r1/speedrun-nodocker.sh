#!/usr/bin/env bash

set -e

# Parse command line arguments
ALGORITHM=""
while [[ $# -gt 0 ]]; do
    case $1 in
        ppo|grpo)
            ALGORITHM="$1" 
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [ALGORITHM]"
            echo "ALGORITHM: ppo (default), grpo"
            echo ""
            echo "Examples:"
            echo "  $0              # Run PPO training (default)"
            echo "  $0 ppo          # Run PPO training"
            echo "  $0 grpo         # Run GRPO training"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information."
            exit 1
            ;;
    esac
done

# Default to PPO if no algorithm specified
if [ -z "$ALGORITHM" ]; then
    echo "No algorithm specified. Defaulting to PPO."
    ALGORITHM="ppo"
fi


#!/usr/bin/env bash
set -euo pipefail

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
rm -rf ~/verl_env
python3.10 -m venv ~/verl_env
source ~/verl_env/bin/activate

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

# Clone VERL from official repo
echo "Cloning VERL from official repo..."
if [ ! -d 'src/verl' ]; then
    git config --global --add safe.directory '*' && cd src && git clone https://github.com/volcengine/verl.git && cd verl && git checkout a43ead6
else
    echo 'VERL already exists, skipping clone.'
fi || {
    echo "Failed to clone VERL from official repo."
    exit 1
}      

# Install VERL
echo "Installing VERL..."

pip3 install -e . || {
    echo "Failed to install VERL."
    
    exit 1
}

cd ../../


echo "=========================================="
echo " Libraries installation complete!"
echo "=========================================="



# Download and preprocess HotpotQA dataset
echo "Downloading and preprocessing HotpotQA dataset..."
# Use data directory
DATA_DIR="data"


if [ ! -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR/hotpotqa"

else
    echo 'Data directory already exists, skipping creation.'

fi

wait

# Download and preprocess HotpotQA dataset
python src/examples/data_preprocess/hotpotqa.py --local_dir "$DATA_DIR/hotpotqa" || {
    echo "Failed to download and preprocess HotpotQA dataset."
    exit 1
}

# Build HotpotQA search index
echo "Building HotpotQA search index..."
if [ ! -f "$DATA_DIR/corpus/hotpotqa/hpqa_corpus.jsonl" ]; then
    mkdir -p "$DATA_DIR/corpus/hotpotqa" && wget -q https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/corpus.jsonl.gz -O "$DATA_DIR/corpus/hotpotqa/corpus.jsonl.gz" && gunzip -c "$DATA_DIR/corpus/hotpotqa/corpus.jsonl.gz" > "$DATA_DIR/corpus/hotpotqa/hpqa_corpus.jsonl"
else
    echo 'HotpotQA corpus already exists, skipping download.'
fi || {
    echo "Failed to download corpus data."
    exit 1
}

# Build FAISS search index if it doesn't exist
if [ ! -f "$DATA_DIR/corpus/hotpotqa/index.bin" ]; then
    echo 'Building FAISS search index...'
    python src/scripts/hotpotqa_search/process_hotpotqa.py || {
        echo "Failed to build search index."
        exit 1
    }
else
    echo 'FAISS index already exists, skipping search index processing.'
fi

# Configure Weights & Biases if API key is set
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "Configuring Weights & Biases..."
    # Set wandb to use local directory
    export WANDB_DIR="./wandb"
    mkdir -p "$WANDB_DIR"
    wandb login $WANDB_API_KEY || {
        echo "Failed to login to Weights & Biases."
        exit 1
    }
fi

# Ensure outputs directory is writable

if [ ! -d "outputs" ]; then
    mkdir -p outputs
fi

sudo chmod -R 777 outputs

pwd

# Run training based on selected algorithm
case "$ALGORITHM" in
    ppo)
        echo "=========================================="
        echo "Starting PPO Training on HotpotQA"
        echo "This will take approximately 12 hours on 4xH100 80GB GPUs"
        echo "=========================================="

        cp src/examples/trainer/run_ppo_hotpotqa.sh ./ && bash run_ppo_hotpotqa.sh || {
            echo "Training failed."
            exit 1
        }
        ;;
    grpo)
        echo "=========================================="
        echo "Starting GRPO Training on HotpotQA"
        echo "This will take approximately 18-24 hours on 4xH100 80GB GPUs"
        echo "=========================================="

        cp src/examples/trainer/run_grpo_hotpotqa.sh ./ && bash run_grpo_hotpotqa.sh || {
            echo "Training failed."
            exit 1
        }
        ;;
    *)
        echo "Unknown algorithm: $ALGORITHM"
        exit 1
        ;;
esac

echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Check the results and logs in the agent_r1 directory."
echo "You can also check Weights & Biases for training metrics if configured."


