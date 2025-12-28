#!/bin/bash

# Speedrun script for agent_r1 setup and training
# Combines commands from docs/getting_started/quickstart.md

set -e  # Exit on any error


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

echo "=========================================="
echo "agent_r1 Speedrun Setup and Training"
echo "=========================================="
echo "Selected algorithm: $ALGORITHM"

# Check if WANDB_API_KEY is set 
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY environment variable is not set."
    echo "Training will proceed without Weights & Biases logging."
    echo "To enable logging, set: export WANDB_API_KEY='your_key_here'"
    echo "Get your key from: https://wandb.ai/settings"
    exit 1
fi

# Pull Docker image if not already present
echo "Pulling Docker image..."
sudo docker pull hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0 || {
    echo "Failed to pull Docker image. Please check your Docker installation and network."
    
}



# Start Docker container
echo "Checking for existing Docker container..."

# Check if container is already running
if [ "$(sudo docker ps -q -f name=verl-agent-r1)" ]; then
    echo "Container 'verl-agent-r1' is already running. Reusing existing container."
# Check if container exists but is stopped
elif [ "$(sudo docker ps -aq -f name=verl-agent-r1)" ]; then
    echo "Container 'verl-agent-r1' exists but is stopped. Starting existing container..."
    sudo docker start verl-agent-r1 || {
        echo "Failed to start existing container. Please check Docker status."
        exit 1
    }
    echo "Container started."
else
    echo "Starting new Docker container..."
    sudo docker run -d --gpus all --name verl-agent-r1 \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v $(pwd)/..:/workspace \
        hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0 \
        sleep infinity || {
        echo "Failed to start Docker container. Please check GPU availability and Docker setup."
        exit 1
    }
    echo "Waiting for container to start..."
    sleep 10
fi

# Install agent_r1 dependencies
echo "Installing agent_r1 dependencies..."
sudo docker exec verl-agent-r1 bash -c "cd /workspace/agent_r1 && pip3 install -e ." || {
    echo "Failed to install agent_r1 dependencies."
    exit 1
}

# Clone VERL from official repo
echo "Cloning VERL from official repo..."
sudo docker exec verl-agent-r1 bash -c "if [ ! -d '/workspace/agent_r1/src/verl' ]; then git config --global --add safe.directory '*' && cd /workspace/agent_r1/src && git clone https://github.com/volcengine/verl.git && cd verl && git checkout a43ead6; else echo 'VERL already exists, skipping clone.'; fi" || {
    echo "Failed to clone VERL from official repo."
    exit 1
}

# Install VERL
echo "Installing VERL..."
sudo docker exec verl-agent-r1 bash -c "cd /workspace/agent_r1/src/verl && pip3 install -e ." || {
    echo "Failed to install VERL."
    exit 1
}

wait 

# Download and preprocess HotpotQA dataset
echo "Downloading and preprocessing HotpotQA dataset..."
sudo docker exec verl-agent-r1 bash -c "cd /workspace/agent_r1 && mkdir -p data/hotpotqa && python src/examples/data_preprocess/hotpotqa.py --local_dir data/hotpotqa" || {
    echo "Failed to download and preprocess HotpotQA dataset."
    exit 1
}

# Build HotpotQA search index
echo "Building HotpotQA search index..."
sudo docker exec verl-agent-r1 bash -c "cd /workspace/agent_r1 && if [ ! -f 'data/corpus/hotpotqa/hpqa_corpus.jsonl' ]; then mkdir -p data/corpus/hotpotqa && wget -q https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/corpus.jsonl.gz -O data/corpus/hotpotqa/corpus.jsonl.gz && gunzip -c data/corpus/hotpotqa/corpus.jsonl.gz > data/corpus/hotpotqa/hpqa_corpus.jsonl; else echo 'HotpotQA corpus already exists, skipping download.'; fi" || {
    echo "Failed to download corpus data."
    exit 1
}

sudo docker exec verl-agent-r1 bash -c "cd /workspace/agent_r1 && if [ -f 'data/corpus/hotpotqa/index.bin' ]; then echo 'HotpotQA search index already exists, skipping index build.'; else echo 'Building FAISS search index (this may take some time)...'; python src/scripts/hotpotqa_search/process_hotpotqa.py; fi" || {
    echo "Failed to build search index."
    exit 1
}

# Configure Weights & Biases if API key is set
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "Configuring Weights & Biases..."
    sudo docker exec verl-agent-r1 bash -c "wandb login $WANDB_API_KEY" || {
        echo "Failed to login to Weights & Biases."
        exit 1
    }
fi

# Set up environment variables for Docker exec
DOCKER_ENV=""
if [ ! -z "$HYDRA_FULL_ERROR" ]; then
    echo "HYDRA_FULL_ERROR is set, enabling full error traces..."
    DOCKER_ENV="export HYDRA_FULL_ERROR=1 && "
fi

# Run training based on selected algorithm
case "$ALGORITHM" in
    ppo)
        echo "=========================================="
        echo "Starting PPO Training on HotpotQA"
        echo "This will take approximately 22 hours on 4xH100 80GB GPUs"
        echo "=========================================="

        sudo docker exec verl-agent-r1 bash -c "cd /workspace/agent_r1 && ${DOCKER_ENV}cp src/examples/trainer/run_ppo_hotpotqa.sh ./ && bash run_ppo_hotpotqa.sh" || {
            echo "Training failed."
            exit 1
        }
        ;;
    grpo)
        echo "=========================================="
        echo "Starting GRPO Training on HotpotQA"
        echo "This will take approximately 20-22 hours on 4xH100 80GB GPUs"
        echo "=========================================="

        sudo docker exec verl-agent-r1 bash -c "cd /workspace/agent_r1 && ${DOCKER_ENV}cp src/examples/trainer/run_grpo_hotpotqa.sh ./ && bash run_grpo_hotpotqa.sh" || {
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

