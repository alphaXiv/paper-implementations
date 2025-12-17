#!/bin/bash

# Speedrun script for agent_r1 setup and training
# Combines commands from docs/getting_started/quickstart.md

set -e  # Exit on any error


echo "=========================================="
echo "agent_r1 Speedrun Setup and Training"
echo "=========================================="

# Check if WANDB_API_KEY is set (optional but recommended for logging)
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY environment variable is not set."
    echo "Training will proceed without Weights & Biases logging."
    echo "To enable logging, set: export WANDB_API_KEY='your_key_here'"
    echo "Get your key from: https://wandb.ai/settings"
else
    echo "WANDB_API_KEY is set. Training will log to Weights & Biases."
fi

# Pull Docker image if not already present
echo "Pulling Docker image..."
sudo docker pull hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0 || {
    echo "Failed to pull Docker image. Please check your Docker installation and network."
    exit 1
}



# Start Docker container
echo "Starting Docker container..."
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

# Wait for container to be ready
echo "Waiting for container to start..."
sleep 10

# Install agent_r1 dependencies
echo "Installing agent_r1 dependencies..."
sudo docker exec verl-agent-r1 bash -c "cd /workspace/agent_r1 && pip3 install -e ." || {
    echo "Failed to install agent_r1 dependencies."
    exit 1
}

# Clone VERL from official repo
echo "Cloning VERL from official repo..."
sudo docker exec verl-agent-r1 bash -c "git config --global --add safe.directory '*' && cd /workspace/agent_r1/src && git clone https://github.com/volcengine/verl.git && cd verl && git checkout a43ead6" || {
    echo "Failed to clone VERL from official repo."
    exit 1
}

# Install VERL
echo "Installing VERL..."
sudo docker exec verl-agent-r1 bash -c "cd /workspace/agent_r1/src/verl && pip3 install -e ." || {
    echo "Failed to install VERL."
    exit 1
}

# Download and preprocess HotpotQA dataset
echo "Downloading and preprocessing HotpotQA dataset..."
sudo docker exec verl-agent-r1 bash -c "cd /workspace/agent_r1 && mkdir -p data/hotpotqa && python src/examples/data_preprocess/hotpotqa.py --local_dir ./data/hotpotqa" || {
    echo "Failed to download and preprocess HotpotQA dataset."
    exit 1
}

# Build HotpotQA search index
echo "Building HotpotQA search index..."
sudo docker exec verl-agent-r1 bash -c "cd /workspace/agent_r1 && mkdir -p data/corpus/hotpotqa && wget -q https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/corpus.jsonl.gz -O data/corpus/hotpotqa/corpus.jsonl.gz && gunzip -c data/corpus/hotpotqa/corpus.jsonl.gz > data/corpus/hotpotqa/hpqa_corpus.jsonl" || {
    echo "Failed to download corpus data."
    exit 1
}

sudo docker exec verl-agent-r1 bash -c "cd /workspace/agent_r1/src/scripts/hotpotqa_search && python process_hotpotqa.py" || {
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

# Run PPO training (default choice for speedrun)
echo "=========================================="
echo "Starting PPO Training on HotpotQA"
echo "This will take approximately 12 hours on 4xA100 80GB GPUs"
echo "=========================================="

sudo docker exec verl-agent-r1 bash -c "cd /workspace/agent_r1 && cp src/examples/trainer/run_ppo_hotpotqa.sh ./ && bash run_ppo_hotpotqa.sh" || {
    echo "Training failed."
    exit 1
}



echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Check the results and logs in the agent_r1 directory."
echo "You can also check Weights & Biases for training metrics if configured."

wait # Wait for all background processes to finish


# Run GRPO training (default choice for speedrun)
echo "=========================================="
echo "Starting GRPO Training on HotpotQA"
echo "This will take approximately 18-24 hours on 4xA100 80GB GPUs"
echo "=========================================="

sudo docker exec verl-agent-r1 bash -c "cd /workspace/agent_r1 && cp src/examples/trainer/run_grpo_hotpotqa.sh ./ && bash run_grpo_hotpotqa.sh" || {
    echo "Training failed."
    exit 1
}


echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Check the results and logs in the agent_r1 directory."
echo "You can also check Weights & Biases for training metrics if configured."</content>

wait # Wait for all background processes to finish



# Run RPP training (default choice for speedrun)
echo "=========================================="
echo "Starting RPP Training on HotpotQA"
echo "This will take approximately 10-12 hours on 4xA100 80GB GPUs"
echo "=========================================="

sudo docker exec verl-agent-r1 bash -c "cd /workspace/agent_r1 && cp src/examples/trainer/run_rpp_hotpotqa.sh ./ && bash run_rpp_hotpotqa.sh" || {
    echo "Training failed."
    exit 1
}



echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Check the results and logs in the agent_r1 directory."
echo "You can also check Weights & Biases for training metrics if configured."

wait # Wait for all background processes to finish

