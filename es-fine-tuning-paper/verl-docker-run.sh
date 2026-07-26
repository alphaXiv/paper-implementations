#!/bin/bash

# Speedrun script for es-fine-tuning-paper setup and training
# Combines commands from docs/getting_started/quickstart.md

set -e  # Exit on any error


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
if [ "$(sudo docker ps -q -f name=verl-es-fine-tuning-paper)" ]; then
    echo "Container 'verl-es-fine-tuning-paper' is already running. Reusing existing container."
# Check if container exists but is stopped
elif [ "$(sudo docker ps -aq -f name=verl-es-fine-tuning-paper)" ]; then
    echo "Container 'verl-es-fine-tuning-paper' exists but is stopped. Starting existing container..."
    sudo docker start verl-es-fine-tuning-paper || {
        echo "Failed to start existing container. Please check Docker status."
        exit 1
    }
    echo "Container started."
else
    echo "Starting new Docker container..."
    sudo docker run -d --gpus all --name verl-es-fine-tuning-paper \
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

# # Install es-fine-tuning-paper dependencies
# echo "Installing es-fine-tuning-paper dependencies..."
# sudo docker exec verl-es-fine-tuning-paper bash -c "cd /workspace/es-fine-tuning-paper && pip3 install -e ." || {
#     echo "Failed to install es-fine-tuning-paper dependencies."
#     exit 1
# }

# Clone VERL from official repo
echo "Cloning VERL from official repo..."
sudo docker exec verl-es-fine-tuning-paper bash -c "if [ ! -d '/workspace/es-fine-tuning-paper/src/verl' ]; then git config --global --add safe.directory '*' && cd /workspace/es-fine-tuning-paper/src && git clone https://github.com/volcengine/verl.git && cd verl && git checkout a43ead6; else echo 'VERL already exists, skipping clone.'; fi" || {
    echo "Failed to clone VERL from official repo."
    exit 1
}

# Patch main_ppo.py to support separate tokenizer path
echo "Patching main_ppo.py to support separate tokenizer path..."
sudo docker exec verl-es-fine-tuning-paper bash -c "
cd /workspace/es-fine-tuning-paper/src/verl
# Check if patch is already applied
if grep -q 'tokenizer_path = config.actor_rollout_ref.model.get' verl/trainer/main_ppo.py; then
    echo 'Patch already applied to main_ppo.py, skipping.'
else
    echo 'Applying tokenizer patch to main_ppo.py...'
    sed -i '/trust_remote_code = config.data.get(\"trust_remote_code\", False)/,/processor = hf_processor(local_path, use_fast=True)/c\        trust_remote_code = config.data.get(\"trust_remote_code\", False)\n        tokenizer_path = config.actor_rollout_ref.model.get(\"tokenizer_path\", None)\n        if tokenizer_path is not None:\n            tokenizer_local_path = copy_to_local(tokenizer_path)\n            print(f\"Using separate tokenizer from: {tokenizer_path}\")\n        else:\n            tokenizer_local_path = local_path\n        tokenizer = hf_tokenizer(tokenizer_local_path, trust_remote_code=trust_remote_code)\n        processor = hf_processor(tokenizer_local_path, use_fast=True)  # used for multimodal LLM, could be none' verl/trainer/main_ppo.py
fi
" || {
    echo "Failed to patch main_ppo.py."
    exit 1
}

# Fix permissions for verl directory
echo "Fixing file permissions..."
sudo chown -R ubuntu:ubuntu /home/ubuntu/alphaxiv-sandbox/paper-implementations/es-fine-tuning-paper/src/verl/ || {
    echo "Warning: Failed to fix permissions, but continuing..."
}

# Install VERL
echo "Installing VERL..."
sudo docker exec verl-es-fine-tuning-paper bash -c "cd /workspace/es-fine-tuning-paper/src/verl && pip3 install -e ." || {
    echo "Failed to install VERL."
    exit 1
}

wait 

# # Download and preprocess HotpotQA dataset
# echo "Downloading and preprocessing HotpotQA dataset..."
# sudo docker exec verl-es-fine-tuning-paper bash -c "cd /workspace/es-fine-tuning-paper && mkdir -p data/hotpotqa && python src/examples/data_preprocess/hotpotqa.py --local_dir data/hotpotqa" || {
#     echo "Failed to download and preprocess HotpotQA dataset."
#     exit 1
# }

# # Build HotpotQA search index
# echo "Building HotpotQA search index..."
# sudo docker exec verl-es-fine-tuning-paper bash -c "cd /workspace/es-fine-tuning-paper && if [ ! -f 'data/corpus/hotpotqa/hpqa_corpus.jsonl' ]; then mkdir -p data/corpus/hotpotqa && wget -q https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/corpus.jsonl.gz -O data/corpus/hotpotqa/corpus.jsonl.gz && gunzip -c data/corpus/hotpotqa/corpus.jsonl.gz > data/corpus/hotpotqa/hpqa_corpus.jsonl; else echo 'HotpotQA corpus already exists, skipping download.'; fi" || {
#     echo "Failed to download corpus data."
#     exit 1
# }

# sudo docker exec verl-es-fine-tuning-paper bash -c "cd /workspace/es-fine-tuning-paper && if [ -f 'data/corpus/hotpotqa/index.bin' ]; then echo 'HotpotQA search index already exists, skipping index build.'; else echo 'Building FAISS search index (this may take some time)...'; cd src/scripts/hotpotqa_search/ && python process_hotpotqa.py; fi" || {
#     echo "Failed to build search index."
#     exit 1
# }

sudo docker exec verl-es-fine-tuning-paper bash -c "pip install --upgrade wandb"
# Configure Weights & Biases if API key is set
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "Configuring Weights & Biases..."
    sudo docker exec verl-es-fine-tuning-paper bash -c "wandb login $WANDB_API_KEY" || {
        echo "Failed to login to Weights & Biases."
        exit 1
    }
fi

# Final comprehensive permission fix for all created files
echo "Fixing all file permissions..."
sudo chown -R ubuntu:ubuntu /home/ubuntu/alphaxiv-sandbox/paper-implementations/es-fine-tuning-paper/src/ 2>/dev/null || true
sudo chown -R ubuntu:ubuntu /home/ubuntu/alphaxiv-sandbox/paper-implementations/es-fine-tuning-paper/data/ 2>/dev/null || true
echo "Setup complete!"

# # Set up environment variables for Docker exec
# DOCKER_ENV=""
# if [ ! -z "$HYDRA_FULL_ERROR" ]; then
#     echo "HYDRA_FULL_ERROR is set, enabling full error traces..."
#     DOCKER_ENV="export HYDRA_FULL_ERROR=1 && "
# fi

# # Run training based on selected algorithm
# case "$ALGORITHM" in
#     ppo)
#         echo "=========================================="
#         echo "Starting PPO Training on HotpotQA"
#         echo "This will take approximately 22 hours on 4xH100 80GB GPUs"
#         echo "=========================================="

#         sudo docker exec verl-es-fine-tuning-paper bash -c "cd /workspace/es-fine-tuning-paper && ${DOCKER_ENV}cp src/examples/trainer/run_ppo_hotpotqa.sh ./ && bash run_ppo_hotpotqa.sh" || {
#             echo "Training failed."
#             exit 1
#         }
#         ;;
#     grpo)
#         echo "=========================================="
#         echo "Starting GRPO Training on HotpotQA"
#         echo "This will take approximately 20-22 hours on 4xH100 80GB GPUs"
#         echo "=========================================="

#         sudo docker exec verl-es-fine-tuning-paper bash -c "cd /workspace/es-fine-tuning-paper && ${DOCKER_ENV}cp src/examples/trainer/run_grpo_hotpotqa.sh ./ && bash run_grpo_hotpotqa.sh" || {
#             echo "Training failed."
#             exit 1
#         }
#         ;;
#     *)
#         echo "Unknown algorithm: $ALGORITHM"
#         exit 1
#         ;;
# esac

# echo "=========================================="
# echo "Training Complete!"
# echo "=========================================="
# echo "Check the results and logs in the es-fine-tuning-paper directory."
# echo "You can also check Weights & Biases for training metrics if configured."

