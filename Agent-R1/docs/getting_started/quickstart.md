### Quick Start: Try Default Search Tool on HotpotQA

#### 1. Start Docker Container
```bash
sudo docker run -d --gpus all --name verl-agent-r1 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v Agent-R1:/workspace/Agent-R1 hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0 sleep infinity
```

#### 2. Install Dependencies
```bash
sudo docker exec verl-agent-r1 bash -c "cd /workspace/Agent-R1 && pip3 install -e ."
```

#### 3. Initialize Git Submodules
```bash
sudo docker exec verl-agent-r1 bash -c "git config --global --add safe.directory '*' && cd /workspace/Agent-R1 && git submodule update --init --recursive"
```

#### 4. Install VERL
```bash
sudo docker exec verl-agent-r1 bash -c "cd /workspace/Agent-R1/verl && pip3 install -e ."
```

#### 5. Download and Preprocess HotpotQA Dataset
```bash
sudo docker exec verl-agent-r1 bash -c "cd /workspace/Agent-R1 && mkdir -p data/hotpotqa && python examples/data_preprocess/hotpotqa.py --local_dir ./data/hotpotqa"
```

This script will:
- Download the HotpotQA dataset directly from the source
- Process the data into the format required by Agent-R1
- Save the processed data as train.parquet and validation.parquet in the specified directory

#### 6. Build HotpotQA Search Index
```bash
sudo docker exec verl-agent-r1 bash -c "cd /workspace/Agent-R1 && mkdir -p data/corpus/hotpotqa && wget -q https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/corpus.jsonl.gz -O data/corpus/hotpotqa/corpus.jsonl.gz && gunzip -c data/corpus/hotpotqa/corpus.jsonl.gz > data/corpus/hotpotqa/hpqa_corpus.jsonl"
```

```bash
sudo docker exec verl-agent-r1 bash -c "cd /workspace/Agent-R1/scripts/hotpotqa_search && python process_hotpotqa.py"
```

This script will:
- Load the corpus data
- Generate embeddings using the BAAI/bge-large-en-v1.5 model
- Build a FAISS index for efficient similarity search
- Save the embeddings and index files in the data/corpus/hotpotqa directory

#### 7. Configure Weights & Biases (Optional)
```bash
sudo docker exec verl-agent-r1 bash -c "wandb login YOUR_API_KEY"
```

#### 8. Run PPO/REINFORCE++/GRPO Training with Qwen2.5-1.5B-Instruct
```bash
# Run the PPO training script (Will take around 10-12 hours on 4xA100 80 GB)
sudo docker exec verl-agent-r1 bash -c "cd /workspace/Agent-R1 && cp examples/trainer/run_ppo_hotpotqa.sh ./ && bash run_ppo_hotpotqa.sh"

# Run the REINFORCE++ training script (Will take around 12 hours on 4xA100 80GB)
sudo docker exec verl-agent-r1 bash -c "cd /workspace/Agent-R1 && cp examples/trainer/run_rpp_hotpotqa.sh ./ && bash run_rpp_hotpotqa.sh"

# Run the GRPO training script (Will take around 15 hours on 4xA100 80GB)
sudo docker exec verl-agent-r1 bash -c "cd /workspace/Agent-R1 && cp examples/trainer/run_grpo_hotpotqa.sh ./ && bash run_grpo_hotpotqa.sh"
```

### 5. Results on HotpotQA

#### PPO

![ppo](../../image/ppo.jpg)

#### REINFORCE++

![rpp](../../image/rpp.jpg)

#### GRPO

![grpo](../../image/grpo.jpg)

We can see that the model (Qwen2.5-1.5B-Instruct) effectively learns to think and then invoke the tool in multiple rounds when faced with challenging multi-hop questions, ultimately achieving improved the EM results. The effectiveness of different reinforcement learning algorithms varies, but the general trend is the same.

Notably, our experiments reveal a striking correlation: EM scores, number of tool calls (turns), and final response length all display consistent trends across training. This demonstrates a novel dimension of scaling laws—one that relates to the frequency of agent-environment interactions. As the agent learns to interact more effectively with its environment through multiple tool calls, performance improves proportionally, suggesting that the ability to engage in multiple rounds of environment interaction may be as crucial to agent performance as traditional scaling factors.
