# Agent-R1: Training LLM Agents with RL

Welcome to **Agent-R1**! This repository is an easy-to-follow implementation of the [Agent-R1](https://alphaxiv.org/abs/2511.14460) paper which provides a framework for training LLMs with RL. The original implementation can be found [here](https://github.com/0russwest0/Agent-R1/tree/main/agent_r1) and is built on top of [VERL](https://github.com/volcengine/verl).

This repo provides a straightforward guide to training your own deep research agent with the Agent-R1 framework. Rather than being a framework or library, we structure the codebase as an annotated guide to easily follow the different components required to train an agent with RL.

## What are we trying to solve?

Training agents is hard. Unlike standard chatbots that just generate text, agents need to:
1.  **Reason** about a problem.
2.  **Act** by calling tools (search, calculators, code execution).
3.  **Observe** the results of those actions.
4.  **Iterate** until the task is solved.

Standard RLHF (Reinforcement Learning from Human Feedback) pipelines are typically designed for what are called "single-turn" interactions:

**Single-turn example:**
```
User: "Write a poem about the ocean"
Model: [generates poem]
Reward: Quality score based on poem
```

On the other hand, agents operate in a **multi-turn** fashion:

**Multi-turn example:**
```
User: "Which magazine was started first, Arthur's Magazine or First for Women?"
Model: [searches for "Arthur's Magazine"]
Environment: [returns Wikipedia article with founding date 1844]
Model: [searches for "First for Women"]
Environment: [returns Wikipedia article with founding date 1989]
Model: "Arthur's Magazine was started first (1844)"
Reward: Correctness of final answer
```

We need a framework that treats the entire interaction trajectory as a training episode.

## Built on VERL

Agent-R1 is built on top of **VERL** (VolcEngine Reinforcement Learning). VERL provides the infrastructure needed for distributed RL training (handling the "heavy lifting" of PPO/GRPO across GPUs). RL training has a lot of moving components: training the critic, training the agent, sampling the agent on the environment, sampling the reference agent, etc. A framework like VERL helps juggle these different moving parts while trying to maximize GPU utilization. More on this later.

**Agent-R1** adds the "Agent" logic on top of this infrastructure.

## Agentic RL Concepts

To make RL work for agents, Agent-R1 introduces two critical concepts that you'll see annotated throughout the codebase:

### 1. Formulation: Agent vs. LLM RL

The key distinction in RL training for agents as opposed to a traditional LLM task like chatbots or reasoners is the multi-turn interaction. An agent is required to make several tool calls to interact with the environment and solve user queries. This distinction can be formalized using the Markov Decision Process (MDP) framework:

**In the context of an LLM:**
- **State**: Simply the sequence of the input prompt and all generated text so far.
- **Action**: Selecting the next token from the vocabulary to add to the sequence.
- **Transitions**: Straightforward addition of the selected token to the existing sequence.
- **Rewards**: Typically only provided at the end of the sequence generation.

**For Agents, the components are more complex:**
- **State**: Includes not only the input and generated text, but also all tool responses from previous interactions.
- **Action**: Still involves selecting the next token, but some tokens can trigger tool calls.
- **Transitions**:
  - *Regular tokens*: Simply add to the sequence like in traditional LLM training.
  - *Tool-triggering tokens*: Cause external tool execution that produces responses, **introducing significant stochasticity** into state transitions unlike the deterministic nature of standard LLM generation.
- **Rewards**: Can be provided at multiple points:
  - After each tool call, which **naturally creates process-level rewards** based on the quality and effectiveness of tool usage.
  - At the end of the complete interaction based on overall task completion.

**Formal Definition:**

The probability of a trajectory can now be written out as:

$$ P(\tau) = P(X) \prod_{j=1}^{m} \left( P(C_j | X, a_{1:t_{j-1}}, C_{1:j-1}) \prod_{i=1}^{t_j} \pi_\theta(a_i | X, a_{1:i-1}, C_{1:j}) \right) $$

where:
- $X$ is the sequence of the current prompt
- $C_j$ is the result of the $j$-th tool call
- $m$ is the number of tool calls
- $a_t$ is the token selected from the vocabulary
- $t_j$ is the number of token responses between the $j-1$-th and $j$-th tool calls, $0<t_1+t_2+...+t_m<t$

This richer reinforcement learning framework allows Agent-R1 to train LLMs that learn effective strategies for when and how to use tools across multi-turn interactions. By optimizing over entire trajectories rather than single responses, we can apply algorithms like PPO and GRPO to develop agents that reason effectively before taking actions.

### 2. Advantage & Loss Masks
This is one of the main contributions of the Agent-R1 work. In a multi-turn trajectory, the context window contains a mix of:
- User instructions (Input)
- Model generations (Thoughts/Actions)
- Tool outputs (Environment Observations)

We **only** want to update the model based on its own actions. We don't want to calculate gradients for the tool outputs (the model didn't generate them, the environment did).

Agent-R1 implements straightforward **masking** to handle this:
- **Loss Mask**: Ensures we only calculate loss on the tokens the model generated.
- **Advantage Mask**: Ensures that the "credit" for a good outcome is properly attributed to the specific actions the model took, skipping over the deterministic tool outputs.

## 3. Multi-GPU Hybrid RL Training Architecture

A good chunk of the Agent-R1 codebase, or for that matter any RL LLM framework, is infrastructure for efficiently training in a multi-GPU setting. Unlike traditional supervised finetuning where we can use libraries like Torch FSDP or Megatron-LM out of the box, there's a little bit more wrestling we have to do to get things to work. A lot of this is handled by libraries like VERL, but we figured it would be useful to walk you through how and why RL training is different from SFT.

### The Challenge: Why Hybrid?

Training LLM agents with RL involves three computationally distinct phases that don't share the same optimal parallelization strategy:

1. **Rollout (Inference)**: Generate agent trajectories with tool calls
   - Requires high throughput and low latency
   - Benefits from tensor parallelism (TP) for large batch inference
   - Memory-bound (KV cache dominates)

2. **Training (Backward)**: Update model parameters via PPO/GRPO
   - Requires gradient computation across long sequences  
   - Benefits from data parallelism (DP) and FSDP
   - Compute-bound (gradient accumulation, optimizer updates)

3. **Reference Policy**: Compute reference log probabilities
   - Similar to rollout but read-only
   - Can be offloaded to CPU or share GPUs strategically

### Architecture Overview

To coordinate across these different phases, Agent-R1 uses [Ray](https://github.com/ray-project/ray) for orchestration. Ray lets us spawn different workers (actor, critic, reference policy) that can dynamically share the same physical GPUs by switching between training and inference modes. The architecture has three key components:

#### Resource Pool Management
The training process dynamically allocates GPUs to different roles through `ResourcePoolManager`:

```python
# Example: 4 GPUs per node, 1 node
resource_pool_spec = {
    "global_pool": [4]  # 4 GPUs available
}

# Map different computational roles to the pool
mapping = {
    Role.ActorRollout: "global_pool",   # Training + Inference
    Role.Critic: "global_pool",          # Value function (PPO only)
    Role.RefPolicy: "global_pool",       # Reference policy
}
```

**Key insight**: All roles can share the same physical GPUs by "time-multiplexing" (running different phases sequentially). The system transitions between:
- **Training mode**: Model wrapped in FSDP for gradient computation
- **Inference mode**: Model loaded into vLLM with tensor parallelism for fast generation

#### Worker Architecture

Three specialized workers handle different computational phases:

1. **ActorRolloutRefWorkerR1** (Actor + Rollout)
   - **Training**: Uses FSDP to shard model parameters across GPUs
   - **Inference**: Dynamically loads weights into vLLM engine with TP
   - **Transition**: `FSDPVLLMShardingManager` synchronizes weights between FSDP ↔ vLLM
   - Handles both actor updates and trajectory rollouts

2. **CriticWorkerR1** (PPO only)
   - Learns value function $V(s)$ for GAE advantage estimation
   - Wrapped in FSDP for training
   - Can be CPU-offloaded when memory is tight

3. **RefPolicy Worker** (Reference policy)
   - Frozen copy of the policy for KL divergence computation
   - Shares architecture with ActorRollout but no gradient updates
   - Can be CPU-offloaded to save GPU memory

#### The Hybrid Engine: FSDP ↔ vLLM

The most intricate part is the **hybrid engine** that switches between training and inference modes:

**Training Mode (FSDP)**:
- Model sharded across GPUs using PyTorch FSDP
- Each GPU holds a fraction of parameters
- Supports gradient checkpointing to reduce memory
- Configuration: `actor.fsdp_config.param_offload`, `optimizer_offload`

**Inference Mode (vLLM)**:
- Model loaded with Tensor Parallelism for fast generation
- KV cache managed by vLLM for batched inference  
- Configuration: `rollout.tensor_model_parallel_size`

**Synchronization**:
```python
# Managed by FSDPVLLMShardingManager
# 1. Before rollout: FSDP state_dict → vLLM weights
rollout_sharding_manager.sync_weights_to_inference()

# 2. After actor update: vLLM stays dormant
# FSDP handles all gradient updates

# 3. Repeat for next iteration
```

### Training Loop Workflow

Here's how a single training step flows across the distributed system:

```
1. Rollout (vLLM): Generate trajectories with tool calls
   ├─ Actor in inference mode (TP=2, for example)
   ├─ Tool environment executes actions  
   └─ Create action_mask (1=model tokens, 0=tool outputs)

2. Compute Rewards: Evaluate trajectory quality
   └─ Reward function runs on driver process

3. Reference Policy: Compute KL penalty
   └─ RefPolicy worker (vLLM mode)

4. Critic (PPO only): Estimate values
   └─ Critic worker (FSDP mode)

5. Advantage Estimation: Compute GAE or GRPO advantages
   ├─ Uses action_mask to only credit model actions
   └─ Driver process (lightweight computation)

6. Actor Update: Train policy via PPO
   ├─ Switch to FSDP training mode
   ├─ Mini-batch iteration with gradient accumulation
   ├─ Clip gradients and update parameters
   └─ Sync weights back to vLLM for next rollout
```

### Parallelism Strategy Details

**Data Parallelism (DP)**: Implicit through FSDP's sharding across GPUs. Each GPU processes different micro-batches.

**Tensor Parallelism (TP)**: Used in vLLM rollout. Model layers split across GPUs.
- Example: With TP=2, each attention head subset runs on different GPUs
- Configured via: `rollout.tensor_model_parallel_size=2`

**Fully Sharded Data Parallel (FSDP)**: PyTorch's native sharding for training.
- Parameters sharded across GPUs
- Gradients computed locally, then all-gathered
- Supports CPU offloading: `fsdp_config.param_offload=True`

**Sequence Parallelism**: Optional, via Ulysses SP in FSDP for very long sequences.

### Memory Optimization Techniques

Agent-R1 employs several strategies to fit large models in limited GPU memory:

1. **Gradient Checkpointing**: Recompute activations during backward pass
   ```python
   actor_rollout_ref.model.enable_gradient_checkpointing=True
   ```

2. **CPU Offloading**: Move parameters/optimizer states to CPU when not needed
   ```python
   actor.fsdp_config.param_offload=True
   actor.fsdp_config.optimizer_offload=True
   ```

3. **Dynamic Batch Sizing**: Automatically adjust batch size based on sequence length
   ```python
   actor.use_dynamic_bsz=True
   actor.ppo_max_token_len_per_gpu=8192
   ```

4. **Response Length Truncation**: Limit agent responses to prevent memory explosion
   ```python
   data.max_response_length=8192
   data.max_response_length_single_turn=1024
   ```

5. **Prefix Caching**: vLLM caches common prompt prefixes (enabled by default)


**Breakdown**:
- 4 GPUs share all three roles (Actor+Rollout, Ref, Critic)
- Rollout uses TP=2 → 2 GPUs for vLLM inference
- Training uses all 4 GPUs via FSDP
- Gradient accumulation: 64 mini-batch / 2 per GPU = 32 steps
- Reference policy offloaded to CPU to reduce memory pressure

### Why This Complexity?

This hybrid architecture might seem over-engineered, but it solves real problems:

- **Agent trajectories are long**: Multi-turn tool usage creates 4-8K+ token sequences
- **Inference must be fast**: Agents generate iteratively; slow rollouts bottleneck training  
- **Training needs gradients**: Can't use vLLM's inference-optimized graph for backprop
- **Memory is scarce**: 7B models with long contexts barely fit on consumer GPUs

The hybrid FSDP+vLLM approach gives you:
- **Fast rollouts** via vLLM's optimized inference
- **Efficient training** via FSDP's gradient computation
- **Flexible resource sharing** via Ray's orchestration

## Simplicity First

Inspired by projects like `Nanochat`, we aim for code readability and ease of understanding.
- **Focused Algorithms**: We strictly support **PPO** (Proximal Policy Optimization) and **GRPO** (Group Relative Policy Optimization). We've removed support for older or less stable algorithms (like REINFORCE, ReMax) to keep the codebase clean.
- **Readable code**: A lot of the original Agent-R1 codebase can inherit classes within the VERL library. This reduces line count and makes code easier to follow.
- **Tutorial Style**: The code is heavily commented to explain *why* we are doing things, not just *what* is happening.

## Training your own Deep Research Agent

With the Agent-R1 framework we can now train a Qwen-2.5 model on HotpotQA, a multi-hop Q&A dataset. To answer questions like "Which magazine was started first Arthur's Magazine or First for Women?", the agent has access to a tool that can search a Wikipedia-style corpus. The answers to these questions also require the agent to make several search calls, thus requiring "multi-turn" interactions with the environment. We provide three main scripts to help you get started. For training, we used 4 H100 (80GB SXM) GPUs on [Lambda Labs](https://lambda.ai/), with training taking just under 6 hours.

### 1. Running our trained checkpoints!

Use the below script to run inference with our provided trained checkpoints.
Available at [Agent-R1 Trained Checkpoints](https://huggingface.co/collections/alphaXiv/agent-r1)

```bash
# Run inference (interactive chat)
bash inference.sh --use-hf-model --hf-model-path alphaXiv/Qwen-2.5-1.5b-instruct-grpo
```

### 2. Dockerized Setup (Recommended)
If you want a completely isolated environment with all dependencies handled for you, use the standard speedrun script. This requires Docker with GPU support.

```bash
# Run PPO (default)
./speedrun.sh

# Or specify the algorithm
./speedrun.sh grpo
```

This script will:
- Pull the VERL Docker image.
- Start a container.
- Install dependencies and download data.
- Launch the training job inside the container.

### 3. Non-Docker Setup ("Speedrun No Docker")
If you prefer to run everything directly on your machine (e.g., you are already in a configured environment like Lambda Labs GPU Base image), use the no-docker version.

We used GPU Base Image 22.4 on Lambda Labs with 4XH100s GPUs (80GB SXM) and the batch sizes has been configured accordingly.

```bash
# Run PPO (default)
bash speedrun-nodocker.sh

# Or specify the algorithm
bash speedrun-nodocker.sh grpo
```

This script will:
- Install system and Python dependencies locally.
- Set up a virtual environment (`~/verlenv`).
- Download data and build indices.
- Launch the training job directly.

### 4. Inference
Once training is complete, you can interact with your trained agent using the inference script. 

```bash
# Run inference (interactive chat)
bash inference.sh
```

This script will:
- Convert your training checkpoints to Hugging Face format.
- Start a vLLM server to host the model.
- Launch an interactive chat interface where you can test the agent.

You can also customize the paths if your checkpoints are in a different location:
```bash
bash inference.sh --checkpoint-dir /path/to/checkpoints
```

## Where to Look in the Code

- **`src/agent_r1/`**: The core logic.
    - **`training/`**: Where the RL magic happens (look for the masking logic here).
    - **`tool/`**: How we define tools and environments.
- **`src/verl/`**: The underlying infrastructure (submodule).

## Extending Agent-R1

Agent-R1 provides a flexible architecture for creating custom tools and tool environments.

- **BaseTool**: Individual tools that agents can use.
- **BaseToolEnv**: Tool environments that define the state transition function.

For detailed guidance:
- [Infrerence](docs/inference/inference.md)
- [Customizing Tools for Multi-hop QA](docs/tutorial/multihopqa.md)
- [Customizing Tool Environment for ReTool](docs/tutorial/retool.md)

Happy training!
