# Agent-R1: A Tutorial-Style Guide to Training LLM Agents with RL

<p align="center">
  <a href="https://arxiv.org/abs/2511.14460">
  <img src="https://img.shields.io/badge/Paper-Arxiv-b31b1b?logo=arxiv&logoColor=white" alt="Paper Arxiv">
</a>
</p>

Welcome to **Agent-R1**! This repository is designed not just as a tool, but as a guide to understanding how to train Large Language Model (LLM) agents using Reinforcement Learning (RL).

If you've ever wondered how to move beyond simple prompting and supervised fine-tuning to agents that can *learn* from their interactions with tools and environments, you're in the right place.

## What are we trying to solve?

Training agents is hard. Unlike standard chatbots that just generate text, agents need to:
1.  **Reason** about a problem.
2.  **Act** by calling tools (search, calculators, code execution).
3.  **Observe** the results of those actions.
4.  **Iterate** until the task is solved.

Standard RLHF (Reinforcement Learning from Human Feedback) pipelines are typically designed for single-turn interactions:
`Prompt -> Response -> Reward`

But agents operate in a **multi-turn** loop. We need a framework that treats the entire interaction trajectory as a training episode.

## Built on VERL

Agent-R1 is built on top of **VERL** (VolcEngine Reinforcement Learning). VERL provides the robust, scalable infrastructure needed for distributed RL training (handling the "heavy lifting" of PPO/GRPO across GPUs).

**Agent-R1** adds the "Agent" logic on top of this infrastructure. Think of VERL as the engine, and Agent-R1 as the car designed for a specific type of race (agentic tasks).

## Key Concepts & Contributions

To make RL work for agents, Agent-R1 introduces two critical concepts that you'll see annotated throughout the codebase:

### 1. Algorithm: Agent vs. LLM RL

Reinforcement learning for Agents differs significantly from LLM (Chatbots, Reasoners). The key distinction lies in the fact that: **a complete Agent trajectory typically involves multi-turn interaction**, requiring **multiple tool calls** to solve user queries. Below, we formalize this distinction using the Markov Decision Process (MDP) framework.

**In the context of LLM:**
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

To better understand the difference, we can define the probability of a trajectory:

$$ P(\tau) = P(X) \prod_{j=1}^{m} \left( P(C_j | X, a_{1:t_{j-1}}, C_{1:j-1}) \prod_{i=1}^{t_j} \pi_\theta(a_i | X, a_{1:i-1}, C_{1:j}) \right) $$

where:
- $X$ is the sequence of the current prompt
- $C_j$ is the result of the $j$-th tool call and $m$ is the number of tool calls
- $a_t$ is the token selected from the vocabulary
- $t_j$ is the number of token responses between the $j-1$th and $j$-th tool calls, $0<t_1+t_2+...+t_m<t$

This richer reinforcement learning framework allows Agent-R1 to train LLMs that learn effective strategies for when and how to use tools across multi-turn interactions. By optimizing over entire trajectories rather than single responses, we can apply algorithms like PPO and GRPO to develop agents that reason effectively before taking actions.

### 2. Advantage & Loss Masks
This is the technical "secret sauce". In a multi-turn trajectory, the context window contains a mix of:
- User instructions (Input)
- Model generations (Thoughts/Actions)
- Tool outputs (Environment Observations)

We **only** want to update the model based on its own actions. We don't want to calculate gradients for the tool outputs (the model didn't generate them, the environment did).

Agent-R1 implements sophisticated **masking**:
- **Loss Mask**: Ensures we only calculate loss on the tokens the model generated.
- **Advantage Mask**: Ensures that the "credit" for a good outcome is properly attributed to the specific actions the model took, skipping over the deterministic tool outputs.

## Simplicity First

Inspired by projects like `Nanochat`, we aim for code readability and ease of understanding.
- **Focused Algorithms**: We strictly support **PPO** (Proximal Policy Optimization) and **GRPO** (Group Relative Policy Optimization). We've removed support for older or less stable algorithms (like REINFORCE, ReMax) to keep the codebase clean.
- **Tutorial Style**: The code is heavily commented to explain *why* we are doing things, not just *what* is happening.

## Running the Code

We provide three main scripts to help you get started, depending on your environment and needs.

### 1. Ruuning our trained checkpoints!

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
Once training is complete, you can interact with your trained agent using the inference script. For more detailed information, please refer to [docs/inference/inference.md](docs/inference/inference.md).

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
- [Infrernce](docs/inference/inference.md)
- [Customizing Tools for Multi-hop QA](docs/tutorial/multihopqa.md)
- [Customizing Tool Environment for ReTool](docs/tutorial/retool.md)

Happy training!
