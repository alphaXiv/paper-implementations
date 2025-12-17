<h1 align="center"> Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning </h1>

<p align="center">
  <a href="https://arxiv.org/abs/2511.14460">
  <img src="https://img.shields.io/badge/Paper-Arxiv-b31b1b?logo=arxiv&logoColor=white" alt="Paper Arxiv">
</a>
 
## Overview

**Agent-R1** is an open-source framework designed to accelerate research and development at the critical intersection of **RL** and **Agent**. Our framework employs **End-to-End** reinforcement learning to train agents in specific environments. Developers need only define domain-specific tools and reward functions to extend Agent-R1 to their unique use cases, eliminating the need for complex workflow engineering. We hope our modest contribution can benefit the open-source community, making it easier for researchers and developers to create and explore agents in their own domains, collectively advancing the development of autonomous agents. For more details on the algorithm, see [algorithm doc](https://github.com/alphaXiv/paper-implementations/tree/omnidocbench/agent_r1/docs/algorithm).


## Key Features

- **Multi-turn Tool Calling**: End-to-end reinforcement learning on complete interaction trajectories, allowing agents to learn from sequences of actions
- **Multi-tool Coordination**: Train agents to effectively coordinate and use multiple tools together to solve complex tasks
- **Process Rewards**: Assign rewards for each tool call based on its effectiveness, balanced with outcome rewards through normalization
- **Custom Tools and Environments**: Compatible with mainstream LLM tool calling formats, making it easy to extend with your own tools and scenarios
- **Multiple RL Algorithms**: Supports diverse reinforcement learning approaches including `PPO`, `GRPO`, and `REINFORCE++`
- **Multi-modal Support**: Compatible with vision-language models (VLMs) and multi-modal reinforcement learning

## Upcoming Features

- **Expanded Model Support**: Integration with more foundation models beyond the currently supported Qwen
- **Additional Use Cases**: More example implementations across diverse scenarios and domains

## Quick Start

The easiest way to set up and train Agent-R1 is using the automated speedrun script:

Use 4xH100 80 SXM (we used Lambda Labs with GPU Base image  22.04 (12.8 CUDA) for this setup)

```bash
./speedrun.sh
```

This script will automatically:
1. Pull the required Docker image (VERL environment with CUDA, PyTorch, vLLM, and FlashInfer)
2. Start a Docker container with GPU support
3. Install Agent-R1 dependencies
4. Initialize Git submodules
5. Install VERL framework
6. Download and preprocess the HotpotQA dataset
7. Build the HotpotQA search index
8. Run training on HotpotQA using PPO, GRPO, and RPP algorithms

**Prerequisites:**
- Docker with GPU support (nvidia-docker or Docker with `--gpus all` flag - best to use already setup docker and CUDA environment like Lambda Labs GPU Base image 22.04)
- NVIDIA CUDA-compatible GPUs (4xA100 80GB recommended for full training)
- At least 500GB free disk space for models and data
- Optional: Weights & Biases API key for logging (`export WANDB_API_KEY="your_key_here"`)

### For Detailed Documentation

- [Environment Setup](https://github.com/alphaXiv/paper-implementations/tree/omnidocbench/agent_r1/docs/getting_started/installation.md)
- [Quick Start: Try Default Search Tool on HotpotQA](https://github.com/alphaXiv/paper-implementations/tree/omnidocbench/agent_r1/docs/getting_started/quickstart.md) (see also: [Results on HotpotQA](https://github.com/alphaXiv/paper-implementations/tree/omnidocbench/agent_r1/docs/getting_started/quickstart.md#5-results-on-hotpotqa))

## Extending Agent-R1 with Your Own Tools and Environments

Agent-R1 provides a flexible architecture for creating custom tools and tool environments to suit various agent applications. Our framework is built on two key abstractions:

1. **BaseTool**: Individual tools that agents can use to interact with external systems
2. **BaseToolEnv**: Tool environments that define the state transition function for agent-tool interactions

For detailed guidance on extending Agent-R1, refer to our tutorials:

- [Customizing Tools for Multi-hop QA](https://github.com/alphaXiv/paper-implementations/tree/omnidocbench/agent_r1/docs/tutorial/multihopqa.md): Learn how to create and customize tools for retrieving information across multiple knowledge sources
- [Customizing Tool Environment for ReTool](https://github.com/alphaXiv/paper-implementations/tree/omnidocbench/agent_r1/docs/tutorial/retool.md): Understand how to implement tool environments that integrate code execution with LLM reasoning

Additional resources are available in the codebase:
- Example tools: `src/agent_r1/tool/tools/`
- Example environments: `src/agent_r1/tool/envs/`
- Data preprocessing: `src/examples/data_preprocess/`
- Reward functions: `src/verl/utils/reward_score/`