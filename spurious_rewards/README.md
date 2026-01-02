<div align="center">

# 💭 Spurious Rewards: Rethinking Training Signals in RLVR

[![Paper](https://img.shields.io/badge/Paper-000000.svg?style=for-the-badge&logo=arxiv&logoColor=white)](http://arxiv.org/abs/2506.10947)

</div>

Welcome to **Spurious Rewards**! This repository provides an implementation of the [Spurious Rewards](http://arxiv.org/abs/2506.10947) paper, which investigates how different reward functions in Reinforcement Learning from Verifier Rewards (RLVR) can lead to spurious correlations that improve performance without truly understanding the task. The implementation is built on top of [TTRL](https://github.com/PRIME-RL/TTRL), a framework for efficient RL training of large language models.

This repo serves as a practical guide to reproducing the experiments from the paper, exploring how reward design impacts learning in mathematical reasoning tasks. Rather than just providing code, we structure it as an annotated walkthrough to understand the effects of different reward signals.

## What are we trying to solve?

Training language models for complex reasoning tasks like mathematics is challenging. Traditional supervised fine-tuning can achieve high accuracy on training data, but often fails to generalize to harder problems or different formats. Reinforcement Learning from Human Feedback (RLHF) has shown promise, but the choice of reward function critically affects what the model actually learns.

The key insight of this work is that **reward functions can create spurious correlations** - patterns that help the model score well on the reward without truly solving the underlying task. For example:

- A reward that gives points for using Python code might encourage models to mention Python even when it's unnecessary
- Format-based rewards might prioritize box notation over correct mathematics
- Random rewards can still improve performance through unintended correlations

We need to understand:
1. **How different rewards affect learning trajectories**
2. **Which rewards lead to genuine improvement vs. spurious gains**
3. **How to design reward functions that promote true mathematical understanding**

## Built on TTRL

Like Agent-R1, this implementation uses **TTRL** (Transformers for RL from PRIME INTELLECT) for the heavy lifting of distributed RL training. TTRL handles the complex orchestration of PPO/GRPO algorithms across multiple GPUs, managing the interplay between:
- Policy training (the main model)
- Value function training (for advantage estimation)
- Reference policy sampling
- Rollout generation

**Spurious Rewards** adds the experimental framework on top of this infrastructure to systematically compare different reward functions.

## Key Concepts: Spurious Correlations in RLVR

The paper introduces several critical concepts for understanding reward design in mathematical reasoning:

### 1. RLVR Framework

Reinforcement Learning from Verifier Rewards (RLVR) treats mathematical problem-solving as a reinforcement learning problem where:
- **State**: The problem statement and current solution attempt
- **Action**: Generating the next token in the solution
- **Reward**: Provided by a verifier function that checks correctness
- **Environment**: The mathematical domain (algebra, geometry, etc.)

### 2. Reward Function Types

The paper examines different reward designs:

- **`math`**: Full mathematical equivalence checking (default)
- **`box_only_format`**: Rewards only for proper boxed answer format
- **`contain_python_wo_backticks`**: Rewards for mentioning Python code
- **`random0.5`**: Random reward with 50% probability

### 3. Spurious Correlations

The key finding is that suboptimal rewards can still improve performance through spurious correlations. For example:
- Models trained with Python-mentioning rewards learn to include Python even for simple problems
- Format rewards teach box notation but may not improve actual math skills
- Even random rewards can create beneficial patterns through trial-and-error

## Experimental Setup

The experiments use:
- **Base Model**: Qwen2.5-Math-7B
- **Dataset**: DeepScaleR (filtered mathematical problems)
- **Training**: GRPO/PPO algorithms
- **Evaluation**: MATH-500, AIME-2024/2025, AMC benchmarks

## Architecture Overview

The codebase follows a similar hybrid architecture to Agent-R1:

### Training Scripts
Located in `src/spurious_rewards/code/scripts/`:
- `rlvr_deepscaler_grpo_qwen_ground_truth.sh`: Main GRPO training script
- `rlvr_deepscaler_grpo_qwen_majority_vote.sh`: Uses majority voting labels
- `rlvr_deepscaler_grpo_qwen_random.sh`: Random reward experiments

### Data Processing
- `data/`: Contains processed datasets
- `scripts/`: Data preparation and labeling scripts
- Supports different data sources via the `TASK` variable

### Evaluation Framework
- `eval_checkpoint.py`: Benchmark evaluation script
- `export_checkpoint.py`: Convert DeepSpeed checkpoints to HF format
- Supports multiple datasets with configurable temperature/shards

## Simplicity First

Following the philosophy of readable RL code:
- **Focused on GRPO/PPO**: Clean implementations without legacy algorithms
- **Well-commented scripts**: Explain hyperparameters and their effects
- **Modular reward functions**: Easy to add new reward types
- **Tutorial-style code**: Heavily annotated for learning

## Reproducing the Experiments

### 1. Quick Evaluation of Pre-trained Models

Test the base Qwen2.5-Math-7B model on benchmarks:

```bash
cd src/spurious_rewards/code
python eval_checkpoint.py --model_path Qwen/Qwen2.5-Math-7B --datasets MATH-500,AIME-2024,AIME-2025,AMC
```

### 2. Dockerized Training (Recommended)

For isolated environment with all dependencies:

```bash
# Run GRPO training with ground truth rewards
./speedrun.sh
```


This will:
- Set up the training environment
- Download/configure the dataset
- Launch distributed training across GPUs
- Save checkpoints every 50 steps

### 4. Custom Reward Experiments

To test different reward functions:

```bash
# Edit the script to change REWARD variable
REWARD="box_only_format"  # or "contain_python_wo_backticks", "random0.5"

```

### 5. Evaluation of Trained Models

After training, evaluate your checkpoints:

```bash
./inference.sh -c /path/to/checkpoint/dir -s 50
```

## Evaluation Results: @k Scores Summary

This document summarizes the evaluation results from the `eval_outputs` and `rlvr_eval_outputs` directories, specifically focusing on the @k score metrics.

## Model Comparison: Base vs RLVR

### Performance Comparison Table (7B model)

| Dataset | Model | avg@8 | pass@8 | avg@1 | pass@1 |
|---------|-------|-------|--------|-------|--------|
| AIME-2024 | Qwen2.5-Math-7B (Base) | 0.121 | 0.333 | - | - |
| AIME-2024 | Qwen2.5-Math-7B (RLVR) | 0.233 | 0.467 | - | - |
| AIME-2025 | Qwen2.5-Math-7B (Base) | 0.054 | 0.200 | - | - |
| AIME-2025 | Qwen2.5-Math-7B (RLVR) | 0.167 | 0.300 | - | - |
| AMC | Qwen2.5-Math-7B (Base) | 0.330 | 0.735 | - | - |
| AMC | Qwen2.5-Math-7B (RLVR) | 0.572 | 0.747 | - | - |
| MATH-500 | Qwen2.5-Math-7B (Base) | - | - | 0.494 | 0.494 |
| MATH-500 | Qwen2.5-Math-7B (RLVR) | - | - | 0.788 | 0.788 |

### Configuration Details

| Dataset | Temperature | Rollouts | GPU Type |
|---------|-------------|----------|-----------|
| AIME-2024, AIME-2025, AMC | 0.6 | 8 | NVIDIA A100-SXM4-40GB |
| MATH-500 | 0.0 | 1 | NVIDIA A100-SXM4-40GB |


### Model Architecture
- **Base Model**: Qwen2.5-Math-7B (standard mathematical reasoning model)
- **RLVR Model**: Qwen2.5-Math-7B enhanced with Reinforcement Learning from Verifier Rewards
- **Hardware**: All evaluations used NVIDIA A100-SXM4-40GB GPUs (Base: 4 shards, RLVR: 2 shards)

## Score Interpretation

The @k scores represent:
- **avg@k**: Average success rate across k attempts/rollouts
- **pass@k**: Probability of at least one success in k attempts

The difference between avg@k and pass@k scores indicates the value of multiple attempts, with larger gaps suggesting the model benefits significantly from additional rollouts.

## Conclusion
These results validate the effectiveness of applying RLVR to the Qwen2.5-Math-7B base model, demonstrating enhanced mathematical reasoning capabilities across diverse problem types and difficulty levels.

## Where to Look in the Code

- **`src/spurious_rewards/code/scripts/`**: Training scripts for different reward experiments
- **`src/spurious_rewards/code/data/`**: Processed datasets and data loading
- **`src/spurious_rewards/code/eval_checkpoint.py`**: Evaluation logic
- **`src/ttrl/`**: Underlying RL infrastructure (submodule)


## Hardware Requirements

- **Training**: 8x H100 GPUs (80GB-SX) recommended (Lambda Labs with GPU Base Image 22.04 was used by us)
- **Evaluation**: Single H100 for base and trained checkpoints
- **Memory**: 80GB+ GPU memory for large models and long sequences

Happy experimenting!
