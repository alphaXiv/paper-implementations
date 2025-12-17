# Evaluation Results: @k Scores Summary

This document summarizes the evaluation results from the `eval_outputs` and `rlvr_eval_outputs` directories, specifically focusing on the @k score metrics.

## Model Comparison: Base vs RLVR

### Performance Comparison Table

| Dataset | Model | avg@8 | pass@8 | avg@1 | pass@1 | Improvement |
|---------|-------|-------|--------|-------|--------|-------------|
| AIME-2024 | Qwen2.5-Math-7B (Base) | 0.121 | 0.333 | - | - | - |
| AIME-2024 | Qwen2.5-Math-7B (RLVR) | 0.233 | 0.467 | - | - | **+92.6% avg@8, +40.2% pass@8** |
| AIME-2025 | Qwen2.5-Math-7B (Base) | 0.054 | 0.200 | - | - | - |
| AIME-2025 | Qwen2.5-Math-7B (RLVR) | 0.167 | 0.300 | - | - | **+209.3% avg@8, +50.0% pass@8** |
| AMC | Qwen2.5-Math-7B (Base) | 0.330 | 0.735 | - | - | - |
| AMC | Qwen2.5-Math-7B (RLVR) | 0.572 | 0.747 | - | - | **+73.3% avg@8, +1.6% pass@8** |
| MATH-500 | Qwen2.5-Math-7B (Base) | - | - | 0.494 | 0.494 | - |
| MATH-500 | Qwen2.5-Math-7B (RLVR) | - | - | 0.788 | 0.788 | **+59.5% avg@1, +59.5% pass@1** |

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