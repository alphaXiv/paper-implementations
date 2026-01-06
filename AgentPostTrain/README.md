# AgentPostTrain: Post-Training LLMs with GPT 5.2

A single-script setup for using GPT 5.2 to post-train Qwen3-1.7B-Base on HumanEval.

> **Note**: This work was inspired by the [PostTrainBench](https://posttrainbench.com) repository, which measures the ability of CLI agents to perform LLM post-training.

## Quick Start

```bash
# 1. Set your API key
export OPENAI_API_KEY="your-key-here"

# 2. Run everything (setup + training + evaluation)
bash run.sh
```

That's it! The script will:
1. Install dependencies (Apptainer, etc.)
2. Build the container (if needed)
3. Download the model (if needed)
4. Run training for 10 hours
5. Evaluate the final model

Results will be in `results/YYYYMMDD_HHMMSS/`.

## What It Does

The script launches GPT 5.2 as an autonomous agent that:
- Researches effective training approaches
- Collects/generates training data  
- Fine-tunes Qwen3-1.7B-Base iteratively
- Evaluates on HumanEval during training
- Saves the best model to `results/*/final_model/`

After training completes, the script automatically runs final evaluation.

## Requirements

- Ubuntu 22.04 (or similar)
- NVIDIA H100 GPU (or compatible)
- ~20GB disk space
- OpenAI API key with GPT 5.2 access
- `sudo` access (for container build)

## Expected Results

GPT 5.2 can improve Qwen3-1.7B-Base from ~7.9% to ~18.3% accuracy on HumanEval in under 10 hours.

## Customization

Edit `run.sh` to change:
- `MODEL`: Model to train (default: `Qwen/Qwen3-1.7B-Base`)
- `HOURS`: Training time limit (default: `10`)

## Files

```
AgentPostTrain/
├── README.md              # This file
├── run.sh                 # Single script that does everything
├── prompt.txt             # Prompt template for the agent
├── containers/
│   └── standard_minimal.def  # Optimized container definition
├── agents/codex/
│   └── solve.sh           # Agent execution
└── eval/
    └── evaluate.py        # HumanEval evaluation
```

## Troubleshooting

**Container build fails**: Ensure you have `sudo` and sufficient disk space (~20GB).

**GPU not detected**: Run `nvidia-smi` to verify drivers are installed.

**API errors**: Verify `OPENAI_API_KEY` is set correctly: `echo $OPENAI_API_KEY`
