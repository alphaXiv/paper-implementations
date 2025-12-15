---
language: en
license: mit
tags:
- trm
- recursive-reasoning
- maze-solving
- pytorch
- huggingface
datasets:
- custom
metrics:
- accuracy
widget:
- text: "Sample maze input here"
---

# TRM Model for Maze Solving

## Model Description

This is a Tiny Recursive Model (TRM) fine-tuned for solving maze navigation tasks. The model implements recursive reasoning to find paths in 30x30 grid mazes.

- **Developed by:** alphaXiv
- **Model type:** TRM-Attention
- **Language(s) (NLP):** N/A (grid-based reasoning)
- **License:** MIT
- **Finetuned from model:** Custom TRM architecture

## Intended Use

### Primary Use

This model is designed to solve maze pathfinding problems by predicting the correct sequence of moves to navigate from start to goal in grid-based mazes.

### Out-of-Scope Use

Not intended for general NLP tasks, image classification, or other domains outside maze solving.

## Limitations and Bias

- Trained only on synthetic maze data
- May not generalize to mazes of different sizes or complexities
- Performance may degrade on mazes with unusual patterns

## Training Data

The model was trained on a dataset of 30x30 grid mazes with hard difficulty levels. The dataset includes:
- Start and goal positions
- Wall configurations
- Correct path sequences



## Evaluation Results

| Metric | Claimed | Achieved |
|--------|---------|----------|
| Exact Accuracy | 85.3% | 83.67% ± 2.28% |

Results from independent reproduction study.

## Repository

https://github.com/alphaXiv/TinyRecursiveModels
